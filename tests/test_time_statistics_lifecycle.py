"""Lifecycle tests for run-scoped time statistics isolation."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detection import orthrus_gnn_testing, orthrus_gnn_training
from factory import requires_time_gap_statistics
from mstc.time_gap import TimeGapStatistics


def _cfg(metadata_dir, *, mode="full_pipeline", time_enabled=True, multiscale=False, run_dir=None):
    context = SimpleNamespace(
        mode="multiscale" if multiscale else "recent",
        multiscale=SimpleNamespace(enabled=multiscale),
    )
    return SimpleNamespace(
        _metadata_dir=str(metadata_dir),
        _run_dir=run_dir,
        pipeline=SimpleNamespace(mode=mode),
        model=SimpleNamespace(variant="mstc"),
        detection=SimpleNamespace(
            gnn_training=SimpleNamespace(
                decoder=SimpleNamespace(
                    time_gap=SimpleNamespace(enabled=time_enabled)
                ),
                encoder=SimpleNamespace(context=context),
            )
        ),
    )


def _train_data(offset=0):
    return [TemporalData(
        src=torch.tensor([0, 0, 0]),
        dst=torch.tensor([1, 1, 1]),
        t=torch.tensor([offset, offset + 10, offset + 30]),
        msg=torch.zeros(3, 1),
    )]


def test_training_fits_train_only_and_saves_time_statistics(tmp_path):
    cfg = _cfg(tmp_path / "metadata")
    train_data = _train_data()

    fitted = orthrus_gnn_training._fit_and_save_time_gap_statistics(train_data, cfg)
    path = tmp_path / "metadata" / "time_statistics.json"

    assert path.exists()
    loaded = TimeGapStatistics.load(str(path))
    assert loaded.time_bucket_boundaries == fitted.time_bucket_boundaries
    assert loaded.scale_boundaries_seconds == fitted.scale_boundaries_seconds


def test_testing_loads_existing_artifact_without_refitting(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path / "metadata")
    path = tmp_path / "metadata" / "time_statistics.json"
    persisted = TimeGapStatistics().fit(_train_data())
    persisted.save(str(path))

    def fail_fit(_train_data):
        raise AssertionError("testing must load existing statistics, not refit")

    monkeypatch.setattr(orthrus_gnn_testing, "fit_time_gap_statistics", fail_fit)
    loaded = orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg, _train_data(100))

    assert loaded.time_bucket_boundaries == persisted.time_bucket_boundaries
    assert loaded.scale_boundaries_seconds == persisted.scale_boundaries_seconds


def test_detection_only_missing_artifact_fails_clearly(tmp_path):
    cfg = _cfg(tmp_path / "missing", mode="detection_only")

    with pytest.raises(FileNotFoundError, match="detection_only.*time_statistics.json"):
        orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg, _train_data())


def test_full_pipeline_fallback_passes_only_train_data_to_fit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path / "missing", mode="full_pipeline")
    train_data = _train_data()
    validation_data = _train_data(1_000)
    test_data = _train_data(2_000)
    marker = object()
    captured = []

    def record_fit(data):
        captured.append(data)
        return marker

    monkeypatch.setattr(orthrus_gnn_testing, "fit_time_gap_statistics", record_fit)
    result = orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg, train_data)

    assert result is marker
    assert captured == [train_data]
    assert all(item is not validation_data for item in captured)
    assert all(item is not test_data for item in captured)


def test_multiscale_context_requires_the_same_statistics_without_time_decoder(tmp_path):
    cfg = _cfg(tmp_path / "metadata", time_enabled=False, multiscale=True)

    assert requires_time_gap_statistics(cfg) is True


def test_training_saves_to_run_dir_metadata_when_run_dir_is_set(tmp_path, monkeypatch):
    """Each run persists and reloads only its own training statistics."""
    shared_metadata = tmp_path / "shared_preprocessing"
    run_a = tmp_path / "run_A"
    run_b = tmp_path / "run_B"

    cfg_a = _cfg(shared_metadata, run_dir=run_a)
    cfg_b = _cfg(shared_metadata, run_dir=run_b)

    # These are genuinely different gap distributions, not absolute-time
    # offsets: A has gaps [10, 10, 10], while B has [100, 200, 300].
    train_data_a = [TemporalData(
        src=torch.tensor([0, 0, 0, 0]),
        dst=torch.tensor([1, 1, 1, 1]),
        t=torch.tensor([0, 10, 20, 30]),
        msg=torch.zeros(4, 1),
    )]
    train_data_b = [TemporalData(
        src=torch.tensor([0, 0, 0, 0]),
        dst=torch.tensor([1, 1, 1, 1]),
        t=torch.tensor([0, 100, 300, 600]),
        msg=torch.zeros(4, 1),
    )]

    fitted_a = orthrus_gnn_training._fit_and_save_time_gap_statistics(train_data_a, cfg_a)
    path_a = run_a / "metadata" / "time_statistics.json"
    path_b = run_b / "metadata" / "time_statistics.json"
    assert path_a.exists()
    saved_a = path_a.read_text()

    fitted_b = orthrus_gnn_training._fit_and_save_time_gap_statistics(train_data_b, cfg_b)
    assert path_b.exists()
    assert path_a != path_b
    assert path_a.read_text() == saved_a

    def fail_fit(_train_data):
        raise AssertionError("run-scoped artifacts must be loaded, not refitted")

    monkeypatch.setattr(orthrus_gnn_testing, "fit_time_gap_statistics", fail_fit)
    loaded_a = orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg_a, train_data_a)
    loaded_b = orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg_b, train_data_b)

    assert loaded_a.time_bucket_boundaries == fitted_a.time_bucket_boundaries
    assert loaded_a.scale_boundaries_seconds == fitted_a.scale_boundaries_seconds
    assert loaded_b.time_bucket_boundaries == fitted_b.time_bucket_boundaries
    assert loaded_b.scale_boundaries_seconds == fitted_b.scale_boundaries_seconds
    assert loaded_a.time_bucket_boundaries != loaded_b.time_bucket_boundaries
    assert loaded_a.scale_boundaries_seconds != loaded_b.scale_boundaries_seconds
    assert not (shared_metadata / "time_statistics.json").exists()

def test_testing_loads_from_run_dir_metadata_when_run_dir_is_set(tmp_path, monkeypatch):
    """Verify testing loads time_statistics.json from run_dir/metadata/ when _run_dir is set."""
    shared_metadata = tmp_path / "shared_preprocessing"
    run_a = tmp_path / "run_a"

    cfg_a = _cfg(shared_metadata, run_dir=run_a)

    train_data_a = _train_data(offset=0)
    fitted_a = orthrus_gnn_training._fit_and_save_time_gap_statistics(train_data_a, cfg_a)

    # Ensure loading succeeds from run-scoped path
    def fail_fit(_train_data):
        raise AssertionError("testing must load existing statistics, not refit")

    monkeypatch.setattr(orthrus_gnn_testing, "fit_time_gap_statistics", fail_fit)

    # Use a fresh cfg with same run_dir
    cfg_load = _cfg(shared_metadata, run_dir=run_a)
    loaded = orthrus_gnn_testing._load_or_fit_time_gap_statistics(cfg_load, _train_data(2000))

    assert loaded.time_bucket_boundaries == fitted_a.time_bucket_boundaries
    assert loaded.scale_boundaries_seconds == fitted_a.scale_boundaries_seconds


def test_run_dir_takes_priority_over_shared_metadata_dir(tmp_path):
    """When both _run_dir and _metadata_dir are set, _run_dir wins."""
    shared_metadata = tmp_path / "shared_preprocessing"
    run_dir = tmp_path / "run_dir"

    cfg = _cfg(shared_metadata, run_dir=run_dir)

    train_data = _train_data()
    orthrus_gnn_training._fit_and_save_time_gap_statistics(train_data, cfg)

    # Must be saved to run_dir/metadata/, not shared_metadata/
    assert (run_dir / "metadata" / "time_statistics.json").exists()
    assert not (shared_metadata / "time_statistics.json").exists()


@pytest.mark.parametrize(
    "resolver",
    [
        orthrus_gnn_training._run_scoped_metadata_dir,
        orthrus_gnn_testing._run_scoped_metadata_dir,
    ],
)
def test_metadata_dir_resolvers_accept_real_string_and_path(resolver, tmp_path):
    path_value = tmp_path / "shared_metadata"

    assert resolver(SimpleNamespace(_run_dir=None, _metadata_dir=str(path_value))) == str(path_value)
    assert resolver(SimpleNamespace(_run_dir=None, _metadata_dir=path_value)) == path_value


@pytest.mark.parametrize(
    "resolver",
    [
        orthrus_gnn_training._run_scoped_metadata_dir,
        orthrus_gnn_testing._run_scoped_metadata_dir,
    ],
)
@pytest.mark.parametrize("invalid_path", [MagicMock(), None, ""])
def test_metadata_dir_resolvers_reject_invalid_paths(resolver, invalid_path):
    cfg = SimpleNamespace(_run_dir=invalid_path, _metadata_dir=invalid_path)

    assert resolver(cfg) is None
