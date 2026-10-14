import csv
import json
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from runtime.runtime_profiler import (
    PROFILE_ENV,
    PROFILE_SAMPLE_ENV,
    RuntimeProfiler,
    profile_requested,
    sample_every_from,
)


def test_disabled_profiler_is_artifact_free(tmp_path):
    profiler = RuntimeProfiler(enabled=False, run_root=tmp_path)
    with profiler.timer("training.total"):
        pass
    profiler.add_counter("training.events", 10)

    assert profiler.flush() is None
    assert not (tmp_path / "profiling").exists()


def test_enabled_profiler_writes_machine_readable_outputs(tmp_path):
    profiler = RuntimeProfiler(
        enabled=True,
        run_root=tmp_path,
        sample_every_n_batches=2,
        metadata={"dataset": "THEIA_E3"},
    )
    profiler.record("training.total", 2.0)
    profiler.add_counter("training.events", 20)
    profiler.add_counter("training.batches", 4)
    profiler.add_counter("training.windows", 2)

    payload = profiler.flush(status="completed")

    assert payload is not None
    assert payload["throughput"]["training.events_per_second"] == 10.0
    profile_dir = tmp_path / "profiling"
    assert json.loads((profile_dir / "perf_profile.json").read_text())["schema_version"] == 2
    assert json.loads((profile_dir / "perf_summary.json").read_text())["status"] == "completed"
    with (profile_dir / "perf_profile.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in rows] == ["training.total"]


def test_totals_include_samples_beyond_raw_retention_limit(tmp_path):
    profiler = RuntimeProfiler(enabled=True, run_root=tmp_path, raw_sample_limit=2)
    for duration in (1.0, 2.0, 3.0):
        profiler.record("training.total", duration)

    metric = profiler.payload(status="completed")["metrics"]["training.total"]

    assert metric["count"] == 3
    assert metric["retained_sample_count"] == 2
    assert metric["samples_truncated"] is True
    assert metric["total_seconds"] == 6.0
    assert metric["mean_seconds"] == 2.0


def test_batch_sampling_is_deterministic():
    profiler = RuntimeProfiler(enabled=True, sample_every_n_batches=2)
    observed = []
    for _ in range(4):
        token = profiler.begin_batch()
        observed.append(profiler.detail_active)
        profiler.end_batch(token, "training.T_total_batch")

    assert observed == [True, False, True, False]
    assert profiler.payload(status="completed")["metrics"]["training.T_total_batch"]["count"] == 4


def test_profile_batches_times_real_retrieval_and_preserves_order():
    profiler = RuntimeProfiler(enabled=True, sample_every_n_batches=2)
    observed = []
    for batch, token in profiler.profile_batches(["a", "b", "c"]):
        observed.append((batch, profiler.detail_active))
        profiler.end_batch(token, "training.T_total_batch")

    payload = profiler.payload(status="completed")
    assert observed == [("a", True), ("b", False), ("c", True)]
    assert payload["metrics"]["training.T_batch_prepare"]["count"] == 2
    assert payload["metrics"]["training.T_total_batch"]["count"] == 3

def test_profile_environment_controls(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "true")
    monkeypatch.setenv(PROFILE_SAMPLE_ENV, "7")
    assert profile_requested(False) is True
    assert sample_every_from(None) == 7

    monkeypatch.setenv(PROFILE_SAMPLE_ENV, "0")
    with pytest.raises(ValueError, match="positive"):
        sample_every_from(None)


def test_nested_cpu_timers_report_inclusive_and_self_time():
    profiler = RuntimeProfiler(enabled=True, sample_every_n_batches=1)
    with profiler.batch("training.T_total_batch"):
        with profiler.timer("training.parent", detail=True):
            with profiler.timer("training.child", detail=True):
                sum(range(100))
    metrics = profiler.payload(status="completed")["metrics"]
    assert metrics["training.parent"]["timing_domain"] == "cpu_wall"
    assert metrics["training.parent"]["total_seconds"] >= metrics["training.child"]["total_seconds"]
    assert 0 <= metrics["training.parent"]["self_total_seconds"] <= metrics["training.parent"]["total_seconds"]


def test_batch_structure_rows_and_observations_are_emitted(tmp_path):
    profiler = RuntimeProfiler(enabled=True, run_root=tmp_path, sample_every_n_batches=1)
    profiler.set_context(epoch=2, window_index=7, window_id="w7")
    token = profiler.begin_batch()
    profiler.record_batch_stat("batch_events", 512)
    profiler.record_batch_stat("short_graph_edges", 123)
    profiler.end_batch(token, "training.T_total_batch")
    payload = profiler.flush()
    rows = list(csv.DictReader((tmp_path / "profiling" / "perf_batches.csv").open()))
    assert rows[0]["window_index"] == "7"
    assert rows[0]["batch_events"] == "512"
    assert payload["observations"]["batch_events"]["median"] == 512


def test_disabled_cuda_timer_does_not_create_cuda_events(monkeypatch):
    import torch
    profiler = RuntimeProfiler(enabled=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "Event",
        lambda **kwargs: pytest.fail("disabled profiler must not create CUDA events"),
    )
    with profiler.cuda_timer("disabled.cuda"):
        pass


def test_basic_timing_is_separate_from_detailed_profiler(tmp_path, monkeypatch):
    import torch

    profiler = RuntimeProfiler(
        enabled=False,
        basic_timing=True,
        run_root=tmp_path,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: pytest.fail("basic timing must not create CUDA events"),
    )
    profiler.set_context(epoch=2, window_index=3, window_id="w3")
    for batch, token in profiler.profile_batches(["batch"]):
        profiler.record_basic_batch_stat("batch_events", 17)
        with profiler.cuda_timer("must_be_disabled"):
            pass
        profiler.end_batch(token, "training.T_total_batch")

    payload = profiler.flush()

    assert profiler.enabled is False
    assert profiler.basic_timing is True
    assert payload["detailed_profiler_enabled"] is False
    assert payload["basic_timing_enabled"] is True
    assert not (tmp_path / "profiling").exists()
    rows = list(csv.DictReader(
        (tmp_path / "benchmark_timing/basic_batches.csv").open()
    ))
    assert rows[0]["epoch"] == "2"
    assert rows[0]["window_index"] == "3"
    assert rows[0]["batch_events"] == "17"
    assert float(rows[0]["total_seconds"]) >= 0.0


def test_pure_disabled_mode_remains_artifact_free_with_basic_timing_off(tmp_path):
    profiler = RuntimeProfiler(
        enabled=False,
        basic_timing=False,
        run_root=tmp_path,
    )
    for batch, token in profiler.profile_batches(["batch"]):
        assert token is None
    assert profiler.flush() is None
    assert not (tmp_path / "benchmark_timing").exists()
    assert not (tmp_path / "profiling").exists()
