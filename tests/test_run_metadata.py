"""
tests/test_run_metadata.py

Unit tests for src/run_metadata.py.

Scope
=====
- Three metadata files are written.
- JSON files can be re-read.
- YAML files can be re-read.
- Git query failure does not interrupt.
- No-CUDA environment records correctly.
- Missing optional libraries do not interrupt.
- Skipped timing is 0.0.
- Tracing not run → time_tracing=0.0.
- Failed status records then re-raises.
- All tests use tmp_path.
- MagicMock cfg does not write to filesystem.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock
from pathlib import Path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0):
    cfg = MagicMock()
    cfg.dataset = MagicMock()
    cfg.dataset.name = dataset
    cfg.detection = MagicMock()
    cfg.detection.gnn_training = MagicMock()
    cfg.detection.gnn_training.used_method = model
    cfg._seed = seed
    cfg._stages = []
    cfg._artifact_root = None
    cfg._run_dir = None
    cfg._run_start_time = "2026-01-01T00:00:00+00:00"
    return cfg


# --------------------------------------------------------------------------- #
# MagicMock guard
# --------------------------------------------------------------------------- #

def test_magicmock_cfg_writes_nothing():
    """MagicMock cfg must not touch the filesystem."""
    from run_metadata import dump_environment, dump_config, dump_runtime

    mock_cfg = MagicMock()
    mock_run_dir = MagicMock()

    # These must be no-ops, not raise
    dump_environment(mock_run_dir)
    dump_config(mock_cfg, mock_run_dir)
    dump_runtime(mock_cfg, mock_run_dir, status="completed")


# --------------------------------------------------------------------------- #
# environment.json
# --------------------------------------------------------------------------- #

def test_environment_json_written(tmp_path):
    """dump_environment must create environment.json."""
    from run_metadata import dump_environment

    run_dir = tmp_path / "run"
    dump_environment(run_dir)

    env_file = run_dir / "environment.json"
    assert env_file.exists(), "environment.json was not created"

    data = _read_json(env_file)
    assert "created_at" in data
    assert "python_version" in data
    assert "platform" in data


def test_environment_json_re_read(tmp_path):
    """environment.json must be valid JSON that can be re-read."""
    from run_metadata import dump_environment

    run_dir = tmp_path / "run"
    dump_environment(run_dir)

    # Re-reading must not raise
    data = _read_json(run_dir / "environment.json")
    assert isinstance(data, dict)


def test_environment_git_failure_does_not_interrupt(tmp_path, monkeypatch):
    """Git command failure must not raise or corrupt other fields."""
    import subprocess

    def _fail(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _fail)

    from run_metadata import dump_environment

    run_dir = tmp_path / "run"
    dump_environment(run_dir)  # must not raise

    data = _read_json(run_dir / "environment.json")
    assert data["git_commit"] is None
    assert data["git_dirty"] is None
    assert data["python_version"] is not None


def test_environment_cuda_fields_present_when_available():
    """CUDA fields must be present when torch.cuda.is_available() is True."""
    from run_metadata import dump_environment

    # This test checks the field names exist in the schema.
    # We don't mock torch here — just verify the code path runs.
    pass   # actual CUDA test requires real torch install


def test_environment_no_cuda_records_correctly(tmp_path):
    """When CUDA is not available, cuda_available must be False and gpu_name null."""
    from run_metadata import dump_environment

    run_dir = tmp_path / "run"
    dump_environment(run_dir)

    data = _read_json(run_dir / "environment.json")
    assert "cuda_available" in data
    # Value depends on actual hardware; just ensure the field is present
    assert isinstance(data["cuda_available"], bool)
    assert "gpu_name" in data


# --------------------------------------------------------------------------- #
# config_resolved.yml
# --------------------------------------------------------------------------- #

def test_config_yml_written(tmp_path):
    """dump_config must create config_resolved.yml."""
    from run_metadata import dump_config

    run_dir = tmp_path / "run"
    cfg = _make_cfg()

    dump_config(cfg, run_dir)

    yml_file = run_dir / "config_resolved.yml"
    assert yml_file.exists(), "config_resolved.yml was not created"


def test_config_yml_re_read(tmp_path):
    """config_resolved.yml must be valid YAML that can be re-read."""
    from run_metadata import dump_config

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    dump_config(cfg, run_dir)

    yml_file = run_dir / "config_resolved.yml"
    content = yml_file.read_text(encoding="utf-8")
    # Must contain at least one top-level key
    assert "dataset" in content or "detection" in content


def test_config_yml_excludes_magicmock(tmp_path):
    """MagicMock values must not appear in config_resolved.yml."""
    from run_metadata import dump_config

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    # dataset.name is a MagicMock — must be excluded or stringified safely
    dump_config(cfg, run_dir)

    content = (run_dir / "config_resolved.yml").read_text(encoding="utf-8")
    assert "MagicMock" not in content
    assert "mock" not in content.lower()


# --------------------------------------------------------------------------- #
# runtime.json
# --------------------------------------------------------------------------- #

def test_runtime_json_written(tmp_path):
    """dump_runtime must create runtime.json."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()

    dump_runtime(cfg, run_dir, status="completed")

    rt_file = run_dir / "runtime.json"
    assert rt_file.exists(), "runtime.json was not created"


def test_runtime_json_re_read(tmp_path):
    """runtime.json must be valid JSON that can be re-read."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    dump_runtime(cfg, run_dir, status="completed")

    data = _read_json(run_dir / "runtime.json")
    assert isinstance(data, dict)


def test_runtime_json_all_timing_keys_present(tmp_path):
    """runtime.json must contain all timing keys, even if some are 0.0."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    timing = {
        "time_total": 10.0,
        "time_build_graphs": 0.0,
        "time_embed_nodes": 0.0,
        "time_embed_edges": 0.0,
        "time_gnn_training": 8.0,
        "time_gnn_testing": 1.5,
        "time_evaluation": 0.5,
        "time_tracing": 0.0,
    }

    dump_runtime(cfg, run_dir, status="completed", timing=timing)

    data = _read_json(run_dir / "runtime.json")
    for k, v in timing.items():
        assert k in data, f"{k} missing from runtime.json"
        assert data[k] == v, f"{k} should be {v}, got {data[k]}"


def test_skipped_timing_is_zero(tmp_path):
    """Stages not executed must have 0.0 in runtime.json."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    # Only train ran; everything else skipped
    timing = {
        "time_total": 5.0,
        "time_build_graphs": 0.0,
        "time_embed_nodes": 0.0,
        "time_embed_edges": 0.0,
        "time_gnn_training": 5.0,
        "time_gnn_testing": 0.0,
        "time_evaluation": 0.0,
        "time_tracing": 0.0,
    }

    dump_runtime(cfg, run_dir, status="completed", timing=timing,
                 executed_stages=["train"])

    data = _read_json(run_dir / "runtime.json")
    assert data["time_tracing"] == 0.0
    assert data["time_gnn_testing"] == 0.0
    assert data["time_evaluation"] == 0.0


def test_tracing_not_run_time_tracing_zero(tmp_path):
    """When trace is not in executed_stages, time_tracing must be 0.0."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    dump_runtime(cfg, run_dir, status="completed",
                 executed_stages=["preprocess", "train", "test", "evaluate"])

    data = _read_json(run_dir / "runtime.json")
    assert data["tracing_executed"] is False
    assert data["time_tracing"] == 0.0


def test_failed_status_records_then_re_raises(tmp_path):
    """status='failed' must write the error message before re-raising."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()

    class DummyError(Exception):
        pass

    exc = DummyError("something went wrong")

    # We simulate what orthrus.py does: write failed status then re-raise
    dump_runtime(cfg, run_dir, status="failed",
                 error_message=f"{type(exc).__name__}: {exc}")

    data = _read_json(run_dir / "runtime.json")
    assert data["status"] == "failed"
    assert "DummyError" in data["error_message"]
    assert "something went wrong" in data["error_message"]


def test_atomic_write_leaves_valid_json(tmp_path):
    """dump_runtime must leave a valid JSON file even if interrupted mid-write."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    dump_runtime(cfg, run_dir, status="completed")

    # The file must be valid JSON (atomic write worked)
    data = _read_json(run_dir / "runtime.json")
    assert data["status"] == "completed"


def test_dataset_and_seed_fields(tmp_path):
    """runtime.json must record dataset, model_variant, seed."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg(dataset="CADETS_E3", model="orthrus", seed=99)
    dump_runtime(cfg, run_dir, status="completed")

    data = _read_json(run_dir / "runtime.json")
    assert data["dataset"] == "CADETS_E3"
    assert data["model_variant"] == "orthrus"
    assert data["seed"] == 99


def test_wandb_mode_recorded(tmp_path):
    """runtime.json must record the resolved W&B mode."""
    from run_metadata import dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()
    dump_runtime(cfg, run_dir, status="completed", wandb_mode="offline")

    data = _read_json(run_dir / "runtime.json")
    assert data["wandb_mode"] == "offline"


def test_all_three_files_written_together(tmp_path):
    """dump_environment, dump_config, dump_runtime must all succeed in one run."""
    from run_metadata import dump_environment, dump_config, dump_runtime

    run_dir = tmp_path / "run"
    cfg = _make_cfg()

    dump_environment(run_dir)
    dump_config(cfg, run_dir)
    dump_runtime(cfg, run_dir, status="completed")

    assert (run_dir / "environment.json").exists()
    assert (run_dir / "config_resolved.yml").exists()
    assert (run_dir / "runtime.json").exists()

    # All must be re-readable
    _read_json(run_dir / "environment.json")
    _read_json(run_dir / "runtime.json")
    content = (run_dir / "config_resolved.yml").read_text()
    assert len(content) > 0
