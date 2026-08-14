"""
tests/test_loader_telemetry_persistence.py

Tests for C8 loader telemetry persistence across pipeline stages.

Tests cover:
1. update_runtime_nested() creates nested dataset_loader schema
2. training and testing telemetry are separate (not overwritten)
3. dump_runtime() preserves dataset_loader section
4. Failed runs retain telemetry written before failure
5. Pipeline authoritative fields (status, timing) override previous values
6. JSON serializability of all telemetry fields
7. RSS phase metadata retention
8. Sidecar manifest hit tracking
9. Source/fallback counters retention
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0):
    """Create a minimal mock cfg for runtime tests."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg._stages = ["preprocess", "train", "test", "evaluate"]
    cfg._seed = seed
    cfg.dataset = MagicMock()
    cfg.dataset.name = dataset
    cfg.detection = MagicMock()
    cfg.detection.gnn_training = MagicMock()
    cfg.detection.gnn_training.used_method = model
    cfg._artifact_root = "/test/artifacts"
    cfg._run_start_time = "2026-01-01T00:00:00+00:00"
    cfg._is_smoke = False
    cfg._max_windows_per_split = None
    return cfg


def _make_sample_loader_telemetry(phase: str) -> dict:
    """Create sample loader telemetry data mimicking real BoundedFullData telemetry."""
    return {
        "architecture": {
            "schema_version": "v4" if phase == "testing" else "v3",
            "load_strategy": "sidecar",
            "source_type": "temporal_data",
        },
        "history_access": {
            "architecture": "compact_index",
            "sidecar_manifest_hit": True,
            "persistent_cache_hit": False,
            "persistent_cache_hit_count": 0,
            "sidecar_tensor_load_count": 42,
            "metadata_source_load_count": 1,
            "compact_build_source_load_count": 0,
            "total_source_artifact_load_count": 1,
            "warm_source_temporaldata_load_count": 0,
            "cold_source_temporaldata_load_count": 1,
            "fallback_full_window_load_count": 0,
            "fallback_full_window_load_bytes_estimate": 0,
        },
        "num_src_active_nodes": 6595 if phase == "training" else 6596,
        "num_dst_active_nodes": 3881 if phase == "training" else 3882,
        "node_feature_storage_bytes": 5489424,
        "node_id_index_bytes": 83808,
        "history_lookup_calls": 150,
        "history_lookup_events": 300,
        "compact_lookup_count": 75,
        "node_lookup_count": 50,
        "msg_lookup_count": 25,
        "rss_mb_by_phase": {
            "before_dataset_loading": 50.0,
            "after_train_loading": 800.0,
            "after_full_data": 1290.0,
            "after_model_construction": 1350.0,
        },
        "peak_rss_mb_by_phase": {
            "loader_phase": 1290.0,
            "testing_phase": 2195.0,
        },
        "dataset_loader_peak_rss_mb": 1290.0,
    }


# ---------------------------------------------------------------------------
# Tests for update_runtime_nested
# ---------------------------------------------------------------------------

class TestUpdateRuntimeNested:
    """Tests for the update_runtime_nested function."""

    def test_creates_nested_structure(self, tmp_path):
        """update_runtime_nested must create dataset_loader.training structure."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        assert runtime_path.exists()

        with runtime_path.open() as f:
            runtime = json.load(f)

        assert "dataset_loader" in runtime
        assert "training" in runtime["dataset_loader"]
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 6595

    def test_training_and_testing_separate(self, tmp_path):
        """Training and testing telemetry must not overwrite each other."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        train_telemetry = _make_sample_loader_telemetry("training")
        test_telemetry = _make_sample_loader_telemetry("testing")

        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)
        update_runtime_nested(run_dir, "dataset_loader", "testing", test_telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        assert "training" in runtime["dataset_loader"]
        assert "testing" in runtime["dataset_loader"]
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 6595
        assert runtime["dataset_loader"]["testing"]["num_src_active_nodes"] == 6596

    def test_testing_does_not_overwrite_training(self, tmp_path):
        """Writing testing must not delete training data."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        train_telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)

        # Simulate testing phase writing after training
        test_telemetry = _make_sample_loader_telemetry("testing")
        update_runtime_nested(run_dir, "dataset_loader", "testing", test_telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Both must exist
        assert "training" in runtime["dataset_loader"]
        assert "testing" in runtime["dataset_loader"]
        # Training data preserved
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 6595

    def test_merge_updates_existing_subsection(self, tmp_path):
        """Update must merge with existing subsection data, not replace entirely."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        initial = {"num_src_active_nodes": 100, "other_field": "initial"}
        update_runtime_nested(run_dir, "dataset_loader", "training", initial)

        update = {"num_src_active_nodes": 200, "new_field": "added"}
        update_runtime_nested(run_dir, "dataset_loader", "training", update)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Latest value wins for same keys
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 200
        # But other_field from initial write is preserved
        assert runtime["dataset_loader"]["training"]["other_field"] == "initial"
        assert runtime["dataset_loader"]["training"]["new_field"] == "added"

    def test_atomic_write_creates_valid_json(self, tmp_path):
        """Atomic write must leave valid JSON even on interruption simulation."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        # Must be valid JSON
        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)
        assert isinstance(runtime, dict)

    def test_preserves_other_sections(self, tmp_path):
        """Updating dataset_loader must not delete other runtime sections."""
        from mstc.experiment_utils import update_runtime_nested, update_runtime
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # First write some other sections via dump_runtime
        cfg = _make_cfg()
        timing = {"time_total": 100.0, "time_gnn_training": 80.0}
        dump_runtime(cfg, run_dir, status="completed", timing=timing)

        # Add a training section (as orthrus_gnn_training.py does)
        update_runtime(run_dir, "training", {"train_seconds_per_epoch": [10.0]})

        # Then write loader telemetry
        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Both sections must exist
        assert "dataset_loader" in runtime
        assert "training" in runtime
        assert runtime["training"]["train_seconds_per_epoch"] == [10.0]

    def test_history_access_fields_preserved(self, tmp_path):
        """history_access nested fields must be fully preserved."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        ha = runtime["dataset_loader"]["training"]["history_access"]
        assert ha["sidecar_manifest_hit"] is True
        assert ha["sidecar_tensor_load_count"] == 42
        assert ha["metadata_source_load_count"] == 1
        assert ha["compact_build_source_load_count"] == 0
        assert ha["fallback_full_window_load_count"] == 0


# ---------------------------------------------------------------------------
# Tests for dump_runtime preservation
# ---------------------------------------------------------------------------

class TestDumpRuntimePreservation:
    """Tests for dump_runtime preserving dataset_loader section."""

    def test_preserves_dataset_loader_after_shutdown(self, tmp_path):
        """dump_runtime must preserve dataset_loader section at shutdown."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Simulate loader telemetry written during pipeline execution
        train_telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)

        test_telemetry = _make_sample_loader_telemetry("testing")
        update_runtime_nested(run_dir, "dataset_loader", "testing", test_telemetry)

        # Simulate pipeline shutdown (dump_runtime call)
        cfg = _make_cfg()
        dump_runtime(cfg, run_dir, status="completed")

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # dataset_loader must still be present after shutdown
        assert "dataset_loader" in runtime
        assert "training" in runtime["dataset_loader"]
        assert "testing" in runtime["dataset_loader"]

    def test_preserves_training_and_testing_telemetry(self, tmp_path):
        """Both training and testing telemetry must survive dump_runtime."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        train_telemetry = _make_sample_loader_telemetry("training")
        test_telemetry = _make_sample_loader_telemetry("testing")

        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)
        update_runtime_nested(run_dir, "dataset_loader", "testing", test_telemetry)

        cfg = _make_cfg()
        dump_runtime(cfg, run_dir, status="completed")

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Both survive
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 6595
        assert runtime["dataset_loader"]["testing"]["num_src_active_nodes"] == 6596

    def test_pipeline_status_not_overwritten(self, tmp_path):
        """Pipeline authoritative status/timing must not be overwritten by previous runtime."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write an old runtime with "completed" status
        old_runtime = {
            "status": "completed",
            "time_total": 999.0,
            "start_time": "2025-01-01T00:00:00+00:00",
        }
        runtime_path = run_dir / "runtime.json"
        runtime_path.write_text(json.dumps(old_runtime))

        # Simulate current pipeline
        cfg = _make_cfg()
        timing = {"time_total": 100.0, "time_gnn_training": 80.0}
        dump_runtime(cfg, run_dir, status="completed", timing=timing)

        with runtime_path.open() as f:
            runtime = json.load(f)

        # Authoritative values must be current, not old
        assert runtime["status"] == "completed"
        assert runtime["time_total"] == 100.0  # Current pipeline value, not 999.0

    def test_failed_run_preserves_telemetry(self, tmp_path):
        """Failed run must preserve telemetry that was written before failure."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write training loader telemetry before failure
        train_telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)

        # Simulate failure and shutdown
        cfg = _make_cfg()
        dump_runtime(cfg, run_dir, status="failed", error_message="OOM")

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Telemetry preserved despite failure
        assert runtime["status"] == "failed"
        assert "dataset_loader" in runtime
        assert runtime["dataset_loader"]["training"]["num_src_active_nodes"] == 6595

    def test_rss_phases_preserved(self, tmp_path):
        """RSS phase telemetry must be preserved."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        assert "rss_mb_by_phase" in runtime["dataset_loader"]["training"]
        assert "peak_rss_mb_by_phase" in runtime["dataset_loader"]["training"]
        assert runtime["dataset_loader"]["training"]["dataset_loader_peak_rss_mb"] == 1290.0

    def test_source_fallback_counters_preserved(self, tmp_path):
        """Source load and fallback counters must be preserved."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        ha = runtime["dataset_loader"]["training"]["history_access"]
        assert ha["metadata_source_load_count"] == 1
        assert ha["total_source_artifact_load_count"] == 1
        assert ha["warm_source_temporaldata_load_count"] == 0
        assert ha["cold_source_temporaldata_load_count"] == 1
        assert ha["fallback_full_window_load_count"] == 0


# ---------------------------------------------------------------------------
# Tests for JSON serializability
# ---------------------------------------------------------------------------

class TestJsonSerializability:
    """Tests for JSON serializability of telemetry data."""

    def test_all_telemetry_fields_json_serializable(self, tmp_path):
        """All telemetry fields must be JSON-serializable."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Must not raise
        json_str = json.dumps(runtime)
        assert isinstance(json_str, str)

    def test_complex_nested_structure_serializable(self, tmp_path):
        """Deeply nested telemetry structures must serialize correctly."""
        from mstc.experiment_utils import update_runtime_nested

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        complex_telemetry = {
            "architecture": {"version": "v4", "features": ["a", "b"]},
            "history_access": {
                "nested": {
                    "deep": {
                        "value": 42,
                    }
                }
            },
            "list_field": [1, 2, 3],
            "nested_arrays": [{"a": 1}, {"b": 2}],
        }
        update_runtime_nested(run_dir, "dataset_loader", "training", complex_telemetry)

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        assert runtime["dataset_loader"]["training"]["architecture"]["version"] == "v4"
        assert runtime["dataset_loader"]["training"]["history_access"]["nested"]["deep"]["value"] == 42


# ---------------------------------------------------------------------------
# Tests for existing sections preservation
# ---------------------------------------------------------------------------

class TestExistingSectionsPreservation:
    """Tests for preserving existing runtime sections."""

    def test_training_testing_model_sections_preserved(self, tmp_path):
        """Existing training, testing, model sections must still be preserved."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write training/testing/model sections via dump_runtime
        cfg = _make_cfg()
        timing = {"time_total": 100.0, "time_gnn_training": 80.0}
        dump_runtime(cfg, run_dir, status="completed", timing=timing)

        # Manually add training and model sections (as orthrus_gnn_training.py does)
        from mstc.experiment_utils import update_runtime
        update_runtime(run_dir, "training", {"train_seconds_per_epoch": [10.0]})
        update_runtime(run_dir, "model", {"parameter_count": 1000000})

        # Write loader telemetry
        train_telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", train_telemetry)

        # Final shutdown
        dump_runtime(cfg, run_dir, status="completed")

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # All sections must exist
        assert "dataset_loader" in runtime
        assert "training" in runtime
        assert "model" in runtime
        assert runtime["training"]["train_seconds_per_epoch"] == [10.0]
        assert runtime["model"]["parameter_count"] == 1000000


# ---------------------------------------------------------------------------
# Tests for smoke metadata
# ---------------------------------------------------------------------------

class TestSmokeMetadata:
    """Tests for is_smoke/max_windows_per_split preservation."""

    def test_smoke_metadata_preserved_with_loader_telemetry(self, tmp_path):
        """Smoke metadata must be preserved alongside loader telemetry."""
        from mstc.experiment_utils import update_runtime_nested
        from run_metadata import dump_runtime

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        # Write loader telemetry
        telemetry = _make_sample_loader_telemetry("training")
        update_runtime_nested(run_dir, "dataset_loader", "training", telemetry)

        # Write smoke runtime
        cfg = _make_cfg()
        cfg._is_smoke = True
        cfg._max_windows_per_split = 2
        dump_runtime(cfg, run_dir, status="completed")

        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        # Both smoke metadata and loader telemetry preserved
        assert runtime["is_smoke"] is True
        assert runtime["max_windows_per_split"] == 2
        assert "dataset_loader" in runtime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
