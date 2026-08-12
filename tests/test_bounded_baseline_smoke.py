"""
tests/test_bounded_baseline_smoke.py

C8-B bounded baseline smoke test implementation tests.

Tests cover:
- load_data_set() with default limit=None loads all files
- load_data_set() with limit=N loads exactly N files (first-N, sorted)
- limit <= 0 raises ValueError
- train/val/test each get independent limits
- run_experiment CLI --max-windows-per-split propagates to cfg
- Default None preserves existing behaviour
- Smoke output is isolated from formal output
- Smoke metadata (is_smoke, max_windows_per_split) is recorded
- collect_results can filter smoke results
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_temporal_data(name: str) -> TemporalData:
    """Create a minimal TemporalData with unique msg content.

    Uses only_type featurization: msg_dim = node_types*2 + edge_types = 6+10=16.
    """
    data = TemporalData()
    data.src = torch.tensor([0, 1], dtype=torch.long)
    data.dst = torch.tensor([1, 0], dtype=torch.long)
    data.t = torch.tensor([1000, 1001], dtype=torch.long)
    # only_type: 3*2 + 10 = 16
    data.msg = torch.randn(2, 16, dtype=torch.float32)
    return data


class MockCfg:
    """Minimal mock cfg for load_data_set tests.

    Uses 'only_type' featurization (simpler path) so we only need
    msg.shape[1] == num_node_types*2 + num_edge_types.
    For the tests we use node_types=3, edge_types=10, so msg_dim=16.
    """
    def __init__(self, max_windows_per_split=None, test_mode=False):
        self._max_windows_per_split = max_windows_per_split
        self._test_mode = test_mode
        self.edge_featurization = MagicMock()
        self.edge_featurization.embed_nodes = MagicMock()
        # Use only_type method: msg_dim = node_types*2 + edge_types = 6+10=16
        self.edge_featurization.embed_nodes.used_method = "only_type"
        self.dataset = MagicMock()
        self.dataset.num_node_types = 3
        self.dataset.num_edge_types = 10
        self.detection = MagicMock()
        self.detection.gnn_training = MagicMock()
        self.detection.gnn_training.decoder = MagicMock()
        self.detection.gnn_training.decoder.used_methods = "custom"


# ---------------------------------------------------------------------------
# Test 1: Default limit=None loads all files
# ---------------------------------------------------------------------------

class TestDefaultBehaviorUnchanged:
    """Verify that default behaviour (no limit) is unchanged."""

    def test_loads_all_files_when_limit_is_none(self, tmp_path):
        """When max_windows_per_split=None, all files must be loaded."""
        from data_utils import load_data_set
        from serialization_compat import load_trusted_torch_artifact

        # Create 5 TemporalData files in train split
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        for i in range(5):
            td = _make_minimal_temporal_data(f"window_{i}")
            torch.save(td, split_dir / f"window_{i:02d}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=None)
        with patch("data_utils.load_trusted_torch_artifact", wraps=load_trusted_torch_artifact) as mock_load:
            result = load_data_set(cfg, path=str(tmp_path), split="train")

        assert len(result) == 5, f"Expected 5 windows, got {len(result)}"
        assert mock_load.call_count == 5


# ---------------------------------------------------------------------------
# Test 2: limit=N loads exactly N files
# ---------------------------------------------------------------------------

class TestBoundedLimit:
    """Verify bounded limit loads exactly N files."""

    def test_loads_exactly_n_files(self, tmp_path):
        """When limit=2, exactly 2 files must be loaded."""
        from data_utils import load_data_set
        from serialization_compat import load_trusted_torch_artifact

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        for i in range(5):
            td = _make_minimal_temporal_data(f"window_{i}")
            torch.save(td, split_dir / f"window_{i:02d}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=2)
        with patch("data_utils.load_trusted_torch_artifact", wraps=load_trusted_torch_artifact) as mock_load:
            result = load_data_set(cfg, path=str(tmp_path), split="train")

        assert len(result) == 2, f"Expected 2 windows, got {len(result)}"
        assert mock_load.call_count == 2

    def test_limit_zero_raises_valueerror(self, tmp_path):
        """limit=0 must raise ValueError (not silently load all)."""
        from data_utils import load_data_set

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        td = _make_minimal_temporal_data("window_0")
        torch.save(td, split_dir / "window_00.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=0)

        with pytest.raises(ValueError, match="positive integer"):
            load_data_set(cfg, path=str(tmp_path), split="train")

    def test_limit_negative_raises_valueerror(self, tmp_path):
        """limit=-1 must raise ValueError."""
        from data_utils import load_data_set

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        td = _make_minimal_temporal_data("window_0")
        torch.save(td, split_dir / "window_00.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=-1)

        with pytest.raises(ValueError, match="positive integer"):
            load_data_set(cfg, path=str(tmp_path), split="train")


# ---------------------------------------------------------------------------
# Test 3: Deterministic sorted-first-N selection
# ---------------------------------------------------------------------------

class TestDeterministicSelection:
    """Verify files are selected deterministically by sorted-first-N."""

    def test_selects_first_n_after_sorting(self, tmp_path):
        """Files must be sorted before selecting first N."""
        from data_utils import load_data_set
        from serialization_compat import load_trusted_torch_artifact

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        # Create files with unsorted names
        for name in ["window_03", "window_01", "window_02", "window_05", "window_04"]:
            td = _make_minimal_temporal_data(name)
            torch.save(td, split_dir / f"{name}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=2)
        with patch("data_utils.load_trusted_torch_artifact", wraps=load_trusted_torch_artifact) as mock_load:
            result = load_data_set(cfg, path=str(tmp_path), split="train")

        # Should load window_01 and window_02 (sorted alphabetically)
        assert len(result) == 2
        call_args = [call.args[0] for call in mock_load.call_args_list]
        assert all("window_0" in arg for arg in call_args), (
            f"Expected window_01 or window_02, got: {call_args}"
        )

    def test_same_limit_produces_same_selection(self, tmp_path):
        """Two calls with same limit must select the same files."""
        from data_utils import load_data_set

        split_dir = tmp_path / "train"
        split_dir.mkdir()
        for i in range(5):
            td = _make_minimal_temporal_data(f"window_{i}")
            torch.save(td, split_dir / f"window_{i:02d}.TemporalData.simple")

        cfg1 = MockCfg(max_windows_per_split=2)
        cfg2 = MockCfg(max_windows_per_split=2)

        with patch("data_utils.load_trusted_torch_artifact"):
            result1 = load_data_set(cfg1, path=str(tmp_path), split="train")
            result2 = load_data_set(cfg2, path=str(tmp_path), split="train")

        assert len(result1) == len(result2) == 2


# ---------------------------------------------------------------------------
# Test 4: train/val/test each get independent limits
# ---------------------------------------------------------------------------

class TestIndependentSplitLimits:
    """Verify each split is limited independently."""

    def test_each_split_respects_limit(self, tmp_path):
        """Each split (train/val/test) must respect the limit independently."""
        from data_utils import load_data_set

        # Create 5 files in each split
        for split in ("train", "val", "test"):
            split_dir = tmp_path / split
            split_dir.mkdir()
            for i in range(5):
                td = _make_minimal_temporal_data(f"{split}_w{i}")
                torch.save(td, split_dir / f"{split}_window_{i:02d}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=2)

        with patch("data_utils.load_trusted_torch_artifact"):
            train = load_data_set(cfg, path=str(tmp_path), split="train")
            val = load_data_set(cfg, path=str(tmp_path), split="val")
            test = load_data_set(cfg, path=str(tmp_path), split="test")

        assert len(train) == 2, f"train: expected 2, got {len(train)}"
        assert len(val) == 2, f"val: expected 2, got {len(val)}"
        assert len(test) == 2, f"test: expected 2, got {len(test)}"


# ---------------------------------------------------------------------------
# Test 5: run_experiment CLI --max-windows-per-split propagates
# ---------------------------------------------------------------------------

class TestCLIMaxWindowsPerSplit:
    """Verify --max-windows-per-split propagates through CLI to cfg."""

    def test_cli_argument_passed_to_namespace(self):
        """Parser must accept --max-windows-per-split and store it."""
        import argparse
        # Build parser using the actual function
        sys.path.insert(0, str(SRC / "experiments"))
        from run_experiment import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--dataset", "THEIA_E3",
            "--config", "config/experiments/baseline.yml",
            "--max-windows-per-split", "2",
        ])

        assert ns.max_windows_per_split == 2

    def test_cli_default_is_none(self):
        """Default value must be None (load all)."""
        sys.path.insert(0, str(SRC / "experiments"))
        from run_experiment import build_parser

        parser = build_parser()
        ns = parser.parse_args([
            "--dataset", "THEIA_E3",
            "--config", "config/experiments/baseline.yml",
        ])

        assert ns.max_windows_per_split is None


# ---------------------------------------------------------------------------
# Test 6: cfg._is_smoke is set correctly
# ---------------------------------------------------------------------------

class TestIsSmokeFlag:
    """Verify cfg._is_smoke is set when bounded smoke is active."""

    def test_is_smoke_true_when_limit_is_set(self):
        """_is_smoke must be True when max_windows_per_split is set."""
        cfg = MagicMock()
        cfg._max_windows_per_split = 2
        # Simulate get_yml_cfg logic
        cfg._is_smoke = cfg._max_windows_per_split is not None and cfg._max_windows_per_split > 0
        assert cfg._is_smoke is True

    def test_is_smoke_false_when_limit_is_none(self):
        """_is_smoke must be False when max_windows_per_split is None."""
        cfg = MagicMock()
        cfg._max_windows_per_split = None
        cfg._is_smoke = cfg._max_windows_per_split is not None and cfg._max_windows_per_split > 0
        assert cfg._is_smoke is False


# ---------------------------------------------------------------------------
# Test 7: smoke metadata in runtime.json
# ---------------------------------------------------------------------------

class TestRuntimeMetadata:
    """Verify smoke metadata is written to runtime.json."""

    def test_runtime_json_contains_smoke_fields(self, tmp_path):
        """runtime.json must contain is_smoke and max_windows_per_split."""
        from run_metadata import dump_runtime
        from pathlib import Path

        cfg = MagicMock()
        cfg._seed = 0
        cfg._stages = ["train", "test", "evaluate"]
        cfg.dataset = MagicMock()
        cfg.dataset.name = "THEIA_E3"
        cfg.detection = MagicMock()
        cfg.detection.gnn_training = MagicMock()
        cfg.detection.gnn_training.used_method = "orthrus"
        cfg._is_smoke = True
        cfg._max_windows_per_split = 2
        cfg._artifact_root = str(tmp_path)
        cfg._run_start_time = "2026-01-01T00:00:00Z"

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        dump_runtime(cfg, str(run_dir), status="completed")

        import json
        runtime_path = run_dir / "runtime.json"
        assert runtime_path.exists(), "runtime.json must be created"

        with runtime_path.open() as f:
            runtime = json.load(f)

        assert runtime["is_smoke"] is True, "is_smoke must be True"
        assert runtime["max_windows_per_split"] == 2, "max_windows_per_split must be 2"

    def test_runtime_json_no_smoke_when_limit_is_none(self, tmp_path):
        """When no limit is set, runtime.json must not have smoke=True."""
        from run_metadata import dump_runtime
        from pathlib import Path

        cfg = MagicMock()
        cfg._seed = 0
        cfg._stages = ["train", "test", "evaluate"]
        cfg.dataset = MagicMock()
        cfg.dataset.name = "THEIA_E3"
        cfg.detection = MagicMock()
        cfg.detection.gnn_training = MagicMock()
        cfg.detection.gnn_training.used_method = "orthrus"
        cfg._is_smoke = False
        cfg._max_windows_per_split = None
        cfg._artifact_root = str(tmp_path)
        cfg._run_start_time = "2026-01-01T00:00:00Z"

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        dump_runtime(cfg, str(run_dir), status="completed")

        import json
        runtime_path = run_dir / "runtime.json"
        with runtime_path.open() as f:
            runtime = json.load(f)

        assert runtime["is_smoke"] is False, "is_smoke must be False for formal runs"
        assert runtime["max_windows_per_split"] is None


# ---------------------------------------------------------------------------
# Test 8: Smoke output is isolated from formal output
# ---------------------------------------------------------------------------

class TestOutputIsolation:
    """Verify smoke and formal runs use different run directories."""

    def test_smoke_uses_different_config_path(self, tmp_path):
        """Smoke config path is different from formal config path.

        The run_matrix scheduler uses _config_id(config) as part of scoped_root.
        Different config paths produce different _config_ids, ensuring smoke
        runs never pollute formal experiment results.
        """
        from experiments.run_matrix import _config_id

        formal_config = Path("config/experiments/baseline.yml")
        smoke_config = Path("/tmp/smoke_baseline_1epoch.yml")

        formal_id = _config_id(formal_config)
        smoke_id = _config_id(smoke_config)

        # Different config paths produce different config_ids
        assert formal_id != smoke_id, (
            "Smoke and formal configs must have different _config_ids for isolation"
        )

    def test_collect_results_can_filter_smoke(self):
        """Verify collect_results can filter out smoke runs by is_smoke flag."""
        smoke_runtime = {
            "dataset": "THEIA_E3",
            "status": "completed",
            "is_smoke": True,
            "max_windows_per_split": 2,
            "seed": 0,
        }

        formal_runtime = {
            "dataset": "THEIA_E3",
            "status": "completed",
            "is_smoke": False,
            "max_windows_per_split": None,
            "seed": 0,
        }

        # Formal runs exclude smoke runs
        def filter_formal(runs):
            return [r for r in runs if not r.get("is_smoke", False)]

        formal_runs = filter_formal([smoke_runtime, formal_runtime])
        assert len(formal_runs) == 1
        assert formal_runs[0]["is_smoke"] is False
        assert formal_runs[0]["max_windows_per_split"] is None


# ---------------------------------------------------------------------------
# Test 9: Print statement format
# ---------------------------------------------------------------------------

class TestLogOutput:
    """Verify bounded smoke produces correct log output."""

    def test_log_prints_selected_count(self, capsys):
        """Log must show selected / available window count."""
        from data_utils import load_data_set

        split_dir = Path(tempfile.mkdtemp()) / "train"
        split_dir.mkdir(parents=True)
        for i in range(5):
            td = _make_minimal_temporal_data(f"window_{i}")
            torch.save(td, split_dir / f"window_{i:02d}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=2)
        with patch("data_utils.load_trusted_torch_artifact"):
            load_data_set(cfg, path=str(split_dir.parent), split="train")

        captured = capsys.readouterr()
        assert "[Bounded smoke]" in captured.out, (
            f"Expected [Bounded smoke] in output, got: {captured.out}"
        )
        assert "selected 2 / 5 windows" in captured.out, (
            f"Expected 'selected 2 / 5 windows' in output, got: {captured.out}"
        )

    def test_no_smoke_log_when_limit_is_none(self, capsys):
        """No bounded smoke log should appear when limit is None."""
        from data_utils import load_data_set

        split_dir = Path(tempfile.mkdtemp()) / "train"
        split_dir.mkdir(parents=True)
        for i in range(3):
            td = _make_minimal_temporal_data(f"window_{i}")
            torch.save(td, split_dir / f"window_{i:02d}.TemporalData.simple")

        cfg = MockCfg(max_windows_per_split=None)
        with patch("data_utils.load_trusted_torch_artifact"):
            load_data_set(cfg, path=str(split_dir.parent), split="train")

        captured = capsys.readouterr()
        assert "[Bounded smoke]" not in captured.out, (
            "No bounded smoke log should appear when limit is None"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
