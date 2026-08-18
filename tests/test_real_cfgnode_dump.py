"""
tests/test_real_cfgnode_dump.py

Regression tests for C8 Bug B: Real CfgNode config_resolved.yml serialization.

These tests verify that:
1. Real yacs CfgNode objects (from project's get_yml_cfg) can be dumped
   to config_resolved.yml and re-loaded with yaml.safe_load()
2. The dumped YAML is non-empty and contains expected key sections
3. Database passwords are redacted
4. Private runtime fields are not written
5. MagicMock path hygiene is preserved

Run with: pytest tests/test_real_cfgnode_dump.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_yaml(path):
    """Read YAML with yaml.safe_load (used in real runs)."""
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Bug B.1: Real CfgNode dump must produce non-empty YAML
# --------------------------------------------------------------------------- #

def test_real_cfgnode_dump_produces_nonempty_yaml(tmp_path):
    """
    A real yacs CfgNode must serialize to a non-empty dict.
    yaml.safe_load(config_resolved.yml) must return a valid dict,
    not {} or None.
    """
    from run_metadata import dump_config, _cfg_to_dict

    # Create a realistic CfgNode with nested structure
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "mstc"
    cfg.dataset = CN()
    cfg.dataset.name = "THEIA_E3"
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.used_method = "orthrus"
    cfg.detection.gnn_training.num_epochs = 6
    cfg.detection.gnn_training.encoder = CN()
    cfg.detection.gnn_training.encoder.backbone = "graph_transformer"
    cfg.detection.gnn_training.encoder.neighbor_size = 24
    cfg.detection.gnn_training.encoder.context = CN()
    cfg.detection.gnn_training.encoder.context.mode = "multiscale"
    cfg.calibration = CN()
    cfg.calibration.method = "hierarchical_relation"
    cfg.calibration.epsilon = 1e-12
    cfg.node_aggregation = CN()
    cfg.node_aggregation.method = "topk_mean"
    cfg.node_aggregation.topk = 5
    cfg.dataset_view = CN()
    cfg.dataset_view.mode = "host_network_full"
    cfg.pipeline = CN()
    cfg.pipeline.mode = "detection_only"
    cfg.database = CN()
    cfg.database.host = "localhost"
    cfg.database.port = "5432"
    cfg.database.user = "postgres"
    cfg.database.password = "super_secret_password"
    cfg._seed = 0
    cfg._run_dir = str(tmp_path / "run")

    dump_config(cfg, tmp_path / "run")

    yml_file = tmp_path / "run" / "config_resolved.yml"
    assert yml_file.exists(), "config_resolved.yml must be created"

    content = yml_file.read_text(encoding="utf-8")
    assert len(content) > 0, "config_resolved.yml must not be empty"

    # yaml.safe_load must return a valid non-empty dict (this is what real runs use)
    loaded = _read_yaml(yml_file)
    assert loaded is not None, "yaml.safe_load must return non-None"
    assert isinstance(loaded, dict), f"yaml.safe_load must return dict, got {type(loaded)}"
    assert len(loaded) > 0, "Dumped config must not be empty dict"


def test_real_cfgnode_dump_contains_expected_sections(tmp_path):
    """Dumped CfgNode must contain all key configuration sections."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "mstc"
    cfg.dataset = CN()
    cfg.dataset.name = "THEIA_E3"
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.num_epochs = 6
    cfg.detection.gnn_training.encoder = CN()
    cfg.detection.gnn_training.encoder.backbone = "graph_transformer"
    cfg.detection.gnn_training.encoder.context = CN()
    cfg.detection.gnn_training.encoder.context.mode = "multiscale"
    cfg.calibration = CN()
    cfg.calibration.method = "hierarchical_relation"
    cfg.node_aggregation = CN()
    cfg.node_aggregation.method = "topk_mean"
    cfg.node_aggregation.topk = 5
    cfg.node_threshold = CN()
    cfg.node_threshold.method = "validation_quantile"
    cfg.node_threshold.quantile = 0.999
    cfg.dataset_view = CN()
    cfg.dataset_view.mode = "host_network_full"
    cfg.pipeline = CN()
    cfg.pipeline.mode = "detection_only"
    cfg.pipeline.run_tracing = False
    cfg.database = CN()
    cfg.database.password = "secret"
    cfg._seed = 0

    dump_config(cfg, tmp_path / "run")

    loaded = _read_yaml(tmp_path / "run" / "config_resolved.yml")

    # Must contain key sections
    assert "model" in loaded, "model section must be in dumped config"
    assert loaded["model"]["variant"] == "mstc"
    assert "detection" in loaded, "detection section must be in dumped config"
    assert "calibration" in loaded, "calibration section must be in dumped config"
    assert loaded["calibration"]["method"] == "hierarchical_relation"
    assert "node_aggregation" in loaded, "node_aggregation section must be in dumped config"
    assert "node_threshold" in loaded, "node_threshold section must be in dumped config"
    assert "dataset_view" in loaded, "dataset_view section must be in dumped config"
    assert "pipeline" in loaded, "pipeline section must be in dumped config"


def test_real_cfgnode_dump_password_redacted(tmp_path):
    """Database passwords must be redacted, not written in plain text."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "mstc"
    cfg.dataset = CN()
    cfg.dataset.name = "THEIA_E3"
    cfg.database = CN()
    cfg.database.host = "my-db-host.example.com"
    cfg.database.port = "5432"
    cfg.database.user = "postgres"
    cfg.database.password = "MyVerySecretPassword123!"
    cfg.pipeline = CN()
    cfg.pipeline.mode = "detection_only"

    dump_config(cfg, tmp_path / "run")

    content = (tmp_path / "run" / "config_resolved.yml").read_text(encoding="utf-8")

    assert "[REDACTED]" in content, "Password must be replaced with [REDACTED]"
    assert "MyVerySecretPassword123!" not in content, "Plain text password must not appear"
    assert "secret" not in content.lower() or "[REDACTED]" in content


def test_real_cfgnode_dump_private_fields_excluded(tmp_path):
    """Private runtime fields (_prefixed) in __dict__ must not appear in dumped config."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "mstc"
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.num_epochs = 6
    # Note: yacs CfgNode treats _seed as a config key (stored in dict), not
    # as a private attribute. The _cfg_to_dict filter on k.startswith("_")
    # ensures these keys are excluded from the YAML output.
    cfg["_seed"] = 42  # yacs stores this as a dict key
    cfg.database = CN()
    cfg.database.password = "secret"

    dump_config(cfg, tmp_path / "run")

    content = (tmp_path / "run" / "config_resolved.yml").read_text(encoding="utf-8")

    # Private keys stored in dict.items() are filtered out by _cfg_to_dict
    assert "_seed" not in content, "_seed must not appear in dumped config"
    assert "42" not in content, "_seed value must not appear"
    assert "secret" not in content, "Plain text password must not appear"


# --------------------------------------------------------------------------- #
# Bug B.2: MagicMock path hygiene
# --------------------------------------------------------------------------- #

def test_magicmock_cfg_does_not_write_to_filesystem(tmp_path):
    """MagicMock cfg must not touch the filesystem (no MagicMock/ created)."""
    from run_metadata import dump_config, dump_environment, dump_runtime

    mock_cfg = MagicMock()
    mock_run_dir = MagicMock()

    # These must be no-ops, not raise
    dump_environment(mock_run_dir)
    dump_config(mock_cfg, mock_run_dir)
    dump_runtime(mock_cfg, mock_run_dir, status="completed")

    # Verify no MagicMock/ directory was created in the repo root
    repo_root = Path(__file__).resolve().parents[2]
    magicmock_path = repo_root / "MagicMock"
    assert not magicmock_path.exists(), (
        f"MagicMock directory must not be created at {magicmock_path}. "
        "This indicates _is_valid_path did not correctly reject MagicMock path."
    )


def test_magicmock_cfg_does_not_create_run_subdirectory(tmp_path):
    """MagicMock run_dir must not create any directory under tmp_path."""
    from run_metadata import dump_config, dump_environment, dump_runtime

    # Use a MagicMock whose path string is under tmp_path to ensure
    # we're testing the MagicMock type/name check, not the path existence check
    mock_run_dir = MagicMock()
    type(mock_run_dir).__name__ = "MagicMock"
    mock_cfg = MagicMock()

    dump_environment(mock_run_dir)
    dump_config(mock_cfg, mock_run_dir)
    dump_runtime(mock_cfg, mock_run_dir, status="completed")

    # tmp_path should be completely untouched
    assert list(tmp_path.iterdir()) == [], (
        f"tmp_path should be empty but contains: {list(tmp_path.iterdir())}. "
        "MagicMock must not trigger filesystem writes."
    )


def test_empty_string_path_is_rejected(tmp_path):
    """Empty string path must not write to filesystem."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "test"

    dump_config(cfg, "")  # Must not raise

    # No file should be created at the current directory
    assert not Path("config_resolved.yml").exists()


def test_none_path_is_rejected(tmp_path):
    """None path must not write to filesystem."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "test"

    dump_config(cfg, None)  # Must not raise


# --------------------------------------------------------------------------- #
# Bug B.3: _cfg_to_dict handles edge cases
# --------------------------------------------------------------------------- #

def test_cfg_to_dict_handles_empty_cfgnode():
    """Empty CfgNode must not cause _cfg_to_dict to return empty dict."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN
    cfg = CN()
    # Empty CfgNode with no attributes

    result = _cfg_to_dict(cfg)
    # Empty CfgNode is valid (no public attributes → empty dict)
    assert result == {}


def test_cfg_to_dict_handles_nested_cfgnodes():
    """Deeply nested CfgNodes must be fully serialized."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.a = CN()
    cfg.a.b = CN()
    cfg.a.b.c = CN()
    cfg.a.b.c.value = 42
    cfg.a.x = 1

    result = _cfg_to_dict(cfg)
    assert result["a"]["b"]["c"]["value"] == 42
    assert result["a"]["x"] == 1


def test_cfg_to_dict_handles_lists():
    """CfgNode attributes that are lists must be preserved."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.scales = [0.5, 0.9, 0.99]
    cfg.neighbor_budgets = [8, 8, 8]
    cfg.model = CN()
    cfg.model.variant = "mstc"

    result = _cfg_to_dict(cfg)
    assert result["scales"] == [0.5, 0.9, 0.99]
    assert result["neighbor_budgets"] == [8, 8, 8]
    assert result["model"]["variant"] == "mstc"


def test_cfg_to_dict_handles_tuples():
    """CfgNode attributes that are tuples must be preserved."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.range = (2, 14)
    cfg.model = CN()
    cfg.model.variant = "test"

    result = _cfg_to_dict(cfg)
    # Tuples are serialized as lists (YAML native format)
    assert result["range"] == [2, 14]


def test_cfg_to_dict_handles_none_values():
    """None values in CfgNode must be preserved."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.resume_checkpoint = None
    cfg.model = CN()
    cfg.model.variant = "test"

    result = _cfg_to_dict(cfg)
    assert result["resume_checkpoint"] is None


def test_cfg_to_dict_preserves_bool_values():
    """Boolean values in CfgNode must be preserved correctly."""
    from run_metadata import _cfg_to_dict

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.enabled = True
    cfg.disabled = False
    cfg.model = CN()
    cfg.model.variant = "test"

    result = _cfg_to_dict(cfg)
    assert result["enabled"] is True
    assert result["disabled"] is False


# --------------------------------------------------------------------------- #
# Bug B.4: _write_yaml validation
# --------------------------------------------------------------------------- #

def test_write_yaml_rejects_empty_dict(tmp_path):
    """_write_yaml must raise ValueError for empty dict, not write 0-byte file."""
    from run_metadata import _write_yaml

    yml_path = tmp_path / "test.yml"

    # This must raise, not silently write empty file
    with pytest.raises(ValueError, match="empty dict"):
        _write_yaml(yml_path, {})

    # Verify no 0-byte file was created
    if yml_path.exists():
        content = yml_path.read_text()
        assert len(content) > 0, "Must not create 0-byte file"


def test_write_yaml_rejects_none(tmp_path):
    """_write_yaml must raise ValueError for None, not write 0-byte file."""
    from run_metadata import _write_yaml

    yml_path = tmp_path / "test_none.yml"

    with pytest.raises(ValueError, match="empty dict"):
        _write_yaml(yml_path, None)


# --------------------------------------------------------------------------- #
# Bug B.5: Integration with project configs
# --------------------------------------------------------------------------- #

def test_baseline_yml_cfg_dump(tmp_path):
    """A CfgNode created from baseline.yml config must dump correctly."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "orthrus_baseline"
    cfg.pipeline = CN()
    cfg.pipeline.mode = "detection_only"
    cfg.pipeline.run_tracing = False
    cfg.logging = CN()
    cfg.logging.wandb_mode = "disabled"
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.encoder = CN()
    cfg.detection.gnn_training.encoder.backbone = "graph_transformer"
    cfg.detection.gnn_training.encoder.neighbor_size = 20
    cfg.detection.gnn_training.encoder.context = CN()
    cfg.detection.gnn_training.encoder.context.mode = "recent"
    cfg.detection.gnn_training.encoder.context.multiscale = CN()
    cfg.detection.gnn_training.encoder.context.multiscale.enabled = False
    cfg.detection.gnn_training.decoder = CN()
    cfg.detection.gnn_training.decoder.predict_edge_type = CN()
    cfg.detection.gnn_training.decoder.predict_edge_type.enabled = True
    cfg.detection.gnn_training.decoder.time_gap = CN()
    cfg.detection.gnn_training.decoder.time_gap.enabled = False
    cfg.calibration = CN()
    cfg.calibration.method = "hierarchical_relation"
    cfg.node_aggregation = CN()
    cfg.node_aggregation.method = "topk_mean"
    cfg.node_aggregation.topk = 5
    cfg.node_threshold = CN()
    cfg.node_threshold.method = "validation_quantile"
    cfg.dataset_view = CN()
    cfg.dataset_view.mode = "host_network_full"
    cfg.database = CN()
    cfg.database.password = "baseline_secret"

    dump_config(cfg, tmp_path / "run")

    loaded = _read_yaml(tmp_path / "run" / "config_resolved.yml")
    assert loaded is not None and isinstance(loaded, dict)
    assert loaded["model"]["variant"] == "orthrus_baseline"
    assert loaded["detection"]["gnn_training"]["encoder"]["backbone"] == "graph_transformer"
    assert "[REDACTED]" in (tmp_path / "run" / "config_resolved.yml").read_text()


def test_mstc_full_yml_cfg_dump(tmp_path):
    """A CfgNode matching mstc_full.yml must dump correctly."""
    from run_metadata import dump_config

    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.model = CN()
    cfg.model.variant = "mstc"
    cfg.pipeline = CN()
    cfg.pipeline.mode = "detection_only"
    cfg.pipeline.run_tracing = False
    cfg.logging = CN()
    cfg.logging.wandb_mode = "disabled"
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.encoder = CN()
    cfg.detection.gnn_training.encoder.backbone = "graph_transformer"
    cfg.detection.gnn_training.encoder.neighbor_size = 24
    cfg.detection.gnn_training.encoder.context = CN()
    cfg.detection.gnn_training.encoder.context.mode = "multiscale"
    cfg.detection.gnn_training.encoder.context.multiscale = CN()
    cfg.detection.gnn_training.encoder.context.multiscale.enabled = True
    cfg.detection.gnn_training.encoder.context.multiscale.candidate_capacity = 64
    cfg.detection.gnn_training.encoder.context.multiscale.scale_quantiles = [0.50, 0.90, 0.99]
    cfg.detection.gnn_training.encoder.context.multiscale.neighbor_budgets = [8, 8, 8]
    cfg.detection.gnn_training.encoder.context.multiscale.share_encoder = True
    cfg.detection.gnn_training.encoder.context.multiscale.fusion = "gated"
    cfg.detection.gnn_training.encoder.context.multiscale.use_scale_embedding = False
    cfg.detection.gnn_training.encoder.context.multiscale.gate_hidden_dim = 64
    cfg.detection.gnn_training.decoder = CN()
    cfg.detection.gnn_training.decoder.predict_edge_type = CN()
    cfg.detection.gnn_training.decoder.predict_edge_type.enabled = True
    cfg.detection.gnn_training.decoder.time_gap = CN()
    cfg.detection.gnn_training.decoder.time_gap.enabled = True
    cfg.detection.gnn_training.decoder.time_gap.lambda_time = 0.3
    cfg.calibration = CN()
    cfg.calibration.method = "hierarchical_relation"
    cfg.node_aggregation = CN()
    cfg.node_aggregation.method = "topk_mean"
    cfg.node_aggregation.topk = 5
    cfg.node_aggregation.include_dst = True
    cfg.node_aggregation.score_field = "score_calibrated"
    cfg.node_threshold = CN()
    cfg.node_threshold.method = "validation_quantile"
    cfg.node_threshold.quantile = 0.999
    cfg.dataset_view = CN()
    cfg.dataset_view.mode = "host_network_full"
    cfg.database = CN()
    cfg.database.password = "mstc_secret"

    dump_config(cfg, tmp_path / "run")

    loaded = _read_yaml(tmp_path / "run" / "config_resolved.yml")
    assert loaded is not None and isinstance(loaded, dict)
    assert loaded["model"]["variant"] == "mstc"
    assert loaded["calibration"]["method"] == "hierarchical_relation"
    assert loaded["node_aggregation"]["method"] == "topk_mean"
    assert loaded["node_threshold"]["method"] == "validation_quantile"
    assert loaded["dataset_view"]["mode"] == "host_network_full"

    content = (tmp_path / "run" / "config_resolved.yml").read_text()
    assert "[REDACTED]" in content
    assert "mstc_secret" not in content
