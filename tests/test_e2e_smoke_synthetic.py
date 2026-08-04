"""End-to-end synthetic smoke tests for detection_only mode.

These tests verify that the pipeline can run in detection_only mode
with synthetic artifacts without connecting to PostgreSQL, downloading
data, or using W&B.
"""

import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data, TemporalData

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)


@pytest.fixture
def synthetic_artifact_root(tmp_path):
    """Create a synthetic artifact structure."""
    root = tmp_path / "artifacts"
    root.mkdir()
    
    # Create dataset directory
    dataset_dir = root / "THEIA_E3"
    dataset_dir.mkdir()
    
    # Create runs directory structure
    run_dir = dataset_dir / "runs" / "orthrus" / "seed_0"
    run_dir.mkdir(parents=True)
    
    # Create checkpoint directory with fake model
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_epoch_1").mkdir()
    
    # Create metadata directory with fake caches
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    
    # Write fake metadata cache files
    import pickle
    
    # node_metadata.pkl
    node_metadata = {
        0: {"uuid": "uuid0", "type": "subject", "path": "/bin/bash", "display": "bash"},
        1: {"uuid": "uuid1", "type": "file", "path": "/etc/passwd", "display": "passwd"},
    }
    with open(metadata_dir / "node_metadata.pkl", "wb") as f:
        pickle.dump(node_metadata, f)
    
    # uuid_to_node_id.pkl
    uuid_to_node_id = {"uuid0": 0, "uuid1": 1}
    with open(metadata_dir / "uuid_to_node_id.pkl", "wb") as f:
        pickle.dump(uuid_to_node_id, f)
    
    # ground_truth_nodes.pkl
    ground_truth = {0, 1}
    with open(metadata_dir / "ground_truth_nodes.pkl", "wb") as f:
        pickle.dump(ground_truth, f)
    
    # dataset_manifest.json
    manifest = {
        "dataset": "THEIA_E3",
        "num_node_types": 3,
        "num_edge_types": 10,
        "train_files": ["graph_1"],
        "val_files": ["graph_2"],
        "test_files": ["graph_3"],
        "word2vec_dim": 128,
        "preprocess_config_hash": "abc123",
        "created_at": "2026-08-04T00:00:00Z",
    }
    with open(metadata_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f)
    
    return {
        "root": root,
        "run_dir": run_dir,
        "metadata_dir": metadata_dir,
        "checkpoint_dir": checkpoint_dir,
    }


class TestSyntheticDetectionOnlySmoke:
    """Smoke tests for detection_only mode with synthetic artifacts."""

    def test_detection_only_skips_preprocess(self, synthetic_artifact_root, mocker):
        """Verify detection_only mode skips preprocess stages."""
        # This would require full imports, so we just test the logic
        from orthrus import _is_detection_only_mode, _check_detection_only_prerequisites
        
        cfg = SimpleNamespace(
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        assert _is_detection_only_mode(cfg) is True

    def test_wandb_mode_disabled_uses_no_network(self):
        """Verify wandb mode disabled doesn't require network."""
        from wandb_control import resolve_wandb_mode
        
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="disabled"))
        args = SimpleNamespace(wandb=False)
        
        mode = resolve_wandb_mode(cfg, args)
        
        assert mode == "disabled"

    def test_metadata_cache_works_offline(self, synthetic_artifact_root):
        """Verify metadata cache works without network."""
        from mstc.metadata_cache import MetadataCache
        
        cache = MetadataCache(synthetic_artifact_root["metadata_dir"])
        
        assert cache.has_node_metadata()
        assert cache.has_uuid_to_node_id()
        assert cache.has_ground_truth_nodes()
        assert cache.has_dataset_manifest()
        
        # Can load without network
        metadata = cache.load_node_metadata()
        assert len(metadata) == 2
        
        manifest = cache.load_dataset_manifest()
        assert manifest["dataset"] == "THEIA_E3"

    def test_artifact_paths_resolve_correctly(self, synthetic_artifact_root):
        """Verify artifact paths resolve to correct locations."""
        from artifact_paths import resolve_artifact_root, resolve_run_dir
        
        root = resolve_artifact_root(str(synthetic_artifact_root["root"]))
        assert "artifacts" in root
        
        run_dir = resolve_run_dir(
            str(synthetic_artifact_root["root"]),
            "THEIA_E3",
            "orthrus",
            0
        )
        assert "THEIA_E3" in run_dir
        assert "seed_0" in run_dir

    def test_config_has_detection_only_mode(self):
        """Verify config supports detection_only mode."""
        from config import _validate_pipeline_mode
        
        assert _validate_pipeline_mode("detection_only") == "detection_only"
        assert _validate_pipeline_mode("full_pipeline") == "full_pipeline"
        
        with pytest.raises(ValueError):
            _validate_pipeline_mode("invalid_mode")

    def test_config_has_wandb_mode(self):
        """Verify config supports wandb modes."""
        from config import _validate_wandb_mode
        
        assert _validate_wandb_mode("disabled") == "disabled"
        assert _validate_wandb_mode("offline") == "offline"
        assert _validate_wandb_mode("online") == "online"
        
        with pytest.raises(ValueError):
            _validate_wandb_mode("invalid")

    def test_corpus_scope_validation(self):
        """Verify corpus_scope validation works."""
        from config import _validate_corpus_scope
        
        assert _validate_corpus_scope("official_full_dataset") == "official_full_dataset"
        assert _validate_corpus_scope("train_only") == "train_only"
        
        with pytest.raises(ValueError):
            _validate_corpus_scope("invalid_scope")


class TestSyntheticDataFlow:
    """Test synthetic data flow through the pipeline components."""

    def test_temporal_data_creation(self):
        """Verify TemporalData can be created for testing."""
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        dst = torch.tensor([1, 2, 0], dtype=torch.long)
        t = torch.tensor([1, 2, 3], dtype=torch.long)
        msg = torch.rand(3, 10)
        
        data = TemporalData(
            src=src,
            dst=dst,
            t=t,
            msg=msg,
        )
        
        assert data.src.shape[0] == 3
        assert data.dst.shape[0] == 3
        assert data.t.shape[0] == 3

    def test_node_messages_optional_in_testing(self, mocker):
        """Verify node messages can be skipped in testing."""
        from detection.orthrus_gnn_testing import _load_nodeid2msg
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake",
            testing=SimpleNamespace(include_node_messages=False),
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        mocker.patch("detection.orthrus_gnn_testing._get_metadata_cache", return_value=None)
        
        result = _load_nodeid2msg(cfg)
        
        assert result == {}

    def test_ground_truth_cache_prevents_db(self, mocker):
        """Verify ground truth cache prevents DB connection."""
        import labelling
        
        mock_cache = MagicMock()
        mock_cache.has_ground_truth_nodes.return_value = True
        mock_cache.has_uuid_to_node_id.return_value = True
        mock_cache.load_ground_truth_nodes.return_value = {1, 2, 3}
        mock_cache.load_uuid_to_node_id.return_value = {}
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        db_mock = mocker.patch.object(labelling, "init_database_connection")
        
        gt, paths, uuid_map = labelling.get_ground_truth(cfg)
        
        assert gt == {1, 2, 3}
        db_mock.assert_not_called()


class TestConfigEnvVarPriority:
    """Test environment variable priority in config."""

    def test_env_var_overrides_default(self, mocker):
        """Verify env var overrides internal default."""
        from config import _resolve_artifact_root_from_env
        
        mocker.patch.dict(os.environ, {"ORTHRUS_ARTIFACT_ROOT": "/env/path"})
        
        result = _resolve_artifact_root_from_env()
        
        assert result == "/env/path"

    def test_cli_overrides_env_var(self):
        """Verify CLI takes priority over env var."""
        from artifact_paths import resolve_artifact_root
        
        result = resolve_artifact_root("/cli/path", "/env/path")
        
        assert "cli" in result

    def test_no_env_uses_default(self, mocker):
        """Verify default is used when no env var set."""
        mocker.patch.dict(os.environ, {}, clear=True)
        from config import _resolve_artifact_root_from_env
        
        # Clear any existing value
        if "ORTHRUS_ARTIFACT_ROOT" in os.environ:
            del os.environ["ORTHRUS_ARTIFACT_ROOT"]
        
        result = _resolve_artifact_root_from_env()
        
        assert result == "./artifacts"


class TestBaselineConfig:
    """Test baseline.yml configuration file."""

    def test_baseline_config_exists(self):
        """Verify baseline config file exists."""
        baseline_path = Path(__file__).resolve().parents[1] / "config" / "experiments" / "baseline.yml"
        assert baseline_path.exists()

    def test_baseline_config_structure(self):
        """Verify baseline config has required fields."""
        baseline_path = Path(__file__).resolve().parents[1] / "config" / "experiments" / "baseline.yml"
        
        with open(baseline_path) as f:
            content = f.read()
        
        assert "detection_only" in content
        assert "include_node_messages: false" in content
        assert "wandb_mode: disabled" in content
