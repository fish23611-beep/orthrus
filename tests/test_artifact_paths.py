"""Tests for artifact_paths.py."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

from artifact_paths import (
    resolve_artifact_root,
    resolve_run_dir,
    extract_cfg_fields,
    ArtifactCfgFields,
    create_stage_directories,
)


class TestResolveArtifactRoot:
    """Tests for resolve_artifact_root function."""

    def test_default_artifact_root(self):
        """Without CLI or env, returns default."""
        result = resolve_artifact_root(None, None)
        assert result == "./artifacts"

    def test_cli_takes_priority(self):
        """CLI argument takes priority over env var."""
        result = resolve_artifact_root("/custom/cli/path", "/env/path")
        assert result == str(Path("/custom/cli/path").resolve())

    def test_env_fallback(self):
        """Environment variable used when no CLI."""
        result = resolve_artifact_root(None, "/env/path")
        assert result == str(Path("/env/path").resolve())

    def test_empty_cli_raises(self):
        """Empty CLI argument raises ValueError."""
        with pytest.raises(ValueError, match="must not be an empty string"):
            resolve_artifact_root("", None)

    def test_strips_whitespace(self):
        """Whitespace is stripped from CLI path."""
        result = resolve_artifact_root("  /custom/path  ", None)
        assert "custom" in result


class TestResolveRunDir:
    """Tests for resolve_run_dir function."""

    def test_basic_run_dir(self, tmp_path):
        """Basic run directory construction."""
        result = resolve_run_dir(str(tmp_path), "THEIA_E3", "orthrus", 42)
        assert "THEIA_E3" in result
        assert "runs" in result
        assert "orthrus" in result
        assert "seed_42" in result

    def test_invalid_dataset_raises(self, tmp_path):
        """Non-string dataset raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty str"):
            resolve_run_dir(str(tmp_path), 123, "orthrus", 42)

    def test_invalid_model_raises(self, tmp_path):
        """Non-string model raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty str"):
            resolve_run_dir(str(tmp_path), "THEIA_E3", None, 42)

    def test_invalid_seed_raises(self, tmp_path):
        """Non-int seed raises ValueError."""
        with pytest.raises(ValueError, match="must be an int"):
            resolve_run_dir(str(tmp_path), "THEIA_E3", "orthrus", "42")


class TestExtractCfgFields:
    """Tests for extract_cfg_fields function."""

    def test_extracts_valid_fields(self):
        """Extracts dataset, model, seed from cfg."""
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(name="THEIA_E3"),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(used_method="orthrus")
            ),
            _seed=42,
        )
        fields = extract_cfg_fields(cfg)
        assert fields.dataset == "THEIA_E3"
        assert fields.model_variant == "orthrus"
        assert fields.seed == 42

    def test_missing_dataset_raises(self):
        """Missing dataset.name raises ValueError."""
        cfg = SimpleNamespace(dataset=SimpleNamespace(name=None))
        with pytest.raises(ValueError, match="cfg.dataset.name is None"):
            extract_cfg_fields(cfg)

    def test_missing_model_raises(self):
        """Missing model raises ValueError."""
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(name="THEIA_E3"),
            detection=SimpleNamespace(gnn_training=SimpleNamespace()),
        )
        with pytest.raises(ValueError, match="used_method is None"):
            extract_cfg_fields(cfg)


class TestCreateStageDirectories:
    """Tests for create_stage_directories function."""

    def test_creates_checkpoints_dir(self, tmp_path):
        """Creates checkpoints directory for train stage."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        create_stage_directories(str(run_dir), ["train"])
        assert (run_dir / "checkpoints").is_dir()

    def test_creates_edge_scores_dir(self, tmp_path):
        """Creates edge_scores directory for test stage."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        create_stage_directories(str(run_dir), ["test"])
        assert (run_dir / "edge_scores").is_dir()

    def test_creates_node_scores_dir(self, tmp_path):
        """Creates node_scores directory for evaluate stage."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        create_stage_directories(str(run_dir), ["evaluate"])
        assert (run_dir / "node_scores").is_dir()

    def test_creates_multiple_dirs(self, tmp_path):
        """Creates multiple directories for multiple stages."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        create_stage_directories(str(run_dir), ["train", "test", "evaluate"])
        assert (run_dir / "checkpoints").is_dir()
        assert (run_dir / "edge_scores").is_dir()
        assert (run_dir / "node_scores").is_dir()

    def test_existing_dirs_ok(self, tmp_path):
        """Does not fail if directories already exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "checkpoints").mkdir()
        # Should not raise
        create_stage_directories(str(run_dir), ["train"])
        assert (run_dir / "checkpoints").is_dir()


class TestArtifactCfgFields:
    """Tests for ArtifactCfgFields class."""

    def test_slots(self):
        """Fields are stored in slots."""
        fields = ArtifactCfgFields("THEIA_E3", "orthrus", 42)
        assert fields.dataset == "THEIA_E3"
        assert fields.model_variant == "orthrus"
        assert fields.seed == 42
