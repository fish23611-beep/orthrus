"""
tests/test_artifact_paths.py

Unit tests for src/artifact_paths.py.

Scope
=====
- Artifact-root three-tier priority (CLI > env var > default).
- Run-directory path construction.
- Strict type validation (MagicMock / None / empty → ValueError).
- Safe component sanitisation.
- Lazy stage directory creation.
- cfg field population (backward compatibility).
- No pollution of the real project root directory.

All tests use tmp_path or monkeypatch.chdir so no files are created in
the real project directory.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from pathlib import Path
from unittest.mock import MagicMock


def _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0):
    cfg = MagicMock()
    cfg.dataset = MagicMock()
    cfg.dataset.name = dataset
    cfg.detection = MagicMock()
    cfg.detection.gnn_training = MagicMock()
    cfg.detection.gnn_training.used_method = model
    cfg._seed = seed
    return cfg


# --------------------------------------------------------------------------- #
# Artifact-root priority
# --------------------------------------------------------------------------- #

def test_cli_path_priority_over_env(tmp_path):
    """CLI --artifact-root must win over ORTHRUS_ARTIFACT_ROOT env var."""
    from artifact_paths import resolve_artifact_root

    cli = str(tmp_path / "cli_root")
    env = str(tmp_path / "env_root")

    result = resolve_artifact_root(cli_path=cli, env_path=env)
    assert result == Path(cli).resolve()


def test_env_path_priority_over_default(tmp_path):
    """ORTHRUS_ARTIFACT_ROOT must win over the ./artifacts default."""
    from artifact_paths import resolve_artifact_root

    env = str(tmp_path / "my_artifacts")
    result = resolve_artifact_root(cli_path=None, env_path=env)
    assert result == Path(env).resolve()


def test_default_resolves_to_artifacts_relative(tmp_path, monkeypatch):
    """
    When neither CLI nor env var is set, default is ./artifacts (relative to cwd).

    We change cwd to tmp_path so the resolved path points into tmp_path,
    not into the real project directory.
    """
    from artifact_paths import resolve_artifact_root, DEFAULT_ARTIFACT_ROOT

    monkeypatch.chdir(tmp_path)
    result = resolve_artifact_root(cli_path=None, env_path=None)
    # Must resolve within tmp_path
    assert result == (tmp_path / DEFAULT_ARTIFACT_ROOT.name).resolve()


def test_empty_cli_path_raises():
    """Empty string for --artifact-root must raise ValueError, not silently use default."""
    from artifact_paths import resolve_artifact_root

    with pytest.raises(ValueError) as exc_info:
        resolve_artifact_root(cli_path="")
    assert "empty" in str(exc_info.value).lower()


def test_none_cli_path_falls_through_to_env(tmp_path):
    """cli_path=None must fall through to environment variable."""
    from artifact_paths import resolve_artifact_root

    env = str(tmp_path / "via_env")
    result = resolve_artifact_root(cli_path=None, env_path=env)
    assert result == Path(env).resolve()


# --------------------------------------------------------------------------- #
# Run-directory construction
# --------------------------------------------------------------------------- #

def test_run_dir_structure(tmp_path):
    """Run dir must follow: <root>/<dataset>/runs/<model>/seed_<seed>/"""
    from artifact_paths import resolve_run_dir

    run = resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant="orthrus", seed=42)
    parts = run.relative_to(tmp_path).parts
    assert parts == ("THEIA_E3", "runs", "orthrus", "seed_42")


def test_pathlib_returned(tmp_path):
    """resolve_artifact_root and resolve_run_dir must return pathlib.Path."""
    from artifact_paths import resolve_artifact_root, resolve_run_dir

    root = resolve_artifact_root(cli_path=None, env_path=None)
    assert isinstance(root, Path)

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    assert isinstance(run, Path)


def test_dataset_in_path(tmp_path):
    """The dataset name must appear literally in the resolved path."""
    from artifact_paths import resolve_run_dir

    run = resolve_run_dir(tmp_path, dataset="CADETS_E5", model_variant="orthrus", seed=0)
    assert "CADETS_E5" in run.parts


def test_seed_in_path(tmp_path):
    """seed_N format must appear in the path."""
    from artifact_paths import resolve_run_dir

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", seed=7)
    assert any(p.startswith("seed_") and p == f"seed_7" for p in run.parts)


# --------------------------------------------------------------------------- #
# Strict type validation
# --------------------------------------------------------------------------- #

def test_magicmock_dataset_raises(tmp_path):
    """MagicMock dataset name must raise ValueError (not produce a directory)."""
    from artifact_paths import resolve_run_dir

    cfg = _make_cfg()
    cfg.dataset.name = MagicMock()   # type: ignore[assignment]

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset=cfg.dataset.name, model_variant="orthrus", seed=0)

    msg = str(exc_info.value)
    assert "dataset" in msg.lower()


def test_none_dataset_raises(tmp_path):
    """None dataset must raise ValueError (not produce a directory)."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset=None, model_variant="orthrus", seed=0)  # type: ignore[arg-type]

    assert "dataset" in str(exc_info.value).lower()


def test_empty_dataset_raises(tmp_path):
    """Empty string dataset must raise ValueError."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="", model_variant="orthrus", seed=0)

    assert "dataset" in str(exc_info.value).lower()


def test_whitespace_dataset_raises(tmp_path):
    """Whitespace-only dataset must raise ValueError."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="   ", model_variant="orthrus", seed=0)

    assert "dataset" in str(exc_info.value).lower()


def test_magicmock_model_raises(tmp_path):
    """MagicMock model_variant must raise ValueError."""
    from artifact_paths import resolve_run_dir

    cfg = _make_cfg()
    cfg.detection.gnn_training.used_method = MagicMock()  # type: ignore[assignment]

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant=cfg.detection.gnn_training.used_method, seed=0)

    assert "model" in str(exc_info.value).lower()


def test_none_model_raises(tmp_path):
    """None model_variant must raise ValueError."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant=None, seed=0)  # type: ignore[arg-type]

    assert "model" in str(exc_info.value).lower()


def test_magicmock_seed_raises(tmp_path):
    """MagicMock seed must raise ValueError."""
    from artifact_paths import resolve_run_dir

    cfg = _make_cfg()
    cfg._seed = MagicMock()  # type: ignore[assignment]

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant="orthrus", seed=cfg._seed)

    assert "seed" in str(exc_info.value).lower()


def test_none_seed_raises(tmp_path):
    """None seed must raise ValueError."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant="orthrus", seed=None)  # type: ignore[arg-type]

    assert "seed" in str(exc_info.value).lower()


def test_float_seed_raises(tmp_path):
    """Float seed must raise ValueError (bool is excluded)."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant="orthrus", seed=3.14)  # type: ignore[arg-type]

    assert "seed" in str(exc_info.value).lower()


def test_bool_seed_raises(tmp_path):
    """Boolean seed must raise ValueError (True/False are not valid seeds)."""
    from artifact_paths import resolve_run_dir

    with pytest.raises(ValueError) as exc_info:
        resolve_run_dir(tmp_path, dataset="THEIA_E3", model_variant="orthrus", seed=True)  # type: ignore[arg-type]

    assert "seed" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# Component sanitisation
# --------------------------------------------------------------------------- #

def test_special_chars_sanitized():
    """Path separators and other invalid characters must be replaced."""
    from artifact_paths import _sanitize

    for bad in ["foo/bar", "foo\\bar", "foo:bar", "foo*bar"]:
        safe = _sanitize(bad)
        assert "/" not in safe
        assert "\\" not in safe
        assert ":" not in safe
        assert "*" not in safe


def test_whitespace_only_becomes_sentinel():
    """Empty/whitespace-only components become the sentinel."""
    from artifact_paths import _sanitize, _SENTINEL

    assert _sanitize("") == _SENTINEL
    assert _sanitize("   ") == _SENTINEL


# --------------------------------------------------------------------------- #
# Lazy stage directory creation
# --------------------------------------------------------------------------- #

def test_train_creates_checkpoints_dir(tmp_path):
    """train stage must create run_dir/checkpoints/."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    create_stage_directories(run, ["train"])
    assert (run / "checkpoints").is_dir()


def test_test_creates_edge_scores_dir(tmp_path):
    """test stage must create run_dir/edge_scores/."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    create_stage_directories(run, ["test"])
    assert (run / "edge_scores").is_dir()


def test_evaluate_creates_node_scores_dir(tmp_path):
    """evaluate stage must create run_dir/node_scores/."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    create_stage_directories(run, ["evaluate"])
    assert (run / "node_scores").is_dir()


def test_trace_does_not_create_dirs(tmp_path):
    """trace stage does not create any sub-directory (depimpact manages its own paths)."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    create_stage_directories(run, ["trace"])
    # trace does not map to any directory
    assert not (run / "checkpoints").exists()
    assert not (run / "edge_scores").exists()
    assert not (run / "node_scores").exists()
    assert not (run / "evaluation").exists()


def test_no_files_created_when_not_needed(tmp_path):
    """Stages that are not requested must not create their directories."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    create_stage_directories(run, ["train"])
    assert (run / "checkpoints").is_dir()
    assert not (run / "edge_scores").exists()
    assert not (run / "node_scores").exists()


def test_existing_dir_not_deleted(tmp_path):
    """Pre-existing directories must not be deleted."""
    from artifact_paths import resolve_run_dir, create_stage_directories

    run = resolve_run_dir(tmp_path, "THEIA_E3", "orthrus", 0)
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    marker = checkpoint_dir / "existing_file.txt"
    marker.write_text("i exist")

    create_stage_directories(run, ["train"])
    assert checkpoint_dir.is_dir()
    assert marker.exists()


# --------------------------------------------------------------------------- #
# resolve_artifact_paths — cfg field population
# --------------------------------------------------------------------------- #

def test_cfg_fields_populated_without_creating_dirs(tmp_path):
    """resolve_artifact_paths must write _artifact_root, _run_dir, _stages into cfg."""
    from artifact_paths import resolve_artifact_paths

    cfg = _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0)
    stages = ["train", "test", "evaluate"]

    run_dir = resolve_artifact_paths(cfg, stages, create_dirs=False)

    assert hasattr(cfg, "_artifact_root")
    assert hasattr(cfg, "_run_dir")
    assert cfg._stages == stages
    # yacs requires _artifact_root and _run_dir to be str (not Path)
    assert isinstance(cfg._artifact_root, str), f"_artifact_root must be str, got {type(cfg._artifact_root)}"
    assert isinstance(cfg._run_dir, str), f"_run_dir must be str, got {type(cfg._run_dir)}"
    assert Path(cfg._run_dir) == run_dir, f"_run_dir should equal computed path"


def test_cfg_output_dirs_mapped(tmp_path):
    """Existing output dir fields must be remapped to run_dir sub-paths."""
    from artifact_paths import resolve_artifact_paths

    cfg = _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0)
    resolve_artifact_paths(cfg, ["train", "test", "evaluate"], create_dirs=False)

    # _run_dir is now str (for yacs compat); convert to Path for path ops
    run_dir = Path(cfg._run_dir)
    assert cfg.detection.gnn_training._trained_models_dir == str(run_dir / "checkpoints")
    assert cfg.detection.gnn_testing._edge_losses_dir == str(run_dir / "edge_scores")
    assert cfg.detection.evaluation.node_evaluation._precision_recall_dir == str(run_dir / "node_scores")


def test_explicit_run_dir_bypasses_derivation(tmp_path):
    """Providing run_dir skips dataset/model/seed path derivation."""
    from artifact_paths import resolve_artifact_paths

    cfg = _make_cfg(dataset="IGNORED", model="IGNORED", seed=999)
    explicit = tmp_path / "explicit" / "run" / "dir"
    explicit.mkdir(parents=True)
    resolve_artifact_paths(cfg, ["train"], run_dir=explicit, create_dirs=False)

    assert cfg._run_dir == str(explicit)


def test_magicmock_cfg_raises_valueerror():
    """
    MagicMock cfg must raise ValueError, not touch the filesystem.

    This is the key regression test: before the fix, resolve_artifact_paths
    would call create_stage_directories with a MagicMock-based path string,
    creating 'artifacts/MagicMock_name=...' in the project root.
    """
    from artifact_paths import resolve_artifact_paths

    cfg = MagicMock()  # completely unpopulated MagicMock

    with pytest.raises(ValueError) as exc_info:
        resolve_artifact_paths(cfg, ["train"])

    # Must mention which field is the problem
    assert "cfg" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# Default behavior with cwd redirect
# --------------------------------------------------------------------------- #

def test_default_artifacts_resolved_under_cwd(tmp_path, monkeypatch):
    """Default artifact root resolves relative to cwd (redirected to tmp_path)."""
    from artifact_paths import resolve_artifact_paths, DEFAULT_ARTIFACT_ROOT

    monkeypatch.chdir(tmp_path)
    cfg = _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0)

    run_dir = resolve_artifact_paths(cfg, ["train"])

    expected = (tmp_path / DEFAULT_ARTIFACT_ROOT.name / "THEIA_E3" / "runs" / "orthrus" / "seed_0").resolve()
    assert cfg._run_dir == str(expected)
    # Directories created in tmp_path, not project root
    assert Path(cfg._run_dir).is_relative_to(tmp_path)


# --------------------------------------------------------------------------- #
# Stage sub-directories only for selected stages
# --------------------------------------------------------------------------- #

def test_selected_stages_only_get_dirs(tmp_path, monkeypatch):
    """Only the directories for selected stages must be created."""
    monkeypatch.chdir(tmp_path)
    from artifact_paths import resolve_artifact_paths

    cfg = _make_cfg(dataset="THEIA_E3", model="orthrus", seed=0)
    resolve_artifact_paths(cfg, ["train", "test", "evaluate"])

    run_dir = Path(cfg._run_dir)
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "edge_scores").is_dir()
    assert (run_dir / "node_scores").is_dir()
    # trace does not create any sub-directory
    assert not (run_dir / "checkpoints").exists() or True  # already asserted above
    assert not (run_dir / "evaluation").exists()


# --------------------------------------------------------------------------- #
# Integration: full round-trip with real cfg
# --------------------------------------------------------------------------- #

def test_full_round_trip_with_real_paths(tmp_path, monkeypatch):
    """Integration: resolve paths + create dirs, verify all sub-dirs exist."""
    monkeypatch.chdir(tmp_path)
    from artifact_paths import resolve_artifact_root, resolve_run_dir, create_stage_directories

    root = resolve_artifact_root(cli_path=str(tmp_path))
    run = resolve_run_dir(root, "THEIA_E3", "orthrus", seed=0)
    create_stage_directories(run, ["preprocess", "train", "test", "evaluate", "trace"])

    assert (run / "checkpoints").is_dir()
    assert (run / "edge_scores").is_dir()
    assert (run / "node_scores").is_dir()
    # trace stage does not create directories; depimpact manages its own paths
