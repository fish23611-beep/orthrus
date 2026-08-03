"""
src/artifact_paths.py

Centralized artifact-root and per-run directory resolution for the ORTHRUS pipeline.

Design goals
============
- Single resolution point: all artifact paths are derived from a single
  canonical `artifact_root` value, never hand-crafted by individual stages.
- Three-tier priority for artifact-root: CLI flag > env var > default.
- All paths use ``pathlib.Path``; no "/" or "\\" in user code.
- Run directory structure::

    <artifact_root>/
      <dataset>/
        runs/
          <model_variant>/
            seed_<seed>/
              config_resolved.yml
              environment.json
              runtime.json
              checkpoints/          ← train stage
              edge_scores/         ← test stage
              node_scores/         ← evaluate stage

  Sub-directories are created lazily — only when the respective stages run.

Path resolution vs directory creation
====================================
- ``resolve_artifact_root(cli_path, env_path)`` → ``Path`` — no filesystem side effects.
- ``resolve_run_dir(artifact_root, dataset, model_variant, seed)`` → ``Path`` —
  validates all inputs strictly; no filesystem side effects.
- ``extract_cfg_fields(cfg)`` → ``ArtifactCfgFields`` — reads cfg fields without
  touching the filesystem; raises ``ValueError`` for invalid/missing fields.
- ``create_stage_directories(run_dir, stages)`` → creates directories; call it
  separately when you actually need the dirs.
- ``resolve_artifact_paths(cfg, stages, ...)` → ``Path`` — the full orchestrator:
  resolves root, derives run_dir, maps cfg fields, and (by default) creates stage dirs.
  Pass ``create_dirs=False`` to skip directory creation (useful in tests or when
  you only need the paths).

Backward compatibility
======================
After calling ``resolve_artifact_paths(cfg, stages)`` the following cfg fields are
populated so existing modules continue to work:

  cfg._artifact_root
  cfg._run_dir
  cfg._stages (the canonical resolved stage list)
  cfg.detection.gnn_training._trained_models_dir  = run_dir / "checkpoints"
  cfg.detection.gnn_testing._edge_losses_dir      = run_dir / "edge_scores"
  cfg.detection.evaluation.node_evaluation._precision_recall_dir = run_dir / "node_scores"
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Safe component sanitisation
# --------------------------------------------------------------------------- #

_INVALID_COMPONENT_RE = re.compile(r"[\\/:\*\?\"<>|]")
_SENTINEL = "_empty"


def _sanitize(component: object) -> str:
    """
    Turn any object into a safe, single-level directory component.

    - Cast to string, strip surrounding whitespace.
    - Replace each ``_INVALID_COMPONENT_RE`` character with underscore.
    - Collapse multiple underscores to one.
    - If the result is empty or all-whitespace → ``_SENTINEL``.

    This function is a last-resort safety net.  All callers SHOULD validate
    their inputs *before* calling this function.
    """
    raw = str(component).strip()
    safe = _INVALID_COMPONENT_RE.sub("_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = _SENTINEL
    return safe


# --------------------------------------------------------------------------- #
# Artifact-root resolution (pure, no filesystem side effects)
# --------------------------------------------------------------------------- #

DEFAULT_ARTIFACT_ROOT = Path("./artifacts")
ARTIFACT_ROOT_ENV_VAR = "ORTHRUS_ARTIFACT_ROOT"


def resolve_artifact_root(
    cli_path: object | None,
    env_path: str | None = None,
) -> Path:
    """
    Resolve the canonical artifact-root path.  No filesystem side effects.

    Priority (highest → lowest):
        1. ``cli_path`` — value from ``--artifact-root`` CLI argument
        2. ``ORTHRUS_ARTIFACT_ROOT`` environment variable
        3. ``DEFAULT_ARTIFACT_ROOT`` ("./artifacts")

    Returns an absolute, case-normalised Path.
    Raises ``ValueError`` if ``cli_path`` is set but empty after stripping.
    """
    if cli_path is not None:
        raw = str(cli_path).strip()
        if not raw:
            raise ValueError(
                "--artifact-root must not be an empty string. "
                "Omit the flag to use the default or set ORTHRUS_ARTIFACT_ROOT."
            )
        resolved = Path(raw).expanduser().resolve()
        return resolved

    if env_path:
        resolved = Path(env_path).expanduser().resolve()
        return resolved

    return DEFAULT_ARTIFACT_ROOT.resolve()


# --------------------------------------------------------------------------- #
# Strict type validation for run-dir components
# --------------------------------------------------------------------------- #

def _require_string(value: object, name: str) -> str:
    """
    Return ``value`` if it is a non-empty str; otherwise raise ``ValueError``.

    MagicMock, None, int, float, bool, empty string, and whitespace-only string
    all raise ``ValueError`` with a clear message.  No mock-type detection is
    performed — the ``isinstance(value, str)`` check is sufficient.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name!r} must be a non-empty str; got {type(value).__name__!r}."
        )
    s = value.strip()
    if not s:
        raise ValueError(f"{name!r} must not be blank.")
    return s


def _require_int(value: object, name: str) -> int:
    """
    Return ``value`` as int if valid; otherwise raise ``ValueError``.

    True and False are rejected because ``isinstance(True, int)`` is True in Python.
    MagicMock, None, float, and other types raise ``ValueError``.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(
        f"{name!r} must be an int (not bool); got {type(value).__name__!r}."
    )


# --------------------------------------------------------------------------- #
# Run-directory resolution (strict inputs, no filesystem side effects)
# --------------------------------------------------------------------------- #

def resolve_run_dir(
    artifact_root: Path,
    dataset: str,
    model_variant: str,
    seed: int,
) -> Path:
    """
    Build the run directory path::

        <artifact_root>/<dataset>/runs/<model_variant>/seed_<seed>/

    Parameters
    ----------
    artifact_root:
        Resolved absolute artifact root (from ``resolve_artifact_root``).
    dataset:
        Non-empty str dataset name.  MagicMock / None / empty → ``ValueError``.
    model_variant:
        Non-empty str model variant name.  MagicMock / None / empty → ``ValueError``.
    seed:
        Integer seed.  MagicMock / None / non-int / bool → ``ValueError``.

    Returns
    -------
    Path
        Absolute run directory path.  No directories are created.
    """
    dataset_s = _require_string(dataset, "dataset")
    model_s   = _require_string(model_variant, "model_variant")
    seed_i    = _require_int(seed, "seed")

    parts = [
        dataset_s,
        "runs",
        model_s,
        f"seed_{seed_i}",
    ]
    return artifact_root.joinpath(*parts)


# --------------------------------------------------------------------------- #
# Cfg field extraction (pure, no filesystem side effects)
# --------------------------------------------------------------------------- #

class ArtifactCfgFields:
    """
    Container for fields extracted from cfg.

    Attributes
    ----------
    dataset : str
    model_variant : str
    seed : int
    """
    __slots__ = ("dataset", "model_variant", "seed")

    def __init__(self, dataset: str, model_variant: str, seed: int):
        self.dataset      = dataset
        self.model_variant = model_variant
        self.seed        = seed


def extract_cfg_fields(cfg) -> ArtifactCfgFields:
    """
    Extract dataset, model_variant, and seed from a cfg object.

    Strict type validation is applied — only real str/int values are accepted.
    MagicMock or any non-string/non-int value raises ``ValueError`` with a clear
    message indicating which field failed and why.

    Returns
    -------
    ArtifactCfgFields
    """
    # dataset
    _dataset = getattr(getattr(cfg, "dataset", None), "name", None)
    if _dataset is None:
        raise ValueError(
            "cfg.dataset.name is None or missing. Provide a valid non-empty dataset name."
        )
    dataset = _require_string(_dataset, "cfg.dataset.name")

    # model_variant
    _model = (
        getattr(getattr(cfg, "detection", None), "gnn_training", None)
        and getattr(cfg.detection.gnn_training, "used_method", None)
    )
    if _model is None:
        raise ValueError(
            "cfg.detection.gnn_training.used_method is None or missing. "
            "Provide a valid non-empty model variant name."
        )
    model_variant = _require_string(_model, "cfg.detection.gnn_training.used_method")

    # seed
    _seed = getattr(cfg, "_seed", None)
    if _seed is None:
        raise ValueError(
            "cfg._seed is None or missing. Provide a valid integer seed."
        )
    seed = _require_int(_seed, "cfg._seed")

    return ArtifactCfgFields(dataset=dataset, model_variant=model_variant, seed=seed)


# --------------------------------------------------------------------------- #
# Lazy stage-directory creation
# --------------------------------------------------------------------------- #

# Stage → sub-directory name(s).
# A stage may need multiple sub-directories; they are created as a list.
_STAGE_DIRS: dict[str, list[Path]] = {
    "train":     [Path("checkpoints")],
    "test":      [Path("edge_scores")],
    "evaluate":  [Path("node_scores")],
}


def create_stage_directories(
    run_dir: Path,
    stages: Sequence[str],
) -> None:
    """
    Create only the sub-directories required by the requested stages.

    Existing directories are NOT deleted; ``exist_ok=True`` throughout.

    Parameters
    ----------
    run_dir:
        The canonical run directory (result of ``resolve_run_dir``).
    stages:
        Ordered list of canonical stage names that will execute.
    """
    to_create: set[Path] = set()
    for stage in stages:
        subs = _STAGE_DIRS.get(stage)
        if subs is None:
            continue
        for sub in subs:
            to_create.add(run_dir / sub)

    for d in sorted(to_create):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Full orchestrator
# --------------------------------------------------------------------------- #

def resolve_artifact_paths(
    cfg,
    stages: Sequence[str],
    *,
    cli_artifact_root: object | None = None,
    env_artifact_root: str | None = None,
    run_dir: Path | None = None,
    create_dirs: bool = True,
) -> Path:
    """
    Resolve artifact-root and run-dir, write resolved paths into ``cfg``,
    then (by default) create stage sub-directories.

    Parameters
    ----------
    cfg:
        The yacs CfgNode.  Invalid/missing fields raise ``ValueError``.
    stages:
        Ordered list of stages that will run.
    cli_artifact_root:
        Value of ``--artifact-root`` CLI argument (may be None).
    env_artifact_root:
        Value of ``ORTHRUS_ARTIFACT_ROOT`` environment variable (may be None).
    run_dir:
        If provided, use this exact path as the run directory (reuse a
        previous run).  In this case ``create_dirs`` controls directory creation.
    create_dirs:
        If True (default), call ``create_stage_directories`` to create
        stage sub-directories.  Set to False when only path resolution is
        needed (e.g., in unit tests or when the caller manages dirs separately).

    Returns
    -------
    Path
        The resolved run directory path.

    Raises
    ------
    ValueError
        If cfg fields cannot be resolved to valid non-empty strings/ints
        (e.g., MagicMock, None, empty string, bool).
    """
    # 1. Resolve artifact root (pure, no filesystem)
    artifact_root = resolve_artifact_root(cli_artifact_root, env_artifact_root)

    # 2. Resolve run dir (strict inputs; raises ValueError for MagicMock etc.)
    if run_dir is None:
        fields = extract_cfg_fields(cfg)
        run_dir = resolve_run_dir(
            artifact_root,
            dataset=fields.dataset,
            model_variant=fields.model_variant,
            seed=fields.seed,
        )
    else:
        run_dir = run_dir.expanduser().resolve()

    # 3. Write back to cfg so existing modules continue to work
    cfg._artifact_root = str(artifact_root)
    cfg._run_dir      = str(run_dir)
    cfg._stages       = list(stages)

    cfg.detection.gnn_training._trained_models_dir = str(run_dir / "checkpoints")
    cfg.detection.gnn_testing._edge_losses_dir     = str(run_dir / "edge_scores")
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = str(run_dir / "node_scores")

    # 4. Lazy directory creation
    if create_dirs:
        create_stage_directories(run_dir, stages)

    return run_dir
