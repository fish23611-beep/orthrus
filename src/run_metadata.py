"""
src/run_metadata.py

Run-metadata persistence for the ORTHRUS pipeline.

Produces three files per completed (or failed) run inside the resolved run directory:

1. ``config_resolved.yml``
       The final runtime configuration — all CfgNode fields at the moment the
       pipeline started, including CLI overrides and defaults merged from the yml
       file.  MagicMock objects are excluded so the output is always valid YAML.

2. ``environment.json``
       Captures the software environment at import time.  Missing optional
       dependencies (torch, wandb, torch_geometric, cuda) are recorded as ``null``
       rather than causing the function to fail.

3. ``runtime.json``
       Records the actual pipeline execution: requested stages, executed stages,
       timing, W&B mode, and run status.  Written via a temporary file +
       atomic rename so a crashed run never leaves a half-written JSON blob.
       On failure the status is set to ``"failed"`` with an error message before
       the original exception is re-raised.

Key design decisions
=====================
- All three files are written **only** when ``cfg._run_dir`` is a valid,
  non-empty path.  In unit tests where ``cfg._run_dir`` is a MagicMock, no
  filesystem operations occur.
- ``dump_environment`` / ``dump_config`` / ``dump_runtime`` are intentionally
  separate functions so callers can write them independently.
- ``dump_runtime`` accepts a ``status`` keyword so the caller can set
  ``"completed"`` or ``"failed"`` based on its own exception handling.
- Database passwords are NOT written to config_resolved.yml or environment.json.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# YAML dumping (no external yaml dependency required)
# --------------------------------------------------------------------------- #

def _cfg_to_dict(cfg, seen: set | None = None) -> dict[str, Any]:
    """
    Recursively convert a yacs CfgNode (or any object with ``__dict__`` and
    ``is_copy`` / ``is_frozen`` attributes) to a plain dict.

    Objects with ``is_copy`` or ``is_frozen`` attributes (yacs internals) are
    replaced by their plain ``__dict__`` to avoid serialising yacs proxy objects.
    """
    if seen is None:
        seen = set()
    obj_id = id(cfg)
    if obj_id in seen:
        return None   # cycle guard
    seen.add(obj_id)

    # yacs / CfgNode objects store config values as dict items (not __dict__ attrs).
    # CfgNode is a dict subclass, so we iterate via dict.items() to get
    # the actual config key-value pairs.
    # CfgNode has is_frozen() as a bound method; calling it returns a bool.
    # MagicMock also has is_frozen (as a MagicMock instance, not callable as expected).
    # SimpleNamespace/other objects may have neither.
    def _is_yacs(obj) -> bool:
        frozen = getattr(obj, "is_frozen", None)
        if frozen is not None:
            if callable(frozen):
                try:
                    return isinstance(frozen(), bool)
                except Exception:
                    return False
            return isinstance(frozen, bool)
        return False

    if _is_yacs(cfg):
        result = {}
        for k, v in cfg.items():
            if k.startswith("_"):
                continue
            try:
                result[k] = _cfg_to_dict(v, seen)
            except Exception:
                continue
        return result

    if isinstance(cfg, dict):
        return {k: _cfg_to_dict(v, seen) for k, v in cfg.items()}

    if isinstance(cfg, (list, tuple)):
        # Use a fresh seen set for sibling items to avoid false cycle
        # detection (e.g., [8, 8, 8] would skip the 2nd/3rd int if we
        # reused the parent's seen set, since all three 8s share the same id).
        return [_cfg_to_dict(item, set()) for item in cfg]

    # Objects with __dict__ but not yacs proxies
    if hasattr(cfg, "__dict__") and not isinstance(cfg, type):
        result = {}
        for k, v in vars(cfg).items():
            if k.startswith("_"):
                continue
            result[k] = _cfg_to_dict(v, seen)
        return result

    # Primitives / serialisable leaf nodes
    return cfg


# --------------------------------------------------------------------------- #
# Environment snapshot
# --------------------------------------------------------------------------- #

def dump_environment(run_dir: Path | str | None) -> None:
    """
    Write ``environment.json`` to ``run_dir``.

    Records: git commit, git dirty flag, Python version, platform, torch version,
    torch_geometric version, CUDA availability, CUDA runtime version, GPU name,
    wandb version, and a UTC timestamp.

    If any tool is unavailable or git commands fail, the corresponding field is
    set to ``None`` — no exception is raised.
    """
    if not _is_valid_path(run_dir):
        return

    env: dict[str, Any] = {
        "git_commit": None,
        "git_dirty": None,
        "python_version": sys.version,
        "platform": sys.platform,
        "torch_version": None,
        "torch_geometric_version": None,
        "cuda_available": False,
        "cuda_runtime_version": None,
        "gpu_name": None,
        "wandb_version": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # git
    try:
        import subprocess
        git_dir_arg = ["-C", os.fspath(run_dir)] if False else []  # noqa: F504
        # Use repo root heuristic: run_dir may be deep; search upward
        rev_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=os.fspath(run_dir),
            timeout=10,
        )
        if rev_out.returncode == 0:
            env["git_commit"] = rev_out.stdout.strip()
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True,
            cwd=os.fspath(run_dir),
            timeout=10,
        )
        env["git_dirty"] = dirty_out.returncode == 0 and bool(dirty_out.stdout.strip())
    except Exception:
        pass

    # torch
    try:
        import torch as _torch
        env["torch_version"] = _torch.__version__
        env["cuda_available"] = _torch.cuda.is_available()
        if env["cuda_available"]:
            try:
                env["cuda_runtime_version"] = _torch.version.cuda
                env["gpu_name"] = _torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass

    # torch_geometric
    try:
        import torch_geometric as _pyg
        env["torch_geometric_version"] = _pyg.__version__
    except Exception:
        pass

    # wandb
    try:
        import wandb as _wandb
        env["wandb_version"] = _wandb.__version__
    except Exception:
        pass

    _write_json(run_dir / "environment.json", env)


# --------------------------------------------------------------------------- #
# Config dump (with password redaction)
# --------------------------------------------------------------------------- #

def dump_config(cfg, run_dir: Path | str | None) -> None:
    """
    Write ``config_resolved.yml`` to ``run_dir``.

    Serialises the full resolved CfgNode as a plain dict, excluding MagicMock
    objects and private (``_``-prefixed) keys. Database passwords are redacted.
    """
    if not _is_valid_path(run_dir):
        return

    cfg_dict = _cfg_to_dict(cfg)

    # Redact database password
    cfg_dict = _redact_database_password(cfg_dict)

    # B8 guard: When cfg is a real yacs CfgNode (is_frozen() returns bool),
    # _cfg_to_dict must not silently return {}.  Raising ValueError makes the
    # failure visible rather than producing a silent 0-byte file.
    # When cfg is MagicMock/SimpleNamespace, {} is acceptable (no real config).
    def _is_real_yacs(obj) -> bool:
        frozen = getattr(obj, "is_frozen", None)
        if frozen is not None:
            if callable(frozen):
                try:
                    return isinstance(frozen(), bool)
                except Exception:
                    return False
            return isinstance(frozen, bool)
        return False

    is_real_cfg = _is_real_yacs(cfg)
    if is_real_cfg and (cfg_dict is None or cfg_dict == {}):
        raise ValueError(
            f"cfg serialised to empty dict; aborting to avoid 0-byte "
            f"config_resolved.yml in {run_dir}.  Check that the CfgNode "
            f"is a valid yacs CfgNode."
        )

    # Write as YAML — raises ValueError if data is {} for real CfgNode
    _write_yaml(run_dir / "config_resolved.yml", cfg_dict)


def _redact_database_password(cfg_dict: dict) -> dict:
    """Recursively redact database.password from config dict."""
    if isinstance(cfg_dict, dict):
        result = {}
        for k, v in cfg_dict.items():
            if k == "password":
                result[k] = "[REDACTED]"
            else:
                result[k] = _redact_database_password(v)
        return result
    elif isinstance(cfg_dict, (list, tuple)):
        return [_redact_database_password(item) for item in cfg_dict]
    return cfg_dict


# --------------------------------------------------------------------------- #
# Runtime record
# --------------------------------------------------------------------------- #

def dump_runtime(
    cfg,
    run_dir: Path | str | None,
    *,
    status: str = "completed",
    executed_stages: list[str] | None = None,
    timing: dict[str, float] | None = None,
    wandb_mode: str | None = None,
    error_message: str | None = None,
) -> None:
    """
    Write ``runtime.json`` to ``run_dir``.

    Parameters
    ----------
    cfg:
        The resolved yacs CfgNode.
    run_dir:
        Target directory (None / MagicMock → no-op).
    status:
        ``"completed"`` or ``"failed"``.
    executed_stages:
        List of stages that actually ran.  If None, inferred from ``cfg._stages``.
    timing:
        Timing dict with keys such as ``time_total``, ``time_gnn_training``, …
        Skipped stages have value ``0.0``.
    wandb_mode:
        Resolved W&B mode (``disabled``, ``offline``, or ``online``).
    error_message:
        Short error string set when ``status == "failed"``.
    """
    if not _is_valid_path(run_dir):
        return

    run_dir = Path(run_dir)

    # Collect requested stages
    requested = list(getattr(cfg, "_stages", []))
    if not requested and hasattr(cfg, "dataset") and hasattr(cfg.dataset, "name"):
        requested = []   # pre-parse placeholder

    # Collect executed stages
    executed = executed_stages if executed_stages is not None else requested

    # Fill in timing with zeros for missing keys
    _ALL_TIMING_KEYS = [
        "time_total",
        "time_build_graphs",
        "time_embed_nodes",
        "time_embed_edges",
        "time_gnn_training",
        "time_gnn_testing",
        "time_evaluation",
        "time_tracing",
    ]
    timing = dict(timing) if timing else {}
    for k in _ALL_TIMING_KEYS:
        timing.setdefault(k, 0.0)

    # MagicMock/yacs proxy attributes must never leak into JSON telemetry.
    # Only resolved primitive values are truthful runtime metadata.
    is_smoke_raw = getattr(cfg, "_is_smoke", False)
    is_smoke = is_smoke_raw if isinstance(is_smoke_raw, bool) else False
    max_windows_raw = getattr(cfg, "_max_windows_per_split", None)
    max_windows_per_split = (
        max_windows_raw
        if isinstance(max_windows_raw, int) and not isinstance(max_windows_raw, bool)
        else None
    )

    runtime: dict[str, Any] = {
        "dataset": str(getattr(cfg.dataset, "name", None) if hasattr(cfg, "dataset") else None),
        "model_variant": str(
            getattr(cfg.detection, "gnn_training", None)
            and getattr(cfg.detection.gnn_training, "used_method", None)
            if hasattr(cfg, "detection") else None
        ),
        "seed": int(getattr(cfg, "_seed", 0)),
        "artifact_root": str(getattr(cfg, "_artifact_root", None) or ""),
        "run_dir": str(run_dir),
        "requested_stages": requested,
        "executed_stages": executed,
        "start_time": getattr(cfg, "_run_start_time", None),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "tracing_executed": "trace" in executed,
        "wandb_mode": wandb_mode,
        "time_total": timing.get("time_total", 0.0),
        "time_build_graphs": timing.get("time_build_graphs", 0.0),
        "time_embed_nodes": timing.get("time_embed_nodes", 0.0),
        "time_embed_edges": timing.get("time_embed_edges", 0.0),
        "time_gnn_training": timing.get("time_gnn_training", 0.0),
        "time_gnn_testing": timing.get("time_gnn_testing", 0.0),
        "time_evaluation": timing.get("time_evaluation", 0.0),
        "time_tracing": timing.get("time_tracing", 0.0),
        # C8-B: Bounded smoke metadata
        "is_smoke": is_smoke,
        "max_windows_per_split": max_windows_per_split,
    }

    if error_message:
        runtime["error_message"] = error_message

    # C8 stages may already have recorded detailed runtime sections.  Preserve
    # them when the pipeline-level summary is written at shutdown.
    # Preserve structured telemetry sections that may have been written by
    # individual pipeline stages (e.g., dataset_loader with nested training/testing).
    runtime_path = run_dir / "runtime.json"
    try:
        with runtime_path.open(encoding="utf-8") as handle:
            previous_runtime = json.load(handle)
    except (OSError, json.JSONDecodeError):
        previous_runtime = {}
    if isinstance(previous_runtime, dict):
        for section in ("dataset_loader", "training", "testing", "model"):
            if isinstance(previous_runtime.get(section), dict):
                runtime[section] = previous_runtime[section]

    _write_json(runtime_path, runtime)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _is_valid_path(path: Path | str | None) -> bool:
    """
    Return True only when ``path`` is a real, non-empty, non-MagicMock string/Path.

    This guard makes every dump function a no-op in unit tests that pass
    MagicMock objects as ``run_dir``.
    """
    if path is None:
        return False
    try:
        s = os.fspath(path)
    except Exception:
        return False
    if not s or "mock" in s.lower() or "magicmock" in type(path).__name__.lower():
        return False
    return True


def _write_json(path: Path | str, data: dict) -> None:
    """Write ``data`` as UTF-8 JSON, using a temp-file + atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=path.parent, delete=False, suffix=".tmp",
        ) as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            tmp_path = fh.name
        os.replace(tmp_path, path)   # atomic on POSIX and Windows
    except Exception:
        # If atomic write fails, try direct write as last resort
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)


def _write_yaml(path: Path | str, data: dict) -> None:
    """
    Write ``data`` as a human-readable YAML file without requiring pyyaml.

    Format: top-level keys are output as ``key:`` with nested dicts indented
    by two spaces.  Lists use ``-`` prefix.  This is sufficient for config
    readability; full YAML spec is not required here.

    Raises
    ------
    ValueError:
        If data is empty (None or {}) after stripping private keys.
        A 0-byte or empty YAML file is not acceptable for experiment
        reproducibility — callers must receive a clear failure rather than
        a silent empty artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # B8: detect silent empty serialization
    if data is None or data == {}:
        raise ValueError(
            "dump_config: cfg resolved to empty dict (None or {{}}). "
            "This indicates a serialization failure. Check that the CfgNode "
            "is a valid yacs CfgNode and that _cfg_to_dict handles all its "
            "attribute types. Will not write a 0-byte config_resolved.yml."
        )

    with open(path, "w", encoding="utf-8") as fh:
        _dump_yaml_node(fh, data, indent=0)


def _dump_yaml_node(fh, node, indent: int) -> None:
    """Recursively write a Python dict/list/primitive as YAML."""
    prefix = "  " * indent
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                fh.write(f"{prefix}{k}:\n")
                _dump_yaml_node(fh, v, indent + 1)
            else:
                fh.write(f"{prefix}{k}: {json.dumps(v) if isinstance(v, str) else v}\n")
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                fh.write(f"{prefix}- \n")
                _dump_yaml_node(fh, item, indent + 1)
            else:
                fh.write(f"{prefix}- {item}\n")
    else:
        fh.write(f"{prefix}{node}\n")
