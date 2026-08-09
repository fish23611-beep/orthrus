"""Small, dependency-light utilities shared by C8 experiment stages.

The helpers deliberately keep experiment bookkeeping separate from model and
temporal-history state.  In particular, runtime updates are merged by section
so training and testing can run independently without erasing one another.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


_SENSITIVE_KEY_PARTS = ("password", "secret", "token", "api_key", "credential")
_RUNTIME_CONFIG_KEYS = {"resume_checkpoint", "resume_from"}


def _is_sensitive_key(key: object) -> bool:
    lowered = str(key).lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _to_plain(value: Any, *, redact: bool = True, for_hash: bool = False) -> Any:
    """Convert cfg-like objects to stable, JSON/YAML-safe plain values.

    This returns a new object and never mutates the live configuration.
    """
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.startswith("_") or (for_hash and key_text in _RUNTIME_CONFIG_KEYS):
                continue
            result[key_text] = "[REDACTED]" if redact and _is_sensitive_key(key_text) else _to_plain(
                item, redact=redact, for_hash=for_hash
            )
        return result
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item, redact=redact, for_hash=for_hash) for item in value]
    if hasattr(value, "items"):
        try:
            return _to_plain(dict(value.items()), redact=redact, for_hash=for_hash)
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        attrs = {
            key: item for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if attrs:
            return _to_plain(attrs, redact=redact, for_hash=for_hash)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitized_config(cfg: Any) -> dict[str, Any]:
    """Return a redacted copy of the resolved configuration."""
    plain = _to_plain(cfg, redact=True)
    return plain if isinstance(plain, dict) else {"config": plain}


def stable_config_hash(cfg: Any) -> str:
    """Hash effective configuration deterministically, excluding secrets/runtime paths."""
    plain = _to_plain(cfg, redact=True, for_hash=True)
    payload = json.dumps(plain, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cfg_value(cfg: Any, *path: str) -> Any:
    current = cfg
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _environment_scalar(value: Any) -> str | int | float | bool | None:
    """Keep environment metadata JSON-safe when test doubles are supplied."""
    return value if isinstance(value, (str, int, float, bool)) or value is None else None


def _valid_out_dir(value: Any) -> bool:
    """Reject mocks and non-path objects so instrumentation never creates junk paths."""
    return isinstance(value, (str, Path)) and bool(os.fspath(value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        yaml.safe_dump(dict(value), handle, allow_unicode=True, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def dump_environment(cfg: Any, out_dir: str | Path) -> Path:
    """Write ``environment.json`` and sanitized ``config_resolved.yml``.

    Optional software, CUDA, and Git metadata are best-effort; unavailable
    values are represented as ``null`` instead of failing CPU-only runs.
    """
    if not _valid_out_dir(out_dir):
        return Path()
    out_dir = Path(out_dir)
    cuda_available = bool(torch.cuda.is_available())
    pyg_version = None
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except Exception:
        pass
    environment = {
        "dataset": _environment_scalar(_cfg_value(cfg, "dataset", "name")),
        "model": _environment_scalar(_cfg_value(cfg, "model", "variant")) or _environment_scalar(_cfg_value(cfg, "detection", "gnn_training", "used_method")),
        "experiment_name": _environment_scalar(getattr(cfg, "_experiment_name", None)) or _environment_scalar(getattr(cfg, "exp", None)),
        "seed": _environment_scalar(getattr(cfg, "_seed", None)),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "pyg_version": pyg_version,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "start_time": _environment_scalar(getattr(cfg, "_run_start_time", None)) or datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(out_dir / "environment.json", environment)
    _atomic_yaml(out_dir / "config_resolved.yml", sanitized_config(cfg))
    return out_dir


def update_runtime(out_dir: str | Path, section: str, values: Mapping[str, Any]) -> Path:
    """Atomically merge one C8 runtime section without deleting other sections."""
    if not _valid_out_dir(out_dir):
        return Path()
    path = Path(out_dir) / "runtime.json"
    existing: dict[str, Any] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded
    except (OSError, json.JSONDecodeError):
        pass
    current = existing.get(section)
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(_to_plain(dict(values), redact=False))
    existing[section] = merged
    _atomic_json(path, existing)
    return path


def events_per_second(processed_event_count: int, measured_seconds: float) -> float:
    """Return true event throughput, or NaN when elapsed time is not positive."""
    return float(processed_event_count) / measured_seconds if measured_seconds > 0 else float("nan")


def peak_cpu_memory_mb() -> float | None:
    """Best-effort process peak RSS; it is process-wide rather than stage-exact."""
    try:
        import resource
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return None


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and available CUDA Torch RNG state."""
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    """Restore all RNG states present in a structured checkpoint."""
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    cuda_states = checkpoint.get("torch_cuda_rng_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
