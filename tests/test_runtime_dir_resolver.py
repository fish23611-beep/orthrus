"""
tests/test_runtime_dir_resolver.py

Regression tests for the runtime metadata directory resolver.

These tests pin the contract that the resolver must NEVER return a directory
whose parent is the filesystem root.  When ``cfg._run_dir`` is missing /
invalid AND the only available fallback would resolve to the filesystem root,
the resolver must return ``None`` so the metadata instrumentation is a safe
no-op rather than attempting to write ``/environment.json`` (which a
non-root user cannot do).

Coverage:
  1. Explicit valid ``cfg._run_dir`` is preserved verbatim.
  2. Normal legacy fallback (``fallback_dir`` whose parent is a real path).
  3. Dangerous fallback (``fallback_dir`` whose parent is filesystem root).
  4. MagicMock ``cfg._run_dir`` never produces a junk path.
  5. The metadata write functions never attempt to write to filesystem root,
     even when the resolver returned ``None`` upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mstc.experiment_utils import (  # noqa: E402
    _is_filesystem_root,
    _is_valid_runtime_path,
    dump_environment,
    resolve_runtime_dir,
    update_runtime,
    update_runtime_nested,
)
from run_metadata import _is_valid_path  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Explicit valid cfg._run_dir is preserved verbatim
# --------------------------------------------------------------------------- #

def test_explicit_run_dir_is_preserved(tmp_path):
    """An explicit cfg._run_dir wins over any fallback."""
    cfg = MagicMock()
    cfg._run_dir = str(tmp_path / "run1")

    result = resolve_runtime_dir(cfg, str(tmp_path / "run1" / "trained_models"))

    assert result == str(tmp_path / "run1"), (
        f"Expected explicit _run_dir to win, got {result!r}"
    )


def test_explicit_run_dir_path_object_is_preserved(tmp_path):
    """cfg._run_dir as a pathlib.Path must also win."""
    cfg = MagicMock()
    cfg._run_dir = tmp_path / "run1"

    result = resolve_runtime_dir(cfg, str(tmp_path / "run1" / "trained_models"))

    assert result == str(tmp_path / "run1"), (
        f"Expected explicit _run_dir to win, got {result!r}"
    )


# --------------------------------------------------------------------------- #
# 2. Normal legacy fallback
# --------------------------------------------------------------------------- #

def test_normal_legacy_fallback_returns_parent(tmp_path):
    """When _run_dir is invalid and fallback has a non-root parent, return parent."""
    cfg = MagicMock()
    cfg._run_dir = None

    fallback = str(tmp_path / "run1" / "trained_models")
    result = resolve_runtime_dir(cfg, fallback)

    assert result == str(tmp_path / "run1"), (
        f"Expected parent of fallback, got {result!r}"
    )


# --------------------------------------------------------------------------- #
# 3. Dangerous fallback (parent is filesystem root)
# --------------------------------------------------------------------------- #

def test_dangerous_fallback_returns_none():
    """fallback_dir='/fake' has parent='/' — resolver must return None."""
    cfg = MagicMock()
    cfg._run_dir = None

    result = resolve_runtime_dir(cfg, "/fake")

    assert result is None, (
        f"Resolver must reject root-parent fallback, got {result!r}. "
        "Writing to '/' is a runtime metadata path safety violation."
    )


def test_filesystem_root_path_is_rejected():
    """_is_filesystem_root must identify '/'."""
    assert _is_filesystem_root("/") is True
    assert _is_filesystem_root("/fake") is False
    assert _is_filesystem_root("/a/b") is False
    assert _is_filesystem_root("") is False
    assert _is_filesystem_root(None) is False


def test_is_valid_runtime_path_rejects_root():
    """An explicit root path must not be treated as a runtime directory."""
    assert _is_valid_runtime_path("/") is False
    assert _is_valid_runtime_path("/some/real/run") is True
    assert _is_valid_runtime_path("") is False
    assert _is_valid_runtime_path(None) is False


# --------------------------------------------------------------------------- #
# 4. MagicMock cfg._run_dir never produces a junk path
# --------------------------------------------------------------------------- #

def test_magicmock_run_dir_does_not_produce_junk_path():
    """MagicMock cfg._run_dir must NOT synthesize a junk __fspath__ string."""
    cfg = MagicMock()
    cfg._run_dir = MagicMock(name="magicmock_run_dir")

    result = resolve_runtime_dir(cfg, "/fake")

    # The dangerous fallback must short-circuit before any path synthesis.
    assert result is None, (
        f"MagicMock _run_dir must not let the fallback chain reach /, got {result!r}"
    )


def test_magicmock_run_dir_with_valid_fallback_uses_parent():
    """When MagicMock _run_dir exists but fallback is valid, parent wins."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MagicMock()
        cfg._run_dir = MagicMock(name="magicmock_run_dir")
        fallback = str(Path(tmpdir) / "run1" / "trained_models")

        result = resolve_runtime_dir(cfg, fallback)
        assert result == str(Path(tmpdir) / "run1"), (
            f"MagicMock rejection should fall through to legacy fallback, got {result!r}"
        )


def test_is_valid_path_in_run_metadata_rejects_root():
    """The legacy _is_valid_path in run_metadata must also reject root."""
    assert _is_valid_path("/") is False
    assert _is_valid_path("/some/real/run") is True
    assert _is_valid_path(None) is False
    assert _is_valid_path("") is False
    assert _is_valid_path(MagicMock()) is False


# --------------------------------------------------------------------------- #
# 5. Write functions never touch filesystem root
# --------------------------------------------------------------------------- #

def test_dump_environment_never_writes_to_root(monkeypatch):
    """dump_environment(cfg, None) must not write anywhere — including '/'.

    Simulates the production bug path: when resolve_runtime_dir returns
    None (because the fallback would resolve to root), the upstream caller
    passes ``None`` to ``dump_environment``.  The write function must treat
    this as a no-op without writing to ``/``.
    """
    # Use a sentinel that fails the test if any IO reaches '/'.
    written = []

    real_path_init = Path.__init__

    def tracking_path_init(self, *args, **kwargs):
        real_path_init(self, *args, **kwargs)
        # If any Path is constructed with the filesystem root, capture it.
        if str(self) == "/":
            written.append(("/path-init", args, kwargs))

    monkeypatch.setattr(Path, "__init__", tracking_path_init)

    cfg = MagicMock()
    cfg.dataset = MagicMock()
    cfg.dataset.name = "TEST"
    cfg.model = MagicMock()
    cfg.model.variant = "mstc"
    cfg.detection = MagicMock()
    cfg.detection.gnn_training = MagicMock()
    cfg.detection.gnn_training.used_method = "mstc"

    # Calling dump_environment with None (the value resolve_runtime_dir
    # returns when fallback is unsafe) must be a no-op.
    result = dump_environment(cfg, None)
    assert result == Path(), (
        f"dump_environment(None) must return Path() (no-op), got {result!r}"
    )
    assert written == [], (
        f"dump_environment must not construct any root path, got {written!r}"
    )


def test_update_runtime_never_writes_to_root(monkeypatch):
    """update_runtime(None, ...) must not touch the filesystem."""
    written = []

    real_path_init = Path.__init__

    def tracking_path_init(self, *args, **kwargs):
        real_path_init(self, *args, **kwargs)
        if str(self) == "/":
            written.append("/path-init")

    monkeypatch.setattr(Path, "__init__", tracking_path_init)

    update_runtime(None, "testing", {"events_per_second": 1.0})
    assert written == [], (
        f"update_runtime(None, ...) must not construct root paths, got {written!r}"
    )


def test_update_runtime_nested_never_writes_to_root(monkeypatch):
    """update_runtime_nested(None, ...) must not touch the filesystem."""
    written = []

    real_path_init = Path.__init__

    def tracking_path_init(self, *args, **kwargs):
        real_path_init(self, *args, **kwargs)
        if str(self) == "/":
            written.append("/path-init")

    monkeypatch.setattr(Path, "__init__", tracking_path_init)

    update_runtime_nested(None, "dataset_loader", "testing", {"events": 1})
    assert written == [], (
        f"update_runtime_nested(None, ...) must not construct root paths, got {written!r}"
    )


def test_dump_environment_with_root_dir_is_noop():
    """Even if a caller passes '/' directly, dump_environment must refuse."""
    cfg = MagicMock()
    cfg.dataset = MagicMock()
    cfg.dataset.name = "TEST"

    result = dump_environment(cfg, "/")
    # Must return the no-op sentinel and not raise / not write.
    assert result == Path()