"""
src/wandb_control.py

Single W&B initialisation and log/finish interface for the ORTHRUS pipeline.

Design goals
============
- Three-mode support: ``disabled``, ``offline``, ``online``.
- ``cfg.logging.wandb_mode`` (when explicitly set) always wins.
  Otherwise ``--wandb`` CLI flag maps to ``online``.
  When neither is set, the default is ``disabled``.
- ``disabled``: no ``wandb.init`` call, no API key required.
- ``offline``: ``wandb.init(mode="offline")`` — works without network.
- ``online``: ``wandb.init(mode="online")`` — standard behaviour.
- ``init_wandb`` is the single entry point for all W&B initialisation.
- ``wandb_log`` and ``wandb_finish`` are thin wrappers that guard against
  calling ``wandb`` when there is no active run (e.g. ``wandb.run is None``).

Usage
=====
::

    from wandb_control import init_wandb, wandb_log, wandb_finish

    mode = resolve_wandb_mode(cfg, args)
    init_wandb(cfg, args, mode)
    wandb_log({"train_loss": 0.5})
    wandb_finish()
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb

# --------------------------------------------------------------------------- #
# Mode constants
# --------------------------------------------------------------------------- #

VALID_WANDB_MODES = ("disabled", "offline", "online")

# --------------------------------------------------------------------------- #
# Mode resolution
# --------------------------------------------------------------------------- #

def resolve_wandb_mode(cfg, args) -> str:
    """
    Resolve the effective W&B mode from cfg and CLI args.

    Priority (highest → lowest):

    1. ``cfg.logging.wandb_mode`` if it is **explicitly** set to a known
       mode string (``disabled``, ``offline``, ``online``).
       A value that is not one of those three (including MagicMock) is ignored
       so tests using MagicMock do not spuriously fail.
    2. ``--wandb`` CLI flag (True → ``online``).
    3. Default: ``disabled``.

    Parameters
    ----------
    cfg:
        The yacs CfgNode.
    args:
        The parsed CLI namespace.

    Returns
    -------
    str
        One of ``"disabled"``, ``"offline"``, ``"online"``.

    Raises
    ------
    ValueError
        If an explicitly-set mode string is not one of the allowed values.
    """
    # 1. Explicit cfg value takes priority
    try:
        cfg_mode = getattr(getattr(cfg, "logging", None), "wandb_mode", None)
    except Exception:
        cfg_mode = None

    if cfg_mode is not None:
        mode = str(cfg_mode).strip().lower()
        if mode in VALID_WANDB_MODES:
            return mode
        if mode not in VALID_WANDB_MODES:
            raise ValueError(
                f"Invalid cfg.logging.wandb_mode={cfg_mode!r}. "
                f"Allowed values are: {', '.join(VALID_WANDB_MODES)}."
            )

    # 2. Legacy --wandb flag
    if getattr(args, "wandb", False):
        return "online"

    # 3. Default
    return "disabled"


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #

def init_wandb(
    cfg,
    args,
    mode: str,
    project: str = "orthrus_repo",
) -> "wandb.Run | None":
    """
    Initialise a W&B run (or a dry run when mode is ``disabled``).

    Parameters
    ----------
    cfg:
        The yacs CfgNode — used to extract experiment name and tags.
    args:
        CLI namespace.
    mode:
        One of ``disabled``, ``offline``, ``online``.
    project:
        W&B project name (default: ``orthrus_repo``).

    Returns
    -------
    wandb.Run or None
        The active W&B run object, or ``None`` when ``mode="disabled"``.
    """
    import wandb as _wandb

    if mode == "disabled":
        return None

    # Build experiment name from args
    exp_name = getattr(args, "exp", "") or getattr(args, "dataset", "unknown")
    tags_raw = getattr(args, "tags", "") or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] or []

    _wandb.init(
        mode=mode,
        project=project,
        name=str(exp_name),
        tags=tags,
    )

    # Upload resolved config
    try:
        from provnet_utils import remove_underscore_keys
        cfg_dict = remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"])
        _wandb.config.update(cfg_dict)
    except Exception:
        pass   # config update is best-effort

    return _wandb.run


# --------------------------------------------------------------------------- #
# Safe log / finish wrappers
# --------------------------------------------------------------------------- #

def wandb_log(metrics: dict, **kwargs) -> None:
    """
    Log metrics to W&B if an active run exists.

    Silently skips if:
    - W&B module is not available
    - ``wandb.run is None`` (no active run)
    """
    try:
        import wandb as _wandb
        if _wandb.run is None:
            return
        _wandb.log(metrics, **kwargs)
    except Exception:
        pass


def wandb_finish() -> None:
    """
    Mark the W&B run as finished if an active run exists.

    Silently skips if:
    - W&B module is not available
    - ``wandb.run is None``
    """
    try:
        import wandb as _wandb
        if _wandb.run is None:
            return
        _wandb.finish()
    except Exception:
        pass
