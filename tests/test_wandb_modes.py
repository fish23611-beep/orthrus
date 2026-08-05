"""
tests/test_wandb_modes.py

Unit tests for src/wandb_control.py.

Scope
=====
- disabled: no init/log/finish calls.
- offline: correct mode passed to wandb.init.
- online: correct mode passed to wandb.init.
- illegal mode raises ValueError listing allowed values.
- legacy --wandb flag maps to online when no explicit cfg mode.
- wandb.run=None: wandb_log and wandb_finish are no-ops.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from wandb_control import VALID_WANDB_MODES


torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch not installed",
)


def _make_cfg(wandb_mode=None):
    cfg = MagicMock()
    cfg.logging = MagicMock()
    # Use a real string so the cfg path works in tests
    if wandb_mode is not None:
        cfg.logging.wandb_mode = wandb_mode
    else:
        cfg.logging.wandb_mode = None
    return cfg


# --------------------------------------------------------------------------- #
# resolve_wandb_mode
# --------------------------------------------------------------------------- #

def test_disabled_default_when_no_cfg_no_cli():
    """Default is 'disabled' when no cfg.wandb_mode and no --wandb flag."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode=None)
    args = MagicMock()
    args.wandb = False

    assert resolve_wandb_mode(cfg, args) == "disabled"


def test_cfg_mode_wins_over_cli():
    """Explicit cfg.logging.wandb_mode takes priority over --wandb CLI flag."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode="offline")
    args = MagicMock()
    args.wandb = True   # would normally mean "online"

    assert resolve_wandb_mode(cfg, args) == "offline"


def test_cfg_mode_wins_over_default():
    """Explicit cfg.logging.wandb_mode wins even when --wandb is False."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode="online")
    args = MagicMock()
    args.wandb = False

    assert resolve_wandb_mode(cfg, args) == "online"


def test_wandb_flag_maps_to_online():
    """--wandb=True with no explicit cfg mode must resolve to 'online'."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode=None)
    args = MagicMock()
    args.wandb = True

    assert resolve_wandb_mode(cfg, args) == "online"


def test_illegal_mode_raises_valueerror():
    """Illegal wandb_mode value must raise ValueError listing allowed values."""
    from wandb_control import resolve_wandb_mode, VALID_WANDB_MODES

    cfg = _make_cfg(wandb_mode="bad_mode")
    args = MagicMock()
    args.wandb = False

    with pytest.raises(ValueError) as exc_info:
        resolve_wandb_mode(cfg, args)

    msg = str(exc_info.value)
    assert "bad_mode" in msg
    # Must list the allowed modes
    for mode in VALID_WANDB_MODES:
        assert mode in msg


def test_explicit_none_cfg_mode_falls_through_to_cli():
    """cfg.wandb_mode=None means 'unset', so --wandb flag still applies."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode=None)
    args = MagicMock()
    args.wandb = True

    assert resolve_wandb_mode(cfg, args) == "online"


def test_explicit_disabled_cfg():
    """cfg.wandb_mode='disabled' must be accepted and return 'disabled'."""
    from wandb_control import resolve_wandb_mode

    cfg = _make_cfg(wandb_mode="disabled")
    args = MagicMock()
    args.wandb = True  # would mean online, but cfg takes priority

    assert resolve_wandb_mode(cfg, args) == "disabled"


# --------------------------------------------------------------------------- #
# init_wandb
# --------------------------------------------------------------------------- #

@patch("wandb.init")
def test_init_disabled_returns_none(m_init):
    """mode='disabled' must not call wandb.init and must return None."""
    from wandb_control import init_wandb

    cfg = _make_cfg(wandb_mode="disabled")
    args = MagicMock()
    args.exp = ""
    args.tags = ""

    result = init_wandb(cfg, args, mode="disabled")

    assert result is None
    m_init.assert_not_called()


@patch("wandb.init")
def test_init_offline_calls_wandb_init(m_init):
    """mode='offline' must call wandb.init with mode='offline'."""
    from wandb_control import init_wandb

    cfg = _make_cfg(wandb_mode=None)
    args = MagicMock()
    args.exp = "my_exp"
    args.tags = "tag1,tag2"

    result = init_wandb(cfg, args, mode="offline")

    m_init.assert_called_once()
    call_kwargs = m_init.call_args
    assert call_kwargs[1].get("mode") == "offline"


@patch("wandb.init")
def test_init_online_calls_wandb_init(m_init):
    """mode='online' must call wandb.init with mode='online'."""
    from wandb_control import init_wandb

    cfg = _make_cfg(wandb_mode=None)
    args = MagicMock()
    args.exp = "my_exp"
    args.tags = ""

    result = init_wandb(cfg, args, mode="online")

    m_init.assert_called_once()
    call_kwargs = m_init.call_args
    assert call_kwargs[1].get("mode") == "online"


# --------------------------------------------------------------------------- #
# wandb_log safe wrapper
# --------------------------------------------------------------------------- #

@patch("wandb.log")
def test_wandb_log_noop_when_run_is_none(m_log):
    """wandb_log must not call wandb.log when wandb.run is None."""
    import wandb as real_wandb
    real_wandb.run = None
    from wandb_control import wandb_log

    wandb_log({"train_loss": 0.5})

    m_log.assert_not_called()


@patch("wandb.log")
def test_wandb_log_calls_wandb_when_run_active(m_log):
    """wandb_log must call wandb.log when wandb.run is not None."""
    import wandb as real_wandb
    real_wandb.run = MagicMock()
    from wandb_control import wandb_log

    wandb_log({"train_loss": 0.5}, step=1)

    m_log.assert_called_once()
    args, kwargs = m_log.call_args
    assert args[0]["train_loss"] == 0.5


@patch("wandb.log")
def test_wandb_log_noop_on_exception(m_log):
    """wandb_log must not raise if wandb.log raises."""
    import wandb as real_wandb
    real_wandb.run = MagicMock()
    from wandb_control import wandb_log

    m_log.side_effect = RuntimeError("wandb error")

    # Must not raise
    wandb_log({"metric": 1.0})


# --------------------------------------------------------------------------- #
# wandb_finish safe wrapper
# --------------------------------------------------------------------------- #

@patch("wandb.finish")
def test_wandb_finish_noop_when_run_is_none(m_finish):
    """wandb_finish must not call wandb.finish when wandb.run is None."""
    import wandb as real_wandb
    real_wandb.run = None
    from wandb_control import wandb_finish

    wandb_finish()

    m_finish.assert_not_called()


@patch("wandb.finish")
def test_wandb_finish_calls_finish_when_run_active(m_finish):
    """wandb_finish must call wandb.finish when wandb.run is not None."""
    import wandb as real_wandb
    real_wandb.run = MagicMock()
    from wandb_control import wandb_finish

    wandb_finish()

    m_finish.assert_called_once()


@patch("wandb.finish")
def test_wandb_finish_noop_on_exception(m_finish):
    """wandb_finish must not raise if wandb.finish raises."""
    import wandb as real_wandb
    real_wandb.run = MagicMock()
    from wandb_control import wandb_finish

    m_finish.side_effect = RuntimeError("wandb finish error")

    # Must not raise
    wandb_finish()
    """Tests for resolve_wandb_mode function."""

    def test_disabled_mode_default(self):
        """Without wandb flag, returns disabled."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="disabled"))
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "disabled"

    def test_offline_mode(self):
        """Offline mode from cfg is respected."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="offline"))
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "offline"

    def test_cfg_mode_takes_priority(self):
        """Cfg mode takes priority over --wandb flag."""
        # Test that cfg.logging.wandb_mode takes priority
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="offline"))
        args = SimpleNamespace(wandb=True)
        result = resolve_wandb_mode(cfg, args)
        assert result == "offline"  # cfg takes priority

    def test_cfg_overrides_wandb_flag(self):
        """Cfg mode takes priority over --wandb flag."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="offline"))
        args = SimpleNamespace(wandb=True)
        result = resolve_wandb_mode(cfg, args)
        assert result == "offline"  # cfg takes priority over CLI flag

    def test_case_insensitive(self):
        """Mode comparison is case-insensitive."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="OFFLINE"))
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "offline"

    def test_whitespace_stripped(self):
        """Whitespace is stripped from mode."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="  offline  "))
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "offline"

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        cfg = SimpleNamespace(logging=SimpleNamespace(wandb_mode="invalid_mode"))
        args = SimpleNamespace(wandb=False)
        with pytest.raises(ValueError, match="Invalid cfg.logging.wandb_mode"):
            resolve_wandb_mode(cfg, args)

    def test_no_logging_section(self):
        """Missing logging section returns default."""
        cfg = SimpleNamespace()
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "disabled"

    def test_no_wandb_mode_attr(self):
        """Missing wandb_mode attribute returns default."""
        cfg = SimpleNamespace(logging=SimpleNamespace())
        args = SimpleNamespace(wandb=False)
        result = resolve_wandb_mode(cfg, args)
        assert result == "disabled"


class TestValidModes:
    """Tests for VALID_WANDB_MODES constant."""

    def test_valid_modes_defined(self):
        """Valid modes are defined."""
        assert "disabled" in VALID_WANDB_MODES
        assert "offline" in VALID_WANDB_MODES
        assert "online" in VALID_WANDB_MODES
        assert len(VALID_WANDB_MODES) == 3
