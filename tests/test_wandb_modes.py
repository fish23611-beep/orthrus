"""Tests for wandb_control.py."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

from wandb_control import (
    resolve_wandb_mode,
    VALID_WANDB_MODES,
)


class TestResolveWandbMode:
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
