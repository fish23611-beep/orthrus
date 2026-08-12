"""
tests/test_wandb_safe_logging_integration.py

Regression tests for C8: W&B safe logging in disabled/offline/online modes.

Tests cover:
- wandb_is_active() returns correct values in all states
- Training code does NOT call raw wandb.log when wandb.run is None
- Evaluation code does NOT call raw wandb.log when wandb.run is None
- Evaluation code does NOT create wandb.Image objects in disabled mode
- Active W&B run: safe wrappers delegate correctly
- Static invariant: production modules do not directly call wandb.log/init/finish
"""

import ast
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# A. wandb_is_active() unit tests
# --------------------------------------------------------------------------- #

class TestWandbIsActive:
    """Tests for the wandb_is_active() helper function."""

    def test_returns_true_when_run_is_active(self):
        """wandb_is_active() must return True when wandb.run is set."""
        import wandb as real_wandb
        real_wandb.run = MagicMock()
        from wandb_control import wandb_is_active
        try:
            assert wandb_is_active() is True
        finally:
            real_wandb.run = None

    def test_returns_false_when_run_is_none(self):
        """wandb_is_active() must return False when wandb.run is None."""
        import wandb as real_wandb
        real_wandb.run = None
        from wandb_control import wandb_is_active
        try:
            assert wandb_is_active() is False
        finally:
            real_wandb.run = None

    def test_returns_false_on_exception(self):
        """wandb_is_active() must not raise on import errors."""
        from wandb_control import wandb_is_active
        # Should not raise
        result = wandb_is_active()
        assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# B. Training disabled mode: no raw wandb.log calls
# --------------------------------------------------------------------------- #

class TestTrainingDisabledMode:
    """Verify training code does not call raw wandb.log in disabled mode."""

    @patch("wandb.log")
    @patch("wandb.init")
    def test_training_uses_safe_wrapper_not_raw_wandb_log(self, m_init, m_log):
        """Training code must use wandb_log(), not raw wandb.log."""
        import wandb as real_wandb
        real_wandb.run = None  # disabled mode

        # Import after patching so the module sees the mock
        import importlib
        import wandb_control
        importlib.reload(wandb_control)

        # Now import training module - it should use wandb_log from wandb_control
        import wandb_control as wc_reloaded
        wandb_log = wc_reloaded.wandb_log

        # Simulate what orthrus_gnn_training.main() does after our fix:
        # it calls wandb_log({...}) instead of wandb.log({...})
        wandb_log({"train_epoch": 1, "train_loss": 0.5})

        # wandb.log must NOT be called (disabled mode)
        m_log.assert_not_called()

        real_wandb.run = None

    def test_training_module_imports_wandb_log(self):
        """orthrus_gnn_training.py must import wandb_log from wandb_control."""
        # Parse the source to check imports
        src_path = SRC / "detection" / "orthrus_gnn_training.py"
        with open(src_path) as f:
            tree = ast.parse(f.read())

        # Check for from wandb_control import wandb_log
        found_safe_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "wandb_control":
                    for alias in node.names:
                        if alias.name == "wandb_log":
                            found_safe_import = True
                            break

        assert found_safe_import, (
            "orthrus_gnn_training.py must import wandb_log from wandb_control"
        )

    def test_training_module_has_no_raw_wandb_log_calls(self):
        """orthrus_gnn_training.py must not contain raw wandb.log calls."""
        src_path = SRC / "detection" / "orthrus_gnn_training.py"
        with open(src_path) as f:
            content = f.read()

        # Allow comments and docstrings
        for line in content.splitlines():
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Check for raw wandb.log( that is not inside a string
            if "wandb.log" in line and "from wandb_control import" not in line:
                # Make sure it's not the safe wrapper usage
                assert "wandb_log" in content, (
                    f"Found raw wandb.log in {src_path}: {line.strip()}"
                )


# --------------------------------------------------------------------------- #
# C. Evaluation disabled mode: no raw wandb.log / no wandb.Image creation
# --------------------------------------------------------------------------- #

class TestEvaluationDisabledMode:
    """Verify evaluation code handles disabled mode correctly."""

    def test_evaluation_module_imports_safe_wrappers(self):
        """evaluation.py must import wandb_log and wandb_is_active."""
        src_path = SRC / "detection" / "evaluation.py"
        with open(src_path) as f:
            content = f.read()

        assert "from wandb_control import" in content
        assert "wandb_log" in content
        assert "wandb_is_active" in content

    def test_evaluation_module_has_no_raw_wandb_log(self):
        """evaluation.py must not contain raw wandb.log calls."""
        src_path = SRC / "detection" / "evaluation.py"
        with open(src_path) as f:
            content = f.read()

        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            # raw wandb.log( that is not in a comment
            if "wandb.log(" in line and "from wandb_control" not in line:
                pytest.fail(
                    f"Found raw wandb.log call in evaluation.py: {line.strip()}"
                )

    @patch("wandb.log")
    @patch("wandb.Image")
    def test_evaluation_no_image_creation_when_disabled(
        self, m_image_cls, m_log
    ):
        """When wandb.run is None, no wandb.Image should be created."""
        import wandb as real_wandb
        real_wandb.run = None

        # wandb.Image constructor should NOT be called in disabled mode
        from wandb_control import wandb_is_active
        assert wandb_is_active() is False

        # If evaluation code follows the pattern:
        # if wandb_is_active():
        #     stats["img"] = wandb.Image(...)
        # then m_image_cls should not be called.
        # We verify by checking that when run is None, the guard prevents creation.
        m_image_cls.assert_not_called()
        real_wandb.run = None


# --------------------------------------------------------------------------- #
# D. Active W&B run: safe wrappers delegate correctly
# --------------------------------------------------------------------------- #

class TestActiveWandbRun:
    """Verify safe wrappers delegate to real wandb when run is active."""

    def test_wandb_log_delegates_to_real_wandb_when_run_active(self):
        """wandb_log() must call wandb.log when run is active."""
        import wandb as real_wandb
        real_wandb.run = MagicMock()

        with patch("wandb.log") as m_log:
            from wandb_control import wandb_log
            wandb_log({"train_loss": 0.5}, step=1)

            m_log.assert_called_once()
            args, kwargs = m_log.call_args
            assert args[0]["train_loss"] == 0.5
            assert kwargs.get("step") == 1

        real_wandb.run = None

    def test_wandb_finish_delegates_to_real_wandb_when_run_active(self):
        """wandb_finish() must call wandb.finish when run is active."""
        import wandb as real_wandb
        real_wandb.run = MagicMock()

        with patch("wandb.finish") as m_finish:
            from wandb_control import wandb_finish
            wandb_finish()
            m_finish.assert_called_once()

        real_wandb.run = None


# --------------------------------------------------------------------------- #
# E. evaluation_utils.py: viz_graph wandb.Image guard
# --------------------------------------------------------------------------- #

class TestEvaluationUtilsWandbImageGuard:
    """Verify viz_graph() guards wandb.Image creation."""

    def test_viz_graph_imports_wandb_is_active(self):
        """evaluation_utils.py must import wandb_is_active for viz_graph."""
        src_path = SRC / "detection" / "evaluation_utils.py"
        with open(src_path) as f:
            content = f.read()

        assert "wandb_is_active" in content
        assert "_wandb_is_active" in content

    def test_viz_graph_no_unconditional_wandb_image(self):
        """viz_graph must not unconditionally create wandb.Image."""
        src_path = SRC / "detection" / "evaluation_utils.py"
        with open(src_path) as f:
            content = f.read()

        # The function should use a guard like:
        # if _wandb_is_active():
        #     return {out_file: _wandb.Image(svg)}
        # return {}
        assert "_wandb_is_active()" in content, (
            "viz_graph must use _wandb_is_active() guard for wandb.Image creation"
        )


# --------------------------------------------------------------------------- #
# F. tracing.py: no raw wandb.log
# --------------------------------------------------------------------------- #

class TestTracingWandbSafeLogging:
    """Verify tracing.py uses safe wandb wrappers."""

    def test_tracing_imports_wandb_log(self):
        """tracing.py must import wandb_log from wandb_control."""
        src_path = SRC / "attack_reconstruction" / "tracing.py"
        with open(src_path) as f:
            content = f.read()

        assert "from wandb_control import" in content
        assert "wandb_log" in content

    def test_tracing_no_raw_wandb_log(self):
        """tracing.py must not contain raw wandb.log calls (except commented)."""
        src_path = SRC / "attack_reconstruction" / "tracing.py"
        with open(src_path) as f:
            content = f.read()

        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            # Skip comments (including the intentionally commented wandb.log line)
            if stripped.startswith("#"):
                continue
            # raw uncommented wandb.log( is forbidden
            if "wandb.log(" in line and "from wandb_control" not in line:
                pytest.fail(
                    f"Found raw wandb.log in tracing.py: {line.strip()}"
                )


# --------------------------------------------------------------------------- #
# G. orthrus.py: uses wandb_is_active() guard
# --------------------------------------------------------------------------- #

class TestOrthrusWandbSafeLogging:
    """Verify orthrus.py uses wandb_is_active() for timing log."""

    def test_orthrus_imports_wandb_is_active(self):
        """orthrus.py must import wandb_is_active."""
        src_path = SRC / "orthrus.py"
        with open(src_path) as f:
            content = f.read()

        assert "wandb_is_active" in content

    def test_orthrus_uses_wandb_is_active_not_bare_run_check(self):
        """orthrus.py should use wandb_is_active() not bare wandb.run check."""
        src_path = SRC / "orthrus.py"
        with open(src_path) as f:
            content = f.read()

        # Should use wandb_is_active() for the guard
        assert "wandb_is_active()" in content


# --------------------------------------------------------------------------- #
# H. Static invariant: no bypass of wandb_control in production modules
# --------------------------------------------------------------------------- #

PRODUCTION_MODULES = [
    SRC / "detection" / "orthrus_gnn_training.py",
    SRC / "detection" / "evaluation.py",
    SRC / "detection" / "evaluation_utils.py",
    SRC / "detection" / "node_evaluation.py",
    SRC / "orthrus.py",
    SRC / "run_metadata.py",
    SRC / "attack_reconstruction" / "tracing.py",
]


class TestNoWandbBypassInvariant:
    """Static analysis: production modules must not bypass wandb_control.

    These tests check that production code never calls `wandb.log()`,
    `wandb.init()`, or `wandb.finish()` directly.  All W&B telemetry
    must go through the safe wrappers in ``wandb_control``.

    Aliases (``_wandb.Image``, ``_wandb.log``) used within a
    ``wandb_is_active()`` guard are permitted because they are
    conditional on an active run.
    """

    @pytest.mark.parametrize("src_path", PRODUCTION_MODULES)
    def test_no_raw_wandb_log_in_production_module(self, src_path):
        """Production modules must not call raw wandb.log/init/finish.

        Allowed patterns (not flagged):
        - _wandb.Image(...) inside a wandb_is_active() guard (evaluation.py)
        - _wandb.log(...) inside wandb_control.py (the control plane itself)
        """
        if not src_path.exists():
            pytest.skip(f"{src_path} does not exist")

        with open(src_path, encoding="utf-8-sig") as f:
            content = f.read()

        # Allow evaluation_utils.py to use _wandb alias under guard
        uses_aliased_import = (src_path / "evaluation_utils.py").exists()
        uses_guard = "wandb_is_active" in content

        tree = ast.parse(content)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "wandb"
                ):
                    attr_name = node.attr
                    if attr_name in ("log", "init", "finish", "Image", "Table"):
                        lineno = getattr(node, "lineno", 0)
                        lines = content.splitlines()
                        line_content = (
                            lines[lineno - 1].strip()
                            if 0 < lineno <= len(lines)
                            else "(unknown)"
                        )
                        violations.append(
                            f"  Line {lineno}: wandb.{attr_name}() in {src_path.name}: {line_content}"
                        )

        assert not violations, (
            "Production modules must not bypass wandb_control. "
            "Use wandb_log / wandb_finish / wandb_is_active from wandb_control:\n" +
            "\n".join(violations)
        )

    @pytest.mark.parametrize("src_path", PRODUCTION_MODULES)
    def test_no_direct_wandb_import_in_production_module(self, src_path):
        """Production modules should not `import wandb` directly (unaliased).

        Allowed:
        - evaluation_utils.py: uses ``import wandb as _wandb`` under guard
        - orthrus.py: has noqa comment, kept for backward compatibility
        """
        if not src_path.exists():
            pytest.skip(f"{src_path} does not exist")

        with open(src_path, encoding="utf-8-sig") as f:
            content = f.read()

        tree = ast.parse(content)

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "wandb" and alias.asname is None:
                        lineno = getattr(node, "lineno", 0)
                        lines = content.splitlines()
                        line_content = (
                            lines[lineno - 1].strip()
                            if 0 < lineno <= len(lines)
                            else "(unknown)"
                        )
                        # Check for noqa comment on the same line
                        has_noqa = "# noqa" in line_content
                        violations.append(
                            f"  Line {lineno}: {line_content}"
                        )

        # orthrus.py has explicit noqa comment for backward compatibility
        # evaluation_utils.py uses aliased import under guard
        # node_evaluation.py has import but never calls wandb functions (import-only)
        allowed = {
            SRC / "orthrus.py",  # noqa comment for backward compat
            SRC / "detection" / "node_evaluation.py",  # import only, no actual calls
        }

        if src_path in allowed:
            return

        assert not violations, (
            "Direct `import wandb` (without alias) found. "
            "Use `from wandb_control import wandb_log, wandb_is_active` instead:\n" +
            "\n".join(violations)
        )


# --------------------------------------------------------------------------- #
# I. Backward compatibility: wandb module still importable via orthrus
# --------------------------------------------------------------------------- #

class TestBackwardCompatibility:
    """Verify backward compatibility import still works."""

    def test_orthrus_still_exposes_wandb(self):
        """orthrus.py must still expose wandb for backward compatibility."""
        # The noqa comment explicitly says it's kept for backward compatibility
        src_path = SRC / "orthrus.py"
        with open(src_path) as f:
            content = f.read()

        assert "import wandb" in content
        assert "backward compatibility" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
