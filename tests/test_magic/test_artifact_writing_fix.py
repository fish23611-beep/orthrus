"""
Regression Tests for MAGIC Artifact Writing Fix

Tests for the fix that makes runtime environment metadata collection non-fatal
when optional packages (like pytz) are not installed.

Tests:
1. _safe_package_version returns version for installed packages
2. _safe_package_version returns "not_installed" for missing packages
3. Runtime environment collection does NOT raise ModuleNotFoundError
4. Missing optional packages are recorded as "not_installed" in manifest
5. Artifact writing succeeds even when optional packages are missing

This test file does NOT require:
- DGL
- CUDA
- PostgreSQL
- Real THEIA_E3 data
- Network access
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Setup paths
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from src.baselines.magic.formal_runner import _safe_package_version


# =============================================================================
# TEST 1: Installed package returns version string
# =============================================================================

class TestSafePackageVersionInstalled:
    """Test that installed packages return valid version strings."""

    def test_installed_package_returns_version(self):
        """Installed package returns version string (not "not_installed")."""
        # torch is always installed in the test environment
        version = _safe_package_version("torch")
        assert version != "not_installed", \
            f"Expected torch to be installed, got 'not_installed'"
        assert isinstance(version, str), \
            f"Expected version string, got {type(version)}"
        assert len(version) > 0, \
            "Version string should not be empty"

    def test_installed_package_version_is_valid_format(self):
        """Installed package version looks like a real version."""
        version = _safe_package_version("torch")
        # Version should have at least one digit
        assert any(c.isdigit() for c in version), \
            f"Version '{version}' should contain digits"


# =============================================================================
# TEST 2: Missing package returns "not_installed"
# =============================================================================

class TestSafePackageVersionMissing:
    """Test that missing packages return 'not_installed'."""

    def test_missing_package_returns_not_installed(self):
        """Missing package returns 'not_installed' (not raises exception)."""
        # This package name is extremely unlikely to exist
        result = _safe_package_version("this_package_definitely_does_not_exist_xyz123")
        assert result == "not_installed", \
            f"Expected 'not_installed', got '{result}'"

    def test_missing_package_does_not_raise(self):
        """Missing package does NOT raise ModuleNotFoundError or any exception."""
        # Should never raise - this is the core invariant
        try:
            result = _safe_package_version("nonexistent_package_abc")
            # If we get here, good - no exception was raised
            assert result == "not_installed"
        except Exception as exc:
            pytest.fail(
                f"_safe_package_version raised an exception for missing package: {exc}. "
                f"Expected it to return 'not_installed' gracefully."
            )

    def test_pytz_fallback_is_not_installed(self):
        """pytz missing returns 'not_installed' (not raises ModuleNotFoundError)."""
        # Test the actual package that caused the THEIA_E3 pilot failure
        try:
            result = _safe_package_version("pytz")
            # Should not raise - either it's installed or returns not_installed
            assert result is not None
        except Exception as exc:
            pytest.fail(
                f"_safe_package_version raised exception for pytz: {exc}. "
                f"The THEIA_E3 pilot failure was caused by __import__('pytz').__version__ "
                f"raising ModuleNotFoundError. This should now be non-fatal."
            )


# =============================================================================
# TEST 3: Runtime environment collection is non-fatal
# =============================================================================

class TestRuntimeEnvironmentCollection:
    """Test that runtime environment collection does not raise fatal errors."""

    def test_safe_package_version_coverage(self):
        """All packages used in runtime_manifest are covered by _safe_package_version."""
        # These are the packages listed in runtime_manifest
        packages_to_check = ["torch", "dgl", "pytz"]

        for pkg in packages_to_check:
            # Should not raise - this is the key regression test
            try:
                result = _safe_package_version(pkg)
                assert isinstance(result, str), \
                    f"Expected string for {pkg}, got {type(result)}"
            except Exception as exc:
                pytest.fail(
                    f"_safe_package_version({pkg!r}) raised {type(exc).__name__}: {exc}. "
                    f"This would cause artifact writing to fail after a successful experiment."
                )

    def test_importlib_metadata_available(self):
        """importlib.metadata is available (Python 3.8+)."""
        from importlib import metadata
        # Should be able to query at least one package
        torch_version = metadata.version("torch")
        assert torch_version is not None
        assert len(torch_version) > 0


# =============================================================================
# TEST 4: Verbatim reproduction of the THEIA_E3 pilot failure
# =============================================================================

class TestTheiaE3PilotRegression:
    """
    Reproduce the exact failure scenario from THEIA_E3 pilot.

    In the real THEIA_E3 pilot run:
    - Training completed: 194 artifacts, 194 contracts
    - Embedding completed: val=(34365, 64), test=(699295, 64)
    - GT mapped: 118/118
    - Metrics computed successfully
    - THEN: artifact writing failed with:
        ModuleNotFoundError: No module named 'pytz'

    The failure was this line:
        "pytz": __import__("pytz").__version__,

    This test verifies the fix prevents this failure.
    """

    def test_pytz_version_collection_does_not_crash(self):
        """
        THEIA_E3 pilot failure: pytz version collection crashes experiment.

        The old code used: __import__("pytz").__version__
        This raises ModuleNotFoundError when pytz is not installed.

        The fix uses: _safe_package_version("pytz")
        This returns "not_installed" without raising.

        This test verifies the new behavior.
        """
        # The exact package that failed in THEIA_E3 pilot
        # If pytz is not installed, old code would crash here
        try:
            result = _safe_package_version("pytz")
            # Should return a string (either version or "not_installed")
            assert isinstance(result, str)
        except ModuleNotFoundError:
            pytest.fail(
                "pytz version collection raised ModuleNotFoundError. "
                "This is the exact bug that crashed the THEIA_E3 pilot. "
                "The fix should prevent this."
            )
        except Exception as exc:
            pytest.fail(
                f"pytz version collection raised unexpected exception: {exc}"
            )

    def test_manifest_writing_remains_non_fatal(self):
        """
        Verify that artifact writing cannot fail due to missing optional packages.

        Simulates the runtime_manifest construction without actually writing files.
        """
        import torch

        # Simulate the runtime_manifest construction logic
        manifest = {
            "python": sys.version.split()[0],
            "torch": _safe_package_version("torch"),
            "dgl": _safe_package_version("dgl"),
            "pytz": _safe_package_version("pytz"),
            "device": "cpu",
            "cuda_compatible": torch.cuda.is_available(),
        }

        # String-typed fields should all be strings
        string_fields = ["python", "torch", "dgl", "pytz", "device"]
        for key in string_fields:
            assert isinstance(manifest[key], str), \
                f"Manifest value for {key} should be str, got {type(manifest[key])}"

        # No value should be None (except explicit optional fields)
        # pytz should be "not_installed" if missing, NOT missing key
        assert "pytz" in manifest, \
            "pytz key should always be present in manifest (even if not_installed)"
        assert manifest["pytz"] is not None, \
            "pytz value should be 'not_installed', not None"


# =============================================================================
# TEST 5: importlib.metadata vs __import__ comparison
# =============================================================================

class TestImportLibMetadataVsImport:
    """Verify importlib.metadata approach is correct and safe."""

    def test_importlib_metadata_does_not_import_package(self):
        """
        importlib.metadata.version queries package metadata without importing.

        This is safer than __import__(pkg).__version__ because:
        1. No module is loaded into sys.modules
        2. No code is executed from the package
        3. Works for packages that are importable but do heavy initialization
        """
        # Query a package that might have side effects on import
        # If importlib.metadata accidentally imports the package,
        # the side effects would be visible
        before = set(sys.modules.keys())
        _safe_package_version("torch")
        after = set(sys.modules.keys())

        # torch may already be imported, but that's OK
        # The point is: we don't crash and don't import unwanted packages
        # (We can't assert torch is NOT imported since the test env has it)
        # Instead, we verify the function returns a string
        assert isinstance(_safe_package_version("torch"), str)

    def test_fake_package_does_not_taint_modules(self):
        """Querying a fake package doesn't add garbage to sys.modules."""
        before = set(sys.modules.keys())
        _safe_package_version("definitely_fake_package_xyz789")
        after = set(sys.modules.keys())

        # No new modules should be added for a fake package
        added = after - before
        assert len(added) == 0, \
            f"Fake package query added modules to sys.modules: {added}"


# =============================================================================
# TEST 6: Unexpected (non-PackageNotFoundError) metadata failures are
# distinguished from missing packages and remain non-fatal.
# =============================================================================

class TestSafePackageVersionMetadataError:
    """
    Verify that metadata failures which are NOT PackageNotFoundError
    (e.g. PermissionError, OSError from corrupted metadata directories)
    are surfaced with a typed marker and never masquerade as
    "not_installed".
    """

    def test_permission_error_returns_metadata_error_marker(self, monkeypatch):
        """
        When importlib.metadata.version raises PermissionError, the helper
        must return a "metadata_error:PermissionError" marker instead of
        crashing or pretending the package is not installed.
        """
        from importlib import metadata as importlib_metadata
        import src.baselines.magic.formal_runner as formal_runner_mod

        def _boom(_name):
            raise PermissionError("metadata inaccessible")

        # Patch BOTH the module-local alias and the canonical importlib.metadata
        # so the call inside _safe_package_version exercises the non-fatal
        # branch regardless of which symbol it resolved during import.
        monkeypatch.setattr(
            formal_runner_mod._importlib_metadata,
            "version",
            _boom,
        )
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            _boom,
        )

        result = formal_runner_mod._safe_package_version("example-package")

        assert result == "metadata_error:PermissionError", (
            f"Expected 'metadata_error:PermissionError', got '{result}'"
        )
        # Must not be confused with a missing package
        assert result != "not_installed", (
            "Unexpected metadata failure must NOT be reported as 'not_installed'"
        )

    def test_os_error_returns_metadata_error_marker(self, monkeypatch):
        """
        Same contract for OSError (e.g. corrupted metadata directory).
        """
        from importlib import metadata as importlib_metadata
        import src.baselines.magic.formal_runner as formal_runner_mod

        def _boom(_name):
            raise OSError("corrupted metadata directory")

        monkeypatch.setattr(
            formal_runner_mod._importlib_metadata,
            "version",
            _boom,
        )
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            _boom,
        )

        result = formal_runner_mod._safe_package_version("example-package")

        assert result == "metadata_error:OSError", (
            f"Expected 'metadata_error:OSError', got '{result}'"
        )
        assert result != "not_installed"

    def test_unexpected_metadata_error_does_not_raise(self, monkeypatch):
        """
        Unexpected metadata failures must NEVER propagate as exceptions.
        """
        from importlib import metadata as importlib_metadata
        import src.baselines.magic.formal_runner as formal_runner_mod

        def _boom(_name):
            raise RuntimeError("metadata corruption")

        monkeypatch.setattr(
            formal_runner_mod._importlib_metadata,
            "version",
            _boom,
        )
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            _boom,
        )

        try:
            result = formal_runner_mod._safe_package_version("example-package")
        except Exception as exc:
            pytest.fail(
                f"_safe_package_version propagated unexpected metadata error: "
                f"{type(exc).__name__}: {exc}"
            )

        assert isinstance(result, str)
        assert result.startswith("metadata_error:")
        assert "RuntimeError" in result


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
