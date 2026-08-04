"""Tests for run_metadata.py."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

from run_metadata import (
    dump_environment,
    dump_config,
    dump_runtime,
    _is_valid_path,
    _redact_database_password,
)


class TestIsValidPath:
    """Tests for _is_valid_path function."""

    def test_none_is_invalid(self):
        """None path is invalid."""
        assert _is_valid_path(None) is False

    def test_mock_is_invalid(self):
        """MagicMock path is invalid."""
        mock = MagicMock()
        assert _is_valid_path(mock) is False

    def test_empty_string_is_invalid(self):
        """Empty string is invalid."""
        assert _is_valid_path("") is False

    def test_valid_path_is_valid(self, tmp_path):
        """Valid path string is valid."""
        assert _is_valid_path(str(tmp_path)) is True


class TestRedactDatabasePassword:
    """Tests for _redact_database_password function."""

    def test_redacts_password(self):
        """Database password is redacted."""
        data = {"database": {"password": "secret123"}}
        result = _redact_database_password(data)
        assert result["database"]["password"] == "[REDACTED]"

    def test_preserves_other_fields(self):
        """Other fields are preserved."""
        data = {"database": {"user": "postgres", "password": "secret"}}
        result = _redact_database_password(data)
        assert result["database"]["user"] == "postgres"
        assert result["database"]["password"] == "[REDACTED]"

    def test_nested_password(self):
        """Nested password is redacted."""
        data = {"a": {"b": {"password": "secret"}}}
        result = _redact_database_password(data)
        assert result["a"]["b"]["password"] == "[REDACTED]"

    def test_list_preserved(self):
        """Lists are preserved."""
        data = [{"password": "secret"}]
        result = _redact_database_password(data)
        assert result[0]["password"] == "[REDACTED]"


class TestDumpEnvironment:
    """Tests for dump_environment function."""

    def test_skips_invalid_path(self):
        """Skips writing for invalid path."""
        dump_environment(None)  # Should not raise
        dump_environment("")  # Should not raise

    def test_writes_environment_json(self, tmp_path):
        """Writes environment.json for valid path."""
        dump_environment(tmp_path)
        env_file = tmp_path / "environment.json"
        assert env_file.exists()
        
        with open(env_file) as f:
            env = json.load(f)
        
        assert "python_version" in env
        assert "platform" in env
        assert "created_at" in env


class TestDumpConfig:
    """Tests for dump_config function."""

    def test_skips_invalid_path(self):
        """Skips writing for invalid path."""
        cfg = SimpleNamespace()
        dump_config(cfg, None)  # Should not raise

    def test_writes_config_yml(self, tmp_path):
        """Writes config_resolved.yml for valid path."""
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(name="THEIA_E3"),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(
                    used_method="orthrus",
                    num_epochs=6,
                )
            ),
            _seed=42,
            _task_path="/test/path",
            database=SimpleNamespace(
                host="localhost",
                user="postgres",
                password="secret_password",
            ),
        )
        dump_config(cfg, tmp_path)
        config_file = tmp_path / "config_resolved.yml"
        assert config_file.exists()

    def test_password_redacted_in_config(self, tmp_path):
        """Database password is redacted in config."""
        # Use a dict for proper serialization
        cfg = {
            "dataset": {"name": "THEIA_E3"},
            "detection": {"gnn_training": {"used_method": "orthrus"}},
            "_seed": 42,
            "database": {"password": "secret123"},
        }
        dump_config(cfg, tmp_path)
        config_file = tmp_path / "config_resolved.yml"
        
        with open(config_file) as f:
            content = f.read()
        
        # The password should be redacted in the YAML output
        assert "[REDACTED]" in content
        assert "secret123" not in content


class TestDumpRuntime:
    """Tests for dump_runtime function."""

    def test_skips_invalid_path(self):
        """Skips writing for invalid path."""
        cfg = SimpleNamespace(_stages=[], _seed=0)
        dump_runtime(cfg, None)  # Should not raise

    def test_writes_runtime_json(self, tmp_path):
        """Writes runtime.json for valid path."""
        cfg = SimpleNamespace(
            _stages=["train", "test"],
            _seed=42,
            dataset=SimpleNamespace(name="THEIA_E3"),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(used_method="orthrus")
            ),
            _artifact_root="/artifacts",
            _run_start_time=None,
        )
        timing = {"time_total": 100.0, "time_gnn_training": 50.0}
        
        dump_runtime(
            cfg, tmp_path,
            status="completed",
            executed_stages=["train", "test"],
            timing=timing,
            wandb_mode="disabled",
        )
        
        runtime_file = tmp_path / "runtime.json"
        assert runtime_file.exists()
        
        with open(runtime_file) as f:
            runtime = json.load(f)
        
        assert runtime["status"] == "completed"
        assert runtime["wandb_mode"] == "disabled"
        assert "train" in runtime["executed_stages"]
        assert runtime["time_total"] == 100.0

    def test_error_message_in_failed_status(self, tmp_path):
        """Error message included in failed status."""
        cfg = SimpleNamespace(
            _stages=["train"],
            _seed=0,
            dataset=SimpleNamespace(name="THEIA_E3"),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(used_method="orthrus")
            ),
            _artifact_root="/artifacts",
            _run_start_time=None,
        )
        dump_runtime(
            cfg, tmp_path,
            status="failed",
            error_message="Connection refused",
        )
        
        runtime_file = tmp_path / "runtime.json"
        with open(runtime_file) as f:
            runtime = json.load(f)
        
        assert runtime["status"] == "failed"
        assert runtime["error_message"] == "Connection refused"
