from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

from src.baselines.magic import formal_runner
from src.baselines.magic.real_backend import (
    DEFAULT_UPSTREAM_PATH,
    MAGICRealBackend,
    MAGICModelConfig,
)


def test_parser_accepts_upstream_path() -> None:
    args = formal_runner.build_parser().parse_args([
        "--dataset", "THEIA_E3",
        "--output-dir", "/tmp/magic-output",
        "--upstream-path", "/some/path",
    ])

    assert args.upstream_path == Path("/some/path")


def test_main_forwards_upstream_path(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(
        formal_runner,
        "run_formal_experiment",
        lambda **kwargs: captured.update(kwargs) or {"status": "mocked"},
    )

    upstream = tmp_path / "magic-upstream"
    formal_runner.main([
        "--dataset", "THEIA_E5",
        "--seed", "7",
        "--output-dir", str(tmp_path / "output"),
        "--upstream-path", str(upstream),
    ])

    assert captured["upstream_path"] == upstream
    assert captured["dataset"] == "THEIA_E5"
    assert captured["seed"] == 7


def test_explicit_upstream_path_reaches_backend(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(formal_runner, "MAGICRealBackend", RecordingBackend)
    upstream = tmp_path / "magic-upstream"

    formal_runner._create_magic_backend(
        seed=3,
        model_config=MAGICModelConfig(),
        device="cpu",
        upstream_path=upstream,
    )

    assert captured["upstream_path"] == upstream


def test_omitted_upstream_path_preserves_backend_default(monkeypatch) -> None:
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(formal_runner, "MAGICRealBackend", RecordingBackend)
    formal_runner._create_magic_backend(
        seed=0,
        model_config=MAGICModelConfig(),
        device="cpu",
        upstream_path=None,
    )

    parameter = inspect.signature(MAGICRealBackend).parameters["upstream_path"]
    assert "upstream_path" not in captured
    assert parameter.default == DEFAULT_UPSTREAM_PATH == Path("/opt/magic-upstream")


def test_runtime_environment_uses_actual_versions_and_is_cpu_safe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class CPUOnlyCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            raise AssertionError("device_count must not be queried without CUDA")

        @staticmethod
        def get_device_name(_index):
            raise AssertionError("GPU names must not be queried without CUDA")

    fake_torch = SimpleNamespace(
        __version__="9.8.7+test",
        version=SimpleNamespace(cuda="99.1"),
        cuda=CPUOnlyCuda(),
    )
    fake_dgl = SimpleNamespace(__version__="6.5.4-test")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "dgl", fake_dgl)

    upstream = tmp_path / "parent" / ".." / "magic-upstream"
    manifest = formal_runner._collect_runtime_environment(
        SimpleNamespace(upstream_path=upstream),
        "cpu",
    )

    assert manifest["torch"] == "9.8.7+test"
    assert manifest["torch_cuda_runtime"] == "99.1"
    assert manifest["dgl"] == "6.5.4-test"
    assert manifest["python_executable"] == sys.executable
    assert manifest["python_prefix"] == sys.prefix
    assert manifest["cuda_available"] is False
    assert manifest["cuda_device_count"] == 0
    assert manifest["cuda_device_names"] == []
    assert manifest["upstream_path"] == str(upstream.resolve())
