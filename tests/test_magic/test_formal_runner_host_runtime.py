from pathlib import Path

from src.baselines.magic import formal_runner


def test_explicit_upstream_path_reaches_backend(monkeypatch, tmp_path: Path):
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(formal_runner, "MAGICRealBackend", RecordingBackend)

    upstream = tmp_path / "magic-upstream"

    formal_runner._create_magic_backend(
        seed=0,
        model_config=formal_runner.MAGICModelConfig(max_epoch=1),
        device="cuda",
        upstream_path=upstream,
    )

    assert captured["upstream_path"] == upstream


def test_omitted_upstream_path_preserves_backend_default(monkeypatch):
    captured = {}

    class RecordingBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(formal_runner, "MAGICRealBackend", RecordingBackend)

    formal_runner._create_magic_backend(
        seed=0,
        model_config=formal_runner.MAGICModelConfig(max_epoch=1),
        device="cpu",
        upstream_path=None,
    )

    assert "upstream_path" not in captured


def test_main_forwards_upstream_path(monkeypatch, tmp_path: Path):
    captured = {}

    monkeypatch.setattr(
        formal_runner,
        "run_formal_experiment",
        lambda **kwargs: captured.update(kwargs) or {"status": "mocked"},
    )

    upstream = tmp_path / "magic-upstream"

    formal_runner.main([
        "--dataset", "THEIA_E5",
        "--seed", "0",
        "--output-dir", str(tmp_path / "output"),
        "--upstream-path", str(upstream),
        "--max-epoch", "1",
        "--device", "cuda",
    ])

    assert captured["dataset"] == "THEIA_E5"
    assert captured["upstream_path"] == upstream
    assert captured["max_epoch"] == 1
    assert captured["device"] == "cuda"
