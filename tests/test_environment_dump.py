import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mstc.experiment_utils import dump_environment


def test_environment_dump_is_stable_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("ORTHRUS_DB_PASSWORD", "SUPER_SECRET_TEST_VALUE")
    monkeypatch.setenv("WANDB_API_KEY", "SUPER_SECRET_WANDB_VALUE")
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(name="synthetic"),
        model=SimpleNamespace(variant="mstc"),
        database=SimpleNamespace(password="SUPER_SECRET_TEST_VALUE"),
        api_token="SUPER_SECRET_WANDB_VALUE",
        _seed=4,
    )
    dump_environment(cfg, tmp_path)
    environment = (tmp_path / "environment.json").read_text(encoding="utf-8")
    config = (tmp_path / "config_resolved.yml").read_text(encoding="utf-8")
    assert (tmp_path / "environment.json").is_file() and (tmp_path / "config_resolved.yml").is_file()
    for key in ("git_commit", "python_version", "torch_version", "pyg_version", "cuda_version"):
        assert key in environment
    assert "SUPER_SECRET_TEST_VALUE" not in environment + config
    assert "SUPER_SECRET_WANDB_VALUE" not in environment + config
