from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments import run_experiment


def write_yaml(tmp_path: Path, content: str = "pipeline: {mode: full_pipeline}\n") -> Path:
    path = tmp_path / "experiment.yml"
    path.write_text(content, encoding="utf-8")
    return path


def parse(tmp_path: Path, *extra: str) -> argparse.Namespace:
    config = write_yaml(tmp_path)
    return run_experiment.build_parser().parse_args([
        "--dataset", "THEIA_E3", "--config", str(config), *extra,
    ])


def test_cli_parses_and_forwards_required_parameters(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(run_experiment, "_run_pipeline", lambda args: captured.setdefault("args", args))
    config = write_yaml(tmp_path)

    run_experiment.main([
        "--dataset", "THEIA_E3", "--config", str(config), "--seed", "17",
        "--artifact-root", str(tmp_path / "artifacts"), "--stages", "train,test,evaluate",
    ])

    args = captured["args"]
    assert args.dataset == "THEIA_E3"
    assert args.config == str(config.resolve())
    assert args.seed == 17
    assert args.artifact_root == str(tmp_path / "artifacts")
    assert args.stages == "train,test,evaluate"
    assert args.run_from_training is False


def test_explicit_config_path_is_preserved(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(run_experiment, "_run_pipeline", lambda args: captured.setdefault("args", args))
    config = write_yaml(tmp_path)

    run_experiment.main(["--dataset", "THEIA_E3", "--config", str(config)])

    assert captured["args"].config == str(config.resolve())


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        ("train", ["train"]),
        ("test,evaluate", ["test", "evaluate"]),
        ("train,test,evaluate", ["train", "test", "evaluate"]),
        ("all", ["preprocess", "train", "test", "evaluate", "trace"]),
    ],
)
def test_stages_keep_pipeline_semantics(tmp_path, stages, expected):
    args = run_experiment.build_pipeline_args(parse(tmp_path, "--stages", stages))
    assert run_experiment.parse_stages(args.stages, run_from_training=False) == expected


def test_train_checkpoint_maps_to_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    args = run_experiment.build_pipeline_args(parse(
        tmp_path, "--stages", "train,test,evaluate", "--checkpoint", str(checkpoint),
    ))

    assert args.resume_checkpoint == str(checkpoint.resolve())
    assert not hasattr(args, "inference_checkpoint")


def test_test_only_checkpoint_maps_to_inference_not_resume(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    args = run_experiment.build_pipeline_args(parse(
        tmp_path, "--stages", "test,evaluate", "--checkpoint", str(checkpoint),
    ))

    assert args.inference_checkpoint == str(checkpoint.resolve())
    assert not hasattr(args, "resume_checkpoint")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--dataset", "THEIA_E3", "--config", "missing.yml"], "Config file does not exist"),
        (["--dataset", "UNKNOWN", "--config", "missing.yml"], "Unknown dataset"),
    ],
)
def test_invalid_paths_and_dataset_fail_clearly(argv, message):
    namespace = run_experiment.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        run_experiment.build_pipeline_args(namespace)


def test_invalid_yaml_fails_clearly(tmp_path):
    config = write_yaml(tmp_path, "pipeline: [unterminated\n")
    namespace = run_experiment.build_parser().parse_args([
        "--dataset", "THEIA_E3", "--config", str(config),
    ])
    with pytest.raises(ValueError, match="Unable to parse YAML config"):
        run_experiment.build_pipeline_args(namespace)


def test_invalid_stage_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="Invalid stage"):
        run_experiment.build_pipeline_args(parse(tmp_path, "--stages", "train,unknown"))


def test_legacy_runtime_parser_keeps_positional_config_behavior():
    from config import get_runtime_required_args

    args = get_runtime_required_args(args=["THEIA_E3"])
    assert args.dataset == "THEIA_E3"
    assert args.model == "orthrus"
    assert args.config is None


def test_config_loader_uses_an_explicit_file_path(tmp_path, monkeypatch):
    import config as config_module

    explicit = write_yaml(tmp_path, Path("config/orthrus.yml").read_text(encoding="utf-8"))
    monkeypatch.setattr(config_module, "set_task_paths", lambda cfg: None)
    args = SimpleNamespace(
        dataset="THEIA_E3", model="orthrus", config=str(explicit), cpu=True,
        from_weights=False, seed=0, skip_tracing=False, artifact_root=None,
    )

    cfg = config_module.get_yml_cfg(args)

    assert cfg.dataset.name == "THEIA_E3"
    assert cfg.detection.gnn_training.used_method == "orthrus"
