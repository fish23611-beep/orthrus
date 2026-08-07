from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def load_evaluation(monkeypatch):
    detection = types.ModuleType("detection")
    detection.__path__ = []
    legacy = types.ModuleType("detection.node_evaluation")
    legacy.main = lambda *args, **kwargs: {}
    utils = types.ModuleType("detection.evaluation_utils")
    utils.__all__ = ["compute_tw_labels"]
    utils.compute_tw_labels = lambda cfg: {}
    data_utils = types.ModuleType("data_utils")
    data_utils.__all__ = []
    provnet = types.ModuleType("provnet_utils")
    provnet.log = lambda *args: None
    wandb = types.ModuleType("wandb")
    wandb.logged = []
    wandb.log = lambda stats: wandb.logged.append(dict(stats))
    wandb.Image = lambda path: path
    for name, module in {"detection": detection, "detection.node_evaluation": legacy,
                         "detection.evaluation_utils": utils, "data_utils": data_utils,
                         "provnet_utils": provnet, "wandb": wandb}.items():
        monkeypatch.setitem(sys.modules, name, module)
    detection.node_evaluation = legacy
    spec = importlib.util.spec_from_file_location("detection.selection_evaluation", SRC_ROOT / "detection" / "evaluation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, wandb


def cfg(method, legacy=False):
    return SimpleNamespace(
        model=SimpleNamespace(variant="orthrus_baseline"),
        model_selection=SimpleNamespace(method=method, legacy_test_selection_enabled=legacy),
        detection=SimpleNamespace(
            gnn_testing=SimpleNamespace(_edge_losses_dir="/fake"),
            evaluation=SimpleNamespace(node_evaluation=SimpleNamespace(_precision_recall_dir="/fake/pr")),
        ),
    )


def stats_fn(stats):
    return lambda val, test, epoch, cfg, **kwargs: dict(stats[epoch])


def test_min_val_mean_edge_loss_ignores_test_metrics(monkeypatch):
    module, wandb = load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2"]
    stats = {"model_epoch_1": {"val_mean_edge_loss": 2.0, "mcc": .99},
             "model_epoch_2": {"val_mean_edge_loss": 1.0, "mcc": .01}}
    module.standard_evaluation(cfg("min_val_mean_edge_loss"), stats_fn(stats))
    assert wandb.logged[-1]["epoch"] == 2
    assert wandb.logged[-1]["mcc"] == .01


def test_last_epoch_selects_final_processed_epoch(monkeypatch):
    module, wandb = load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2"]
    stats = {"model_epoch_1": {"val_mean_edge_loss": .1}, "model_epoch_2": {"val_mean_edge_loss": 9.0}}
    module.standard_evaluation(cfg("last_epoch"), stats_fn(stats))
    assert wandb.logged[-1]["epoch"] == 2


def test_legacy_test_selection_is_rejected(monkeypatch):
    module, _ = load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1"]
    with pytest.raises(ValueError, match="leakage"):
        module.standard_evaluation(cfg("min_val_mean_edge_loss", legacy=True), stats_fn({"model_epoch_1": {"val_mean_edge_loss": 1.0, "mcc": 1.0}}))
