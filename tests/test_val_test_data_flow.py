"""Tests for production checkpoint selection from validation metrics."""

import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def evaluation_module():
    """Import production evaluation with reversible path and network patches."""
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "path", [src_dir, *sys.path]))
        stack.enter_context(patch("nltk.download", return_value=True))
        from detection import evaluation

        yield evaluation


def _make_cfg(tmp_path):
    return SimpleNamespace(
        detection=SimpleNamespace(
            gnn_testing=SimpleNamespace(
                _edge_losses_dir=str(tmp_path / "edge_scores")
            ),
            evaluation=SimpleNamespace(
                node_evaluation=SimpleNamespace(
                    _precision_recall_dir=str(tmp_path / "node_scores")
                )
            ),
        ),
        model_selection=SimpleNamespace(
            method="min_val_mean_edge_loss",
            legacy_test_selection_enabled=False,
        ),
        model=SimpleNamespace(variant="orthrus_baseline"),
    )


def _create_checkpoint_dirs(cfg, epochs=(1, 2, 3)):
    edge_scores_dir = Path(cfg.detection.gnn_testing._edge_losses_dir)
    for epoch in epochs:
        for split in ("val", "test"):
            (edge_scores_dir / split / f"model_epoch_{epoch}").mkdir(parents=True)


def _run_standard_evaluation(evaluation, cfg, stats_by_epoch):
    evaluation_calls = []

    def controlled_evaluation_fn(
        val_tw_path, test_tw_path, model_epoch_dir, received_cfg, **kwargs
    ):
        evaluation_calls.append(
            (val_tw_path, test_tw_path, model_epoch_dir, received_cfg, kwargs)
        )
        return dict(stats_by_epoch[model_epoch_dir])

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(evaluation, "compute_tw_labels", return_value={})
        )
        stack.enter_context(patch.object(evaluation, "log"))
        stack.enter_context(
            patch.object(evaluation.wandb, "Image", return_value=MagicMock())
        )
        wandb_log = stack.enter_context(patch.object(evaluation, "wandb_log"))

        evaluation.standard_evaluation(cfg, evaluation_fn=controlled_evaluation_fn)

    assert len(evaluation_calls) == 3
    assert wandb_log.call_count == 4
    return dict(wandb_log.call_args_list[-1].args[0])


def test_evaluation_selects_checkpoint_by_lowest_val_loss(
    tmp_path, evaluation_module
):
    """Read the chosen epoch from standard_evaluation's final production log."""
    cfg = _make_cfg(tmp_path)
    _create_checkpoint_dirs(cfg)
    stats_by_epoch = {
        "model_epoch_1": {"val_mean_edge_loss": 2.5, "mcc": 0.90},
        "model_epoch_2": {"val_mean_edge_loss": 0.8, "mcc": 0.20},
        "model_epoch_3": {"val_mean_edge_loss": 1.4, "mcc": 0.95},
    }

    best_summary = _run_standard_evaluation(
        evaluation_module, cfg, stats_by_epoch
    )

    assert best_summary["epoch"] == 2
    assert best_summary["val_mean_edge_loss"] == 0.8


def test_evaluation_selection_ignores_test_metrics(tmp_path, evaluation_module):
    """Changing test-MCC ordering must not change the production-selected epoch."""
    cfg = _make_cfg(tmp_path)
    _create_checkpoint_dirs(cfg)
    val_losses = {
        "model_epoch_1": 0.5,
        "model_epoch_2": 1.0,
        "model_epoch_3": 2.0,
    }
    run_a_stats = {
        "model_epoch_1": {
            "val_mean_edge_loss": val_losses["model_epoch_1"], "mcc": 0.10
        },
        "model_epoch_2": {
            "val_mean_edge_loss": val_losses["model_epoch_2"], "mcc": 0.20
        },
        "model_epoch_3": {
            "val_mean_edge_loss": val_losses["model_epoch_3"], "mcc": 0.99
        },
    }
    run_b_stats = {
        "model_epoch_1": {
            "val_mean_edge_loss": val_losses["model_epoch_1"], "mcc": 0.05
        },
        "model_epoch_2": {
            "val_mean_edge_loss": val_losses["model_epoch_2"], "mcc": 0.98
        },
        "model_epoch_3": {
            "val_mean_edge_loss": val_losses["model_epoch_3"], "mcc": 0.15
        },
    }

    run_a_summary = _run_standard_evaluation(
        evaluation_module, cfg, run_a_stats
    )
    run_b_summary = _run_standard_evaluation(
        evaluation_module, cfg, run_b_stats
    )

    assert run_a_summary["epoch"] == 1
    assert run_b_summary["epoch"] == 1
    assert run_a_summary["mcc"] == 0.10
    assert run_b_summary["mcc"] == 0.05
