"""
tests/test_epoch_selection.py

Unit tests for epoch/model selection logic in evaluation.py.

Design rules (strict, paper-grade):
  1. Only validation metrics may drive best-epoch selection.
  2. Test-set MCC, F1, loss, and labels MUST NOT influence selection.
  3. Default method 'min_val_mean_edge_loss' selects the epoch with the
     lowest val_mean_edge_loss.
  4. 'last_epoch' selects the final processed checkpoint, regardless of metrics.
  5. legacy_test_selection_enabled=True raises ValueError — test-set leakage
     is hard-disabled; no silent fallback.
  6. Missing val_mean_edge_loss OR non-finite (None/NaN/inf) value in ANY
     participating epoch raises ValueError.  Test metrics are not consulted
     as a fallback.  Partial-coverage is no longer tolerated.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, patch, call


torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch not installed",
)


def _make_eval_fn(stats_by_epoch):
    """Return a mock evaluation_fn that returns stats for each called epoch."""
    def mock_fn(val_path, test_path, epoch_dir, cfg, **kwargs):
        epoch_num = int(epoch_dir.split("_")[-1])
        return stats_by_epoch.get(epoch_dir, {
            "val_mean_edge_loss": 0.0, "mcc": 0.0, "epoch": epoch_num
        })
    return mock_fn


# --------------------------------------------------------------------------- #
# Default: min validation loss
# --------------------------------------------------------------------------- #
@requires_torch
def test_default_selects_min_val_loss():
    """
    Default 'min_val_mean_edge_loss' picks the epoch with the lowest
    val_mean_edge_loss — even if that epoch has the worst test MCC.
    """
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 2.5, "mcc": 0.75, "epoch": 1},
        "model_epoch_3": {"val_mean_edge_loss": 1.8, "mcc": 0.80, "epoch": 3},
        "model_epoch_2": {"val_mean_edge_loss": 2.1, "mcc": 0.78, "epoch": 2},
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    # Patch wandb_log to capture all logged calls
    logged_calls = []
    def mock_wandb_log(stats, **kwargs):
        logged_calls.append(dict(stats))

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2", "model_epoch_3"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb_log", side_effect=mock_wandb_log), \
         patch("detection.evaluation.wandb") as m_wandb, \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        standard_evaluation(cfg, evaluation_fn=mock_fn)

    # The last wandb_log call should be the best epoch (epoch 3 with val_loss=1.8)
    best_call = logged_calls[-1]
    assert best_call.get("val_mean_edge_loss") == 1.8, (
        f"Expected val_mean_edge_loss=1.8 (epoch_3), got {best_call}"
    )
    assert best_call.get("epoch") == 3, (
        f"Expected epoch=3, got {best_call.get('epoch')}"
    )


@requires_torch
def test_val_better_test_worse_epoch_selected():
    """
    An epoch with better validation loss but worse test MCC MUST be
    selected — test metrics must not influence the decision.
    """
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 0.9, "mcc": 0.60, "epoch": 1},  # val-best, test-worst
        "model_epoch_2": {"val_mean_edge_loss": 1.5, "mcc": 0.80, "epoch": 2},
        "model_epoch_3": {"val_mean_edge_loss": 2.0, "mcc": 0.95, "epoch": 3},  # val-worst, test-best
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    logged_calls = []
    def mock_wandb_log(stats, **kwargs):
        logged_calls.append(dict(stats))

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2", "model_epoch_3"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb_log", side_effect=mock_wandb_log), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        standard_evaluation(cfg, evaluation_fn=mock_fn)

    best_call = logged_calls[-1]
    assert best_call.get("epoch") == 1, (
        f"Val-best epoch (epoch_1) must be selected even though it has worst "
        f"test MCC; got epoch={best_call.get('epoch')}"
    )
    assert best_call.get("mcc") == 0.60, (
        f"Expected mcc=0.60 (epoch_1 test MCC), got {best_call.get('mcc')}"
    )


# --------------------------------------------------------------------------- #
# last_epoch
# --------------------------------------------------------------------------- #
@requires_torch
def test_last_epoch_selects_final():
    """method='last_epoch' picks the last listed epoch, regardless of metrics."""
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 1.0, "mcc": 0.90, "epoch": 1},
        "model_epoch_2": {"val_mean_edge_loss": 0.5, "mcc": 0.95, "epoch": 2},  # best val
        "model_epoch_3": {"val_mean_edge_loss": 2.0, "mcc": 0.85, "epoch": 3},
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "last_epoch"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    logged_calls = []
    def mock_wandb_log(stats, **kwargs):
        logged_calls.append(dict(stats))

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2", "model_epoch_3"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb_log", side_effect=mock_wandb_log), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        standard_evaluation(cfg, evaluation_fn=mock_fn)

    best_call = logged_calls[-1]
    assert best_call.get("epoch") == 3, (
        f"Expected epoch=3 (last in list), got {best_call.get('epoch')}"
    )


# --------------------------------------------------------------------------- #
# legacy_test_selection_enabled=True raises ValueError
# --------------------------------------------------------------------------- #
@requires_torch
def test_legacy_test_selection_raises_valueerror():
    """
    legacy_test_selection_enabled=True MUST raise ValueError because
    selecting by test metrics is test-set leakage.  No silent fallback.
    """
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 1.0, "mcc": 0.90, "epoch": 1},
        "model_epoch_2": {"val_mean_edge_loss": 0.5, "mcc": 0.80, "epoch": 2},
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = True    # ← the forbidden flag
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        with pytest.raises(ValueError) as exc_info:
            standard_evaluation(cfg, evaluation_fn=mock_fn)

        msg = str(exc_info.value).lower()
        assert "leakage" in msg or "test" in msg, (
            f"ValueError message should mention leakage/test; got: {exc_info.value}"
        )


# --------------------------------------------------------------------------- #
# Strict val-metric integrity — every epoch must have a finite value
# --------------------------------------------------------------------------- #
@requires_torch
def test_missing_val_metric_raises_valueerror():
    """
    When NO epoch has val_mean_edge_loss, selection MUST raise ValueError.
    Test metrics (mcc) MUST NOT be used as a fallback.
    """
    stats = {
        "model_epoch_1": {"mcc": 0.90, "epoch": 1},
        "model_epoch_2": {"mcc": 0.80, "epoch": 2},
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        with pytest.raises(ValueError) as exc_info:
            standard_evaluation(cfg, evaluation_fn=mock_fn)

        msg = str(exc_info.value).lower()
        assert "val_mean_edge_loss" in msg or "val" in msg, (
            f"ValueError should mention val_mean_edge_loss; got: {exc_info.value}"
        )


@requires_torch
def test_partial_val_metric_coverage_raises_valueerror():
    """
    STRICT (paper-grade) — when even ONE participating epoch is missing
    the val_mean_edge_loss key, selection MUST raise ValueError and list
    the offending epoch(s).  Partial-coverage is no longer tolerated:
    silently picking from the remaining epochs could mask incomplete or
    corrupted evaluation results.
    """
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 1.2, "mcc": 0.95, "epoch": 1},  # highest mcc, ignored
        "model_epoch_2": {"mcc": 0.50, "epoch": 2},                              # missing val metric
        "model_epoch_3": {"val_mean_edge_loss": 0.9, "mcc": 0.70, "epoch": 3},  # best val
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2", "model_epoch_3"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        with pytest.raises(ValueError) as exc_info:
            standard_evaluation(cfg, evaluation_fn=mock_fn)

        msg = str(exc_info.value)
        # The error message must clearly identify the bad epoch(s)
        assert "model_epoch_2" in msg, (
            f"Error message must list the invalid epoch(s); got: {msg}"
        )


@requires_torch
@pytest.mark.parametrize("bad_val", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_val_metric_raises_valueerror(bad_val):
    """
    val_mean_edge_loss must be a finite number.  None / NaN / ±inf are all
    invalid and MUST raise ValueError.
    """
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 1.2, "mcc": 0.95, "epoch": 1},
        "model_epoch_2": {"val_mean_edge_loss": bad_val, "mcc": 0.80, "epoch": 2},
        "model_epoch_3": {"val_mean_edge_loss": 0.9, "mcc": 0.70, "epoch": 3},
    }
    mock_fn = _make_eval_fn(stats)

    cfg = MagicMock()
    cfg.model_selection.method = "min_val_mean_edge_loss"
    cfg.model_selection.legacy_test_selection_enabled = False
    cfg.detection.gnn_testing._edge_losses_dir = "/fake"
    cfg.detection.evaluation.node_evaluation._precision_recall_dir = "/fake/pr"

    with patch("detection.evaluation.listdir_sorted", return_value=[
        "model_epoch_1", "model_epoch_2", "model_epoch_3"
    ]), \
         patch("detection.evaluation.compute_tw_labels", return_value={}), \
         patch("detection.evaluation.wandb"), \
         patch("detection.evaluation.log"):

        from detection.evaluation import standard_evaluation
        with pytest.raises(ValueError) as exc_info:
            standard_evaluation(cfg, evaluation_fn=mock_fn)

        msg = str(exc_info.value)
        assert "model_epoch_2" in msg, (
            f"Error message must identify model_epoch_2 (val={bad_val!r}); got: {msg}"
        )
