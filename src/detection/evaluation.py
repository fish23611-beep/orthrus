import os
import wandb
import numpy as np
from collections import defaultdict
from pprint import pprint

from . import node_evaluation
from data_utils import *
from provnet_utils import log
from .evaluation_utils import *


def standard_evaluation(cfg, evaluation_fn):
    """
    Evaluate all model checkpoints and select the best one based on cfg.model_selection.

    Selection strategies (cfg.model_selection.method):
        "min_val_mean_edge_loss" (default): pick epoch with lowest mean edge loss on val.
        "last_epoch":                   pick the last processed epoch.

    When cfg.model_selection.legacy_test_selection=True, falls back to selecting by
    test MCC — this constitutes test-set leakage and is kept only for compatibility
    with the official ORTHRUS baseline.  It must NOT be used for paper results.
    """
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")

    tw_to_malicious_nodes = compute_tw_labels(cfg)

    # Collect per-epoch stats so we can pick the best one
    epoch_results = []   # list of (model_epoch_dir, stats)

    for model_epoch_dir in listdir_sorted(test_losses_dir):
        log(f"\nEvaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path  = os.path.join(val_losses_dir,  model_epoch_dir)

        stats = evaluation_fn(
            val_tw_path, test_tw_path, model_epoch_dir, cfg,
            tw_to_malicious_nodes=tw_to_malicious_nodes
        )

        # Annotate epoch number for convenience
        stats["epoch"] = int(model_epoch_dir.split("_")[-1])

        out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
        stats["simple_scores_img"] = wandb.Image(
            os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
        )

        scores_img = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
        if os.path.exists(scores_img):
            stats["scores_img"] = wandb.Image(scores_img)

        dor_img = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
        if os.path.exists(dor_img):
            stats["dor_img"] = wandb.Image(dor_img)

        pr_img = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
        if os.path.exists(pr_img):
            stats["precision_recall_img"] = wandb.Image(pr_img)

        # Log every epoch so the full history is visible in W&B
        wandb.log(stats)

        epoch_results.append((model_epoch_dir, stats))

    # ------------------------------------------------------------------ #
    # Select the best epoch
    # ------------------------------------------------------------------ #
    method = cfg.model_selection.method
    legacy = cfg.model_selection.legacy_test_selection_enabled

    if legacy:
        raise ValueError(
            "cfg.model_selection.legacy_test_selection_enabled=True is not allowed: "
            "selecting the best epoch by test-set metrics constitutes test-set leakage "
            "and is prohibited.  Use cfg.model_selection.method='min_val_mean_edge_loss' "
            "(default) or cfg.model_selection.method='last_epoch' instead."
        )

    if method == "last_epoch":
        best_epoch_dir = epoch_results[-1][0]
        best_stats     = epoch_results[-1][1]
    else:
        # Strict val-metric integrity check before any selection.
        # Every epoch must have a finite val_mean_edge_loss.  Missing,
        # None, NaN, or inf values are all treated as invalid; selection
        # MUST NOT silently skip such epochs or fall back to test metrics.
        import math
        invalid = []
        for epoch_dir, stats in epoch_results:
            v = stats.get("val_mean_edge_loss", None)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                invalid.append((epoch_dir, v))
        if invalid:
            lines = "\n".join(f"  - {ep}: {val!r}" for ep, val in invalid)
            raise ValueError(
                "Invalid or missing 'val_mean_edge_loss' for one or more epochs "
                "under method 'min_val_mean_edge_loss'.  Test-set metrics are "
                "NOT consulted as a fallback.  Affected epochs:\n"
                f"{lines}\n"
                "Ensure node_evaluation.main computes and returns a finite "
                "'val_mean_edge_loss' for every checkpoint."
            )

        val_losses = {r[0]: r[1]["val_mean_edge_loss"] for r in epoch_results}
        best_epoch_dir = min(val_losses, key=val_losses.get)
        best_stats = next(r[1] for r in epoch_results if r[0] == best_epoch_dir)

    log(f"Best epoch selected ({method}): {best_epoch_dir}  "
        f"val_mean_edge_loss={best_stats.get('val_mean_edge_loss', float('nan')):.6f}")

    wandb.log(best_stats)


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
