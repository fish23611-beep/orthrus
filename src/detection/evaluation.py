import os
import sys
import numpy as np
from collections import defaultdict
from pprint import pprint
import json as _json
import tempfile
from typing import Union, Optional

from . import node_evaluation
from mstc.calibration_runner import (
    load_event_records_from_csv as _cal_load_csv,
    load_event_records_from_csv_directory as _cal_load_csv_dir,
    run_calibration as _cal_run,
)
from data_utils import *
from provnet_utils import log
from .evaluation_utils import *
from wandb_control import wandb_log, wandb_is_active
from labelling import get_GP_of_each_attack

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------- #
# Safe path validation (rejects MagicMock / Mock / synthetic PathLike objects)
# --------------------------------------------------------------------------- #

def _is_real_path(path: Optional[PathLike]) -> bool:
    """
    Return True only when *path* is a real, non-empty, non-Mock string or Path.

    This guard makes every persistence helper a no-op in unit tests that pass
    MagicMock objects as run_dir, preventing filesystem pollution such as
    "MagicMock/mock._run_dir/<id>/node_scores/metrics.json".

    Unlike the broader ``isinstance(path, os.PathLike)`` check (which returns
    True for MagicMock due to its ``__fspath__`` implementation), this
    function explicitly rejects Mock-like objects by inspecting the type name.
    """
    if path is None:
        return False
    try:
        s = os.fspath(path)
    except Exception:
        return False
    if not s:
        return False
    # Reject any Mock-like objects that may implement __fspath__
    type_name = type(path).__name__.lower()
    if "mock" in type_name:
        return False
    return True

# Use sys.modules for lazy wandb import so that test patches on
# "detection.evaluation.wandb" work correctly.  Importing wandb at the
# module level with a plain "import wandb" statement would create a binding
# that test patches cannot intercept.
wandb = sys.modules.get("wandb")


# --------------------------------------------------------------------------- #
# Safe JSON serialisation helpers (used for canonical artifact persistence)
# --------------------------------------------------------------------------- #

def _is_serializable_json_leaf(value):
    """Return True when value is a JSON primitive (or NaN/inf which JSON allows)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, float) and (_json.is_nan(value) or _json.is_inf(value)):
        return True
    return False


def _strip_non_serializable(stats):
    """
    Return a copy of *stats* with all non-JSON-serialisable objects removed.

    W&B Image/Video/etc. objects raise TypeError on json.dump so we probe
    with json.JSONEncoder().encode() which raises for non-serialisable values.
    """
    import json as _json

    result = {}
    encoder = _json.JSONEncoder()
    for key, value in stats.items():
        try:
            encoder.encode(value)
            result[key] = value
        except TypeError:
            # Non-JSON value (W&B media, lambda, etc.) — skip it
            pass
    return result


def _persist_canonical_metrics(run_dir, best_stats, best_epoch_dir, method, cfg):
    """
    Atomically write the selected-epoch metrics to <run_dir>/node_scores/metrics.json.

    The file represents the authoritative test-set metrics of the best-epoch
    selected by *method* on the validation set.

    Parameters
    ----------
    run_dir:
        Resolved run directory path.
    best_stats:
        The evaluation_fn result dict for the selected best epoch.
    best_epoch_dir:
        The model_epoch directory name of the selected epoch.
    method:
        The model-selection method string (e.g. "min_val_mean_edge_loss").
    cfg:
        The resolved yacs CfgNode (used to extract val_mean_edge_loss if
        not already present in best_stats).
    """
    if not _is_real_path(run_dir):
        log(f"[evaluation] _persist_canonical_metrics: run_dir={run_dir!r} is not valid; skipping persistence.")
        return

    run_dir = os.fspath(run_dir)
    node_scores_dir = os.path.join(run_dir, "node_scores")
    os.makedirs(node_scores_dir, exist_ok=True)
    metrics_path = os.path.join(node_scores_dir, "metrics.json")

    # Build the canonical payload
    payload = _strip_non_serializable(best_stats)

    # Ensure required fields are present
    payload.setdefault("selected_epoch", int(best_epoch_dir.split("_")[-1]))
    payload.setdefault("model_selection_method", method)

    # Fall back to val_mean_edge_loss from cfg if not already in stats
    if "val_mean_edge_loss" not in payload:
        val_path = os.path.join(
            getattr(cfg.detection.gnn_testing, "_edge_losses_dir", ""),
            "val", best_epoch_dir
        )
        try:
            val_loss_list = []
            if os.path.isdir(val_path):
                for fname in sorted(os.listdir(val_path)):
                    import pandas as _pd
                    df = _pd.read_csv(os.path.join(val_path, fname))
                    val_loss_list.extend(df["loss"].tolist())
            if val_loss_list:
                payload["val_mean_edge_loss"] = float(np.mean(val_loss_list))
        except Exception:
            pass

    # Atomic write: temp file + rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=node_scores_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=2, allow_nan=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, metrics_path)
        log(f"[evaluation] Canonical metrics written to {metrics_path}")
    except Exception as exc:
        log(f"[evaluation] Failed to write canonical metrics: {exc}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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

    tw_to_malicious_nodes = (
        compute_tw_labels(cfg)
        if cfg.model.variant == "orthrus_baseline"
        else {}
    )

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

        # Only create W&B media objects when there is an active run.
        # In disabled mode wandb.run is None and raw wandb.log would raise
        # "You must call wandb.init() before wandb.log()".
        if wandb_is_active():
            import wandb as _wandb
            out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
            stats["simple_scores_img"] = _wandb.Image(
                os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png")
            )

            scores_img = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
            if os.path.exists(scores_img):
                stats["scores_img"] = _wandb.Image(scores_img)

            dor_img = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
            if os.path.exists(dor_img):
                stats["dor_img"] = _wandb.Image(dor_img)

            pr_img = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
            if os.path.exists(pr_img):
                stats["precision_recall_img"] = _wandb.Image(pr_img)

        # Log every epoch so the full history is visible in W&B
        wandb_log(stats)

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

    wandb_log(best_stats)

    # Persist the canonical selected-epoch metrics to <run_dir>/node_scores/metrics.json
    runtime_dir = getattr(cfg, "_run_dir", None)
    if runtime_dir:
        _persist_canonical_metrics(runtime_dir, best_stats, best_epoch_dir, method, cfg)

    # For MSTC: copy the selected-epoch predictions to the canonical node_scores/ location
    # so the paper artifact is at <run_dir>/node_scores/node_predictions.csv.
    # Per-epoch files under evaluation_results/calibration/model_epoch_X/ are preserved.
    if getattr(cfg.model, "variant", None) == "mstc":
        _persist_mstc_predictions(runtime_dir, best_epoch_dir, cfg)


def _persist_mstc_predictions(run_dir, best_epoch_dir, cfg):
    """Copy the selected epoch's MSTC predictions to node_scores/node_predictions.csv."""
    if not _is_real_path(run_dir):
        return
    run_dir = os.fspath(run_dir)
    eval_results_dir = getattr(cfg.detection.evaluation, "_evaluation_results_dir", "")
    if not eval_results_dir:
        eval_results_dir = os.path.join(run_dir, "evaluation_results")
    source = os.path.join(eval_results_dir, "calibration", best_epoch_dir, "node_predictions.csv")
    dest_dir = os.path.join(run_dir, "node_scores")
    dest = os.path.join(dest_dir, "node_predictions.csv")
    if os.path.exists(source):
        os.makedirs(dest_dir, exist_ok=True)
        import shutil
        shutil.copy2(source, dest)
        log(f"[evaluation] MSTC predictions copied: {dest}")


def mstc_evaluation_main(val_tw_path, test_tw_path, model_epoch_dir, cfg, **kwargs):
    """Inject legacy GT and metric providers into the lightweight C6 runner."""
    from mstc.evaluation_runner import mstc_evaluation_main as run_mstc_evaluation
    from mstc.node_evaluation import get_mstc_node_predictions_from_cfg

    # _cal_load_csv, _cal_load_csv_dir, and _cal_run are module-level imports.
    # They are accessible from the class body because module-level names are in
    # the module's global namespace, which the class body's LEGB lookup can reach.
    class CalibrationModule:
        load_event_records_from_csv = staticmethod(_cal_load_csv)
        load_event_records_from_csv_directory = staticmethod(_cal_load_csv_dir)
        run_calibration = staticmethod(_cal_run)

    result = run_mstc_evaluation(
        val_tw_path,
        test_tw_path,
        model_epoch_dir,
        cfg,
        calibration_module=CalibrationModule,
        node_prediction_fn=get_mstc_node_predictions_from_cfg,
        ground_truth_fn=get_ground_truth_nids,
        classifier_evaluation_fn=classifier_evaluation,
        attack_to_nodes_fn=get_GP_of_each_attack,
    )
    # Return the flat stats dict so evaluation.py can access val_mean_edge_loss
    # and other metrics at the top level (consistent with node_evaluation.main).
    # The per-epoch predictions remain in evaluation_results/calibration/model_epoch_X/.
    return result["stats"]


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    variant = cfg.model.variant
    if method != "node_evaluation":
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")
    if variant == "orthrus_baseline":
        standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif variant == "mstc":
        standard_evaluation(cfg, evaluation_fn=mstc_evaluation_main)
    else:
        raise ValueError(f"Invalid model variant {variant}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
