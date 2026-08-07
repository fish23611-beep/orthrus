"""Lightweight C6 evaluation orchestration, independent of ``detection``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np


_SUPPORTED_CALIBRATION_METHODS = frozenset(
    {"global_empirical", "relation_triplet", "hierarchical_relation"}
)


def run_mstc_epoch(
    val_tw_path: str | Path,
    test_tw_path: str | Path,
    model_epoch_dir: str,
    cfg: Any,
    *,
    calibration_module: Any,
    node_prediction_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], Path, float]:
    """Calibrate one matched epoch and fix its node predictions."""
    val_path, test_path = Path(val_tw_path), Path(test_tw_path)
    if val_path.name != model_epoch_dir or test_path.name != model_epoch_dir:
        raise ValueError("validation and test raw artifacts must match model_epoch_dir")
    if cfg.calibration.method not in _SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(f"Unsupported C6 calibration method: {cfg.calibration.method}")
    validation_records = calibration_module.load_event_records_from_csv_directory(val_path)
    test_records = calibration_module.load_event_records_from_csv_directory(test_path)
    if not validation_records:
        raise ValueError("validation records cannot be empty")
    output_dir = Path(cfg.detection.evaluation._evaluation_results_dir) / "calibration" / model_epoch_dir
    calibration_module.run_calibration(
        validation_records, test_records, output_dir,
        min_triplet_samples=cfg.calibration.min_triplet_samples,
        min_type_pair_samples=cfg.calibration.min_type_pair_samples,
        epsilon=cfg.calibration.epsilon,
        method=cfg.calibration.method,
    )
    validation_calibrated = calibration_module.load_event_records_from_csv(output_dir / "validation_calibrated.csv")
    test_calibrated = calibration_module.load_event_records_from_csv(output_dir / "test_calibrated.csv")
    predictions = node_prediction_fn(validation_calibrated, test_calibrated, cfg)
    return predictions, output_dir, float(np.mean([float(r["score_raw"]) for r in validation_records]))


def mstc_evaluation_main(
    val_tw_path: str | Path, test_tw_path: str | Path, model_epoch_dir: str, cfg: Any,
    *, calibration_module: Any, node_prediction_fn: Callable[..., dict[str, Any]],
    ground_truth_fn: Callable[[Any], tuple[set[Any], Any]],
    classifier_evaluation_fn: Callable[[list[int], list[int], list[float]], dict[str, Any]],
) -> dict[str, Any]:
    """Run C6 then attach labels and metrics only after predictions are fixed."""
    prediction_result, output_dir, val_mean_edge_loss = run_mstc_epoch(
        val_tw_path, test_tw_path, model_epoch_dir, cfg,
        calibration_module=calibration_module, node_prediction_fn=node_prediction_fn,
    )
    ground_truth_nids, _ = ground_truth_fn(cfg)
    rows = [
        {"node_id": node_id, "score": score,
         "y_hat": prediction_result["test_node_predictions"][node_id],
         "y_true": int(node_id in ground_truth_nids)}
        for node_id, score in prediction_result["test_node_scores"].items()
    ]
    stats = classifier_evaluation_fn(
        [row["y_true"] for row in rows], [row["y_hat"] for row in rows], [row["score"] for row in rows]
    ) if rows else {}
    stats["val_mean_edge_loss"] = val_mean_edge_loss
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "node_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "score", "y_hat", "y_true"])
        writer.writeheader(); writer.writerows(rows)
    return {"stats": stats, "prediction_result": prediction_result, "output_dir": output_dir, "rows": rows}
