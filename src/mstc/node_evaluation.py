"""MSTC node evaluation helper for calibrated event anomaly detection.

This module provides a clean MSTC path that:
- Consumes pre-calibrated event records (validation_calibrated.csv, test_calibrated.csv)
- Aggregates events into node scores using NodeScoreAggregator (B3)
- Computes thresholds from validation node scores only (B4)
- Applies thresholds to test node scores to produce predictions

The baseline ORTHRUS path in node_evaluation.py remains completely unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Import B3 aggregation (same pattern as existing MSTC modules)
_agg_path = Path(__file__).resolve().parent / "aggregation.py"
_agg_spec = importlib.util.spec_from_file_location("_aggregation", _agg_path)
assert _agg_spec and _agg_spec.loader
_agg_mod = importlib.util.module_from_spec(_agg_spec)
sys.modules[_agg_spec.name] = _agg_mod
_agg_spec.loader.exec_module(_agg_mod)
NodeScoreAggregator = _agg_mod.NodeScoreAggregator
AggregationMethod = _agg_mod.AggregationMethod

# Import B4 thresholding
_thr_path = Path(__file__).resolve().parent / "thresholding.py"
_thr_spec = importlib.util.spec_from_file_location("_thresholding", _thr_path)
assert _thr_spec and _thr_spec.loader
_thr_mod = importlib.util.module_from_spec(_thr_spec)
sys.modules[_thr_spec.name] = _thr_mod
_thr_spec.loader.exec_module(_thr_mod)
compute_node_threshold = _thr_mod.compute_node_threshold
apply_node_threshold = _thr_mod.apply_node_threshold
NodeThresholdMethod = _thr_mod.NodeThresholdMethod

# Type aliases
MSTCAggregationMethod = Literal["mean", "max", "topk_mean", "topk_sum"]
MSTCNodeThresholdMethod = Literal["validation_quantile", "max_validation", "kmeans"]


def get_legacy_compute_kmeans_labels():
    """Load the existing ORTHRUS KMeans implementation without copying it."""
    src_root = str(Path(__file__).resolve().parents[1])
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from detection.evaluation_utils import compute_kmeans_labels

    return compute_kmeans_labels


def load_calibrated_events_from_csv(
    csv_path: Path | str,
) -> list[dict[str, Any]]:
    """Load calibrated event records from a CSV file.

    Returns a list of dictionaries with all CSV columns preserved.
    """
    import csv as csv_lib

    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            record: dict[str, Any] = {}
            for key, value in row.items():
                if value == "" or value is None:
                    record[key] = None
                    continue
                try:
                    if "." in value:
                        record[key] = float(value)
                    else:
                        record[key] = int(value)
                except (ValueError, TypeError):
                    record[key] = value
            records.append(record)
    return records


def get_mstc_node_predictions(
    validation_event_records: list[dict[str, Any]] | None,
    test_event_records: list[dict[str, Any]] | None,
    *,
    aggregation_method: MSTCAggregationMethod = "topk_mean",
    topk: int = 5,
    include_dst: bool = True,
    score_field: str = "score_calibrated",
    threshold_method: MSTCNodeThresholdMethod = "validation_quantile",
    threshold_quantile: float = 0.999,
    legacy_kmeans_topk: int = 20,
) -> dict[str, Any]:
    """Compute MSTC node scores and predictions from calibrated event records.

    This function implements the clean MSTC evaluation path:
    1. Aggregate validation events → validation node scores
    2. Compute threshold from validation node scores only
    3. Aggregate test events → test node scores
    4. Apply threshold to test node scores → test predictions

    Parameters
    ----------
    validation_event_records : list[dict[str, Any]] | None
        Pre-calibrated validation event records. Each record must contain:
        - srcnode: source node ID
        - dstnode: destination node ID
        - score_field: the score to use for aggregation
    test_event_records : list[dict[str, Any]] | None
        Pre-calibrated test event records with the same structure.
    aggregation_method : str
        Aggregation method: "topk_mean", "topk_sum", "mean", "max".
        Default: "topk_mean".
    topk : int
        Number of top events to consider for topk methods.
        Default: 5.
    include_dst : bool
        Whether to include destination nodes in aggregation.
        Default: True.
    score_field : str
        Which score field to use: "score_calibrated" (default) or "score_raw".
        Default: "score_calibrated".
    threshold_method : str
        Threshold method: "validation_quantile" or "max_validation".
        Default: "validation_quantile".
    threshold_quantile : float
        Quantile for validation_quantile threshold method.
        Default: 0.999.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - validation_node_scores: dict[node_id -> float]
        - test_node_scores: dict[node_id -> float]
        - threshold: float
        - threshold_method: str
        - test_node_predictions: dict[node_id -> int] (0 or 1)
        - metadata: dict with configuration used

    Raises
    ------
    ValueError
        If validation_event_records is empty or threshold cannot be computed.
    KeyError
        If required fields are missing from event records.
    """
    aggregator = NodeScoreAggregator()

    # Step 1: Aggregate validation events → validation node scores
    validation_node_scores: dict[Hashable, float]
    if validation_event_records and len(validation_event_records) > 0:
        validation_node_scores = aggregator.aggregate_events(
            validation_event_records,
            method=aggregation_method,  # type: ignore
            topk=topk,
            include_dst=include_dst,
            score_field=score_field,
        )
    else:
        raise ValueError(
            "validation_event_records cannot be empty. "
            "Cannot compute a valid threshold without validation node scores."
        )

    if not validation_node_scores:
        raise ValueError(
            "validation_event_records produced no nodes after aggregation. "
            "Cannot compute a valid threshold."
        )

    # Step 2: Aggregate test events → test node scores
    test_node_scores: dict[Hashable, float]
    if test_event_records and len(test_event_records) > 0:
        test_node_scores = aggregator.aggregate_events(
            test_event_records,
            method=aggregation_method,  # type: ignore
            topk=topk,
            include_dst=include_dst,
            score_field=score_field,
        )
    else:
        # Empty test: return empty predictions
        test_node_scores = {}

    # Step 3: select predictions. KMeans is a legacy comparison path: it
    # clusters test node scores and has no validation-derived scalar.
    test_node_predictions: dict[Hashable, int]
    if threshold_method == "kmeans":
        threshold = None
        if not test_node_scores:
            test_node_predictions = {}
        else:
            if (
                not isinstance(legacy_kmeans_topk, int)
                or isinstance(legacy_kmeans_topk, bool)
                or legacy_kmeans_topk < 2
            ):
                raise ValueError("legacy_kmeans_topk must be an integer of at least 2")
            if len(test_node_scores) < 2:
                raise ValueError("legacy KMeans requires at least two test nodes")
            legacy_results = {
                node_id: {"score": score, "y_hat": 0}
                for node_id, score in test_node_scores.items()
            }
            legacy_kmeans = get_legacy_compute_kmeans_labels()
            updated_results = legacy_kmeans(legacy_results, legacy_kmeans_topk)
            updated_results = legacy_results if updated_results is None else updated_results
            test_node_predictions = {
                node_id: int(result["y_hat"])
                for node_id, result in updated_results.items()
            }
    else:
        threshold = compute_node_threshold(
            validation_node_scores,
            method=threshold_method,
            quantile=threshold_quantile,
        )
        test_node_predictions = (
            apply_node_threshold(test_node_scores, threshold)
            if test_node_scores
            else {}
        )

    # Build metadata
    metadata = {
        "aggregation_method": aggregation_method,
        "topk": topk,
        "include_dst": include_dst,
        "score_field": score_field,
        "threshold_method": threshold_method,
        "threshold_quantile": threshold_quantile,
        "legacy_kmeans_topk": legacy_kmeans_topk if threshold_method == "kmeans" else None,
        "prediction_mode": "legacy_kmeans" if threshold_method == "kmeans" else "validation_threshold",
        "num_validation_nodes": len(validation_node_scores),
        "num_test_nodes": len(test_node_scores),
        "num_test_predictions": sum(test_node_predictions.values()),
    }

    return {
        "validation_node_scores": validation_node_scores,
        "test_node_scores": test_node_scores,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "test_node_predictions": test_node_predictions,
        "metadata": metadata,
    }


def get_mstc_node_predictions_from_cfg(
    validation_event_records: list[dict[str, Any]] | None,
    test_event_records: list[dict[str, Any]] | None,
    cfg: Any,
) -> dict[str, Any]:
    """Read only C6 algorithm settings from cfg and call the parameterized core."""
    return get_mstc_node_predictions(
        validation_event_records,
        test_event_records,
        aggregation_method=cfg.node_aggregation.method,
        topk=cfg.node_aggregation.topk,
        include_dst=cfg.node_aggregation.include_dst,
        score_field=cfg.node_aggregation.score_field,
        threshold_method=cfg.node_threshold.method,
        threshold_quantile=cfg.node_threshold.quantile,
        legacy_kmeans_topk=cfg.detection.evaluation.node_evaluation.kmeans_top_K,
    )


def get_mstc_node_predictions_from_paths(
    validation_csv_path: Path | str | None,
    test_csv_path: Path | str | None,
    *,
    aggregation_method: MSTCAggregationMethod = "topk_mean",
    topk: int = 5,
    include_dst: bool = True,
    score_field: str = "score_calibrated",
    threshold_method: NodeThresholdMethod = "validation_quantile",
    threshold_quantile: float = 0.999,
) -> dict[str, Any]:
    """Load calibrated events from CSV paths and compute MSTC node predictions.

    This is a convenience wrapper that loads CSV files and calls
    get_mstc_node_predictions.

    Parameters
    ----------
    validation_csv_path : Path | str | None
        Path to validation_calibrated.csv. Can be None for testing empty validation.
    test_csv_path : Path | str | None
        Path to test_calibrated.csv. Can be None for testing empty test.
    Other parameters are passed directly to get_mstc_node_predictions.

    Returns
    -------
    dict[str, Any]
        Same as get_mstc_node_predictions return value.

    Raises
    ------
    FileNotFoundError
        If CSV files do not exist.
    ValueError
        If validation events cannot be loaded or are empty.
    """
    # Load validation events
    validation_event_records: list[dict[str, Any]] | None = None
    if validation_csv_path is not None:
        validation_csv_path = Path(validation_csv_path)
        if validation_csv_path.exists():
            validation_event_records = load_calibrated_events_from_csv(
                validation_csv_path
            )

    # Load test events
    test_event_records: list[dict[str, Any]] | None = None
    if test_csv_path is not None:
        test_csv_path = Path(test_csv_path)
        if test_csv_path.exists():
            test_event_records = load_calibrated_events_from_csv(test_csv_path)

    return get_mstc_node_predictions(
        validation_event_records,
        test_event_records,
        aggregation_method=aggregation_method,
        topk=topk,
        include_dst=include_dst,
        score_field=score_field,
        threshold_method=threshold_method,
        threshold_quantile=threshold_quantile,
    )
