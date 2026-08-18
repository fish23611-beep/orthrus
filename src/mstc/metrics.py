"""Stable, testable metrics for C8 experiment artifacts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def compute_classification_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], scores: Iterable[float] | None = None,
) -> dict[str, float | int]:
    """Compute binary classification metrics, using NaN for undefined values."""
    truth = np.asarray(list(y_true), dtype=int)
    prediction = np.asarray(list(y_pred), dtype=int)
    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have the same length")
    score_values = list(scores) if scores is not None else None
    if score_values is not None and len(score_values) != truth.size:
        raise ValueError("scores and y_true must have the same length")
    score_array = np.asarray(score_values, dtype=float) if score_values is not None else None
    if truth.size and (not np.isin(truth, (0, 1)).all() or not np.isin(prediction, (0, 1)).all()):
        raise ValueError("classification labels and predictions must be binary (0 or 1)")

    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if not (math.isnan(precision) or math.isnan(recall)) else float("nan")
    mcc = _safe_divide(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    fpr = _safe_divide(fp, fp + tn)

    auroc = float("nan")
    auprc = float("nan")
    if score_array is not None and truth.size and np.unique(truth).size == 2:
        auroc = float(roc_auc_score(truth, score_array))
    if score_array is not None and truth.size and np.any(truth == 1):
        auprc = float(average_precision_score(truth, score_array))

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "mcc": mcc,
        "auprc": auprc, "auroc": auroc, "fpr": fpr,
    }


def compute_fp_per_million(fp: int, num_benign_nodes: int) -> float:
    """False positives per one million benign nodes."""
    return float(fp) / num_benign_nodes * 1_000_000 if num_benign_nodes else float("nan")


def compute_attack_detection_rate(
    attack_to_nodes: Mapping[Any, Iterable[Any]], predicted_positive_nodes: Iterable[Any],
) -> float:
    """Fraction of attack scenarios containing at least one detected node."""
    if not attack_to_nodes:
        return float("nan")
    predicted = set(predicted_positive_nodes)
    detected = sum(bool(set(nodes) & predicted) for nodes in attack_to_nodes.values())
    return detected / len(attack_to_nodes)


def compute_inspected_nodes_per_attack(
    num_predicted_positive_nodes: int,
    num_attacks: int,
) -> float:
    """
    Average number of detection-positive nodes an analyst must inspect per attack.

    Operational definition (C8 paper artifact contract):
        inspected_nodes_per_attack = (TP + FP) / num_attacks
                                   = num_predicted_positive_nodes / num_attacks

    This metric quantifies the analyst inspection burden: how many detection-alerting
    nodes must be investigated to cover all predicted detections, divided by the
    number of distinct attack scenarios in the dataset.

    Parameters
    ----------
    num_predicted_positive_nodes:
        Number of nodes with y_hat=1 (true positives + false positives).
        Equivalent to count of detection alerts.
    num_attacks:
        Number of ground-truth attack scenarios (len(attack_to_nodes)).

    Returns
    -------
    float
        Average inspected nodes per attack, or NaN when num_attacks==0.

    Examples
    --------
    >>> compute_inspected_nodes_per_attack(9, 3)   # 9 alerts / 3 attacks
    3.0
    >>> compute_inspected_nodes_per_attack(0, 0)  # no attacks → NaN
    nan
    >>> compute_inspected_nodes_per_attack(5, 0)  # zero attacks → NaN
    nan
    """
    return _safe_divide(num_predicted_positive_nodes, num_attacks)
