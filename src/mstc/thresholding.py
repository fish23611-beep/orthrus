"""Pure threshold selection and application for continuous node scores."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal

import numpy as np


NodeThresholdMethod = Literal["validation_quantile", "max_validation"]


def _as_finite_scores(
    node_scores: Mapping[Hashable, Any] | Iterable[Any], *, name: str
) -> np.ndarray:
    """Return finite score values without mutating the caller's collection."""
    values = node_scores.values() if isinstance(node_scores, Mapping) else node_scores
    try:
        scores = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric scores") from exc
    if scores.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError(f"{name} must contain finite numeric scores")
    return scores


def _validate_quantile(quantile: float) -> float:
    if isinstance(quantile, bool):
        raise ValueError("quantile must be a finite value in [0, 1]")
    try:
        value = float(quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantile must be a finite value in [0, 1]") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("quantile must be a finite value in [0, 1]")
    return value


def _validate_threshold(threshold: float) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite float") from exc
    if not np.isfinite(value):
        raise ValueError("threshold must be a finite float")
    return value


def compute_validation_quantile_threshold(
    validation_node_scores: Mapping[Hashable, Any] | Iterable[Any],
    quantile: float = 0.999,
) -> float:
    """Compute a threshold from normal validation node scores only."""
    scores = _as_finite_scores(validation_node_scores, name="validation_node_scores")
    quantile = _validate_quantile(quantile)
    return float(np.quantile(scores, quantile))


def compute_max_validation_threshold(
    validation_node_scores: Mapping[Hashable, Any] | Iterable[Any],
) -> float:
    """Return the maximum of normal validation node scores only."""
    scores = _as_finite_scores(validation_node_scores, name="validation_node_scores")
    return float(np.max(scores))


def compute_node_threshold(
    validation_node_scores: Mapping[Hashable, Any] | Iterable[Any],
    *,
    method: NodeThresholdMethod = "validation_quantile",
    quantile: float = 0.999,
) -> float:
    """Compute a node threshold without accepting test scores or labels."""
    if method == "validation_quantile":
        return compute_validation_quantile_threshold(validation_node_scores, quantile)
    if method == "max_validation":
        return compute_max_validation_threshold(validation_node_scores)
    raise ValueError("method must be one of: validation_quantile, max_validation")


def apply_node_threshold(
    node_scores: Mapping[Hashable, Any], threshold: float
) -> dict[Hashable, int]:
    """Convert continuous node scores to 0/1 predictions using strict ``>``."""
    if not isinstance(node_scores, Mapping):
        raise TypeError("node_scores must be a mapping of node_id to score")
    threshold = _validate_threshold(threshold)
    predictions: dict[Hashable, int] = {}
    for node_id, score in node_scores.items():
        try:
            hash(node_id)
        except TypeError as exc:
            raise TypeError("node_id must be hashable") from exc
        value = _as_finite_scores([score], name="node_scores")[0]
        predictions[node_id] = int(value > threshold)
    return predictions
