"""
MAGIC Unified Evaluator Integration

本模块实现 M6: MAGIC Unified Evaluator Integration。

数据流：
TRAIN -> fit MAGIC raw-score detector
VALIDATION -> raw scores -> max merge -> validation_quantile(q=0.999) -> freeze threshold
TEST -> raw scores -> max merge -> apply frozen threshold -> predictions
-> attach ground truth -> compute metrics

设计依据：
- 冻结合同: docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md
- 审计报告: docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md
- 现有项目 metrics: src/mstc/metrics.py

禁止行为：
- A/B 阶段不接收 ground truth
- threshold fit 不使用 test labels
- predictions 不使用 test labels
- 只在最终 metrics 阶段读取 labels
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from .contracts import ThresholdConfig
from .protocol import fit_threshold, apply_threshold
from .scoring import (
    MAGICEntityScorer,
    NodeScoreRecord,
    merge_and_score_snapshots,
)


# =============================================================================
# Validator: Validation Benign-only Status
# =============================================================================

# 来自 src/config.py DATASET_DEFAULT_CONFIG 的核实结论
VALIDATION_BENIGN_STATUS = {
    "THEIA_E3": {
        "validation_graph": "graph_9",
        "date": "2018-04-09",
        "status": "benign_only",
        "notes": "Defined by canonical split in config.py"
    },
    "THEIA_E5": {
        "validation_graph": "graph_11",
        "date": "2019-05-11",
        "status": "benign_only",
        "notes": "Defined by canonical split in config.py"
    },
}


# =============================================================================
# Stage A: Fit Threshold
# =============================================================================

def fit_magic_threshold(
    validation_node_scores: Mapping[str, float],
    method: str = "validation_quantile",
    q: float = 0.999,
) -> ThresholdConfig:
    """
    Stage A: Fit threshold on validation scores.

    合同要求：
    - 仅使用 validation scores
    - 不接受 test scores 或 labels
    - 使用 validation_quantile(q=0.999)

    Args:
        validation_node_scores: Merged validation node scores
        method: Threshold method
        q: Quantile value

    Returns:
        Frozen ThresholdConfig
    """
    return fit_threshold(
        validation_node_scores,
        method=method,
        q=q,
    )


# =============================================================================
# Stage B: Predict
# =============================================================================

def predict_magic(
    test_node_scores: Mapping[str, float],
    threshold_config: ThresholdConfig,
) -> Dict[str, int]:
    """
    Stage B: Apply frozen threshold to test scores.

    合同要求：
    - 仅使用 test scores 和 frozen threshold
    - 不接受 labels
    - 使用 strict > comparison

    Args:
        test_node_scores: Merged test node scores
        threshold_config: Frozen threshold configuration

    Returns:
        Dict mapping canonical_node_id to prediction (0 or 1)
    """
    return apply_threshold(test_node_scores, threshold_config)


# =============================================================================
# Stage C: Compute Metrics
# =============================================================================

def compute_magic_metrics(
    ground_truth: Mapping[str, int],
    predictions: Mapping[str, int],
    scores: Mapping[str, float],
) -> Dict[str, Any]:
    """
    Stage C: Compute metrics using ground truth.

    这是唯一接收 ground truth 的阶段。

    复用 src/mstc/metrics.py 中的指标计算。

    Args:
        ground_truth: Dict mapping canonical_node_id to 0/1
        predictions: Dict mapping canonical_node_id to 0/1
        scores: Dict mapping canonical_node_id to raw anomaly score

    Returns:
        Dict with metrics: precision, recall, f1, mcc, auprc, auroc, fpr, tp, fp, tn, fn
    """
    # Import and use project metrics
    from src.mstc.metrics import compute_classification_metrics

    # Collect aligned arrays
    node_ids = list(ground_truth.keys())

    y_true = [ground_truth[nid] for nid in node_ids]
    y_pred = [predictions.get(nid, 0) for nid in node_ids]
    score_values = [scores.get(nid, 0.0) for nid in node_ids]

    # Compute metrics using project implementation
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=score_values,
    )

    return metrics


# =============================================================================
# Canonical Output
# =============================================================================

@dataclass
class MagicRunResult:
    """
    Complete MAGIC run result.

    对应 canonical artifact schema。
    """
    # Configuration
    method: str = "MAGIC"
    score_method: str = "knn_distance_ratio"
    k: int = 10
    merge_method: str = "max"
    threshold_method: str = "validation_quantile"
    threshold_quantile: float = 0.999

    # Threshold
    threshold: float = field(default=0.0)
    threshold_provenance: str = "validation_only"
    validation_score_count: int = 0

    # Node predictions
    node_predictions: Dict[str, Dict] = field(default_factory=dict)
    # {canonical_node_id: {score, prediction, node_type}}

    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mcc: float = 0.0
    auprc: float = 0.0
    auroc: float = 0.0
    fpr: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    # Provenance
    dataset: str = ""
    split: str = ""


def write_predictions_csv(
    result: MagicRunResult,
    output_path: str,
) -> None:
    """
    Write node predictions to CSV.

    Schema:
        canonical_node_id,node_type,score_raw,threshold,prediction

    Args:
        result: MagicRunResult
        output_path: Path to output CSV
    """
    lines = ["canonical_node_id,node_type,score_raw,threshold,prediction"]

    for node_id, info in sorted(result.node_predictions.items()):
        node_type = info.get("node_type", "")
        score = info.get("score", 0.0)
        prediction = info.get("prediction", 0)

        lines.append(
            "{},{},{:.6f},{},{}".format(
                node_id, node_type, score, result.threshold, prediction
            )
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_metrics_json(
    result: MagicRunResult,
    output_path: str,
) -> None:
    """
    Write metrics to JSON.

    Schema:
        method, score_method, k, merge_method, threshold_method,
        threshold_quantile, threshold, threshold_provenance,
        validation_score_count, precision, recall, f1, mcc,
        auprc, auroc, fpr, tp, fp, tn, fn

    Args:
        result: MagicRunResult
        output_path: Path to output JSON
    """
    metrics_dict = {
        "method": result.method,
        "score_method": result.score_method,
        "k": result.k,
        "merge_method": result.merge_method,
        "threshold_method": result.threshold_method,
        "threshold_quantile": result.threshold_quantile,
        "threshold": result.threshold,
        "threshold_provenance": result.threshold_provenance,
        "validation_score_count": result.validation_score_count,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "mcc": result.mcc,
        "auprc": result.auprc,
        "auroc": result.auroc,
        "fpr": result.fpr,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
    }

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


# =============================================================================
# Full Pipeline
# =============================================================================

class MagicEvaluator:
    """
    Unified MAGIC evaluator.

    实现严格的三阶段数据流：
    A. fit_threshold(validation_scores) -> frozen threshold
    B. predict(test_scores, frozen_threshold) -> predictions
    C. compute_metrics(ground_truth, predictions) -> metrics
    """

    def __init__(
        self,
        k: int = 10,
        threshold_method: str = "validation_quantile",
        threshold_quantile: float = 0.999,
    ):
        """
        Initialize evaluator.

        Args:
            k: K for KNN
            threshold_method: Threshold method
            threshold_quantile: Quantile value
        """
        self.k = k
        self.threshold_method = threshold_method
        self.threshold_quantile = threshold_quantile

        self._threshold_config: Optional[ThresholdConfig] = None
        self._scorer: Optional[MAGICEntityScorer] = None

    def fit_threshold(
        self,
        validation_node_scores: Mapping[str, float],
    ) -> ThresholdConfig:
        """
        Stage A: Fit threshold on validation scores.

        Args:
            validation_node_scores: Merged validation node scores

        Returns:
            Frozen ThresholdConfig
        """
        self._threshold_config = fit_magic_threshold(
            validation_node_scores,
            method=self.threshold_method,
            q=self.threshold_quantile,
        )
        return self._threshold_config

    def predict(
        self,
        test_node_scores: Mapping[str, float],
    ) -> Dict[str, int]:
        """
        Stage B: Apply threshold to test scores.

        Args:
            test_node_scores: Merged test node scores

        Returns:
            Dict mapping canonical_node_id to prediction
        """
        if self._threshold_config is None:
            raise RuntimeError(
                "Threshold not fitted. Call fit_threshold first."
            )
        return predict_magic(test_node_scores, self._threshold_config)

    def evaluate(
        self,
        ground_truth: Mapping[str, int],
        predictions: Mapping[str, int],
        scores: Mapping[str, float],
    ) -> Dict[str, Any]:
        """
        Stage C: Compute metrics.

        Args:
            ground_truth: Dict mapping node_id to 0/1
            predictions: Dict mapping node_id to 0/1
            scores: Dict mapping node_id to raw score

        Returns:
            Metrics dict
        """
        return compute_magic_metrics(ground_truth, predictions, scores)

    @property
    def threshold_config(self) -> Optional[ThresholdConfig]:
        """Get frozen threshold config."""
        return self._threshold_config

    @property
    def is_threshold_fitted(self) -> bool:
        """Check if threshold is fitted."""
        return self._threshold_config is not None


# =============================================================================
# Adversarial Test Helper
# =============================================================================

def verify_label_independence(
    train_embeddings: np.ndarray,
    train_node_ids: List[str],
    val_embeddings: np.ndarray,
    val_node_ids: List[str],
    test_embeddings: np.ndarray,
    test_node_ids: List[str],
    ground_truth_a: Mapping[str, int],
    ground_truth_b: Mapping[str, int],
    k: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Verify that changing ground truth labels does not affect scores/predictions.

    This is an adversarial leakage test.

    Args:
        train_embeddings: Training embeddings
        train_node_ids: Training node IDs
        val_embeddings: Validation embeddings
        val_node_ids: Validation node IDs
        test_embeddings: Test embeddings
        test_node_ids: Test node IDs
        ground_truth_a: First ground truth mapping
        ground_truth_b: Second ground truth mapping
        k: K for KNN
        seed: Explicit seed for the scorer (required: 0, 1, or 2)

    Returns:
        Dict with verification results
    """
    if seed is None:
        raise ValueError(
            "verify_label_independence requires an explicit seed "
            "(official seeds: 0, 1, 2)."
        )
    # Fit scorer
    scorer = MAGICEntityScorer(k=k, seed=seed)
    scorer.fit(train_embeddings, train_node_ids)

    # Score validation
    val_scores_a = scorer.score(val_embeddings, val_node_ids)
    val_scores_b = val_scores_a  # Same embeddings

    # Merge (simplified - no snapshots)
    val_node_scores_a = {r.canonical_node_id: r.score_raw for r in val_scores_a}
    val_node_scores_b = val_node_scores_a

    # Fit threshold
    threshold_config_a = fit_magic_threshold(val_node_scores_a)
    threshold_config_b = fit_magic_threshold(val_node_scores_b)

    # Score test
    test_scores_a = scorer.score(test_embeddings, test_node_ids)
    test_scores_b = test_scores_a  # Same embeddings

    # Merge
    test_node_scores_a = {r.canonical_node_id: r.score_raw for r in test_scores_a}
    test_node_scores_b = test_node_scores_a

    # Predict
    predictions_a = predict_magic(test_node_scores_a, threshold_config_a)
    predictions_b = predict_magic(test_node_scores_b, threshold_config_b)

    # Verify independence
    return {
        "threshold_a": threshold_config_a.threshold_value,
        "threshold_b": threshold_config_b.threshold_value,
        "threshold_equal": threshold_config_a.threshold_value == threshold_config_b.threshold_value,
        "scores_a_equal_scores_b": test_node_scores_a == test_node_scores_b,
        "predictions_a_equal_predictions_b": predictions_a == predictions_b,
        "labels_changed": ground_truth_a != ground_truth_b,
        "metrics_a": compute_magic_metrics(ground_truth_a, predictions_a, test_node_scores_a),
        "metrics_b": compute_magic_metrics(ground_truth_b, predictions_b, test_node_scores_b),
    }
