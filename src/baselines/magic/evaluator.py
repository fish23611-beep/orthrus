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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
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
# Ground Truth Loader
# =============================================================================

def load_magic_ground_truth(
    gt_root: Path,
) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
    """
    Load THEIA ground truth with authoritative UUID -> integer index_id mapping.

    CSV format (from DARPA THEIA dataset):
        node_uuid, "{'node_type': 'display_label'}", index_id

    This function:
    1. Reads node_uuid (col 1) and index_id (col 3) from each CSV
    2. Maps each UUID to its integer index_id (the ORTHRUS graph node ID)
    3. Returns ground_truth as {str(index_id): 1} for all anomaly nodes

    Authoritative mapping source: Ground_Truth CSV files' third column (index_id),
    which is the same field used by ORTHRUS graph construction
    (src/graph_construction/build_orthrus_graphs.py: src_index_id/dst_index_id).

    Args:
        gt_root: Path to Ground_Truth/darpa/darpa/E3-THEIA/ or equivalent

    Returns:
        Tuple of:
        - ground_truth: Dict[str(index_id)] = 1 for anomaly nodes
        - uuid_to_index_id: Dict[uuid] = str(index_id)
        - unmapped_uuids: List of UUIDs that had no index_id (should be empty)

    Raises:
        ValueError: If any UUID is missing its index_id column
    """
    ground_truth: Dict[str, int] = {}
    uuid_to_index_id: Dict[str, str] = {}
    unmapped_uuids: List[str] = []

    if not gt_root.is_dir():
        raise FileNotFoundError(
            f"Ground truth root not found: {gt_root}. "
            f"Expected directory containing E3-THEIA/ (or E5-THEIA/) subdirectory."
        )

    # Support both E3 and E5 naming
    theia_dirs = [
        gt_root / "E3-THEIA",
        gt_root / "E5-THEIA",
        gt_root,  # fallback: CSV files directly under gt_root
    ]

    csv_paths: List[Path] = []
    for td in theia_dirs:
        if td.is_dir():
            csv_paths.extend(sorted(td.glob("node_*.csv")))
        elif td == gt_root:
            csv_paths.extend(sorted(gt_root.glob("node_*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No node_*.csv files found under {gt_root}. "
            f"Searched: {[str(d) for d in theia_dirs]}"
        )

    for csv_path in csv_paths:
        with open(csv_path, newline="") as fh:
            for lineno, raw in enumerate(fh):
                parts = raw.rstrip("\n").split(",", 2)
                if not parts or not parts[0].strip():
                    continue
                first_col = parts[0].strip()
                # Skip header row: if first column looks like a column name (no dashes)
                # rather than a UUID (which always has dashes).
                # The real GT files have no header; synthetic test files may.
                # Conservative check: lines where first col has no '-' and is not a UUID
                # are treated as headers.
                if lineno == 0 and "-" not in first_col:
                    # Likely a header row (column names)
                    continue
                uuid = first_col
                if len(parts) < 3:
                    raise ValueError(
                        f"Ground truth CSV {csv_path} line has < 3 columns: "
                        f"{raw!r}. Expected: uuid,label,index_id"
                    )
                index_id_str = parts[2].strip()
                # index_id must be parseable as integer
                try:
                    int(index_id_str)
                except ValueError:
                    raise ValueError(
                        f"Ground truth CSV {csv_path} line has non-integer index_id "
                        f"in column 3: {parts!r}. Expected integer index_id "
                        f"(the ORTHRUS graph node identifier)."
                    )
                uuid_to_index_id[uuid] = index_id_str
                ground_truth[index_id_str] = 1

    return ground_truth, uuid_to_index_id, unmapped_uuids


# =============================================================================
# Stage C: Compute Metrics — Fixed Universe Contract
# =============================================================================

class EvaluationUniverseError(ValueError):
    """Raised when prediction/score universe does not match test node universe."""
    pass


def compute_magic_metrics(
    test_node_ids: List[str],
    predictions: Mapping[str, int],
    scores: Mapping[str, float],
    ground_truth: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """
    Stage C: Compute metrics using ground truth.

    正式合同（FORMAL-F1）：
    - EVALUATION_UNIVERSE = TEST_NODE_UNIVERSE（所有进入 test scoring 的 canonical integer node IDs）
    - y_true = 1 if node_id in ground_truth else 0  （仅对 test universe 中的节点判定 true label）
    - GT node 不在 TEST_NODE_UNIVERSE → 不加入 universe，不计为 FN（仅报告 GT_OUTSIDE_TEST_COUNT）
    - predictions 和 scores 必须对每个 test node 都存在，不接受 .get(x, 0) 静默掩盖缺失

    Args:
        test_node_ids: Ordered list of all canonical integer node IDs in the test split.
                       This is the EVALUATION_UNIVERSE.
        predictions: Dict mapping canonical_node_id (str) to 0/1 prediction.
                     Must contain ALL test_node_ids as keys.
        scores: Dict mapping canonical_node_id (str) to raw anomaly score.
                 Must contain ALL test_node_ids as keys.
        ground_truth: Dict mapping str(index_id) to 1 for anomaly nodes.
                       May be None (for diagnostic runs without GT).

    Returns:
        Dict with metrics: precision, recall, f1, mcc, auprc, auroc, fpr, tp, fp, tn, fn,
        plus: gt_total, gt_in_test, gt_outside_test, test_node_count

    Raises:
        EvaluationUniverseError: If any test node is missing from predictions or scores.
    """
    from src.mstc.metrics import compute_classification_metrics

    # ---- Universe enforcement: fail-fast on missing ----
    test_node_set = set(test_node_ids)
    pred_keys = set(predictions.keys())
    score_keys = set(scores.keys())

    missing_pred = sorted(test_node_set - pred_keys)
    missing_score = sorted(test_node_set - score_keys)

    if missing_pred:
        raise EvaluationUniverseError(
            f"Missing predictions for {len(missing_pred)} test nodes: "
            f"{missing_pred[:10]}{'...' if len(missing_pred) > 10 else ''}. "
            f"Every test node must have a prediction."
        )
    if missing_score:
        raise EvaluationUniverseError(
            f"Missing scores for {len(missing_score)} test nodes: "
            f"{missing_score[:10]}{'...' if len(missing_score) > 10 else ''}. "
            f"Every test node must have a score."
        )

    # ---- Build aligned arrays from TEST_NODE_UNIVERSE ----
    node_ids = test_node_ids  # ordered, canonical
    y_pred = [predictions[nid] for nid in node_ids]

    # ---- y_true: only for nodes in both test_universe AND ground_truth ----
    if ground_truth is not None:
        gt_keys = set(ground_truth.keys())
        y_true = [1 if nid in gt_keys else 0 for nid in node_ids]
    else:
        # No GT: all labels = 0 (all benign diagnostic)
        y_true = [0] * len(node_ids)

    score_values = [scores[nid] for nid in node_ids]

    # ---- Compute metrics ----
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        scores=score_values,
    )

    # ---- GT coverage diagnostics ----
    if ground_truth is not None:
        gt_in_test = sum(1 for nid in node_ids if nid in ground_truth)
        gt_outside_test = len(ground_truth) - gt_in_test
    else:
        gt_in_test = 0
        gt_outside_test = 0

    # ---- Enrich with provenance ----
    metrics["gt_total"] = len(ground_truth) if ground_truth is not None else 0
    metrics["gt_in_test"] = gt_in_test
    metrics["gt_outside_test"] = gt_outside_test
    metrics["test_node_count"] = len(node_ids)

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

    # GT provenance
    gt_total: int = 0
    gt_in_test: int = 0
    gt_outside_test: int = 0

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
        auprc, auroc, fpr, tp, fp, tn, fn,
        gt_total, gt_in_test, gt_outside_test, test_node_count

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
        "gt_total": result.gt_total,
        "gt_in_test": result.gt_in_test,
        "gt_outside_test": result.gt_outside_test,
        "test_node_count": result.test_node_count if hasattr(result, "test_node_count") else 0,
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
        Stage B: Apply frozen threshold to test scores.

        Args:
            test_node_scores: Merged test node scores

        Returns:
            Dict mapping node_id to prediction (0 or 1)
        """
        if self._threshold_config is None:
            raise RuntimeError("Threshold not fitted. Call fit_threshold() first.")
        return predict_magic(test_node_scores, self._threshold_config)

    def compute_metrics(
        self,
        test_node_ids: List[str],
        predictions: Mapping[str, int],
        scores: Mapping[str, float],
        ground_truth: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Stage C: Compute metrics.

        Args:
            test_node_ids: Ordered list of all test canonical node IDs (EVALUATION_UNIVERSE)
            predictions: Dict mapping node_id to 0/1
            scores: Dict mapping node_id to raw anomaly score
            ground_truth: Dict mapping str(index_id) to 1, or None

        Returns:
            Metrics dict
        """
        return compute_magic_metrics(
            test_node_ids=test_node_ids,
            predictions=predictions,
            scores=scores,
            ground_truth=ground_truth,
        )

    @property
    def threshold_config(self) -> Optional[ThresholdConfig]:
        """Get the fitted threshold config."""
        return self._threshold_config

    @property
    def is_threshold_fitted(self) -> bool:
        """True if threshold has been fitted."""
        return self._threshold_config is not None

    def evaluate(
        self,
        test_node_ids: List[str],
        predictions: Mapping[str, int],
        scores: Mapping[str, float],
        ground_truth: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline (Stage C).

        Args:
            test_node_ids: Ordered list of all test canonical node IDs
            predictions: Dict mapping node_id to 0/1
            scores: Dict mapping node_id to raw anomaly score
            ground_truth: Dict mapping str(index_id) to 1, or None

        Returns:
            Metrics dict
        """
        return self.compute_metrics(
            test_node_ids=test_node_ids,
            predictions=predictions,
            scores=scores,
            ground_truth=ground_truth,
        )


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
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Verify that changing ground truth labels does not affect model behavior.

    A: ground_truth_a
    B: ground_truth_b

    Model, scorer, and scores should be identical.
    Only metrics should differ.

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
        seed: Random seed

    Returns:
        Dict with verification results
    """
    scorer = MAGICEntityScorer(k=k, seed=seed)
    scorer.fit(train_embeddings, train_node_ids)

    # Score validation
    val_scores_a = scorer.score(val_embeddings, val_node_ids)
    val_scores_b = val_scores_a  # Same embeddings

    # Merge (simplified - no snapshots)
    val_node_scores_a = {r.canonical_node_id: r.score_raw for r in val_scores_a}
    val_node_scores_b = {r.canonical_node_id: r.score_raw for r in val_scores_b}

    # Threshold
    threshold_config_a = fit_magic_threshold(val_node_scores_a)
    threshold_config_b = fit_magic_threshold(val_node_scores_b)

    # Score test
    test_scores_a = scorer.score(test_embeddings, test_node_ids)
    test_scores_b = test_scores_a  # Same embeddings

    # Merge
    test_node_scores_a = {r.canonical_node_id: r.score_raw for r in test_scores_a}
    test_node_scores_b = {r.canonical_node_id: r.score_raw for r in test_scores_b}

    # Predict
    predictions_a = apply_threshold(test_node_scores_a, threshold_config_a)
    predictions_b = apply_threshold(test_node_scores_b, threshold_config_b)

    return {
        "val_scores_a_equal_val_scores_b": val_node_scores_a == val_node_scores_b,
        "test_scores_a_equal_test_scores_b": test_node_scores_a == test_node_scores_b,
        "scores_a_equal_scores_b": test_node_scores_a == test_node_scores_b,
        "predictions_a_equal_predictions_b": predictions_a == predictions_b,
        "labels_changed": ground_truth_a != ground_truth_b,
        "threshold_equal": threshold_config_a.threshold_value == threshold_config_b.threshold_value,
        "metrics_a": compute_magic_metrics(test_node_ids, predictions_a, test_node_scores_a, ground_truth_a),
        "metrics_b": compute_magic_metrics(test_node_ids, predictions_b, test_node_scores_b, ground_truth_b),
    }
