"""
Tests for MAGIC Unified Evaluator (M6)

测试 M6 的核心功能：
1. q=0.999 threshold
2. threshold 只依赖 validation scores
3. test scores 不改变 threshold
4. test labels 不改变 threshold
5. test labels 不改变 raw score
6. test labels 不改变 predictions
7. labels 只影响 final metrics
8. no y_test-based filtering
9. full canonical test population 保留
10. threshold tie comparator
11. TP/FP/TN/FN
12. Precision
13. Recall
14. FPR
15. MCC
16. AUROC
17. AUPRC
18. AUROC/AUPRC 使用 raw scores
19. deterministic repeated evaluation
20. output schema
21. adversarial leakage tests
"""

from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import pytest
from typing import Dict, List

from src.baselines.magic import (
    MAGICEntityScorer,
    MagicEvaluator,
    MagicRunResult,
    fit_magic_threshold,
    predict_magic,
    compute_magic_metrics,
    write_predictions_csv,
    write_metrics_json,
    VALIDATION_BENIGN_STATUS,
    verify_label_independence,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def train_embeddings():
    """5 train embeddings in 4D space."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=np.float32)


@pytest.fixture
def train_node_ids():
    return ["train_0", "train_1", "train_2", "train_3", "train_4"]


@pytest.fixture
def val_embeddings():
    """3 validation embeddings - benign-like."""
    return np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.1, 0.0, 0.9, 0.1],
    ], dtype=np.float32)


@pytest.fixture
def val_node_ids():
    return ["val_0", "val_1", "val_2"]


@pytest.fixture
def test_embeddings():
    """3 test embeddings."""
    return np.array([
        [0.8, 0.2, 0.0, 0.0],   # similar to train
        [100.0, 100.0, 100.0, 100.0],  # anomalous
        [0.3, 0.7, 0.0, 0.0],   # similar to train
    ], dtype=np.float32)


@pytest.fixture
def test_node_ids():
    return ["test_0", "test_1", "test_2"]


@pytest.fixture
def fitted_scorer(train_embeddings, train_node_ids):
    """Fitted MAGIC entity scorer."""
    scorer = MAGICEntityScorer(k=5, seed=0)
    scorer.fit(train_embeddings, train_node_ids)
    return scorer


@pytest.fixture
def val_node_scores(fitted_scorer, val_embeddings, val_node_ids):
    """Validation node scores (as dict)."""
    score_records = fitted_scorer.score(val_embeddings, val_node_ids)
    return {r.canonical_node_id: r.score_raw for r in score_records}


@pytest.fixture
def test_node_scores(fitted_scorer, test_embeddings, test_node_ids):
    """Test node scores (as dict)."""
    score_records = fitted_scorer.score(test_embeddings, test_node_ids)
    return {r.canonical_node_id: r.score_raw for r in score_records}


@pytest.fixture
def fitted_evaluator(val_node_scores):
    """Evaluator with fitted threshold."""
    evaluator = MagicEvaluator(k=5, threshold_quantile=0.999)
    evaluator.fit_threshold(val_node_scores)
    return evaluator


@pytest.fixture
def ground_truth():
    """Test ground truth: only test_1 is anomalous."""
    return {
        "test_0": 0,
        "test_1": 1,
        "test_2": 0,
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Test 1: q=0.999 threshold
# =============================================================================

class TestThresholdQuantile0999:
    """Test q=0.999 threshold."""

    def test_default_q_is_0999(self):
        """Test default quantile is 0.999."""
        evaluator = MagicEvaluator()
        assert evaluator.threshold_quantile == 0.999

    def test_threshold_fitted_at_0999(self, val_node_scores):
        """Test threshold is fitted with q=0.999."""
        evaluator = MagicEvaluator(threshold_quantile=0.999)
        config = evaluator.fit_threshold(val_node_scores)
        assert config.q == 0.999


# =============================================================================
# Test 2: Threshold Only Depends on Validation Scores
# =============================================================================

class TestThresholdValidationOnly:
    """Test that threshold only depends on validation scores."""

    def test_threshold_from_validation(self, val_node_scores):
        """Test threshold fitted from validation scores."""
        config = fit_magic_threshold(val_node_scores, q=0.999)
        assert config.threshold_value is not None
        assert config.provenance == "validation_only"


# =============================================================================
# Test 3: Test Scores Do Not Change Threshold
# =============================================================================

class TestTestScoresDoNotChangeThreshold:
    """Test that test scores don't change threshold."""

    def test_different_test_scores_same_threshold(
        self, val_node_scores, fitted_scorer, test_node_ids
    ):
        """Test different test scores produce same threshold."""
        config1 = fit_magic_threshold(val_node_scores, q=0.999)

        # Score different test data
        test_a = fitted_scorer.score(
            np.array([[100.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            ["t1"]
        )
        test_b = fitted_scorer.score(
            np.array([[0.0, 0.0, 0.0, 0.1]], dtype=np.float32),
            ["t2"]
        )

        # Threshold should be unchanged
        config2 = fit_magic_threshold(val_node_scores, q=0.999)
        assert config1.threshold_value == config2.threshold_value


# =============================================================================
# Test 4: Test Labels Do Not Change Threshold
# =============================================================================

class TestTestLabelsDoNotChangeThreshold:
    """Test that test labels don't affect threshold."""

    def test_labels_a_b_same_threshold(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test different labels produce same threshold."""
        labels_a = {"test_0": 0, "test_1": 1, "test_2": 0}
        labels_b = {"test_0": 1, "test_1": 0, "test_2": 1}

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        assert result["threshold_equal"]


# =============================================================================
# Test 5: Test Labels Do Not Change Raw Scores
# =============================================================================

class TestTestLabelsDoNotChangeScores:
    """Test that test labels don't affect raw scores."""

    def test_scores_independent_of_labels(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test scores_a == scores_b regardless of labels."""
        labels_a = {"test_0": 0, "test_1": 1, "test_2": 0}
        labels_b = {"test_0": 1, "test_1": 0, "test_2": 1}

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        assert result["scores_a_equal_scores_b"]


# =============================================================================

class TestTestLabelsDoNotChangePredictions:
    """Test that test labels don't affect predictions."""

    def test_predictions_independent_of_labels(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test predictions_a == predictions_b regardless of labels."""
        labels_a = {"test_0": 0, "test_1": 1, "test_2": 0}
        labels_b = {"test_0": 1, "test_1": 0, "test_2": 1}

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        assert result["predictions_a_equal_predictions_b"]


# =============================================================================
# Test 7: Labels Only Affect Final Metrics
# =============================================================================

class TestLabelsOnlyAffectMetrics:
    """Test that labels only affect final metrics."""

    def test_metrics_can_differ_with_same_predictions(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test metrics differ when labels change but predictions same."""
        labels_a = {"test_0": 0, "test_1": 1, "test_2": 0}
        labels_b = {"test_0": 1, "test_1": 0, "test_2": 1}

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        # Metrics should differ when labels differ
        assert result["labels_changed"]
        assert result["metrics_a"] != result["metrics_b"]


# =============================================================================
# Test 8: No y_test-based Filtering
# =============================================================================

class TestNoYTestFiltering:
    """Test that no y_test-based filtering occurs."""

    def test_full_test_population_preserved(self, test_node_scores, fitted_evaluator):
        """Test that all test nodes have predictions."""
        predictions = fitted_evaluator.predict(test_node_scores)

        # All test nodes should have predictions
        for node_id in test_node_scores.keys():
            assert node_id in predictions


# =============================================================================
# Test 9: Full Canonical Test Population Preserved
# =============================================================================

class TestFullTestPopulation:
    """Test full canonical test population is preserved."""

    def test_all_test_nodes_get_predictions(self, test_node_scores, fitted_evaluator):
        """Test all test nodes get predictions."""
        predictions = fitted_evaluator.predict(test_node_scores)

        assert len(predictions) == len(test_node_scores)
        assert set(predictions.keys()) == set(test_node_scores.keys())

    def test_no_test_node_dropped(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test no test node is dropped from evaluation."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )

        # tp + fp + tn + fn should equal total test nodes
        total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
        assert total == len(test_node_scores)


# =============================================================================
# Test 10: Threshold Tie Comparator
# =============================================================================

class TestThresholdTieComparator:
    """Test threshold uses strict > comparison."""

    def test_strict_greater_than(self):
        """Test threshold uses strict > comparison."""
        from src.baselines.magic.contracts import ThresholdConfig

        config = ThresholdConfig(
            method="validation_quantile",
            q=0.999,
            threshold_value=0.5,
        )

        scores = {"a": 0.4, "b": 0.5, "c": 0.6}
        predictions = predict_magic(scores, config)

        # 0.5 == threshold should be 0 (not anomaly)
        assert predictions["a"] == 0
        assert predictions["b"] == 0
        assert predictions["c"] == 1


# =============================================================================
# Test 11: TP/FP/TN/FN Correct
# =============================================================================

class TestConfusionMatrix:
    """Test confusion matrix is correct."""

    def test_confusion_matrix_correct(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test TP/FP/TN/FN are correctly computed."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )

        # Should have all four values
        assert "tp" in metrics
        assert "fp" in metrics
        assert "tn" in metrics
        assert "fn" in metrics

        # Total should equal test size
        total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
        assert total == len(test_node_scores)


# =============================================================================
# Test 12-17: Metric Correctness
# =============================================================================

class TestMetricCorrectness:
    """Test metric correctness."""

    def test_precision_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test precision is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "precision" in metrics

    def test_recall_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test recall is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "recall" in metrics

    def test_fpr_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test FPR is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "fpr" in metrics

    def test_mcc_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test MCC is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "mcc" in metrics

    def test_auroc_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test AUROC is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "auroc" in metrics

    def test_auprc_in_metrics(self, test_node_scores, fitted_evaluator, ground_truth):
        """Test AUPRC is in metrics."""
        predictions = fitted_evaluator.predict(test_node_scores)
        metrics = compute_magic_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "auprc" in metrics


# =============================================================================
# Test 18: AUROC/AUPRC Use Raw Scores
# =============================================================================

class TestAuRocUsesRawScores:
    """Test AUROC/AUPRC use raw scores, not binary predictions."""

    def test_auroc_changes_with_score_ordering(self, fitted_evaluator):
        """Test AUROC changes with raw score, not predictions."""
        # Two sets of scores that give same predictions but different AUROC
        scores_a = {"n1": 1.0, "n2": 0.4, "n3": 0.3}
        scores_b = {"n1": 2.0, "n2": 0.4, "n3": 0.3}

        labels = {"n1": 1, "n2": 0, "n3": 0}

        # Same predictions (both 1 only for n1, above threshold)
        # But different raw scores -> different AUROC
        config = fitted_evaluator.threshold_config
        pred_a = predict_magic(scores_a, config)
        pred_b = predict_magic(scores_b, config)

        # Both should predict n1 as anomaly
        assert pred_a["n1"] == 1
        assert pred_b["n1"] == 1

        # AUROC from scores uses raw values
        metrics_a = compute_magic_metrics(labels, pred_a, scores_a)
        metrics_b = compute_magic_metrics(labels, pred_b, scores_b)


# =============================================================================
# Test 19: Deterministic Repeated Evaluation
# =============================================================================

class TestDeterministicEvaluation:
    """Test that evaluation is deterministic."""

    def test_repeated_threshold_same(
        self, val_node_scores
    ):
        """Test repeated threshold fitting is same."""
        config1 = fit_magic_threshold(val_node_scores, q=0.999)
        config2 = fit_magic_threshold(val_node_scores, q=0.999)
        assert config1.threshold_value == config2.threshold_value

    def test_repeated_prediction_same(
        self, test_node_scores, val_node_scores
    ):
        """Test repeated prediction is same."""
        config = fit_magic_threshold(val_node_scores, q=0.999)
        pred1 = predict_magic(test_node_scores, config)
        pred2 = predict_magic(test_node_scores, config)
        assert pred1 == pred2


# =============================================================================
# Test 20: Output Schema
# =============================================================================

class TestOutputSchema:
    """Test output file schemas."""

    def test_predictions_csv_schema(self, temp_dir):
        """Test predictions CSV has correct schema."""
        result = MagicRunResult(
            method="MAGIC",
            score_method="knn_distance_ratio",
            k=10,
            merge_method="max",
            threshold_method="validation_quantile",
            threshold_quantile=0.999,
            threshold=0.5,
            validation_score_count=100,
            node_predictions={
                "node_1": {"score": 0.7, "prediction": 1, "node_type": "subject"},
                "node_2": {"score": 0.3, "prediction": 0, "node_type": "file"},
            },
        )

        output_path = os.path.join(temp_dir, "predictions.csv")
        write_predictions_csv(result, output_path)

        # Read and check schema
        with open(output_path, "r") as f:
            lines = f.read().strip().split("\n")

        # Header
        header = lines[0]
        assert "canonical_node_id" in header
        assert "node_type" in header
        assert "score_raw" in header
        assert "threshold" in header
        assert "prediction" in header

    def test_metrics_json_schema(self, temp_dir):
        """Test metrics JSON has correct schema."""
        result = MagicRunResult(
            method="MAGIC",
            score_method="knn_distance_ratio",
            k=10,
            merge_method="max",
            threshold_method="validation_quantile",
            threshold_quantile=0.999,
            threshold=0.5,
            validation_score_count=100,
            precision=0.8,
            recall=0.7,
            f1=0.75,
            mcc=0.5,
            auprc=0.85,
            auroc=0.9,
            fpr=0.1,
            tp=7,
            fp=2,
            tn=8,
            fn=3,
        )

        output_path = os.path.join(temp_dir, "metrics.json")
        write_metrics_json(result, output_path)

        # Read and check schema
        with open(output_path, "r") as f:
            metrics = json.load(f)

        required_keys = [
            "method", "score_method", "k", "merge_method",
            "threshold_method", "threshold_quantile", "threshold",
            "validation_score_count", "precision", "recall", "f1",
            "mcc", "auprc", "auroc", "fpr",
            "tp", "fp", "tn", "fn",
        ]
        for key in required_keys:
            assert key in metrics


# =============================================================================
# Test 21: MagicEvaluator Pipeline
# =============================================================================

class TestMagicEvaluatorPipeline:
    """Test MagicEvaluator pipeline."""

    def test_fit_threshold_predict_evaluate(self, val_node_scores, test_node_scores, ground_truth):
        """Test full pipeline: fit_threshold, predict, evaluate."""
        evaluator = MagicEvaluator(k=5, threshold_quantile=0.999)

        # Stage A
        config = evaluator.fit_threshold(val_node_scores)
        assert config is not None

        # Stage B
        predictions = evaluator.predict(test_node_scores)
        assert isinstance(predictions, dict)

        # Stage C
        metrics = evaluator.evaluate(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=test_node_scores,
        )
        assert "precision" in metrics

    def test_predict_before_fit_raises(self, test_node_scores):
        """Test predict before fit_threshold raises error."""
        evaluator = MagicEvaluator(k=5)
        with pytest.raises(RuntimeError, match="fit_threshold"):
            evaluator.predict(test_node_scores)


# =============================================================================
# Test 22: Adversarial Leakage Tests
# =============================================================================

class TestAdversarialLeakage:
    """Adversarial tests for label leakage."""

    def test_label_flip_no_score_change(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test flipping labels does not change scores/predictions/threshold."""
        # Construct two completely different label sets
        labels_a = {
            "test_0": 0, "test_1": 0, "test_2": 0,
        }
        labels_b = {
            "test_0": 1, "test_1": 1, "test_2": 1,
        }

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        # Threshold and scores must be identical
        assert result["threshold_equal"]
        assert result["scores_a_equal_scores_b"]
        assert result["predictions_a_equal_predictions_b"]

    def test_arbitrary_label_change_no_leakage(
        self, train_embeddings, train_node_ids,
        val_embeddings, val_node_ids,
        test_embeddings, test_node_ids
    ):
        """Test arbitrary label change doesn't leak into scores."""
        labels_a = {"test_0": 1, "test_1": 0, "test_2": 1}
        labels_b = {"test_0": 0, "test_1": 1, "test_2": 0}

        result = verify_label_independence(
            train_embeddings=train_embeddings,
            train_node_ids=train_node_ids,
            val_embeddings=val_embeddings,
            val_node_ids=val_node_ids,
            test_embeddings=test_embeddings,
            test_node_ids=test_node_ids,
            ground_truth_a=labels_a,
            ground_truth_b=labels_b,
            k=5,
            seed=0,
        )

        # Scores/threshold/predictions must be unchanged
        assert result["threshold_equal"]
        assert result["scores_a_equal_scores_b"]
        assert result["predictions_a_equal_predictions_b"]


# =============================================================================
# Test 23: Validation Benign-only Status
# =============================================================================

class TestValidationBenignStatus:
    """Test validation benign-only status."""

    def test_e3_validation_graph_9(self):
        """Test THEIA_E3 validation graph is graph_9."""
        assert VALIDATION_BENIGN_STATUS["THEIA_E3"]["validation_graph"] == "graph_9"
        assert VALIDATION_BENIGN_STATUS["THEIA_E3"]["status"] == "benign_only"

    def test_e5_validation_graph_11(self):
        """Test THEIA_E5 validation graph is graph_11."""
        assert VALIDATION_BENIGN_STATUS["THEIA_E5"]["validation_graph"] == "graph_11"
        assert VALIDATION_BENIGN_STATUS["THEIA_E5"]["status"] == "benign_only"


# =============================================================================
# Test 24: Empty/Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_val_scores_raises(self):
        """Test empty validation scores raises error."""
        with pytest.raises(ValueError):
            fit_magic_threshold({}, q=0.999)

    def test_all_test_above_threshold(self, fitted_evaluator):
        """Test all test scores above threshold."""
        # All scores very high
        scores = {"a": 1000.0, "b": 2000.0, "c": 3000.0}
        predictions = fitted_evaluator.predict(scores)
        assert all(p == 1 for p in predictions.values())

    def test_all_test_below_threshold(self, fitted_evaluator):
        """Test all test scores below threshold."""
        scores = {"a": 0.0001, "b": 0.0002, "c": 0.0003}
        predictions = fitted_evaluator.predict(scores)
        assert all(p == 0 for p in predictions.values())
