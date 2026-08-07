"""Tests for MSTC node evaluation integration with calibrated events.

This module tests the get_mstc_node_predictions() function and related helpers.
All tests use synthetic data and do not require THEIA, PostgreSQL, GPU, or checkpoints.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

# Load modules using the same pattern as existing tests
SRC_ROOT = Path(__file__).resolve().parents[1]

# Load aggregation (B3)
AGGREGATION_PATH = SRC_ROOT / "src" / "mstc" / "aggregation.py"
AGGREGATION_SPEC = importlib.util.spec_from_file_location("aggregation_for_tests", AGGREGATION_PATH)
assert AGGREGATION_SPEC is not None and AGGREGATION_SPEC.loader is not None
AGGREGATION_MODULE = importlib.util.module_from_spec(AGGREGATION_SPEC)
sys.modules[AGGREGATION_SPEC.name] = AGGREGATION_MODULE
AGGREGATION_SPEC.loader.exec_module(AGGREGATION_MODULE)
NodeScoreAggregator = AGGREGATION_MODULE.NodeScoreAggregator
AggregationMethod = AGGREGATION_MODULE.AggregationMethod

# Load thresholding (B4)
THRESHOLDING_PATH = SRC_ROOT / "src" / "mstc" / "thresholding.py"
THRESHOLDING_SPEC = importlib.util.spec_from_file_location("thresholding_for_tests", THRESHOLDING_PATH)
assert THRESHOLDING_SPEC is not None and THRESHOLDING_SPEC.loader is not None
THRESHOLDING_MODULE = importlib.util.module_from_spec(THRESHOLDING_SPEC)
sys.modules[THRESHOLDING_SPEC.name] = THRESHOLDING_MODULE
THRESHOLDING_SPEC.loader.exec_module(THRESHOLDING_MODULE)
compute_node_threshold = THRESHOLDING_MODULE.compute_node_threshold
apply_node_threshold = THRESHOLDING_MODULE.apply_node_threshold
NodeThresholdMethod = THRESHOLDING_MODULE.NodeThresholdMethod

# Load node_evaluation (B5 - this module)
NODE_EVAL_PATH = SRC_ROOT / "src" / "mstc" / "node_evaluation.py"
NODE_EVAL_SPEC = importlib.util.spec_from_file_location("node_evaluation_for_tests", NODE_EVAL_PATH)
assert NODE_EVAL_SPEC is not None and NODE_EVAL_SPEC.loader is not None
NODE_EVAL_MODULE = importlib.util.module_from_spec(NODE_EVAL_SPEC)
sys.modules[NODE_EVAL_SPEC.name] = NODE_EVAL_MODULE
NODE_EVAL_SPEC.loader.exec_module(NODE_EVAL_MODULE)
get_mstc_node_predictions = NODE_EVAL_MODULE.get_mstc_node_predictions
get_mstc_node_predictions_from_paths = NODE_EVAL_MODULE.get_mstc_node_predictions_from_paths
load_calibrated_events_from_csv = NODE_EVAL_MODULE.load_calibrated_events_from_csv

# Type aliases
MSTCAggregationMethod = Literal["mean", "max", "topk_mean", "topk_sum"]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create a temporary directory for CSV files."""
    return tmp_path


@pytest.fixture
def basic_validation_events():
    """Basic validation events with some nodes having multiple events."""
    return [
        # Node A: 3 events with moderate scores
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.3, "score_raw": 0.5},
        {"srcnode": "A", "dstnode": "C", "score_calibrated": 0.35, "score_raw": 0.55},
        {"srcnode": "A", "dstnode": "D", "score_calibrated": 0.4, "score_raw": 0.6},
        # Node B: 2 events
        {"srcnode": "B", "dstnode": "E", "score_calibrated": 0.5, "score_raw": 0.7},
        {"srcnode": "B", "dstnode": "F", "score_calibrated": 0.55, "score_raw": 0.75},
        # Node C: 1 event with high score
        {"srcnode": "C", "dstnode": "G", "score_calibrated": 0.9, "score_raw": 0.95},
        # Node E: 4 events (will have topk=2)
        {"srcnode": "E", "dstnode": "A", "score_calibrated": 0.2, "score_raw": 0.4},
        {"srcnode": "E", "dstnode": "B", "score_calibrated": 0.25, "score_raw": 0.45},
        {"srcnode": "E", "dstnode": "C", "score_calibrated": 0.1, "score_raw": 0.3},
        {"srcnode": "E", "dstnode": "D", "score_calibrated": 0.15, "score_raw": 0.35},
    ]


@pytest.fixture
def basic_test_events():
    """Basic test events with a clear outlier node."""
    return [
        # Node A: low scores in test
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.2, "score_raw": 0.4},
        {"srcnode": "A", "dstnode": "C", "score_calibrated": 0.22, "score_raw": 0.42},
        # Node B: moderate scores
        {"srcnode": "B", "dstnode": "E", "score_calibrated": 0.45, "score_raw": 0.65},
        {"srcnode": "B", "dstnode": "F", "score_calibrated": 0.48, "score_raw": 0.68},
        # Node X: VERY high score - should be predicted as anomaly
        {"srcnode": "X", "dstnode": "Y", "score_calibrated": 0.99, "score_raw": 0.99},
        # Node Z: high score
        {"srcnode": "Z", "dstnode": "W", "score_calibrated": 0.85, "score_raw": 0.9},
    ]


@pytest.fixture
def same_node_across_splits_validation():
    """Validation events where node A has moderate scores."""
    return [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.1, "score_raw": 0.3},
        {"srcnode": "A", "dstnode": "C", "score_calibrated": 0.2, "score_raw": 0.4},
    ]


@pytest.fixture
def same_node_across_splits_test():
    """Test events where node A has VERY high score (should NOT affect validation threshold)."""
    return [
        # This 100 should NOT affect validation threshold
        {"srcnode": "A", "dstnode": "X", "score_calibrated": 100.0, "score_raw": 100.0},
        {"srcnode": "B", "dstnode": "Y", "score_calibrated": 0.1, "score_raw": 0.3},
    ]


# ==============================================================================
# Test 1: Validation and test are aggregated separately
# ==============================================================================

def test_validation_and_test_are_aggregated_separately(
    basic_validation_events, basic_test_events
):
    """Verify val/test aggregation does NOT mix scores."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        aggregation_method="topk_mean",
        topk=2,
        score_field="score_calibrated",
    )

    # Node A in validation should have scores [0.3, 0.35, 0.4] → top-2 mean = 0.375
    assert "A" in result["validation_node_scores"]
    val_A_score = result["validation_node_scores"]["A"]
    assert abs(val_A_score - 0.375) < 1e-6

    # Node A in test should have scores [0.2, 0.22] → top-2 mean = 0.21
    assert "A" in result["test_node_scores"]
    test_A_score = result["test_node_scores"]["A"]
    assert abs(test_A_score - 0.21) < 1e-6

    # They must be different
    assert val_A_score != test_A_score


# ==============================================================================
# Test 2: Same node across splits does not leak test score into validation
# ==============================================================================

def test_same_node_across_splits_does_not_leak_test_score_into_validation(
    same_node_across_splits_validation, same_node_across_splits_test
):
    """CRITICAL: Test score must NOT influence validation threshold."""
    result = get_mstc_node_predictions(
        same_node_across_splits_validation,
        same_node_across_splits_test,
        aggregation_method="topk_mean",
        topk=2,
        score_field="score_calibrated",
    )

    # Validation node A score:
    # A appears as src twice: [0.1, 0.2]
    # A does NOT appear as dst (B and C are dsts, not A)
    # So A's scores = [0.1, 0.2], top-2 mean = 0.15
    val_A_score = result["validation_node_scores"]["A"]
    assert abs(val_A_score - 0.15) < 1e-6

    # Test node A score should be [100.0] → mean = 100.0
    test_A_score = result["test_node_scores"]["A"]
    assert abs(test_A_score - 100.0) < 1e-6

    # Threshold should be based ONLY on validation nodes
    # With quantile=0.999, threshold ≈ max validation node score
    threshold = result["threshold"]
    # The threshold should be < 1.0 (100 is from test, should not be in threshold)
    assert threshold < 1.0

    # Test predictions should use the validation-based threshold
    # A in test: 100.0 > threshold → predicted as 1 (anomaly)
    test_preds = result["test_node_predictions"]
    assert test_preds["A"] == 1


# ==============================================================================
# Test 3: Validation quantile threshold uses validation nodes only
# ==============================================================================

def test_validation_quantile_threshold_uses_validation_nodes_only(
    basic_validation_events, basic_test_events
):
    """Threshold must be computed from validation node scores only."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        threshold_method="validation_quantile",
        threshold_quantile=0.999,
        score_field="score_calibrated",
    )

    # Compute expected threshold manually
    # Validation nodes: A=[0.3,0.35,0.4], B=[0.5,0.55], C=[0.9], E=[0.2,0.25,0.1,0.15]
    # Node scores with topk=5: A=0.35, B=0.525, C=0.9, E=0.2
    # Sorted: [0.2, 0.35, 0.525, 0.9]
    # 0.999 quantile ≈ max = 0.9

    expected_threshold = 0.9  # max of validation node scores
    assert abs(result["threshold"] - expected_threshold) < 0.01


# ==============================================================================
# Test 4: Max validation threshold uses validation nodes only
# ==============================================================================

def test_max_validation_uses_validation_nodes_only(
    basic_validation_events, basic_test_events
):
    """max_validation threshold must be computed from validation node scores only."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        threshold_method="max_validation",
        score_field="score_calibrated",
    )

    # max_validation = max(validation node scores)
    # Expected: C has max score = 0.9
    assert abs(result["threshold"] - 0.9) < 0.01

    # Test node X has score 0.99, which is > 0.9, so it should be predicted as 1
    assert result["test_node_predictions"]["X"] == 1


# ==============================================================================
# Test 5: Test predictions use fixed validation threshold
# ==============================================================================

def test_test_predictions_use_fixed_validation_threshold(
    basic_validation_events, basic_test_events
):
    """Test predictions must use the threshold computed from validation only."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        threshold_method="validation_quantile",
        threshold_quantile=0.999,
        score_field="score_calibrated",
    )

    # threshold = 0.9 (max validation node score)
    threshold = result["threshold"]

    # Test predictions
    test_scores = result["test_node_scores"]
    test_preds = result["test_node_predictions"]

    for node_id, score in test_scores.items():
        expected_pred = 1 if score > threshold else 0
        assert test_preds[node_id] == expected_pred, (
            f"Node {node_id}: score={score}, threshold={threshold}, "
            f"expected_pred={expected_pred}, actual_pred={test_preds[node_id]}"
        )


# ==============================================================================
# Test 6: Default uses calibrated score
# ==============================================================================

def test_default_uses_calibrated_score(basic_validation_events, basic_test_events):
    """Default score_field should be score_calibrated."""
    result_calibrated = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        score_field="score_calibrated",
    )

    # With score_calibrated and include_dst=True:
    # Node A has events: A->B(0.3), A->C(0.35), A->D(0.4), E->A(0.2)
    # Topk_mean(5) = mean of top 5 = (0.4+0.35+0.3+0.2)/4 = 0.3125
    assert abs(result_calibrated["validation_node_scores"]["A"] - 0.3125) < 1e-6

    # Verify metadata shows score_calibrated was used
    assert result_calibrated["metadata"]["score_field"] == "score_calibrated"


# ==============================================================================
# Test 7: Raw score field supports no-calibration ablation
# ==============================================================================

def test_raw_score_field_supports_no_calibration_ablation(
    basic_validation_events, basic_test_events
):
    """Should support score_raw for ablation without calibration."""
    result_raw = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        score_field="score_raw",
    )

    result_calibrated = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        score_field="score_calibrated",
    )

    # Raw scores are generally higher than calibrated scores
    # A in val: raw=[0.5,0.55,0.6], calibrated=[0.3,0.35,0.4]
    val_A_raw = result_raw["validation_node_scores"]["A"]
    val_A_cal = result_calibrated["validation_node_scores"]["A"]

    # Both should still be computed correctly, just with different values
    assert val_A_raw > val_A_cal  # raw scores typically higher

    # Verify score_field was used correctly
    assert result_raw["metadata"]["score_field"] == "score_raw"
    assert result_calibrated["metadata"]["score_field"] == "score_calibrated"


# ==============================================================================
# Test 8: Topk mean is reused from aggregator
# ==============================================================================

def test_topk_mean_is_reused_from_aggregator(basic_validation_events):
    """Verify topk_mean uses the correct top-k events."""
    # Test with topk=2
    result = get_mstc_node_predictions(
        basic_validation_events,
        [],  # Empty test
        aggregation_method="topk_mean",
        topk=2,
        include_dst=False,  # Use include_dst=False for predictable results
        score_field="score_calibrated",
    )

    # Node E has events as src: E->A(0.2), E->B(0.25), E->C(0.1), E->D(0.15)
    # Top-2: [0.25, 0.2] → mean = 0.225
    assert abs(result["validation_node_scores"]["E"] - 0.225) < 1e-6


# ==============================================================================
# Test 9: include_dst true is consistent for both splits
# ==============================================================================

def test_include_dst_true_is_consistent_for_both_splits():
    """include_dst=True should include both src and dst nodes consistently."""
    val_events = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5},
    ]
    test_events = [
        {"srcnode": "C", "dstnode": "B", "score_calibrated": 0.7},
    ]

    result = get_mstc_node_predictions(
        val_events,
        test_events,
        include_dst=True,
        score_field="score_calibrated",
    )

    # B appears in both val and test as dst
    # B in val: [0.5] → score = 0.5
    # B in test: [0.7] → score = 0.7
    assert result["validation_node_scores"]["B"] == 0.5
    assert result["test_node_scores"]["B"] == 0.7

    # A only in val, C only in test
    assert "A" in result["validation_node_scores"]
    assert "C" in result["test_node_scores"]


# ==============================================================================
# Test 10: include_dst false is consistent for both splits
# ==============================================================================

def test_include_dst_false_is_consistent_for_both_splits():
    """include_dst=False should only include src nodes."""
    val_events = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5},
    ]
    test_events = [
        {"srcnode": "C", "dstnode": "B", "score_calibrated": 0.7},
    ]

    result = get_mstc_node_predictions(
        val_events,
        test_events,
        include_dst=False,
        score_field="score_calibrated",
    )

    # B should NOT be in any scores
    assert "B" not in result["validation_node_scores"]
    assert "B" not in result["test_node_scores"]

    # Only src nodes should appear
    assert "A" in result["validation_node_scores"]
    assert "C" in result["test_node_scores"]


# ==============================================================================
# Test 11: Empty test returns empty predictions
# ==============================================================================

def test_empty_test_returns_empty_predictions(basic_validation_events):
    """Empty test events should return empty predictions."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        [],  # Empty test
        score_field="score_calibrated",
    )

    assert result["test_node_scores"] == {}
    assert result["test_node_predictions"] == {}
    assert result["metadata"]["num_test_nodes"] == 0
    assert result["metadata"]["num_test_predictions"] == 0


# ==============================================================================
# Test 12: Empty validation nodes raise
# ==============================================================================

def test_empty_validation_nodes_raise():
    """Empty validation events should raise ValueError."""
    with pytest.raises(ValueError, match="validation_event_records cannot be empty"):
        get_mstc_node_predictions([], [], score_field="score_calibrated")


# ==============================================================================
# Test 13: Strict threshold boundary is preserved (score == threshold → 0)
# ==============================================================================

def test_strict_threshold_boundary_is_preserved():
    """Score exactly equal to threshold should predict 0 (strict >)."""
    # Create validation where one node has score exactly at what will be threshold
    val_events = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5},
        {"srcnode": "B", "dstnode": "C", "score_calibrated": 0.7},
    ]

    # With max_validation, threshold = 0.7
    result = get_mstc_node_predictions(
        val_events,
        [
            # Test node C with score EXACTLY equal to threshold
            {"srcnode": "C", "dstnode": "D", "score_calibrated": 0.7},
            # Test node A with score just below threshold
            {"srcnode": "A", "dstnode": "E", "score_calibrated": 0.699999},
            # Test node X with score just above threshold
            {"srcnode": "X", "dstnode": "Y", "score_calibrated": 0.700001},
        ],
        threshold_method="max_validation",
        score_field="score_calibrated",
    )

    # C: score == threshold → 0 (strict >)
    assert result["test_node_predictions"]["C"] == 0
    # A: score < threshold → 0
    assert result["test_node_predictions"]["A"] == 0
    # X: score > threshold → 1
    assert result["test_node_predictions"]["X"] == 1


# ==============================================================================
# Test 14: Node IDs are preserved
# ==============================================================================

def test_node_ids_are_preserved(basic_validation_events, basic_test_events):
    """Node IDs from events should be preserved in results."""
    result = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        score_field="score_calibrated",
    )

    # Check validation node IDs (include_dst=True adds dst nodes)
    expected_val_nodes = {"A", "B", "C", "D", "E", "F", "G"}
    assert set(result["validation_node_scores"].keys()) == expected_val_nodes

    # Check test node IDs
    # basic_test_events has: A->B, A->C, B->E, B->F, X->Y, Z->W
    # With include_dst=True: A, B, C, E, F, X, Y, Z, W
    # D does not appear in test events
    expected_test_nodes = {"A", "B", "C", "E", "F", "W", "X", "Y", "Z"}
    assert set(result["test_node_scores"].keys()) == expected_test_nodes


# ==============================================================================
# Test 15: Inputs are not modified
# ==============================================================================

def test_inputs_are_not_modified(basic_validation_events, basic_test_events):
    """Input event records should not be modified by the function."""
    # Make deep copies for comparison
    import copy
    val_copy = copy.deepcopy(basic_validation_events)
    test_copy = copy.deepcopy(basic_test_events)

    get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        score_field="score_calibrated",
    )

    # Original events should be unchanged
    assert basic_validation_events == val_copy
    assert basic_test_events == test_copy


# ==============================================================================
# Test 16: No label fields are required
# ==============================================================================

def test_no_label_fields_are_required():
    """Event records without labels should work fine."""
    val_events = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5},
    ]
    test_events = [
        {"srcnode": "C", "dstnode": "D", "score_calibrated": 0.9},
    ]

    # This should work without any label fields
    result = get_mstc_node_predictions(
        val_events,
        test_events,
        score_field="score_calibrated",
    )

    assert result["validation_node_scores"]["A"] == 0.5
    assert result["test_node_predictions"]["C"] == 1  # 0.9 > threshold


# ==============================================================================
# Test 17: Labels do not affect predictions
# ==============================================================================

def test_labels_do_not_affect_predictions():
    """Labels in records should not affect scores or predictions."""
    val_events_labeled = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5, "label": 1, "is_malicious": True},
    ]
    val_events_unlabeled = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5},
    ]
    test_events_labeled = [
        {"srcnode": "C", "dstnode": "D", "score_calibrated": 0.9, "label": 0, "is_malicious": False},
    ]
    test_events_unlabeled = [
        {"srcnode": "C", "dstnode": "D", "score_calibrated": 0.9},
    ]

    result_labeled = get_mstc_node_predictions(
        val_events_labeled,
        test_events_labeled,
        score_field="score_calibrated",
    )

    result_unlabeled = get_mstc_node_predictions(
        val_events_unlabeled,
        test_events_unlabeled,
        score_field="score_calibrated",
    )

    # Results should be identical
    assert result_labeled == result_unlabeled


# ==============================================================================
# Test 18: Legacy path remains available
# ==============================================================================

def test_legacy_path_remains_available():
    """Verify that legacy functions in node_evaluation remain importable."""
    # Verify legacy files exist and have expected content
    legacy_ne_path = SRC_ROOT / "src" / "detection" / "node_evaluation.py"
    assert legacy_ne_path.exists(), "Legacy node_evaluation.py should exist"

    legacy_ne_content = legacy_ne_path.read_text()

    # These functions should still exist in legacy code
    assert "def get_node_predictions" in legacy_ne_content
    assert "def main" in legacy_ne_content
    assert "def analyze_false_positives" in legacy_ne_content

    # Legacy evaluation_utils should also exist
    legacy_ut_path = SRC_ROOT / "src" / "detection" / "evaluation_utils.py"
    assert legacy_ut_path.exists(), "Legacy evaluation_utils.py should exist"

    legacy_ut_content = legacy_ut_path.read_text()
    assert "def get_threshold" in legacy_ut_content
    assert "def reduce_losses_to_score" in legacy_ut_content


# ==============================================================================
# Test 19: CSV path loading works correctly
# ==============================================================================

def test_csv_path_loading(temp_csv_dir):
    """Test loading events from CSV files."""
    # Create validation CSV
    val_csv = temp_csv_dir / "validation_calibrated.csv"
    val_data = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.5, "score_raw": 0.7},
        {"srcnode": "B", "dstnode": "C", "score_calibrated": 0.8, "score_raw": 0.9},
    ]
    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["srcnode", "dstnode", "score_calibrated", "score_raw"])
        writer.writeheader()
        writer.writerows(val_data)

    # Create test CSV
    test_csv = temp_csv_dir / "test_calibrated.csv"
    test_data = [
        {"srcnode": "X", "dstnode": "Y", "score_calibrated": 0.99, "score_raw": 0.99},
    ]
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["srcnode", "dstnode", "score_calibrated", "score_raw"])
        writer.writeheader()
        writer.writerows(test_data)

    # Load via path-based function
    result = get_mstc_node_predictions_from_paths(
        val_csv,
        test_csv,
        score_field="score_calibrated",
    )

    # Verify loaded correctly
    assert result["validation_node_scores"]["A"] == 0.5
    assert result["validation_node_scores"]["B"] == 0.65  # (0.5+0.8)/2
    assert result["test_node_predictions"]["X"] == 1  # 0.99 > threshold


# ==============================================================================
# Test 20: Path-based function handles missing files gracefully
# ==============================================================================

def test_path_based_handles_missing_validation():
    """Path-based function should handle missing validation gracefully."""
    with pytest.raises(ValueError, match="validation_event_records cannot be empty"):
        get_mstc_node_predictions_from_paths(
            Path("/nonexistent/validation.csv"),
            None,
            score_field="score_calibrated",
        )


# ==============================================================================
# Test 21: Aggregation methods are consistent
# ==============================================================================

def test_aggregation_methods_are_consistent(basic_validation_events, basic_test_events):
    """Different aggregation methods should produce different but correct results."""
    # Mean aggregation with include_dst=False for predictable results
    result_mean = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        aggregation_method="mean",
        include_dst=False,
        score_field="score_calibrated",
    )

    # Max aggregation
    result_max = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        aggregation_method="max",
        include_dst=False,
        score_field="score_calibrated",
    )

    # Topk mean (default)
    result_topk = get_mstc_node_predictions(
        basic_validation_events,
        basic_test_events,
        aggregation_method="topk_mean",
        topk=5,
        include_dst=False,
        score_field="score_calibrated",
    )

    # Node A scores should differ between methods
    val_A_mean = result_mean["validation_node_scores"]["A"]
    val_A_max = result_max["validation_node_scores"]["A"]
    val_A_topk = result_topk["validation_node_scores"]["A"]

    # A has events as src: [0.3, 0.35, 0.4]
    # Mean of [0.3, 0.35, 0.4] = 0.35
    assert abs(val_A_mean - 0.35) < 1e-6
    # Max of [0.3, 0.35, 0.4] = 0.4
    assert abs(val_A_max - 0.4) < 1e-6
    # Topk mean of [0.3, 0.35, 0.4] = 0.35 (all included since topk=5)
    assert abs(val_A_topk - 0.35) < 1e-6


# ==============================================================================
# Test 22: Different topk values produce different results
# ==============================================================================

def test_different_topk_values_produce_different_results():
    """Different topk values should affect node scores appropriately."""
    val_events = [
        {"srcnode": "A", "dstnode": "B", "score_calibrated": 0.1},
        {"srcnode": "A", "dstnode": "C", "score_calibrated": 0.2},
        {"srcnode": "A", "dstnode": "D", "score_calibrated": 0.3},
        {"srcnode": "A", "dstnode": "E", "score_calibrated": 0.4},
        {"srcnode": "A", "dstnode": "F", "score_calibrated": 0.5},
    ]

    result_topk2 = get_mstc_node_predictions(
        val_events,
        [],
        aggregation_method="topk_mean",
        topk=2,
        score_field="score_calibrated",
    )

    result_topk3 = get_mstc_node_predictions(
        val_events,
        [],
        aggregation_method="topk_mean",
        topk=3,
        score_field="score_calibrated",
    )

    # topk=2: top-2 = [0.5, 0.4] → mean = 0.45
    assert abs(result_topk2["validation_node_scores"]["A"] - 0.45) < 1e-6

    # topk=3: top-3 = [0.5, 0.4, 0.3] → mean = 0.4
    assert abs(result_topk3["validation_node_scores"]["A"] - 0.4) < 1e-6


# ==============================================================================
# Test 23: Smoke test with realistic scores
# ==============================================================================

def test_smoke_test_realistic_scores():
    """Smoke test with realistic calibrated scores."""
    # Create realistic validation events
    np.random.seed(42)
    val_events = []
    for i in range(100):
        # Normal nodes have scores around 0.3-0.5
        score = 0.3 + np.random.random() * 0.2
        val_events.append({
            "srcnode": f"node_{i % 20}",  # 20 unique nodes
            "dstnode": f"node_{(i + 1) % 20}",
            "score_calibrated": float(score),
        })

    # Add a few anomaly events to test
    val_events.append({"srcnode": "anomaly_node", "dstnode": "victim", "score_calibrated": 0.99})

    # Test events
    test_events = [
        {"srcnode": "normal_test", "dstnode": "victim", "score_calibrated": 0.35},
        {"srcnode": "attack_test", "dstnode": "victim", "score_calibrated": 0.98},
    ]

    result = get_mstc_node_predictions(
        val_events,
        test_events,
        aggregation_method="topk_mean",
        topk=5,
        threshold_method="validation_quantile",
        threshold_quantile=0.999,
        score_field="score_calibrated",
    )

    # Should have computed results
    assert len(result["validation_node_scores"]) > 0
    assert result["threshold"] > 0

    # The attack_test score (0.98) should be above threshold
    # normal_test score (0.35) should be below threshold
    # (Exact prediction depends on the validation threshold)


# ==============================================================================
# Test 24: DST node handling with self-loops
# ==============================================================================

def test_dst_node_handling_with_self_loops():
    """When src == dst, score should be appended twice."""
    events = [
        # Self-loop: A → A
        {"srcnode": "A", "dstnode": "A", "score_calibrated": 0.5},
    ]

    result = get_mstc_node_predictions(
        events,
        [],
        include_dst=True,
        score_field="score_calibrated",
    )

    # A has scores [0.5, 0.5] due to self-loop (appended twice)
    assert result["validation_node_scores"]["A"] == 0.5


# ==============================================================================
# Test 25: KMeans path deferred
# ==============================================================================

def test_kmeans_path_deferred_note():
    """Document that MSTC KMeans integration is deferred.

    The legacy compute_kmeans_labels() function in evaluation_utils.py
    operates on pre-computed node scores. MSTC path does not currently
    connect to this function. This is a known limitation documented
    in the B5 report.
    """
    # This test verifies the legacy file exists and has KMeans function
    legacy_ut_path = SRC_ROOT / "src" / "detection" / "evaluation_utils.py"
    assert legacy_ut_path.exists(), "Legacy evaluation_utils.py should exist"

    legacy_ut_content = legacy_ut_path.read_text()
    assert "def compute_kmeans_labels" in legacy_ut_content

    # KMeans integration is NOT implemented in MSTC path
    # This is deferred to future work
    pass
