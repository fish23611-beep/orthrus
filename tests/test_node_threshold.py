"""Synthetic tests for pure C6 node thresholding algorithms."""

import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


thresholding = _load_module("thresholding_for_tests", "src/mstc/thresholding.py")
aggregation = _load_module("aggregation_for_threshold_tests", "src/mstc/aggregation.py")


def test_validation_quantile_matches_numpy():
    scores = {10: 1.0, 20: 2.0, 30: 3.0, 40: 4.0}
    assert thresholding.compute_validation_quantile_threshold(scores, 0.5) == pytest.approx(np.quantile([1, 2, 3, 4], 0.5))


def test_default_high_quantile_matches_numpy_and_is_finite():
    scores = [1.0, 2.0, 3.0, 20.0]
    threshold = thresholding.compute_validation_quantile_threshold(scores)
    assert threshold == pytest.approx(np.quantile(scores, 0.999))
    assert np.isfinite(threshold)


def test_quantile_zero_returns_minimum():
    assert thresholding.compute_validation_quantile_threshold([3.0, 1.0, 2.0], 0.0) == 1.0


def test_quantile_one_returns_maximum():
    assert thresholding.compute_validation_quantile_threshold([3.0, 1.0, 2.0], 1.0) == 3.0


@pytest.mark.parametrize("quantile", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_quantile_raises(quantile):
    with pytest.raises(ValueError, match="quantile"):
        thresholding.compute_validation_quantile_threshold([1.0], quantile)


def test_max_validation_returns_validation_maximum():
    assert thresholding.compute_max_validation_threshold([1.0, 5.0, 3.0]) == 5.0


def test_empty_validation_raises():
    with pytest.raises(ValueError, match="empty"):
        thresholding.compute_node_threshold({})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_validation_score_raises(score):
    with pytest.raises(ValueError, match="finite"):
        thresholding.compute_node_threshold([1.0, score])


def test_strict_threshold_boundary_matches_legacy_baseline():
    predictions = thresholding.apply_node_threshold({"equal": 5.0, "above": 5.0001, "below": 4.0}, 5.0)
    assert predictions == {"equal": 0, "above": 1, "below": 0}


def test_multiple_node_predictions_preserve_node_ids():
    scores = {11: 0.5, 42: 2.0, 999: 2.1}
    assert thresholding.apply_node_threshold(scores, 2.0) == {11: 0, 42: 0, 999: 1}


def test_input_mapping_is_not_modified():
    validation_scores = {"v1": 1.0, "v2": 2.0}
    node_scores = {"n1": 2.0, "n2": 3.0}
    before_validation = copy.deepcopy(validation_scores)
    before_nodes = copy.deepcopy(node_scores)
    thresholding.compute_node_threshold(validation_scores)
    thresholding.apply_node_threshold(node_scores, 2.0)
    assert validation_scores == before_validation
    assert node_scores == before_nodes


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_prediction_score_raises(score):
    with pytest.raises(ValueError, match="finite"):
        thresholding.apply_node_threshold({"node": score}, 1.0)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        thresholding.apply_node_threshold({"node": 1.0}, threshold)


def test_core_api_has_no_label_arguments():
    forbidden = {"label", "labels", "y_true", "ground_truth", "test_scores", "attack"}
    parameters = set(inspect.signature(thresholding.compute_node_threshold).parameters)
    assert not parameters & forbidden


def test_mapping_and_sequence_compute_same_threshold():
    values = [1.0, 4.0, 5.0, 9.0]
    mapping = {10: 1.0, 20: 4.0, 30: 5.0, 40: 9.0}
    assert thresholding.compute_node_threshold(values, quantile=0.75) == thresholding.compute_node_threshold(mapping, quantile=0.75)


def test_threshold_and_predictions_are_deterministic():
    validation = {1: 1.0, 2: 3.0, 3: 5.0}
    nodes = {10: 3.0, 20: 5.0, 30: 6.0}
    first = thresholding.compute_node_threshold(validation, quantile=0.5)
    second = thresholding.compute_node_threshold(validation, quantile=0.5)
    assert first == second
    assert thresholding.apply_node_threshold(nodes, first) == thresholding.apply_node_threshold(nodes, second)


def test_threshold_does_not_depend_on_test_node_scores():
    validation = {1: 1.0, 2: 2.0, 3: 3.0}
    threshold = thresholding.compute_node_threshold(validation, quantile=0.5)
    low_test_predictions = thresholding.apply_node_threshold({10: -100.0}, threshold)
    high_test_predictions = thresholding.apply_node_threshold({10: 10000.0}, threshold)
    assert threshold == pytest.approx(2.0)
    assert low_test_predictions == {10: 0}
    assert high_test_predictions == {10: 1}


def test_b3_aggregation_to_b4_threshold_uses_validation_only():
    aggregator = aggregation.NodeScoreAggregator()
    validation_events = [
        {"srcnode": "v1", "dstnode": "v2", "score_calibrated": 1.0},
        {"srcnode": "v2", "dstnode": "v3", "score_calibrated": 3.0},
    ]
    test_events = [{"srcnode": "t1", "dstnode": "t2", "score_calibrated": 100.0}]
    validation_nodes = aggregator.aggregate_events(validation_events, method="mean")
    test_nodes = aggregator.aggregate_events(test_events, method="mean")
    threshold = thresholding.compute_node_threshold(validation_nodes, quantile=1.0)
    assert threshold == 3.0
    assert thresholding.apply_node_threshold(test_nodes, threshold) == {"t1": 1, "t2": 1}


def test_invalid_threshold_method_raises():
    with pytest.raises(ValueError, match="method"):
        thresholding.compute_node_threshold([1.0], method="kmeans")
