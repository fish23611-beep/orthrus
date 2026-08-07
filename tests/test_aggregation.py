"""Synthetic unit tests for NodeScoreAggregator."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


AGGREGATION_PATH = Path(__file__).resolve().parents[1] / "src" / "mstc" / "aggregation.py"
AGGREGATION_SPEC = importlib.util.spec_from_file_location(
    "aggregation_for_tests", AGGREGATION_PATH
)
assert AGGREGATION_SPEC is not None and AGGREGATION_SPEC.loader is not None
AGGREGATION_MODULE = importlib.util.module_from_spec(AGGREGATION_SPEC)
sys.modules[AGGREGATION_SPEC.name] = AGGREGATION_MODULE
AGGREGATION_SPEC.loader.exec_module(AGGREGATION_MODULE)
NodeScoreAggregator = AGGREGATION_MODULE.NodeScoreAggregator


def _event(score, src="A", dst="B", **extra):
    return {"srcnode": src, "dstnode": dst, "score_calibrated": score, **extra}


@pytest.fixture
def aggregator():
    return NodeScoreAggregator()


def test_topk_mean_less_than_k_uses_all(aggregator):
    result = aggregator.aggregate_events([_event(1), _event(2), _event(3)], topk=5, include_dst=False)
    assert result == {"A": pytest.approx(2.0)}


def test_topk_mean_uses_highest_k(aggregator):
    result = aggregator.aggregate_events([_event(score) for score in [1, 10, 3, 5]], topk=2, include_dst=False)
    assert result["A"] == pytest.approx(7.5)


def test_max(aggregator):
    assert aggregator.aggregate_events([_event(score) for score in [1, 10, 3]], method="max", include_dst=False)["A"] == 10.0


def test_mean(aggregator):
    assert aggregator.aggregate_events([_event(score) for score in [1, 2, 6]], method="mean", include_dst=False)["A"] == pytest.approx(3.0)


def test_topk_sum(aggregator):
    assert aggregator.aggregate_events([_event(score) for score in [1, 10, 3, 5]], method="topk_sum", topk=2, include_dst=False)["A"] == 15.0


def test_topk_sum_less_than_k_uses_all(aggregator):
    assert aggregator.aggregate_events([_event(1), _event(2)], method="topk_sum", topk=5, include_dst=False)["A"] == 3.0


def test_include_dst_true_assigns_score_to_both_nodes(aggregator):
    assert aggregator.build_node_to_event_scores([_event(5)]) == {"A": [5.0], "B": [5.0]}


def test_include_dst_false_assigns_score_only_to_source(aggregator):
    assert aggregator.build_node_to_event_scores([_event(5)], include_dst=False) == {"A": [5.0]}


def test_one_node_multiple_events(aggregator):
    result = aggregator.aggregate_events([_event(1), _event(5), _event(3)], method="max", include_dst=False)
    assert result == {"A": 5.0}


def test_multiple_nodes_are_independent(aggregator):
    events = [_event(1, "A", "B"), _event(9, "C", "D")]
    assert aggregator.aggregate_events(events, method="mean") == {"A": 1.0, "B": 1.0, "C": 9.0, "D": 9.0}


def test_duplicate_scores_are_not_deduplicated(aggregator):
    result = aggregator.aggregate_events([_event(score) for score in [5, 5, 5, 1]], topk=2, include_dst=False)
    assert result["A"] == pytest.approx(5.0)


def test_event_order_does_not_change_result(aggregator):
    events = [_event(1, "A", "B"), _event(10, "A", "C"), _event(3, "B", "C")]
    assert aggregator.aggregate_events(events, topk=2) == aggregator.aggregate_events(list(reversed(events)), topk=2)


def test_empty_input_returns_empty_mapping(aggregator):
    assert aggregator.aggregate_events([]) == {}


def test_invalid_method_raises(aggregator):
    with pytest.raises(ValueError, match="method"):
        aggregator.aggregate_events([_event(1)], method="median")


@pytest.mark.parametrize("topk", [0, -1, 1.5, True])
def test_invalid_topk_raises(aggregator, topk):
    with pytest.raises(ValueError, match="topk"):
        aggregator.aggregate_events([_event(1)], topk=topk)


def test_missing_calibrated_score_raises(aggregator):
    with pytest.raises(KeyError, match="score_calibrated"):
        aggregator.aggregate_events([{"srcnode": "A", "dstnode": "B"}])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_raise(aggregator, score):
    with pytest.raises(ValueError, match="finite"):
        aggregator.aggregate_events([_event(score)])


def test_input_records_are_not_modified(aggregator):
    events = [_event(5, "A", "B"), _event(1, "A", "C")]
    before = copy.deepcopy(events)
    aggregator.aggregate_events(events, topk=1)
    assert events == before


def test_score_field_can_be_switched(aggregator):
    events = [_event(10, score_raw=2), _event(20, score_raw=6)]
    result = aggregator.aggregate_events(events, method="mean", include_dst=False, score_field="score_raw")
    assert result == {"A": 4.0}


def test_src_equals_dst_preserves_existing_double_append_semantics(aggregator):
    events = [_event(5, "A", "A")]
    assert aggregator.build_node_to_event_scores(events, include_dst=True) == {"A": [5.0, 5.0]}
    assert aggregator.build_node_to_event_scores(events, include_dst=False) == {"A": [5.0]}
