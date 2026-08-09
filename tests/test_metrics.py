import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mstc.metrics import compute_attack_detection_rate, compute_classification_metrics, compute_fp_per_million


def test_classification_metrics_and_confusion_matrix():
    metrics = compute_classification_metrics([1, 0, 1, 0], [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["mcc"] == 0.0
    assert metrics["fpr"] == 0.5
    assert metrics["auroc"] == 0.75
    assert metrics["auprc"] > 0


def test_metric_edge_cases_are_nan_not_fake_zero():
    no_positive = compute_classification_metrics([0, 0], [0, 0], [0.1, 0.2])
    assert math.isnan(no_positive["recall"]) and math.isnan(no_positive["auprc"]) and math.isnan(no_positive["auroc"])
    no_predictions = compute_classification_metrics([1, 0], [0, 0], [0.2, 0.1])
    assert math.isnan(no_predictions["precision"]) and math.isnan(no_predictions["f1"])
    empty = compute_classification_metrics([], [], [])
    assert (empty["tp"], empty["fp"], empty["tn"], empty["fn"]) == (0, 0, 0, 0)
    assert math.isnan(empty["fpr"])


def test_fp_per_million_and_attack_detection_rate():
    assert compute_fp_per_million(3, 6) == 500_000
    assert math.isnan(compute_fp_per_million(0, 0))
    assert compute_attack_detection_rate({"a": {1, 2}, "b": {3}}, {2}) == 0.5
    assert compute_attack_detection_rate({"a": set()}, {2}) == 0.0
    assert math.isnan(compute_attack_detection_rate({}, {2}))
