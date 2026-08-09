"""Tests for C6 configuration defaults and legacy KMeans delegation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
from config import get_default_cfg


def _load_node_evaluation():
    path = SRC_ROOT / "mstc" / "node_evaluation.py"
    spec = importlib.util.spec_from_file_location("c6_b6_node_evaluation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


node_evaluation = _load_node_evaluation()


def _args():
    return SimpleNamespace(cpu=True, from_weights=False, seed=0, skip_tracing=False, dataset="THEIA_E5")


def _events(scores, *, labels=False):
    return [
        {
            "srcnode": node_id,
            "dstnode": f"dst-{node_id}",
            "score_calibrated": score,
            "score_raw": score,
            **({"label": int(score > 5), "y_true": int(score > 5)} if labels else {}),
        }
        for node_id, score in scores.items()
    ]


def test_c6_default_configuration_contract_and_baseline_compatibility():
    cfg = get_default_cfg(_args())
    assert cfg.model.variant == "orthrus_baseline"
    assert cfg.calibration.method == "hierarchical_relation"
    assert cfg.calibration.min_triplet_samples == 100
    assert cfg.calibration.min_type_pair_samples == 200
    assert cfg.calibration.epsilon == pytest.approx(1e-12)
    assert cfg.node_aggregation.method == "topk_mean"
    assert cfg.node_aggregation.topk == 5
    assert cfg.node_aggregation.include_dst is True
    assert cfg.node_aggregation.score_field == "score_calibrated"
    assert cfg.node_threshold.method == "validation_quantile"
    assert cfg.node_threshold.quantile == pytest.approx(0.999)
    assert not hasattr(cfg, "use_mstc_node_evaluation")


def test_cfg_wrapper_drives_b5_parameters():
    cfg = get_default_cfg(_args())
    cfg.node_aggregation.method = "mean"
    cfg.node_aggregation.topk = 3
    cfg.node_aggregation.include_dst = False
    cfg.node_aggregation.score_field = "score_raw"
    cfg.node_threshold.method = "max_validation"
    cfg.node_threshold.quantile = 0.9
    validation = [
        {"srcnode": "A", "dstnode": "x", "score_raw": 1.0, "score_calibrated": 10.0},
        {"srcnode": "A", "dstnode": "y", "score_raw": 3.0, "score_calibrated": 20.0},
        {"srcnode": "B", "dstnode": "z", "score_raw": 5.0, "score_calibrated": 30.0},
    ]
    test = [
        {"srcnode": "T", "dstnode": "x", "score_raw": 6.0, "score_calibrated": 0.0},
        {"srcnode": "U", "dstnode": "x", "score_raw": 1.0, "score_calibrated": 99.0},
    ]
    result = node_evaluation.get_mstc_node_predictions_from_cfg(validation, test, cfg)
    assert result["validation_node_scores"] == {"A": 2.0, "B": 5.0}
    assert result["threshold"] == 5.0
    assert result["test_node_predictions"] == {"T": 1, "U": 0}
    assert result["metadata"]["aggregation_method"] == "mean"
    assert result["metadata"]["score_field"] == "score_raw"


def test_kmeans_adapter_calls_legacy_once_and_extracts_predictions(monkeypatch):
    calls = []

    def fake_legacy(results, topk_K):
        calls.append((results, topk_K))
        assert set(results["high"]) == {"score", "y_hat"}
        results["high"]["y_hat"] = 1
        return results

    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: fake_legacy)
    result = node_evaluation.get_mstc_node_predictions(
        _events({"validation": 1.0}),
        _events({"low": 2.0, "high": 10.0}),
        include_dst=False, threshold_method="kmeans", legacy_kmeans_topk=20,
    )
    assert len(calls) == 1
    assert calls[0][1] == 20
    assert result["test_node_predictions"] == {"low": 0, "high": 1}
    assert result["threshold"] is None
    assert result["threshold_method"] == "kmeans"
    assert result["metadata"]["prediction_mode"] == "legacy_kmeans"


def test_kmeans_ignores_labels_and_handles_empty_test(monkeypatch):
    calls = []

    def fake_legacy(results, topk_K):
        calls.append(topk_K)
        results[max(results, key=lambda key: results[key]["score"])]["y_hat"] = 1
        return results

    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: fake_legacy)
    validation = _events({"validation": 1.0})
    scores = {"low": 2.0, "high": 10.0}
    labeled = node_evaluation.get_mstc_node_predictions(validation, _events(scores, labels=True), include_dst=False, threshold_method="kmeans")
    unlabeled = node_evaluation.get_mstc_node_predictions(validation, _events(scores), include_dst=False, threshold_method="kmeans")
    assert labeled["test_node_predictions"] == unlabeled["test_node_predictions"]
    empty = node_evaluation.get_mstc_node_predictions(validation, [], include_dst=False, threshold_method="kmeans")
    assert empty["test_node_scores"] == {}
    assert empty["test_node_predictions"] == {}
    assert empty["threshold"] is None
    assert calls == [20, 20]


def test_aggregation_topk_and_kmeans_topk_are_distinct(monkeypatch):
    received = []

    def fake_legacy(results, topk_K):
        received.append(topk_K)
        return results

    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: fake_legacy)
    node_evaluation.get_mstc_node_predictions(
        _events({"validation": 1.0}), _events({"a": 1.0, "b": 2.0}),
        include_dst=False, topk=5, threshold_method="kmeans", legacy_kmeans_topk=20,
    )
    assert received == [20]


@pytest.mark.parametrize("method", ["validation_quantile", "max_validation"])
def test_validation_threshold_methods_do_not_call_legacy(monkeypatch, method):
    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: pytest.fail("unexpected legacy call"))
    result = node_evaluation.get_mstc_node_predictions(
        _events({"validation": 1.0}), _events({"test": 2.0}),
        include_dst=False, threshold_method=method,
    )
    assert result["threshold"] == 1.0
    assert result["test_node_predictions"] == {"test": 1}


def test_unknown_threshold_method_does_not_fallback_to_kmeans(monkeypatch):
    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: pytest.fail("unexpected legacy call"))
    with pytest.raises(ValueError, match="method"):
        node_evaluation.get_mstc_node_predictions(
            _events({"validation": 1.0}), _events({"test": 2.0}),
            include_dst=False, threshold_method="unknown_method",
        )


def test_cfg_wrapper_uses_legacy_kmeans_topk_not_aggregation_topk(monkeypatch):
    cfg = get_default_cfg(_args())
    cfg.node_aggregation.include_dst = False
    cfg.node_aggregation.topk = 5
    cfg.node_threshold.method = "kmeans"
    cfg.detection.evaluation.node_evaluation.kmeans_top_K = 20
    received = []

    def fake_legacy(results, topk_K):
        received.append(topk_K)
        return results

    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: fake_legacy)
    node_evaluation.get_mstc_node_predictions_from_cfg(
        _events({"validation": 1.0}), _events({"a": 1.0, "b": 2.0}), cfg
    )
    assert received == [20]


def test_kmeans_one_test_node_fails_without_calling_legacy(monkeypatch):
    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: pytest.fail("unexpected legacy call"))
    with pytest.raises(ValueError, match="at least two"):
        node_evaluation.get_mstc_node_predictions(
            _events({"validation": 1.0}), _events({"only": 2.0}),
            include_dst=False, threshold_method="kmeans",
        )


def test_calibration_method_validation_and_experiment_overlays():
    import yaml
    from config import _validate_calibration_method

    expected = {
        "calibration_global_p.yml": "global_empirical",
        "calibration_relation.yml": "relation_triplet",
        "calibration_hierarchical.yml": "hierarchical_relation",
    }
    overlay_dir = Path(__file__).resolve().parents[1] / "config" / "experiments"
    for filename, method in expected.items():
        assert _validate_calibration_method(method) == method
        payload = yaml.safe_load((overlay_dir / filename).read_text(encoding="utf-8"))
        assert payload["calibration"]["method"] == method
    with pytest.raises(ValueError, match="calibration.method"):
        _validate_calibration_method("unsupported")
