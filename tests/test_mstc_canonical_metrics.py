"""
tests/test_mstc_canonical_metrics.py

Regression tests for C8 Bug A: MSTC canonical metrics schema.

These tests verify that:
1. MSTC evaluation emits the full canonical metrics schema (tp/fp/tn/fn,
   precision/recall/f1/mcc/auprc/auroc/fpr/fp_per_million/attack_detection_rate)
2. The selected-epoch metrics.json contains all canonical fields plus
   selected_epoch / model_selection_method / val_mean_edge_loss
3. collect_results correctly maps canonical fields into all_runs.csv
4. Legacy aliases (fscore/ap/auc) are preserved for backward compat

Run with: pytest tests/test_mstc_canonical_metrics.py -v
"""

from __future__ import annotations

import csv
import json
import math
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# --------------------------------------------------------------------------- #
# Canonical metrics expected schema
# --------------------------------------------------------------------------- #

CANONICAL_METRICS = frozenset({
    "tp", "fp", "tn", "fn",
    "precision", "recall", "f1", "mcc",
    "auprc", "auroc", "fpr",
    "fp_per_million", "attack_detection_rate",
})

SELECTION_FIELDS = frozenset({
    "selected_epoch", "model_selection_method", "val_mean_edge_loss",
})

ALL_REQUIRED = CANONICAL_METRICS | SELECTION_FIELDS


# --------------------------------------------------------------------------- #
# mstc.metrics unit tests (sanity check the canonical definitions)
# --------------------------------------------------------------------------- #

def test_mstc_metrics_compute_classification_metrics_returns_all_canonical_fields():
    """compute_classification_metrics must return all canonical fields."""
    from mstc.metrics import compute_classification_metrics

    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.1] * 15

    result = compute_classification_metrics(y_true, y_pred, scores)

    # Core canonical fields from compute_classification_metrics
    for field in frozenset({"tp", "fp", "tn", "fn", "precision", "recall", "f1", "mcc", "auprc", "auroc", "fpr"}):
        assert field in result, f"Missing canonical field: {field}"

    # TP=0, FP=0, TN=10, FN=5
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["tn"] == 10
    assert result["fn"] == 5

    # precision = 0/(0+0) = nan
    assert math.isnan(result["precision"])
    # recall = 0/(0+5) = 0
    assert result["recall"] == 0.0
    # f1 = nan
    assert math.isnan(result["f1"])


def test_mstc_metrics_precision_recall_f1():
    """precision/recall/f1 must be mathematically correct."""
    from mstc.metrics import compute_classification_metrics

    # 3 true positives, 2 false positives, 5 true negatives, 1 false negative
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
    scores = [0.9, 0.85, 0.8, 0.7, 0.3, 0.25, 0.2, 0.15, 0.7, 0.65, 0.1]

    result = compute_classification_metrics(y_true, y_pred, scores)

    # TP=3, FP=2, TN=5, FN=1
    assert result["tp"] == 3
    assert result["fp"] == 2
    assert result["tn"] == 5
    assert result["fn"] == 1

    precision = 3 / (3 + 2)
    recall = 3 / (3 + 1)
    f1 = 2 * precision * recall / (precision + recall)
    mcc_num = (3 * 5) - (2 * 1)
    mcc_den = math.sqrt((3 + 2) * (3 + 1) * (5 + 2) * (5 + 1))
    mcc = mcc_num / mcc_den
    fpr = 2 / (2 + 5)

    assert abs(result["precision"] - precision) < 1e-9
    assert abs(result["recall"] - recall) < 1e-9
    assert abs(result["f1"] - f1) < 1e-9
    assert abs(result["mcc"] - mcc) < 1e-9
    assert abs(result["fpr"] - fpr) < 1e-9


def test_mstc_metrics_auprc_auroc():
    """auprc and auroc must be computed from scores."""
    from mstc.metrics import compute_classification_metrics

    # Perfect separation: all positives score higher than negatives
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    result = compute_classification_metrics(y_true, y_pred, scores)

    assert result["auroc"] == 1.0, "Perfect separation → AUROC = 1.0"
    assert result["auprc"] == 1.0, "Perfect separation → AUPRC = 1.0"


def test_mstc_metrics_no_scores_returns_nan_for_auroc_auprc():
    """When scores=None, AUROC and AUPRC must be NaN (not computed)."""
    from mstc.metrics import compute_classification_metrics

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    result = compute_classification_metrics(y_true, y_pred, scores=None)

    assert math.isnan(result["auroc"]), "auroc must be NaN when scores=None"
    assert math.isnan(result["auprc"]), "auprc must be NaN when scores=None"


def test_mstc_metrics_fp_per_million():
    """fp_per_million must equal fp / benign_count * 1_000_000."""
    from mstc.metrics import compute_fp_per_million

    assert compute_fp_per_million(5, 100_000) == 50.0
    assert compute_fp_per_million(0, 100_000) == 0.0
    assert math.isnan(compute_fp_per_million(5, 0))


def test_mstc_metrics_attack_detection_rate():
    """attack_detection_rate = fraction of attacks with ≥1 detected node."""
    from mstc.metrics import compute_attack_detection_rate

    # 2 attacks: attack1 has detected node, attack2 does not
    attack_to_nodes = {
        "attack1": ["node_a", "node_b"],
        "attack2": ["node_c", "node_d"],
    }
    predicted_positive = ["node_a", "node_x"]  # node_a is in attack1

    rate = compute_attack_detection_rate(attack_to_nodes, predicted_positive)
    assert rate == 0.5, "1 of 2 attacks detected → rate = 0.5"

    # All attacks detected
    rate_all = compute_attack_detection_rate(attack_to_nodes, ["node_a", "node_c"])
    assert rate_all == 1.0, "2 of 2 attacks detected → rate = 1.0"

    # No attacks detected
    rate_none = compute_attack_detection_rate(attack_to_nodes, ["node_x", "node_y"])
    assert rate_none == 0.0, "0 of 2 attacks detected → rate = 0.0"


def test_mstc_metrics_attack_detection_rate_empty():
    """Empty attack_to_nodes must return NaN."""
    from mstc.metrics import compute_attack_detection_rate

    rate = compute_attack_detection_rate({}, ["node_a"])
    assert math.isnan(rate)


def test_mstc_metrics_inspected_nodes_per_attack_basic():
    """inspected_nodes_per_attack = (TP+FP) / num_attacks."""
    from mstc.metrics import compute_inspected_nodes_per_attack

    # 9 positive predictions, 3 attacks → 3.0
    result = compute_inspected_nodes_per_attack(9, 3)
    assert result == 3.0

    # 0 positive predictions, 2 attacks → 0.0
    result = compute_inspected_nodes_per_attack(0, 2)
    assert result == 0.0

    # 5 positive predictions, 1 attack → 5.0
    result = compute_inspected_nodes_per_attack(5, 1)
    assert result == 5.0


def test_mstc_metrics_inspected_nodes_per_attack_zero_attacks():
    """Zero attacks must return NaN."""
    from mstc.metrics import compute_inspected_nodes_per_attack

    assert math.isnan(compute_inspected_nodes_per_attack(0, 0))
    assert math.isnan(compute_inspected_nodes_per_attack(5, 0))
    assert math.isnan(compute_inspected_nodes_per_attack(0, 0))


def test_mstc_metrics_inspected_nodes_per_attack_fractional():
    """Non-integer division must be computed correctly."""
    from mstc.metrics import compute_inspected_nodes_per_attack

    # 10 alerts / 3 attacks = 3.333...
    result = compute_inspected_nodes_per_attack(10, 3)
    assert abs(result - 10 / 3) < 1e-9


# --------------------------------------------------------------------------- #
# Integration test: mstc_evaluation_runner emits canonical metrics
# --------------------------------------------------------------------------- #

def _make_mstc_cfg(tmp_path):
    """Minimal cfg for MSTC evaluation runner test."""
    ns = types.SimpleNamespace
    cfg = ns(
        calibration=ns(
            method="hierarchical_relation",
            min_triplet_samples=2,
            min_type_pair_samples=2,
            epsilon=1e-12,
        ),
        node_aggregation=ns(
            method="mean",
            topk=3,
            include_dst=False,
            score_field="score_calibrated",
        ),
        node_threshold=ns(
            method="validation_quantile",
            quantile=0.75,
        ),
        detection=ns(
            evaluation=ns(
                _evaluation_results_dir=str(tmp_path / "out"),
                node_evaluation=ns(kmeans_top_K=20),
            ),
        ),
        model=ns(variant="mstc"),
    )
    return cfg


def _make_gt_cfg_with_attacks():
    """cfg that returns real attack_to_nodes mapping for attack_detection_rate."""
    ns = types.SimpleNamespace

    def gt_fn(cfg):
        return {"node_1", "node_2", "node_100"}, {}

    def attack_fn(cfg):
        return {
            "Browser_Extension_Drakon_Dropper": ["node_1", "node_2"],
            "Firefox_Backdoor_Drakon": ["node_100", "node_101"],
        }

    return gt_fn, attack_fn


def _write_csv_records(path, records):
    """Write event records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _record(score, src=1, edge=2, dst=3, event_index=0, **extra):
    return {
        "score_raw": score,
        "src_type": src,
        "edge_type_index": edge,
        "dst_type": dst,
        "event_index": event_index,
        "time": 1000 + event_index,
        "srcnode": src * 10 + event_index,
        "dstnode": dst * 10 + event_index,
        **extra,
    }


def test_mstc_evaluation_runner_emits_all_canonical_metrics(tmp_path):
    """MSTC evaluation_runner must emit all canonical metrics in returned stats."""
    import importlib.util

    def _load_src_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Load real modules from source
    cal_mod = _load_src_mod("mstc.calibration_runner", SRC_ROOT / "mstc" / "calibration_runner.py")
    eval_run_mod = _load_src_mod("mstc.evaluation_runner", SRC_ROOT / "mstc" / "evaluation_runner.py")
    node_eval_mod = _load_src_mod("mstc.node_evaluation", SRC_ROOT / "mstc" / "node_evaluation.py")

    mstc_pkg = types.ModuleType("mstc")
    mstc_pkg.__path__ = [str(SRC_ROOT / "mstc")]
    mstc_pkg.calibration_runner = cal_mod
    mstc_pkg.evaluation_runner = eval_run_mod
    mstc_pkg.node_evaluation = node_eval_mod
    mstc_pkg.metrics = _load_src_mod("mstc.metrics", SRC_ROOT / "mstc" / "metrics.py")
    sys.modules["mstc"] = mstc_pkg

    class CalMod:
        load_event_records_from_csv = staticmethod(cal_mod.load_event_records_from_csv)
        load_event_records_from_csv_directory = staticmethod(cal_mod.load_event_records_from_csv_directory)
        run_calibration = staticmethod(cal_mod.run_calibration)

    def node_pred_fn(val, test, cfg):
        return {
            "test_node_scores": {
                "node_1": 0.9,
                "node_2": 0.8,
                "node_3": 0.1,
                "node_4": 0.05,
                "node_5": 0.03,
            },
            "test_node_predictions": {
                "node_1": 1,
                "node_2": 1,
                "node_3": 0,
                "node_4": 0,
                "node_5": 0,
            },
        }

    def gt_fn(cfg):
        return {"node_1", "node_2", "node_100"}, {}

    def attack_fn(cfg):
        return {
            "attack1": ["node_1", "node_2"],
            "attack2": ["node_100", "node_101"],
        }

    def legacy_eval(y_true, y_pred, scores):
        return {"fscore": 0.5, "ap": 0.4, "auc": 0.6}

    cfg = _make_mstc_cfg(tmp_path)
    val_dir = tmp_path / "val1" / "model_epoch_1"
    test_dir = tmp_path / "test1" / "model_epoch_1"
    _write_csv_records(val_dir / "tw0.csv", [
        _record(1.0, event_index=0),
        _record(2.0, event_index=1),
    ])
    _write_csv_records(test_dir / "tw0.csv", [
        _record(1.5, event_index=100),
        _record(2.5, event_index=101),
    ])

    result = eval_run_mod.mstc_evaluation_main(
        str(val_dir),
        str(test_dir),
        "model_epoch_1",
        cfg,
        calibration_module=CalMod,
        node_prediction_fn=node_pred_fn,
        ground_truth_fn=gt_fn,
        classifier_evaluation_fn=legacy_eval,
        attack_to_nodes_fn=attack_fn,
    )

    stats = result["stats"]

    # All canonical metrics must be present
    for field in CANONICAL_METRICS:
        assert field in stats, f"Missing canonical field in MSTC stats: {field}"

    # Legacy aliases must also be preserved
    assert "fscore" in stats
    assert "ap" in stats
    assert "auc" in stats

    # Check values: 2 TP (node_1, node_2), 0 FP, 3 TN, 0 FN
    assert stats["tp"] == 2, f"Expected tp=2, got {stats['tp']}"
    assert stats["fp"] == 0, f"Expected fp=0, got {stats['fp']}"
    assert stats["tn"] == 3, f"Expected tn=3, got {stats['tn']}"
    assert stats["fn"] == 0, f"Expected fn=0, got {stats['fn']}"

    # precision=2/2=1, recall=2/2=1, f1=1
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0
    assert stats["f1"] == 1.0

    # attack_detection_rate: 1 of 2 attacks detected
    assert stats["attack_detection_rate"] == 0.5

    # fp_per_million = 0/3 * 1e6 = 0
    assert stats["fp_per_million"] == 0.0


def test_mstc_evaluation_without_attack_fn_has_nan_attack_detection_rate(tmp_path):
    """When attack_to_nodes_fn=None, attack_detection_rate must be NaN."""
    import importlib.util

    def _load_src_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    cal_mod = _load_src_mod("mstc.calibration_runner", SRC_ROOT / "mstc" / "calibration_runner.py")
    eval_run_mod = _load_src_mod("mstc.evaluation_runner", SRC_ROOT / "mstc" / "evaluation_runner.py")
    node_eval_mod = _load_src_mod("mstc.node_evaluation", SRC_ROOT / "mstc" / "node_evaluation.py")

    mstc_pkg = types.ModuleType("mstc")
    mstc_pkg.__path__ = [str(SRC_ROOT / "mstc")]
    mstc_pkg.calibration_runner = cal_mod
    mstc_pkg.evaluation_runner = eval_run_mod
    mstc_pkg.node_evaluation = node_eval_mod
    mstc_pkg.metrics = _load_src_mod("mstc.metrics", SRC_ROOT / "mstc" / "metrics.py")
    sys.modules["mstc"] = mstc_pkg

    class CalMod:
        load_event_records_from_csv = staticmethod(cal_mod.load_event_records_from_csv)
        load_event_records_from_csv_directory = staticmethod(cal_mod.load_event_records_from_csv_directory)
        run_calibration = staticmethod(cal_mod.run_calibration)

    def node_pred_fn(val, test, cfg):
        return {"test_node_scores": {"n1": 0.9}, "test_node_predictions": {"n1": 1}}

    def gt_fn(cfg):
        return {"n1"}, {}

    def legacy_eval(y_true, y_pred, scores):
        return {}

    cfg = _make_mstc_cfg(tmp_path)
    val_dir = tmp_path / "val2" / "ep1"
    test_dir = tmp_path / "test2" / "ep1"
    _write_csv_records(val_dir / "tw0.csv", [_record(1.0, event_index=0)])
    _write_csv_records(test_dir / "tw0.csv", [_record(0.9, event_index=100)])

    result = eval_run_mod.mstc_evaluation_main(
        str(val_dir), str(test_dir), "ep1", cfg,
        calibration_module=CalMod,
        node_prediction_fn=node_pred_fn,
        ground_truth_fn=gt_fn,
        classifier_evaluation_fn=legacy_eval,
        attack_to_nodes_fn=None,
    )

    assert math.isnan(result["stats"]["attack_detection_rate"])


# --------------------------------------------------------------------------- #
# collect_results: canonical fields mapped correctly to all_runs.csv
# --------------------------------------------------------------------------- #

def test_collect_results_maps_all_canonical_metrics_to_csv(tmp_path):
    """collect_results must map all canonical metrics into all_runs.csv."""
    from experiments import collect_results, run_matrix

    cfg_file = tmp_path / "mstc_full.yml"
    cfg_file.write_text(
        "model:\n  variant: mstc\ndataset_view:\n  mode: host_only\n"
        "detection:\n  gnn_training:\n    encoder:\n      backbone: graphsage\n",
        encoding="utf-8",
    )

    scoped = run_matrix.run_artifact_root(tmp_path, cfg_file)
    dataset = "THEIA_E3"
    seed = 0
    marker = run_matrix.run_status_path(tmp_path, dataset, cfg_file, seed)
    model_variant = "mstc"
    run_dir = scoped / dataset / "runs" / model_variant / f"seed_{seed}"
    run_dir.mkdir(parents=True)

    # Write canonical metrics (from mstc.metrics.compute_classification_metrics output)
    canonical_metrics = {
        "selected_epoch": 6,
        "model_selection_method": "min_val_mean_edge_loss",
        "val_mean_edge_loss": 0.9365,
        # From compute_classification_metrics
        "tp": 5, "fp": 1161, "tn": 698016, "fn": 113,
        "precision": 0.00429, "recall": 0.04237, "f1": 0.00779, "mcc": 0.01296,
        "auprc": 0.000994, "auroc": 0.3121, "fpr": 0.00166,
        # Derived
        "fp_per_million": 1663.0,
        "attack_detection_rate": 0.5,
    }
    (run_dir / "node_scores").mkdir()
    (run_dir / "node_scores" / "metrics.json").write_text(
        json.dumps(canonical_metrics), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"git_commit": "abc", "model": "mstc"}), encoding="utf-8"
    )
    (run_dir / "config_resolved.yml").write_text(
        "model: {variant: mstc}\ndataset_view: {mode: host_only}\n"
        "detection: {gnn_training: {encoder: {backbone: graphsage}}}\n",
        encoding="utf-8",
    )
    (run_dir / "runtime.json").write_text(
        json.dumps({
            "training": {"train_seconds_per_epoch": [1.0]},
            "testing": {"test_seconds": 2.0},
            "model": {"parameter_count": 1000},
        }), encoding="utf-8"
    )

    run_matrix._atomic_json(marker, run_matrix._status_payload(
        dataset, cfg_file, seed, "completed", scoped_root=scoped,
    ))

    rows = collect_results.collect(tmp_path)
    assert len(rows) == 1
    row = rows[0]

    # Verify canonical field mapping (METRIC_MAP normalizes lowercase keys)
    assert float(row["TP"]) == 5, f"Expected TP=5, got {row['TP']}"
    assert float(row["FP"]) == 1161, f"Expected FP=1161, got {row['FP']}"
    assert float(row["TN"]) == 698016, f"Expected TN=698016, got {row['TN']}"
    assert float(row["FN"]) == 113, f"Expected FN=113, got {row['FN']}"
    assert abs(float(row["Precision"]) - 0.00429) < 0.0001, f"Expected Precision~0.00429, got {row['Precision']}"
    assert abs(float(row["Recall"]) - 0.04237) < 0.001, f"Expected Recall~0.04237, got {row['Recall']}"
    assert abs(float(row["F1"]) - 0.00779) < 0.001, f"Expected F1~0.00779, got {row['F1']}"
    assert abs(float(row["MCC"]) - 0.01296) < 0.001, f"Expected MCC~0.01296, got {row['MCC']}"
    assert abs(float(row["AUPRC"]) - 0.000994) < 0.0001, f"Expected AUPRC~0.000994, got {row['AUPRC']}"
    assert abs(float(row["AUROC"]) - 0.3121) < 0.01, f"Expected AUROC~0.3121, got {row['AUROC']}"
    assert abs(float(row["FPR"]) - 0.00166) < 0.0001, f"Expected FPR~0.00166, got {row['FPR']}"
    assert float(row["FP_per_million"]) == 1663.0, f"Expected FP_per_million=1663.0, got {row['FP_per_million']}"
    assert float(row["Attack_Detection_Rate"]) == 0.5, f"Expected ADR=0.5, got {row['Attack_Detection_Rate']}"


def test_collect_results_fp_per_million_normalization():
    """METRIC_MAP must normalize fp_per_million correctly."""
    from experiments.collect_results import METRIC_MAP, norm

    # Verify norm() strips spaces and lowercases
    assert METRIC_MAP.get(norm("fp_per_million")) == "FP_per_million"
    assert METRIC_MAP.get(norm("fppermillion")) == "FP_per_million"
    assert METRIC_MAP.get(norm("f1")) == "F1"
    assert METRIC_MAP.get(norm("mcc")) == "MCC"
