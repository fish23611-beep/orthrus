from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluation_runner = load_file("c6_b7_evaluation_runner", SRC_ROOT / "mstc" / "evaluation_runner.py")
node_evaluation = load_file("c6_b7_node_evaluation", SRC_ROOT / "mstc" / "node_evaluation.py")


def make_cfg(tmp_path, *, threshold="validation_quantile"):
    return SimpleNamespace(
        calibration=SimpleNamespace(method="hierarchical_relation", min_triplet_samples=7, min_type_pair_samples=9, epsilon=1e-7),
        node_aggregation=SimpleNamespace(method="mean", topk=3, include_dst=False, score_field="score_calibrated"),
        node_threshold=SimpleNamespace(method=threshold, quantile=0.75),
        detection=SimpleNamespace(evaluation=SimpleNamespace(
            _evaluation_results_dir=str(tmp_path / "out"),
            used_method="node_evaluation",
            node_evaluation=SimpleNamespace(kmeans_top_K=20),
        )),
        model=SimpleNamespace(variant="mstc"),
    )


class FakeCalibration:
    def __init__(self, calls=None, *, validation_raw=None, test_raw=None, validation_calibrated=None, test_calibrated=None):
        self.calls = [] if calls is None else calls
        self.validation_raw = [{"score_raw": 0.2}] if validation_raw is None else validation_raw
        self.test_raw = [{"score_raw": 8.0}] if test_raw is None else test_raw
        self.validation_calibrated = [{"srcnode": "v", "dstnode": "x", "score_calibrated": 0.2, "calibrated": True}] if validation_calibrated is None else validation_calibrated
        self.test_calibrated = [{"srcnode": "t", "dstnode": "x", "score_calibrated": 0.7, "calibrated": True}] if test_calibrated is None else test_calibrated

    def load_event_records_from_csv_directory(self, path):
        return self.validation_raw if "val" in Path(path).parts else self.test_raw

    def run_calibration(self, validation, test, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append(("calibration", kwargs, output_dir))

    def load_event_records_from_csv(self, path):
        return self.validation_calibrated if Path(path).name.startswith("validation") else self.test_calibrated


def epoch_paths(tmp_path, epoch="model_epoch_1", raw=b"raw artifacts unchanged"):
    val = tmp_path / "raw" / "val" / epoch
    test = tmp_path / "raw" / "test" / epoch
    val.mkdir(parents=True)
    test.mkdir(parents=True)
    (val / "events.csv").write_bytes(raw)
    (test / "events.csv").write_bytes(raw)
    return val, test, raw


def test_runner_orders_calibration_prediction_ground_truth_metrics_and_passes_cfg(tmp_path):
    calls = []
    val, test, raw = epoch_paths(tmp_path)
    calibration = FakeCalibration(calls)

    def node_prediction(validation, test_records, cfg):
        calls.append("node_prediction")
        assert validation[0]["calibrated"] and test_records[0]["calibrated"]
        assert cfg.node_aggregation.method == "mean"
        assert cfg.node_aggregation.include_dst is False
        return {"validation_node_scores": {"v": 0.2}, "test_node_scores": {"t": 0.7}, "threshold": 0.2,
                "threshold_method": "validation_quantile", "test_node_predictions": {"t": 1}}

    def ground_truth(cfg):
        calls.append("ground_truth")
        return {"t"}, {}

    def metrics(y_true, y_hat, scores):
        calls.append("metrics")
        assert (y_true, y_hat, scores) == ([1], [1], [0.7])
        return {"ok": True}

    cfg = make_cfg(tmp_path)
    result = evaluation_runner.mstc_evaluation_main(
        val, test, "model_epoch_1", cfg, calibration_module=calibration,
        node_prediction_fn=node_prediction, ground_truth_fn=ground_truth,
        classifier_evaluation_fn=metrics,
    )
    assert [calls[0][0], *calls[1:]] == ["calibration", "node_prediction", "ground_truth", "metrics"]
    assert calls[0][1] == {"min_triplet_samples": 7, "min_type_pair_samples": 9, "epsilon": 1e-7, "method": "hierarchical_relation"}
    assert result["prediction_result"]["test_node_predictions"] == {"t": 1}
    assert (result["output_dir"] / "node_predictions.csv").exists()
    assert (val / "events.csv").read_bytes() == raw
    assert (test / "events.csv").read_bytes() == raw


def test_ground_truth_changes_only_labels_and_metrics(tmp_path):
    val, test, _ = epoch_paths(tmp_path)
    cfg = make_cfg(tmp_path)
    calibration = FakeCalibration()

    def predict(validation, test_records, cfg):
        return node_evaluation.get_mstc_node_predictions_from_cfg(validation, test_records, cfg)

    outputs = []
    for malicious in (set(), {"t"}):
        outputs.append(evaluation_runner.mstc_evaluation_main(
            val, test, "model_epoch_1", cfg, calibration_module=calibration,
            node_prediction_fn=predict, ground_truth_fn=lambda cfg, m=malicious: (m, {}),
            classifier_evaluation_fn=lambda y, p, s: {"truth": tuple(y)},
        ))
    assert outputs[0]["prediction_result"] == outputs[1]["prediction_result"]
    assert outputs[0]["rows"][0]["y_true"] == 0
    assert outputs[1]["rows"][0]["y_true"] == 1


def test_test_scores_do_not_change_validation_quantile_threshold(tmp_path):
    cfg = make_cfg(tmp_path)
    validation = [{"srcnode": "v1", "dstnode": "x", "score_calibrated": 1.0}, {"srcnode": "v2", "dstnode": "x", "score_calibrated": 3.0}]
    ordinary = [{"srcnode": "t", "dstnode": "x", "score_calibrated": 2.0}]
    extreme = [{"srcnode": "t", "dstnode": "x", "score_calibrated": 1e12}]
    first = node_evaluation.get_mstc_node_predictions_from_cfg(validation, ordinary, cfg)
    second = node_evaluation.get_mstc_node_predictions_from_cfg(validation, extreme, cfg)
    assert first["threshold"] == second["threshold"]
    assert first["test_node_predictions"] != second["test_node_predictions"]


def test_kmeans_uses_b6_legacy_adapter(monkeypatch, tmp_path):
    cfg = make_cfg(tmp_path, threshold="kmeans")
    calls = []

    def legacy(results, topk):
        calls.append(topk)
        results["high"]["y_hat"] = 1
        return results

    monkeypatch.setattr(node_evaluation, "get_legacy_compute_kmeans_labels", lambda: legacy)
    validation = [{"srcnode": "v", "dstnode": "x", "score_calibrated": 1.0}]
    test = [{"srcnode": "low", "dstnode": "x", "score_calibrated": 2.0}, {"srcnode": "high", "dstnode": "x", "score_calibrated": 9.0}]
    result = node_evaluation.get_mstc_node_predictions_from_cfg(validation, test, cfg)
    assert calls == [20]
    assert result["threshold"] is None
    assert result["threshold_method"] == "kmeans"
    assert result["test_node_predictions"] == {"low": 0, "high": 1}


def test_empty_validation_fails_before_prediction(tmp_path):
    val, test, _ = epoch_paths(tmp_path)
    called = []
    with pytest.raises(ValueError, match="validation"):
        evaluation_runner.run_mstc_epoch(
            val, test, "model_epoch_1", make_cfg(tmp_path),
            calibration_module=FakeCalibration(validation_raw=[]),
            node_prediction_fn=lambda *args: called.append(True),
        )
    assert called == []


def test_empty_test_returns_no_nodes_and_skips_metrics(tmp_path):
    val, test, _ = epoch_paths(tmp_path)
    calibration = FakeCalibration(test_raw=[], test_calibrated=[])
    result = evaluation_runner.mstc_evaluation_main(
        val, test, "model_epoch_1", make_cfg(tmp_path), calibration_module=calibration,
        node_prediction_fn=node_evaluation.get_mstc_node_predictions_from_cfg,
        ground_truth_fn=lambda cfg: (set(), {}),
        classifier_evaluation_fn=lambda *args: pytest.fail("metrics must not run for empty predictions"),
    )
    assert result["prediction_result"]["test_node_scores"] == {}
    assert result["prediction_result"]["test_node_predictions"] == {}
    assert result["rows"] == []


def test_runner_rejects_mismatched_epochs(tmp_path):
    with pytest.raises(ValueError, match="match model_epoch_dir"):
        evaluation_runner.run_mstc_epoch(
            tmp_path / "val" / "model_epoch_1", tmp_path / "test" / "model_epoch_2", "model_epoch_1",
            make_cfg(tmp_path), calibration_module=FakeCalibration(), node_prediction_fn=lambda *args: {},
        )


def test_epoch_outputs_are_isolated(tmp_path):
    cfg = make_cfg(tmp_path)
    outputs = []
    for epoch in ("model_epoch_1", "model_epoch_2"):
        val, test, _ = epoch_paths(tmp_path, epoch)
        _, output, _ = evaluation_runner.run_mstc_epoch(
            val, test, epoch, cfg, calibration_module=FakeCalibration(),
            node_prediction_fn=lambda *args: {"test_node_scores": {}, "test_node_predictions": {}},
        )
        outputs.append(output)
    assert outputs == [tmp_path / "out" / "calibration" / "model_epoch_1", tmp_path / "out" / "calibration" / "model_epoch_2"]
    assert outputs[0].is_dir() and outputs[1].is_dir()


def load_detection_evaluation_with_stubs(monkeypatch):
    detection = types.ModuleType("detection")
    detection.__path__ = []
    legacy = types.ModuleType("detection.node_evaluation")
    legacy.main = lambda *args, **kwargs: {"legacy": True}
    utils = types.ModuleType("detection.evaluation_utils")
    utils.__all__ = ["compute_tw_labels", "get_ground_truth_nids", "classifier_evaluation"]
    utils.compute_tw_labels = lambda cfg: {}
    utils.get_ground_truth_nids = lambda cfg: (set(), {})
    utils.classifier_evaluation = lambda *args: {}
    data_utils = types.ModuleType("data_utils")
    data_utils.__all__ = []
    provnet = types.ModuleType("provnet_utils")
    provnet.log = lambda *args: None
    wandb = types.ModuleType("wandb")
    wandb.log = lambda *args: None
    wandb.Image = lambda path: path
    for name, module in {"detection": detection, "detection.node_evaluation": legacy,
                         "detection.evaluation_utils": utils, "data_utils": data_utils,
                         "provnet_utils": provnet, "wandb": wandb}.items():
        monkeypatch.setitem(sys.modules, name, module)
    detection.node_evaluation = legacy
    return load_file("detection.c6_b7_evaluation", SRC_ROOT / "detection" / "evaluation.py")


def test_real_dispatch_baseline_mstc_and_unknown(monkeypatch, tmp_path):
    module = load_detection_evaluation_with_stubs(monkeypatch)
    selected = []
    monkeypatch.setattr(module, "standard_evaluation", lambda cfg, evaluation_fn: selected.append(evaluation_fn))
    cfg = make_cfg(tmp_path)
    cfg.model.variant = "orthrus_baseline"
    module.main(cfg)
    assert selected[-1] is module.node_evaluation.main
    cfg.model.variant = "mstc"
    module.main(cfg)
    assert selected[-1] is module.mstc_evaluation_main
    cfg.model.variant = "unknown"
    with pytest.raises(ValueError, match="Invalid model variant"):
        module.main(cfg)


def write_raw(directory, records):
    directory.mkdir(parents=True)
    fields = ["event_index", "score_raw", "src_type", "dst_type", "edge_type_index", "srcnode", "dstnode"]
    with (directory / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def test_real_b2_b6_synthetic_smoke(tmp_path):
    calibration_runner = load_file("c6_b7_calibration_runner", SRC_ROOT / "mstc" / "calibration_runner.py")
    epoch = "model_epoch_1"
    val = tmp_path / "raw" / "val" / epoch
    test = tmp_path / "raw" / "test" / epoch
    validation = [
        {"event_index": i, "score_raw": score, "src_type": 1, "dst_type": 2, "edge_type_index": 3, "srcnode": f"v{i}", "dstnode": "vx"}
        for i, score in enumerate((0.1, 0.2, 0.3, 0.4), 1)
    ]
    testing = [
        {"event_index": i, "score_raw": score, "src_type": 1, "dst_type": 2, "edge_type_index": 3, "srcnode": f"t{i}", "dstnode": "tx"}
        for i, score in enumerate((0.15, 0.9, 2.0), 11)
    ]
    write_raw(val, validation)
    write_raw(test, testing)
    val_before = (val / "events.csv").read_bytes()
    test_before = (test / "events.csv").read_bytes()
    cfg = make_cfg(tmp_path)
    cfg.calibration.min_triplet_samples = 2
    cfg.calibration.min_type_pair_samples = 2
    result = evaluation_runner.mstc_evaluation_main(
        val, test, epoch, cfg, calibration_module=calibration_runner,
        node_prediction_fn=node_evaluation.get_mstc_node_predictions_from_cfg,
        ground_truth_fn=lambda cfg: ({"t13"}, {}),
        classifier_evaluation_fn=lambda y, p, s: {"count": len(y)},
    )
    output = result["output_dir"]
    for name in ("calibrator.pkl", "calibration_summary.json", "validation_calibrated.csv", "test_calibrated.csv", "node_predictions.csv"):
        assert (output / name).exists()
    assert result["prediction_result"]["threshold"] is not None
    assert set(result["prediction_result"]["test_node_predictions"]) == {"t11", "t12", "t13"}
    assert (val / "events.csv").read_bytes() == val_before
    assert (test / "events.csv").read_bytes() == test_before


@pytest.mark.parametrize(
    "method", ["global_empirical", "relation_triplet", "hierarchical_relation"]
)
def test_evaluation_runner_passes_each_calibration_method_to_b2(tmp_path, method):
    val, test, _ = epoch_paths(tmp_path)
    calibration = FakeCalibration()
    cfg = make_cfg(tmp_path)
    cfg.calibration.method = method
    evaluation_runner.run_mstc_epoch(
        val, test, "model_epoch_1", cfg,
        calibration_module=calibration,
        node_prediction_fn=lambda *args: {"test_node_scores": {}, "test_node_predictions": {}},
    )
    assert calibration.calls[0][1]["method"] == method
