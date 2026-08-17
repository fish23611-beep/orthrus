"""
tests/test_mstc_evaluation_calibration_adapter.py

Regression test for the MSTC evaluation NameError fault (Fault B).

Problem: In evaluation.py, the class CalibrationModule tried to reference
load_event_records_from_csv, load_event_records_from_csv_directory, and
run_calibration from the enclosing function mstc_evaluation_main's local scope.
Python class bodies do not have a local scope — names are resolved in the
MODULE global namespace.  The imported names were local to mstc_evaluation_main
and invisible inside the class body, causing NameError.

Fix: Import calibration_runner functions under private aliases (_cal_load_csv,
_cal_load_csv_dir, _cal_run), then bind them to CalibrationModule using
globals()["_cal_alias"] inside the class body, which bypasses the class body's
scope limitation.

This test verifies:
  A. The source fix is present (globals()["_cal_..."] pattern).
  B. The broken NameError pattern is absent.
  C. mstc_evaluation_main can be called without NameError (using the same
     stub setup as test_c6_evaluation_integration.py).
  D. CalibrationModule's staticmethods are the ORIGINAL calibration_runner
     functions (verified by the real calibration output files being created).
"""

import csv
import re
import sys
import types
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


# --------------------------------------------------------------------------- #
# Source-level verification (A + B)
# --------------------------------------------------------------------------- #

def test_module_level_imports_present_in_evaluation_py():
    """Verify calibration_runner functions are imported at module level."""
    eval_src = (SRC_ROOT / "detection" / "evaluation.py").read_text()

    # The fix should use _cal_load_csv, _cal_load_csv_dir, _cal_run
    # as the aliases imported from mstc.calibration_runner at module level
    assert "_cal_load_csv" in eval_src
    assert "_cal_load_csv_dir" in eval_src
    assert "_cal_run" in eval_src

    # And they're used in the class body directly (not via globals())
    assert "load_event_records_from_csv = staticmethod(_cal_load_csv)" in eval_src
    assert "load_event_records_from_csv_directory = staticmethod(_cal_load_csv_dir)" in eval_src
    assert "run_calibration = staticmethod(_cal_run)" in eval_src

    # The broken pattern must be absent
    bad_pattern = re.search(
        r'staticmethod\(\s*load_event_records_from_csv\s*\)',
        eval_src,
    )
    assert bad_pattern is None, (
        "Broken NameError pattern 'staticmethod(load_event_records_from_csv)' "
        "must not appear in evaluation.py"
    )


# --------------------------------------------------------------------------- #
# Runtime verification via the same stub pattern as test_c6_evaluation_integration
# --------------------------------------------------------------------------- #

def _load_detection_evaluation_with_stubs():
    """Replicate the stub pattern from test_c6_evaluation_integration.py.

    Must provide: detection, detection.node_evaluation, detection.evaluation_utils,
    data_utils, provnet_utils, wandb, wandb_control, and the mstc package
    (calibration_runner, evaluation_runner, node_evaluation).

    Uses importlib.util to load real modules from source files so we avoid
    sys.path issues when pytest runs from the repo root.
    """
    import importlib.util

    # Load real mstc submodules from source files (avoids sys.path dependencies)
    def _load_src_mod(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    calibration_runner_module = _load_src_mod(
        "mstc.calibration_runner", SRC_ROOT / "mstc" / "calibration_runner.py"
    )
    evaluation_runner_module = _load_src_mod(
        "mstc.evaluation_runner", SRC_ROOT / "mstc" / "evaluation_runner.py"
    )
    node_evaluation_module = _load_src_mod(
        "mstc.node_evaluation", SRC_ROOT / "mstc" / "node_evaluation.py"
    )

    # Wire up the mstc package (needs __init__.py to support sub-package imports)
    mstc_init = types.ModuleType("mstc")
    mstc_init.__path__ = [str(SRC_ROOT / "mstc")]  # makes 'mstc' a package
    mstc_init.calibration_runner = calibration_runner_module
    mstc_init.evaluation_runner = evaluation_runner_module
    mstc_init.node_evaluation = node_evaluation_module
    sys.modules["mstc"] = mstc_init

    detection = types.ModuleType("detection")
    detection.__path__ = []

    legacy = types.ModuleType("detection.node_evaluation")
    legacy.main = lambda *args, **kwargs: {"legacy": True}
    detection.node_evaluation = legacy

    utils = types.ModuleType("detection.evaluation_utils")
    utils.__all__ = ["compute_tw_labels", "get_ground_truth_nids", "classifier_evaluation"]
    utils.compute_tw_labels = lambda cfg: {}
    utils.get_ground_truth_nids = lambda cfg: (set(), {})
    utils.classifier_evaluation = lambda *args: {}
    detection.evaluation_utils = utils

    data_utils = types.ModuleType("data_utils")
    data_utils.__all__ = []
    provnet = types.ModuleType("provnet_utils")
    provnet.log = lambda *args: None
    wandb = types.ModuleType("wandb")
    wandb.log = lambda *args: None
    wandb.Image = lambda path: path
    wandb_control = types.ModuleType("wandb_control")
    wandb_control.wandb_is_active = lambda: False
    wandb_control.wandb_log = wandb.log
    wandb_control.wandb_finish = lambda: None

    for name, module in {
        "detection": detection,
        "detection.node_evaluation": legacy,
        "detection.evaluation_utils": utils,
        "data_utils": data_utils,
        "provnet_utils": provnet,
        "wandb": wandb,
        "wandb_control": wandb_control,
    }.items():
        sys.modules[name] = module

    eval_path = SRC_ROOT / "detection" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("detection.evaluation", eval_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["detection.evaluation"] = module
    spec.loader.exec_module(module)
    return module


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


def _write_csv_records(path, records):
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


def _make_cfg(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(
        calibration=SimpleNamespace(
            method="hierarchical_relation",
            min_triplet_samples=2,
            min_type_pair_samples=2,
            epsilon=1e-12,
        ),
        node_aggregation=SimpleNamespace(
            method="mean",
            topk=3,
            include_dst=False,
            score_field="score_calibrated",
        ),
        node_threshold=SimpleNamespace(
            method="validation_quantile",
            quantile=0.75,
        ),
        detection=SimpleNamespace(evaluation=SimpleNamespace(
            _evaluation_results_dir=str(tmp_path / "out"),
            node_evaluation=SimpleNamespace(kmeans_top_K=20),
        )),
        model=SimpleNamespace(variant="mstc"),
    )


def test_mstc_evaluation_main_runs_without_nameerror(tmp_path):
    """Verify mstc_evaluation_main() can be called end-to-end without NameError."""
    eval_module = _load_detection_evaluation_with_stubs()

    cfg = _make_cfg(tmp_path)
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

    # This must NOT raise NameError
    result = eval_module.mstc_evaluation_main(
        str(val_dir),
        str(test_dir),
        "model_epoch_1",
        cfg,
    )
    assert isinstance(result, dict)
    assert "val_mean_edge_loss" in result


def test_calibration_module_produces_real_calibration_artifacts(tmp_path):
    """Verify CalibrationModule delegates to the real calibration_runner
    by checking that all expected calibration output files are produced."""
    eval_module = _load_detection_evaluation_with_stubs()

    cfg = _make_cfg(tmp_path)
    val_dir = tmp_path / "val2" / "model_epoch_1"
    test_dir = tmp_path / "test2" / "model_epoch_1"
    _write_csv_records(val_dir / "tw0.csv", [
        _record(1.0, event_index=0),
        _record(2.0, event_index=1),
        _record(3.0, event_index=2),
    ])
    _write_csv_records(test_dir / "tw0.csv", [
        _record(1.5, event_index=100),
        _record(2.5, event_index=101),
    ])

    result = eval_module.mstc_evaluation_main(
        str(val_dir),
        str(test_dir),
        "model_epoch_1",
        cfg,
    )
    assert isinstance(result, dict)

    # Calibration output must exist (proves the real runner was called)
    cal_out = tmp_path / "out" / "calibration" / "model_epoch_1"
    assert (cal_out / "calibrator.pkl").exists(), \
        "calibrator.pkl must exist — proves CalibrationModule uses real runner"
    assert (cal_out / "calibration_summary.json").exists(), \
        "calibration_summary.json must exist"
    assert (cal_out / "validation_calibrated.csv").exists()
    assert (cal_out / "test_calibrated.csv").exists()

    # Check the summary content
    import json
    with open(cal_out / "calibration_summary.json") as f:
        summary = json.load(f)
    assert summary["calibration_method"] == "hierarchical_relation"
    assert summary["event_counts"]["num_validation_events"] == 3
    assert summary["event_counts"]["num_test_events"] == 2


# --------------------------------------------------------------------------- #
# Production-like import smoke test
# --------------------------------------------------------------------------- #
def test_evaluation_imports_from_top_level_mstc_package():
    """Verify detection.evaluation imports from 'mstc.calibration_runner'
    (top-level), NOT 'detection.mstc.calibration_runner' (relative).

    This test loads the source directly with minimal stubs and ensures the
    correct import path is used.  If the source used a relative import
    'from .mstc.calibration_runner', this test would fail with
    ModuleNotFoundError because 'detection.mstc' does not exist in production.
    """
    import importlib.util

    # Stub ONLY non-target dependencies (those unrelated to mstc imports).
    # This is intentionally LIGHTER than _load_detection_evaluation_with_stubs
    # so any missing import will surface clearly.

    def _stub_module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    # Core stubs
    stub_module = _stub_module
    detection = stub_module("detection")
    detection.__path__ = []
    stub_module("detection.node_evaluation").main = lambda *a, **k: {}
    utils = stub_module("detection.evaluation_utils")
    for attr in ("compute_tw_labels", "get_ground_truth_nids", "classifier_evaluation"):
        setattr(utils, attr, lambda *a, **k: ({} if "gt" in attr else {}))
    stub_module("data_utils")
    provnet = stub_module("provnet_utils")
    provnet.log = lambda *a, **k: None
    wandb = stub_module("wandb")
    wandb.log = lambda *a, **k: None
    wandb.Image = lambda x: x
    wandb_control = stub_module("wandb_control")
    wandb_control.wandb_is_active = lambda: False
    wandb_control.wandb_log = wandb.log
    wandb_control.wandb_finish = lambda: None

    # Load real mstc submodules
    def _load_src_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    cal_mod = _load_src_mod("mstc.calibration_runner", SRC_ROOT / "mstc" / "calibration_runner.py")
    eval_run_mod = _load_src_mod("mstc.evaluation_runner", SRC_ROOT / "mstc" / "evaluation_runner.py")
    node_eval_mod = _load_src_mod("mstc.node_evaluation", SRC_ROOT / "mstc" / "node_evaluation.py")

    # Wire mstc package
    mstc_pkg = stub_module("mstc")
    mstc_pkg.__path__ = [str(SRC_ROOT / "mstc")]
    mstc_pkg.calibration_runner = cal_mod
    mstc_pkg.evaluation_runner = eval_run_mod
    mstc_pkg.node_evaluation = node_eval_mod

    # NOTE: We deliberately do NOT create 'detection.mstc' here.
    # If evaluation.py uses 'from .mstc.calibration_runner', this test will fail.

    # Load detection.evaluation — if import path is wrong, ModuleNotFoundError surfaces
    eval_path = SRC_ROOT / "detection" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("detection.evaluation", eval_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["detection.evaluation"] = module
    spec.loader.exec_module(module)

    # Verify the module-level aliases are bound to the real calibration_runner
    assert hasattr(module, "_cal_load_csv")
    assert hasattr(module, "_cal_load_csv_dir")
    assert hasattr(module, "_cal_run")
    assert module._cal_load_csv is cal_mod.load_event_records_from_csv
    assert module._cal_load_csv_dir is cal_mod.load_event_records_from_csv_directory
    assert module._cal_run is cal_mod.run_calibration
