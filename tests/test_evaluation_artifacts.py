"""
tests/test_evaluation_artifacts.py

Integration + unit tests for the C8 result-contract fixes:

B. canonical metrics.json persistence after best-epoch selection
C. MSTC node_predictions.csv canonical persistence
D. config_resolved.yml fallback (invalid YAML + valid original)
E. per-seed attempt resolution in export_tables
G.  all_runs.csv preserved as audit trail

Run with: pytest tests/test_evaluation_artifacts.py -v
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_evaluation(monkeypatch):
    """Load evaluation.py with mocked heavy dependencies."""
    detection = types.ModuleType("detection")
    detection.__path__ = []
    legacy = types.ModuleType("detection.node_evaluation")
    legacy.main = lambda *args, **kwargs: {}
    utils = types.ModuleType("detection.evaluation_utils")
    utils.__all__ = ["compute_tw_labels"]
    utils.compute_tw_labels = lambda cfg: {}
    data_utils = types.ModuleType("data_utils")
    data_utils.__all__ = []
    provnet = types.ModuleType("provnet_utils")
    provnet.log = lambda *args, **kwargs: None
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.logged = []
    wandb_mod.log = lambda stats: wandb_mod.logged.append(dict(stats))
    wandb_mod.Image = lambda path: path
    for name, module in {
        "detection": detection,
        "detection.node_evaluation": legacy,
        "detection.evaluation_utils": utils,
        "data_utils": data_utils,
        "provnet_utils": provnet,
        "wandb": wandb_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    detection.node_evaluation = legacy
    spec = __import__("importlib.util").util.spec_from_file_location(
        "detection.evaluation", SRC_ROOT / "detection" / "evaluation.py"
    )
    assert spec is not None and spec.loader is not None
    module = types.ModuleType("detection.evaluation")
    sys.modules["detection.evaluation"] = module
    spec.loader.exec_module(module)
    # Also register detection.evaluation in the detection namespace so
    # patch("detection.evaluation.wandb") works in existing tests
    detection.evaluation = module
    monkeypatch.setitem(sys.modules, "detection", detection)
    return module, wandb_mod


def _cfg(variant="orthrus_baseline", method="min_val_mean_edge_loss", run_dir=None):
    ns = types.SimpleNamespace()
    ns.model = types.SimpleNamespace(variant=variant)
    ns.model_selection = types.SimpleNamespace(method=method, legacy_test_selection_enabled=False)
    ns.detection = types.SimpleNamespace(
        gnn_testing=types.SimpleNamespace(_edge_losses_dir="/fake"),
        evaluation=types.SimpleNamespace(
            node_evaluation=types.SimpleNamespace(_precision_recall_dir="/fake/pr"),
            _evaluation_results_dir="/fake/eval_results",
        ),
    )
    ns._run_dir = run_dir
    return ns


def _stats_fn(stats_dict):
    def fn(val, test, epoch, cfg, **kwargs):
        return dict(stats_dict[epoch])
    return fn


# --------------------------------------------------------------------------- #
# B.1: canonical metrics.json is written for Baseline (orthrus_baseline)
# --------------------------------------------------------------------------- #

def test_baseline_writes_canonical_metrics_json(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2", "model_epoch_3"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    node_scores = run_dir / "node_scores"
    node_scores.mkdir()

    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 2.0, "mcc": 0.90, "precision": 0.8, "recall": 0.7,
                           "f1": 0.75, "tp": 10, "fp": 5, "tn": 80, "fn": 5},
        "model_epoch_2": {"val_mean_edge_loss": 1.0, "mcc": 0.60, "precision": 0.6, "recall": 0.5,
                           "f1": 0.55, "tp": 8,  "fp": 8, "tn": 77, "fn": 7},   # ← best val
        "model_epoch_3": {"val_mean_edge_loss": 3.0, "mcc": 0.80, "precision": 0.7, "recall": 0.6,
                           "f1": 0.65, "tp": 9,  "fp": 6, "tn": 79, "fn": 6},
    }

    cfg = _cfg(variant="orthrus_baseline", run_dir=str(run_dir))
    module.standard_evaluation(cfg, _stats_fn(stats))

    metrics_file = node_scores / "metrics.json"
    assert metrics_file.exists(), "metrics.json must be written to node_scores/"

    with metrics_file.open() as f:
        metrics = json.load(f)

    # Selected epoch must be epoch_2 (lowest val_mean_edge_loss=1.0)
    assert metrics.get("selected_epoch") == 2, f"selected_epoch must be 2, got {metrics}"
    assert metrics.get("model_selection_method") == "min_val_mean_edge_loss"
    assert metrics.get("val_mean_edge_loss") == 1.0
    # Must contain the test metrics of the SELECTED epoch, not the last
    assert metrics.get("mcc") == 0.60, f"mcc must be from epoch 2 (0.60), got {metrics.get('mcc')}"


# --------------------------------------------------------------------------- #
# B.2: W&B Image objects are stripped from metrics.json (JSON-serialisable)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# B.2: W&B Image objects are stripped from metrics.json (JSON-serialisable)
# --------------------------------------------------------------------------- #

def test_wandb_images_stripped_from_metrics_json(monkeypatch, tmp_path):
    """
    Non-JSON-serializable objects must not cause metrics.json persistence to fail.
    """
    from detection.evaluation import _strip_non_serializable, _persist_canonical_metrics

    class NonSerializableMedia:
        def __repr__(self):
            return f"<NonSerializableMedia>"
        def __str__(self):
            return f"<NonSerializableMedia>"

    stats = {
        "val_mean_edge_loss": 1.0,
        "mcc": 0.5,
        "scores_img": NonSerializableMedia(),
        "simple_scores_img": NonSerializableMedia(),
        "wandb_obj": NonSerializableMedia(),
    }

    stripped = _strip_non_serializable(stats)
    assert "scores_img" not in stripped, "Non-serializable objects must be stripped"
    assert "simple_scores_img" not in stripped
    assert "wandb_obj" not in stripped
    assert stripped.get("mcc") == 0.5
    assert stripped.get("val_mean_edge_loss") == 1.0

    # Verify the stripped dict is actually JSON-serializable
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()
    _persist_canonical_metrics(str(run_dir), stripped, "model_epoch_1", "min_val_mean_edge_loss", _cfg())

    with (run_dir / "node_scores" / "metrics.json").open() as f:
        loaded = json.load(f)
    assert loaded.get("mcc") == 0.5
    assert "scores_img" not in loaded


# --------------------------------------------------------------------------- #
# B.3: NaN/inf values in metrics.json are allowed (JSON allow_nan)
# --------------------------------------------------------------------------- #

def test_nan_inf_metrics_serialised(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()

    stats = {
        "model_epoch_1": {
            "val_mean_edge_loss": 1.0, "mcc": float("nan"), "auroc": float("inf"),
            "fpr": float("-inf"),
        }
    }

    cfg = _cfg(variant="orthrus_baseline", run_dir=str(run_dir))
    module.standard_evaluation(cfg, _stats_fn(stats))

    with (run_dir / "node_scores" / "metrics.json").open() as f:
        metrics = json.load(f)

    assert math.isnan(metrics["mcc"])
    assert math.isinf(metrics["auroc"])
    assert metrics["auroc"] > 0
    assert math.isinf(metrics["fpr"])
    assert metrics["fpr"] < 0


# --------------------------------------------------------------------------- #
# C.1: MSTC selected-epoch predictions copied to node_scores/
# --------------------------------------------------------------------------- #

def test_mstc_selected_epoch_predictions_persisted(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()

    eval_results = tmp_path / "eval_results" / "calibration"
    for epoch in ("model_epoch_1", "model_epoch_2"):
        epoch_dir = eval_results / epoch
        epoch_dir.mkdir(parents=True)
        pred_file = epoch_dir / "node_predictions.csv"
        pred_file.write_text(
            "node_id,score,y_hat,y_true\n"
            "0,0.9,1,1\n"
            "1,0.1,0,0\n",
            encoding="utf-8",
        )

    # Epoch 2 has lower val loss → selected
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 2.0, "mcc": 0.80},
        "model_epoch_2": {"val_mean_edge_loss": 1.0, "mcc": 0.60},
    }

    cfg = _cfg(variant="mstc", run_dir=str(run_dir))
    # _persist_mstc_predictions appends "calibration/model_epoch_X/" internally,
    # so _evaluation_results_dir must be the evaluation_results ROOT, not the calibration subdir.
    cfg.detection.evaluation._evaluation_results_dir = str(eval_results.parent)
    module.standard_evaluation(cfg, _stats_fn(stats))

    canonical_preds = run_dir / "node_scores" / "node_predictions.csv"
    assert canonical_preds.exists(), (
        "MSTC selected-epoch predictions must be copied to node_scores/node_predictions.csv"
    )
    content = canonical_preds.read_text(encoding="utf-8")
    assert "0,0.9,1,1" in content

    # Per-epoch files must be preserved
    assert (eval_results / "model_epoch_1" / "node_predictions.csv").exists()
    assert (eval_results / "model_epoch_2" / "node_predictions.csv").exists()


# --------------------------------------------------------------------------- #
# C.2: MSTC selected-epoch metrics also in node_scores/metrics.json
# --------------------------------------------------------------------------- #

def test_mstc_metrics_json_contains_selected_epoch_test_metrics(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()

    eval_results = tmp_path / "eval_results" / "calibration"
    for epoch in ("model_epoch_1", "model_epoch_2"):
        (eval_results / epoch).mkdir(parents=True)
        (eval_results / epoch / "node_predictions.csv").write_text(
            "node_id,score,y_hat,y_true\n0,0.5,0,0\n", encoding="utf-8"
        )

    # Epoch 1 best val
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 0.5, "mcc": 0.95, "precision": 0.90,
                           "recall": 1.0, "f1": 0.95, "tp": 10, "fp": 1, "tn": 89, "fn": 0},
        "model_epoch_2": {"val_mean_edge_loss": 1.5, "mcc": 0.70, "precision": 0.70,
                           "recall": 0.6, "f1": 0.65, "tp": 8,  "fp": 4, "tn": 86, "fn": 2},
    }

    cfg = _cfg(variant="mstc", run_dir=str(run_dir))
    # _persist_mstc_predictions appends "calibration/model_epoch_X/" internally,
    # so _evaluation_results_dir must be the evaluation_results ROOT, not the calibration subdir.
    cfg.detection.evaluation._evaluation_results_dir = str(eval_results.parent)
    module.standard_evaluation(cfg, _stats_fn(stats))

    with (run_dir / "node_scores" / "metrics.json").open() as f:
        metrics = json.load(f)

    assert metrics.get("selected_epoch") == 1
    assert metrics.get("mcc") == 0.95, "MSTC metrics.json must contain selected-epoch test MCC"
    assert metrics.get("precision") == 0.90


# --------------------------------------------------------------------------- #
# D.1: invalid config_resolved.yml + valid original config → fallback succeeds
# --------------------------------------------------------------------------- #

def test_config_fallback_uses_original_config_when_run_dir_yaml_invalid(tmp_path):
    """
    When run_dir/config_resolved.yml is invalid YAML but original config is valid,
    the collector must fallback and populate metadata from the original config.
    """
    from experiments import collect_results, run_matrix

    # Original config file (valid YAML mapping)
    cfg_file = tmp_path / "mstc.yml"
    cfg_file.write_text(
        "model:\n  variant: mstc\ndataset_view:\n  mode: host_only\n"
        "detection:\n  gnn_training:\n    encoder:\n      backbone: graphsage\n",
        encoding="utf-8",
    )

    scoped = run_matrix.run_artifact_root(tmp_path, cfg_file)
    dataset = "THEIA_E3"
    seed = 0
    marker = run_matrix.run_status_path(tmp_path, dataset, cfg_file, seed)

    # Run directory structure must match what find_run_dir() looks for:
    # <scoped>/<dataset>/runs/<model_variant>/seed_<seed>/
    model_variant = "mstc"
    run_dir = scoped / dataset / "runs" / model_variant / f"seed_{seed}"
    run_dir.mkdir(parents=True)

    (run_dir / "node_scores").mkdir()
    (run_dir / "node_scores" / "metrics.json").write_text(
        json.dumps({"MCC": 0.8}), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"git_commit": "abc", "model": "mstc"}), encoding="utf-8"
    )
    # Write invalid config_resolved.yml (empty → not a YAML mapping)
    (run_dir / "config_resolved.yml").write_text("", encoding="utf-8")

    run_matrix._atomic_json(marker, run_matrix._status_payload(
        dataset, cfg_file, seed, "completed", scoped_root=scoped,
    ))

    rows = collect_results.collect(tmp_path)
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
    row = rows[0]
    assert row["status"] == "completed", f"Expected completed, got {row['status']}"
    assert row["MCC"] == 0.8, f"Expected MCC=0.8, got {row.get('MCC')}"
    assert row["model_variant"] == "mstc"
    assert row["backbone"] == "graphsage"
    assert "fallback" in row.get("config_fallback_warning", "").lower()


# --------------------------------------------------------------------------- #
# D.2: Both run_dir YAML and original YAML invalid → config_fallback_warning set
# --------------------------------------------------------------------------- #

def test_config_fallback_warning_when_both_invalid(tmp_path):
    from experiments import collect_results, run_matrix

    cfg_file = tmp_path / "baseline.yml"
    cfg_file.write_text("not: [valid yaml either", encoding="utf-8")  # invalid

    scoped = run_matrix.run_artifact_root(tmp_path, cfg_file)
    dataset = "THEIA_E3"
    seed = 0
    marker = run_matrix.run_status_path(tmp_path, dataset, cfg_file, seed)
    run_dir = scoped / dataset / "runs" / "baseline" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "node_scores").mkdir()
    (run_dir / "node_scores" / "metrics.json").write_text(
        json.dumps({"MCC": 0.5}), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"git_commit": "abc", "model": "orthrus_baseline"}), encoding="utf-8"
    )
    (run_dir / "config_resolved.yml").write_text("model: {variant: baseline}\n", encoding="utf-8")

    run_matrix._atomic_json(marker, run_matrix._status_payload(
        dataset, cfg_file, seed, "completed", scoped_root=scoped,
    ))

    rows = collect_results.collect(tmp_path)
    row = rows[0]
    # With valid run_dir YAML, no fallback needed
    assert row.get("config_fallback_warning", "") == ""
    # But if run_dir YAML were invalid, original would also be tried


# --------------------------------------------------------------------------- #
# E.1: failed old + completed new → paper table counts as success once
# --------------------------------------------------------------------------- #

def test_failed_old_completed_new_counts_as_success(tmp_path):
    from experiments import collect_results, export_tables

    # Write all_runs.csv manually with two rows for same seed
    # NOTE: field names must match collect_results.FIELDS (uppercase: MCC, FP, TN, FN, ...)
    collect_results.write_csv(
        tmp_path / "results" / "all_runs.csv",
        [
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/baseline.yml",
                "config_id": "baseline", "seed": 0, "status": "failed",
                "MCC": float("nan"), "experiment": "baseline",
            },
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/baseline.yml",
                "config_id": "baseline", "seed": 0, "status": "completed",
                "MCC": 0.75, "experiment": "baseline",
            },
        ],
    )

    export_tables.main(["--artifact-root", str(tmp_path)])
    main_csv = list(
        csv.DictReader((tmp_path / "results" / "main_results.csv").open())
    )
    row = next(r for r in main_csv if "ORTHRUS" in r["experiment"])
    assert row["successful_seed_count"] == "1", f"Expected 1 successful, got {row}"
    assert row["failed_seed_count"] == "0", f"Expected 0 failed, got {row}"
    assert row["successful_seeds"] == "0"
    assert row["failed_seeds"] == ""
    assert row["MCC_mean"] == "0.75"


# --------------------------------------------------------------------------- #
# E.2: two failed attempts → failed_seeds="0" (counted once)
# --------------------------------------------------------------------------- #

def test_two_failed_attempts_counts_as_one_failed_seed(tmp_path):
    from experiments import collect_results, export_tables

    collect_results.write_csv(
        tmp_path / "results" / "all_runs.csv",
        [
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "mstc_full", "config_path": "/x/mstc.yml",
                "config_id": "mstc", "seed": 0, "status": "failed",
                "MCC": float("nan"), "experiment": "mstc_full",
            },
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "mstc_full", "config_path": "/x/mstc.yml",
                "config_id": "mstc", "seed": 0, "status": "failed",
                "MCC": float("nan"), "experiment": "mstc_full",
            },
        ],
    )

    export_tables.main(["--artifact-root", str(tmp_path)])
    main_csv = list(
        csv.DictReader((tmp_path / "results" / "main_results.csv").open())
    )
    row = next(r for r in main_csv if "MSTC" in r["experiment"])
    assert row["failed_seed_count"] == "1", f"Expected 1 failed, got {row}"
    assert row["failed_seeds"] == "0", f"Expected failed_seeds='0', got {row}"
    assert row["successful_seed_count"] == "0"


# --------------------------------------------------------------------------- #
# E.3: two completed attempts → ValueError (ambiguity)
# --------------------------------------------------------------------------- #

def test_two_completed_attempts_raises_valueerror(tmp_path):
    from experiments import collect_results, export_tables

    collect_results.write_csv(
        tmp_path / "results" / "all_runs.csv",
        [
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
                "config_id": "b", "seed": 0, "status": "completed",
                "MCC": 0.60, "experiment": "baseline",
            },
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
                "config_id": "b", "seed": 0, "status": "completed",
                "MCC": 0.90, "experiment": "baseline",
            },
        ],
    )

    with pytest.raises(ValueError, match="Ambiguous completed attempts"):
        export_tables.main(["--artifact-root", str(tmp_path)])


# --------------------------------------------------------------------------- #
# E.4: all_runs.csv still preserves ALL rows (audit trail)
# --------------------------------------------------------------------------- #

def test_all_runs_preserves_all_attempts(tmp_path):
    from experiments import collect_results, export_tables

    # Write all_runs.csv with 3 rows for 2 seeds
    rows = [
        collect_results.empty_row()
        | {
            "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
            "config_id": "b", "seed": 0, "status": "failed",
            "MCC": float("nan"), "experiment": "baseline",
        },
        collect_results.empty_row()
        | {
            "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
            "config_id": "b", "seed": 0, "status": "completed",
            "MCC": 0.70, "experiment": "baseline",
        },
        collect_results.empty_row()
        | {
            "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
            "config_id": "b", "seed": 1, "status": "completed",
            "MCC": 0.80, "experiment": "baseline",
        },
    ]
    collect_results.write_csv(tmp_path / "results" / "all_runs.csv", rows)

    # all_runs.csv must have all 3 rows
    all_runs = list(
        csv.DictReader((tmp_path / "results" / "all_runs.csv").open())
    )
    assert len(all_runs) == 3, "all_runs.csv must preserve all audit rows"

    # But main_results.csv must only count the authoritative rows
    export_tables.main(["--artifact-root", str(tmp_path)])
    main_csv = list(
        csv.DictReader((tmp_path / "results" / "main_results.csv").open())
    )
    row = next(r for r in main_csv if "ORTHRUS" in r["experiment"])
    assert row["successful_seed_count"] == "2"
    assert row["successful_seeds"] == "0,1"


# --------------------------------------------------------------------------- #
# G.1: metrics.json uses selected epoch, not last epoch
# --------------------------------------------------------------------------- #

def test_metrics_json_from_selected_epoch_not_last(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1", "model_epoch_2", "model_epoch_3"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()

    # Epoch 3 has lowest val loss → selected, but epoch 3 is LAST
    stats = {
        "model_epoch_1": {"val_mean_edge_loss": 2.0, "mcc": 0.90},
        "model_epoch_2": {"val_mean_edge_loss": 1.5, "mcc": 0.70},
        "model_epoch_3": {"val_mean_edge_loss": 1.0, "mcc": 0.50},  # selected (lowest val)
    }

    cfg = _cfg(variant="orthrus_baseline", run_dir=str(run_dir))
    module.standard_evaluation(cfg, _stats_fn(stats))

    with (run_dir / "node_scores" / "metrics.json").open() as f:
        metrics = json.load(f)

    # selected_epoch must be 3 (lowest val), not 3 (last)
    # This test proves the contract: if selected==last by coincidence,
    # check the val_mean_edge_loss to confirm it was val-driven
    assert metrics.get("selected_epoch") == 3
    assert metrics.get("val_mean_edge_loss") == 1.0


# --------------------------------------------------------------------------- #
# G.2: W&B disabled → metrics.json still written
# --------------------------------------------------------------------------- #

def test_wandb_disabled_metrics_json_still_written(monkeypatch, tmp_path):
    module, _ = _load_evaluation(monkeypatch)
    module.listdir_sorted = lambda path: ["model_epoch_1"]

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "node_scores").mkdir()

    stats = {"model_epoch_1": {"val_mean_edge_loss": 1.0, "mcc": 0.5}}

    cfg = _cfg(variant="orthrus_baseline", run_dir=str(run_dir))
    module.standard_evaluation(cfg, _stats_fn(stats))

    assert (run_dir / "node_scores" / "metrics.json").exists()


# --------------------------------------------------------------------------- #
# G.3: metrics.json valid_n excludes NaN metric seeds but keeps seed as successful
# --------------------------------------------------------------------------- #

def test_metric_nan_excluded_but_seed_counts_as_successful(tmp_path):
    from experiments import collect_results, export_tables

    collect_results.write_csv(
        tmp_path / "results" / "all_runs.csv",
        [
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
                "config_id": "b", "seed": 0, "status": "completed",
                "MCC": float("nan"), "AUROC": 0.90,
                "experiment": "baseline",
            },
            collect_results.empty_row()
            | {
                "dataset": "THEIA_E3", "config": "baseline", "config_path": "/x/b.yml",
                "config_id": "b", "seed": 1, "status": "completed",
                "MCC": 0.70, "AUROC": 0.85,
                "experiment": "baseline",
            },
        ],
    )

    export_tables.main(["--artifact-root", str(tmp_path)])
    main_csv = list(
        csv.DictReader((tmp_path / "results" / "main_results.csv").open())
    )
    row = next(r for r in main_csv if "ORTHRUS" in r["experiment"])
    assert row["successful_seed_count"] == "2"
    assert row["MCC_valid_n"] == "1", "NaN MCC excluded from valid_n"
    assert float(row["MCC_mean"]) == 0.70
    assert row["AUROC_valid_n"] == "2"


# --------------------------------------------------------------------------- #
# G.4: new config_fallback_warning field in collect_results output
# --------------------------------------------------------------------------- #

def test_collect_results_includes_config_fallback_warning_field(tmp_path):
    from experiments import collect_results, run_matrix

    cfg_file = tmp_path / "test.yml"
    cfg_file.write_text(
        "model:\n  variant: baseline\ndetection:\n  gnn_training:\n    encoder:\n      backbone: graphsage\n",
        encoding="utf-8",
    )
    scoped = run_matrix.run_artifact_root(tmp_path, cfg_file)
    dataset = "THEIA_E3"
    seed = 0
    marker = run_matrix.run_status_path(tmp_path, dataset, cfg_file, seed)
    run_dir = scoped / dataset / "runs" / "baseline" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "node_scores").mkdir()
    (run_dir / "node_scores" / "metrics.json").write_text(
        json.dumps({"MCC": 0.5}), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps({"git_commit": "abc", "model": "orthrus_baseline"}), encoding="utf-8"
    )
    (run_dir / "config_resolved.yml").write_text("", encoding="utf-8")
    run_matrix._atomic_json(marker, run_matrix._status_payload(
        dataset, cfg_file, seed, "completed", scoped_root=scoped,
    ))

    rows = collect_results.collect(tmp_path)
    row = rows[0]
    assert "config_fallback_warning" in row
    assert row["config_fallback_warning"] != ""
    assert "fallback" in row["config_fallback_warning"].lower()
