import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    _validate_checkpoint_policy,
    get_checkpoint_policy,
    get_default_cfg,
)
from detection import checkpoint_policy
from detection import evaluation
from detection import orthrus_gnn_testing as testing


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def make_cfg(tmp_path, policy="all_checkpoints", method="min_val_objective"):
    edge_root = tmp_path / "edge_scores"
    run_dir = tmp_path / "run"
    return ns(
        _run_dir=str(run_dir),
        _from_weights=False,
        _from_weights_path=str(tmp_path / "weights"),
        _inference_checkpoint=None,
        dataset=ns(name="THEIA_E5"),
        model=ns(variant="mstc"),
        model_selection=ns(
            method=method,
            legacy_test_selection_enabled=False,
        ),
        testing=ns(include_node_messages=False),
        detection=ns(
            gnn_training=ns(
                _trained_models_dir=str(tmp_path / "checkpoints"),
                decoder=ns(
                    predict_edge_type=ns(enabled=True),
                    time_gap=ns(enabled=True),
                ),
            ),
            gnn_testing=ns(_edge_losses_dir=str(edge_root)),
            evaluation=ns(
                checkpoint_policy=policy,
                used_method="node_evaluation",
                _evaluation_results_dir=str(run_dir / "evaluation_results"),
                node_evaluation=ns(
                    _precision_recall_dir=str(run_dir / "precision_recall")
                ),
            ),
        ),
    )


class FakeData:
    def __init__(self, name):
        self.name = name
        self.devices = []

    def to(self, device=None):
        self.devices.append(device)
        return self


class FakeEncoder:
    def __init__(self, epoch, calls):
        self.epoch = epoch
        self.calls = calls

    def reset_state(self):
        self.calls.append((self.epoch, "reset"))


class FakeModel:
    def __init__(self, epoch, calls):
        self.epoch = epoch
        self.encoder = FakeEncoder(epoch, calls)
        self.graph_reindexer = object()


def patch_common_runtime(
    monkeypatch,
    cfg,
    *,
    checkpoints,
    train_data,
    val_data,
    test_data,
    calls,
    build_samples,
    cleanup_calls,
):
    monkeypatch.setattr(testing, "_load_nodeid2msg", lambda cfg: {})
    monkeypatch.setattr(
        testing, "requires_time_gap_statistics", lambda cfg: False
    )
    monkeypatch.setattr(
        testing,
        "listdir_sorted",
        lambda path: list(checkpoints),
    )
    monkeypatch.setattr(
        testing, "get_device", lambda cfg: torch.device("cpu")
    )
    monkeypatch.setattr(testing, "dump_environment", lambda *a, **k: None)
    monkeypatch.setattr(
        testing, "update_runtime_nested", lambda *a, **k: None
    )
    monkeypatch.setattr(testing, "update_runtime", lambda *a, **k: None)
    monkeypatch.setattr(testing, "peak_cpu_memory_mb", lambda: 1.0)
    monkeypatch.setattr(
        testing, "tqdm", lambda values, desc=None: values
    )

    def build_model(*, data_sample, **kwargs):
        build_samples.append(data_sample)
        return FakeModel("unloaded", calls)

    def load_model(model, path):
        epoch = os.path.basename(path)
        return FakeModel(epoch, calls)

    def replay(model, *args, **kwargs):
        calls.append((model.epoch, "replay"))

    monkeypatch.setattr(testing, "build_model", build_model)
    monkeypatch.setattr(testing, "load_model", load_model)
    monkeypatch.setattr(testing, "_replay_train_history", replay)
    monkeypatch.setattr(
        testing,
        "_cleanup_checkpoint_cuda",
        lambda model, device: cleanup_calls.append(model.epoch),
    )


def test_default_checkpoint_policy_is_all_checkpoints():
    args = ns(
        dataset="THEIA_E5",
        cpu=True,
        from_weights=False,
        seed=0,
        max_windows_per_split=None,
        perf_profile=False,
        perf_profile_sample_every_n_batches=None,
        skip_tracing=False,
    )
    cfg = get_default_cfg(args)
    assert cfg.detection.evaluation.checkpoint_policy == "all_checkpoints"
    assert get_checkpoint_policy(ns(detection=ns(evaluation=ns()))) == (
        "all_checkpoints"
    )


@pytest.mark.parametrize(
    "policy",
    ["all_checkpoints", "val_select_then_test_best"],
)
def test_checkpoint_policy_validator_accepts_supported_values(policy):
    assert _validate_checkpoint_policy(policy) == policy


def test_checkpoint_policy_validator_rejects_unknown_value():
    with pytest.raises(ValueError, match="checkpoint_policy"):
        _validate_checkpoint_policy("test_every_other_epoch")


def test_all_checkpoints_keeps_legacy_n_replay_n_val_n_test(
    tmp_path, monkeypatch
):
    cfg = make_cfg(tmp_path)
    checkpoints = [f"model_epoch_{i}" for i in range(1, 4)]
    train = [FakeData("train")]
    val = [FakeData("val")]
    test_data = [FakeData("test")]
    calls, samples, cleanup = [], [], []
    patch_common_runtime(
        monkeypatch,
        cfg,
        checkpoints=checkpoints,
        train_data=train,
        val_data=val,
        test_data=test_data,
        calls=calls,
        build_samples=samples,
        cleanup_calls=cleanup,
    )
    monkeypatch.setattr(
        testing,
        "load_all_datasets",
        lambda cfg: (train, val, test_data, object(), 10),
    )

    def run_split(data, *, split, model, **kwargs):
        calls.append((model.epoch, split))
        return {"processed_event_count": 1}

    monkeypatch.setattr(testing, "test", run_split)
    testing.main(cfg)

    assert sum(step == "replay" for _, step in calls) == 3
    assert sum(step == "val" for _, step in calls) == 3
    assert sum(step == "test" for _, step in calls) == 3
    assert samples == [test_data[0]] * 3
    assert cleanup == checkpoints


def test_optimized_three_checkpoints_has_four_replay_four_val_one_test(
    tmp_path, monkeypatch
):
    cfg = make_cfg(tmp_path, policy="val_select_then_test_best")
    checkpoints = [f"model_epoch_{i}" for i in range(1, 4)]
    train = [FakeData("train")]
    val = [FakeData("val")]
    test_data = [FakeData("test")]
    phase_a_full = object()
    phase_b_full = object()
    calls, samples, cleanup = [], [], []
    objective = {
        "model_epoch_1": 0.30,
        "model_epoch_2": 0.10,
        "model_epoch_3": 0.20,
    }
    patch_common_runtime(
        monkeypatch,
        cfg,
        checkpoints=checkpoints,
        train_data=train,
        val_data=val,
        test_data=test_data,
        calls=calls,
        build_samples=samples,
        cleanup_calls=cleanup,
    )
    monkeypatch.setattr(
        testing,
        "load_train_val_datasets",
        lambda cfg: (train, val, phase_a_full, 8),
    )

    def load_all_after_selection(cfg_arg):
        manifest = Path(cfg._run_dir) / "checkpoint_selection.json"
        assert manifest.exists(), "test split loaded before durable selection"
        payload = json.loads(manifest.read_text())
        assert payload["selected_epoch"] == "model_epoch_2"
        calls.append(("loader", "test_materialized"))
        return train, val, test_data, phase_b_full, 10

    monkeypatch.setattr(testing, "load_all_datasets", load_all_after_selection)

    def run_split(data, *, split, model, model_epoch_file, **kwargs):
        manifest = Path(cfg._run_dir) / "checkpoint_selection.json"
        if split == "test":
            assert manifest.exists(), "test iterated before selection"
        calls.append((model.epoch, split))
        out_dir = (
            Path(cfg.detection.gnn_testing._edge_losses_dir)
            / split
            / model_epoch_file
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        if split == "val":
            rows = pd.DataFrame(
                {
                    "event_index": [10, 11],
                    "score_raw": [
                        objective[model_epoch_file],
                        objective[model_epoch_file],
                    ],
                }
            )
        else:
            rows = pd.DataFrame(
                {
                    "event_index": [20, 21],
                    "score_raw": [9.0, 9.0],
                    "test_mcc": [0.01, 0.01],
                }
            )
        rows.to_csv(out_dir / "window.csv", index=False)
        return {"processed_event_count": len(rows)}

    monkeypatch.setattr(testing, "test", run_split)
    testing.main(cfg)

    assert sum(step == "replay" for _, step in calls) == 4
    assert sum(step == "val" for _, step in calls) == 4
    assert sum(step == "test" for _, step in calls) == 1
    assert samples[:3] == [val[0]] * 3
    assert samples[3] is test_data[0]
    assert cleanup == checkpoints + ["model_epoch_2"]

    test_root = Path(cfg.detection.gnn_testing._edge_losses_dir) / "test"
    assert sorted(path.name for path in test_root.iterdir()) == [
        "model_epoch_2"
    ]
    selected_val = pd.read_csv(
        Path(cfg.detection.gnn_testing._edge_losses_dir)
        / "val"
        / "model_epoch_2"
        / "window.csv"
    )
    assert selected_val["event_index"].tolist() == [10, 11]
    assert len(selected_val) == 2

    loader_index = calls.index(("loader", "test_materialized"))
    selection_val_calls = [
        item for item in calls[:loader_index] if item[1] == "val"
    ]
    assert len(selection_val_calls) == 3
    assert not any(item[1] == "test" for item in calls[:loader_index])

    canonical = calls[loader_index + 1 :]
    assert canonical[:4] == [
        ("model_epoch_2", "reset"),
        ("model_epoch_2", "replay"),
        ("model_epoch_2", "val"),
        ("model_epoch_2", "test"),
    ]


def test_selection_uses_validation_only_not_test_mcc():
    records = {
        "model_epoch_1": {
            "val_objective": 0.30,
            "test_mcc": 0.99,
        },
        "model_epoch_2": {
            "val_objective": 0.10,
            "test_mcc": 0.01,
        },
    }
    selected = checkpoint_policy.select_checkpoint_from_validation(
        {
            epoch: values["val_objective"]
            for epoch, values in records.items()
        }
    )
    assert selected == "model_epoch_2"


def test_joint_validation_objective_is_score_raw_arithmetic_mean(
    tmp_path,
):
    cfg = make_cfg(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    pd.DataFrame({"score_raw": [1.0, 2.0, 6.0]}).to_csv(
        val_dir / "window.csv", index=False
    )

    shared = checkpoint_policy.compute_validation_objective(val_dir, cfg)
    legacy_wrapper = evaluation._compute_validation_objective(val_dir, cfg)
    assert shared == legacy_wrapper
    assert shared == (3.0, "score_raw")
    assert evaluation._validation_objective_field(cfg) == "score_raw"


def test_train_val_loader_requests_no_test_split(monkeypatch):
    from data_utils import _load_datasets, load_train_val_datasets
    import data_utils

    requested = []

    def load_data_set(cfg, *, path, split, lazy):
        requested.append(split)
        return object()

    sentinel = object()
    monkeypatch.setattr(data_utils, "load_data_set", load_data_set)
    monkeypatch.setattr(
        data_utils,
        "_eager_load_selected_datasets",
        lambda cfg, split_names, datasets: sentinel,
    )
    cfg = ns(
        edge_featurization=ns(
            embed_edges=ns(_edge_embeds_dir="/unused")
        )
    )
    assert load_train_val_datasets(cfg) is sentinel
    assert requested == ["train", "val"]


def test_manifest_atomic_roundtrip_and_schema(tmp_path):
    cfg = make_cfg(tmp_path, policy="val_select_then_test_best")
    path = checkpoint_policy.selection_manifest_path(cfg)
    payload = {
        "checkpoint_policy": "val_select_then_test_best",
        "model_selection_method": "min_val_objective",
        "val_objective_field": "score_raw",
        "per_epoch_validation_objective": {
            "model_epoch_1": 0.3,
            "model_epoch_2": 0.1,
            "model_epoch_3": 0.2,
        },
        "selected_epoch": "model_epoch_2",
    }
    checkpoint_policy.write_selection_manifest(path, payload)
    assert checkpoint_policy.read_selection_manifest(path) == payload
    assert checkpoint_policy.validate_selection_manifest(payload, cfg) == (
        "model_epoch_2"
    )
    assert not list(Path(cfg._run_dir).glob("*.tmp"))


def test_stale_non_selected_test_artifact_fails_loudly(
    tmp_path,
):
    cfg = make_cfg(tmp_path, policy="val_select_then_test_best")
    test_root = (
        Path(cfg.detection.gnn_testing._edge_losses_dir) / "test"
    )
    (test_root / "model_epoch_1").mkdir(parents=True)
    (test_root / "model_epoch_2").mkdir()
    with pytest.raises(RuntimeError, match="fresh run root"):
        testing._reject_stale_non_selected_test_artifacts(
            cfg, "model_epoch_2"
        )


def test_selected_evaluation_reads_manifest_and_runs_only_selected(
    tmp_path, monkeypatch
):
    cfg = make_cfg(tmp_path, policy="val_select_then_test_best")
    payload = {
        "checkpoint_policy": "val_select_then_test_best",
        "model_selection_method": "min_val_objective",
        "val_objective_field": "score_raw",
        "per_epoch_validation_objective": {
            "model_epoch_1": 0.3,
            "model_epoch_2": 0.1,
        },
        "selected_epoch": "model_epoch_2",
    }
    checkpoint_policy.write_selection_manifest(
        checkpoint_policy.selection_manifest_path(cfg), payload
    )
    edge_root = Path(cfg.detection.gnn_testing._edge_losses_dir)
    (edge_root / "val" / "model_epoch_2").mkdir(parents=True)
    (edge_root / "test" / "model_epoch_2").mkdir(parents=True)

    calls = []

    def evaluation_fn(val_path, test_path, epoch, cfg_arg, **kwargs):
        calls.append((Path(val_path).name, Path(test_path).name, epoch))
        return {"mcc": 0.5}

    monkeypatch.setattr(evaluation, "wandb_log", lambda *a, **k: None)
    monkeypatch.setattr(
        evaluation, "_persist_canonical_metrics", lambda *a: calls.append("metrics")
    )
    monkeypatch.setattr(
        evaluation, "_persist_mstc_predictions", lambda *a: calls.append("node")
    )
    monkeypatch.setattr(
        evaluation, "_persist_event_predictions", lambda *a: calls.append("event")
    )
    monkeypatch.setattr(evaluation, "update_runtime", lambda *a, **k: None)

    stats = evaluation._selected_checkpoint_evaluation(cfg, evaluation_fn)
    assert calls[0] == (
        "model_epoch_2",
        "model_epoch_2",
        "model_epoch_2",
    )
    assert calls.count("metrics") == 1
    assert calls.count("node") == 1
    assert calls.count("event") == 1
    assert stats["val_objective"] == pytest.approx(0.1)
    assert stats["val_objective_field"] == "score_raw"


def test_selection_manifest_is_validated_before_full_evaluation(
    tmp_path,
):
    cfg = make_cfg(tmp_path, policy="val_select_then_test_best")
    payload = {
        "checkpoint_policy": "val_select_then_test_best",
        "model_selection_method": "min_val_objective",
        "val_objective_field": "score_raw",
        "per_epoch_validation_objective": {"model_epoch_1": 0.1},
        "selected_epoch": "model_epoch_999",
    }
    checkpoint_policy.write_selection_manifest(
        checkpoint_policy.selection_manifest_path(cfg), payload
    )
    called = []

    def forbidden_test_calibration_or_gt(*args, **kwargs):
        called.append(True)
        raise AssertionError("test evaluation accessed before valid selection")

    with pytest.raises(ValueError, match="selected_epoch"):
        evaluation._selected_checkpoint_evaluation(
            cfg, forbidden_test_calibration_or_gt
        )
    assert called == []


def test_actual_validation_writer_overwrites_canonical_rerun(
    tmp_path, monkeypatch
):
    edge_root = tmp_path / "edge_scores"
    cfg = ns(
        testing=ns(include_node_messages=False),
        detection=ns(
            gnn_testing=ns(_edge_losses_dir=str(edge_root))
        ),
    )
    batch = ns(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_type=torch.tensor(
            [[1.0] + [0.0] * 9, [1.0] + [0.0] * 9]
        ),
        edge_type_index=torch.tensor([0, 0]),
        t=torch.tensor([1, 2]),
        global_event_index=torch.tensor([5, 6]),
        src_type=torch.tensor([0, 0]),
        dst_type=torch.tensor([1, 1]),
    )

    class Model:
        graph_reindexer = object()

        def eval(self):
            return self

        def __call__(self, batch_arg, full_data, inference):
            zeros = torch.zeros(2)
            logits = torch.zeros((2, 6))
            return {
                "score_raw": torch.tensor([0.1, 0.2]),
                "loss_type": torch.tensor([0.1, 0.2]),
                "loss_time_src": zeros,
                "loss_time_dst": zeros,
                "loss_time": zeros,
                "src_time_target": torch.tensor([0, 1]),
                "dst_time_target": torch.tensor([1, 2]),
                "src_time_logits": logits,
                "dst_time_logits": logits,
            }

    monkeypatch.setattr(
        testing, "batch_loader_factory", lambda *a, **k: [batch]
    )
    monkeypatch.setattr(
        testing, "ns_time_to_datetime_US", lambda value: str(int(value))
    )

    for _ in range(2):
        testing.test(
            data=batch,
            full_data=object(),
            model=Model(),
            nodeid2msg={},
            split="val",
            model_epoch_file="model_epoch_2",
            cfg=cfg,
            device=torch.device("cpu"),
        )

    csv_files = list(
        (edge_root / "val" / "model_epoch_2").glob("*.csv")
    )
    assert len(csv_files) == 1
    records = pd.read_csv(csv_files[0])
    assert len(records) == 2
    assert records["event_index"].tolist() == [5, 6]


def test_all_and_optimized_selection_match_exactly(
    tmp_path, monkeypatch
):
    cfg = make_cfg(tmp_path)
    edge_root = Path(cfg.detection.gnn_testing._edge_losses_dir)
    objectives = {
        "model_epoch_1": [0.2, 0.4],
        "model_epoch_2": [0.05, 0.15],
    }
    for epoch, values in objectives.items():
        val_dir = edge_root / "val" / epoch
        test_dir = edge_root / "test" / epoch
        val_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        pd.DataFrame({"score_raw": values}).to_csv(
            val_dir / "window.csv", index=False
        )

    optimized_values = {
        epoch: checkpoint_policy.compute_validation_objective(
            edge_root / "val" / epoch, cfg
        )[0]
        for epoch in objectives
    }
    optimized_epoch = checkpoint_policy.select_checkpoint_from_validation(
        optimized_values
    )
    selected = []
    monkeypatch.setattr(evaluation, "wandb_is_active", lambda: False)
    monkeypatch.setattr(evaluation, "wandb_log", lambda *a, **k: None)
    monkeypatch.setattr(
        evaluation,
        "_persist_canonical_metrics",
        lambda run_dir, stats, epoch, method, cfg_arg: selected.append(
            (epoch, stats["val_objective"])
        ),
    )
    monkeypatch.setattr(
        evaluation, "_persist_mstc_predictions", lambda *a, **k: None
    )
    monkeypatch.setattr(
        evaluation, "_persist_event_predictions", lambda *a, **k: None
    )
    monkeypatch.setattr(evaluation, "update_runtime", lambda *a, **k: None)

    test_metric = {"model_epoch_1": 0.99, "model_epoch_2": 0.01}

    def evaluation_fn(val_path, test_path, epoch, cfg_arg, **kwargs):
        return {"mcc": test_metric[epoch]}

    evaluation.standard_evaluation(cfg, evaluation_fn)
    assert selected == [
        (optimized_epoch, optimized_values[optimized_epoch])
    ]
    assert optimized_epoch == "model_epoch_2"


def test_optimized_policy_rejects_incompatible_model_selection(
    tmp_path, monkeypatch
):
    cfg = make_cfg(
        tmp_path,
        policy="val_select_then_test_best",
        method="last_epoch",
    )
    monkeypatch.setattr(
        testing,
        "_load_nodeid2msg",
        lambda cfg: pytest.fail("loader reached before compatibility gate"),
    )
    with pytest.raises(ValueError, match="min_val_objective"):
        testing.main(cfg)
