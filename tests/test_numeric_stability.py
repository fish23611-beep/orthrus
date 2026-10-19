"""CPU contracts for opt-in inference and unchanged empirical calibration."""
import csv
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection.inference_stability import (
    backend_metadata, deterministic_inference, inference_metadata, inference_rng,
)
from mstc.aggregation import NodeScoreAggregator
from mstc.calibration import HierarchicalRelationCalibrator
from mstc.calibration_runner import load_event_records_from_csv
from mstc.thresholding import apply_node_threshold, compute_node_threshold


def cfg(enabled=True):
    return NS(_seed=17, detection=NS(evaluation=NS(deterministic_inference=enabled)))


def draws():
    return random.random(), float(np.random.rand()), torch.rand(4).tolist()


def test_scoped_backend_flags_restored_on_operator_error():
    before = backend_metadata()
    with pytest.raises(RuntimeError, match="unsupported operator probe"):
        with deterministic_inference(cfg(), "cpu"):
            actual = backend_metadata()
            assert actual["deterministic_algorithms"]
            assert not actual["deterministic_warn_only"]
            assert actual["cudnn_deterministic"]
            assert not actual["cudnn_benchmark"]
            for key in ("matmul_allow_tf32", "cudnn_allow_tf32", "float32_matmul_precision"):
                assert actual[key] == before[key]
            raise RuntimeError("unsupported operator probe")
    assert backend_metadata() == before


def test_cuda_workspace_missing_fails_before_cuda_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    with pytest.raises(RuntimeError, match="outer launcher BEFORE"):
        with deterministic_inference(cfg(), "cuda"):
            pytest.fail("missing configuration accepted")


def test_default_mode_does_not_change_flags_or_rng():
    before = backend_metadata()
    state = torch.get_rng_state().clone()
    with deterministic_inference(cfg(False), "cpu"), inference_rng(cfg(False), "cpu", "val"):
        assert backend_metadata() == before
        assert torch.equal(torch.get_rng_state(), state)
    assert not inference_metadata(cfg(False))["enabled"]


def test_boundaries_ignore_prior_rng_consumption_and_restore_callers():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    state = torch.get_rng_state().clone()
    with inference_rng(cfg(), "cpu", "val"):
        first = draws()
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    assert np.array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    assert torch.equal(state, torch.get_rng_state())
    draws()
    with inference_rng(cfg(), "cpu", "val"):
        assert draws() == first
    with inference_rng(cfg(), "cpu", "test"):
        assert draws() != first


def test_rng_restores_after_exception():
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        with inference_rng(cfg(), "cpu", "replay"):
            draws()
            raise ValueError("probe")
    assert torch.equal(state, torch.get_rng_state())


def event(score, index, edge=7, src_type=0, dst_type=2):
    return dict(score_raw=score, event_index=index, edge_type_index=edge,
                src_type=src_type, dst_type=dst_type, srcnode=index % 4,
                dstnode=(index + 1) % 7)


def test_fixed_csv_recalibration_nodes_threshold_predictions_are_exact(tmp_path):
    # Production support thresholds and K=5; no tolerance or rounding.
    references = [event(8.0, i) for i in range(153)]
    references += [event(7.0, i + 153) for i in range(187)]
    references += [event(9.0, i + 340) for i in range(29)]
    queries = [event(score, 1000 + i) for i, score in enumerate([8.0, 8.000000953674316, 10.0])]
    for name, records in (("val", references), ("test", queries)):
        with (tmp_path / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    outputs = []
    for _ in range(2):
        val = load_event_records_from_csv(tmp_path / "val.csv")
        test = load_event_records_from_csv(tmp_path / "test.csv")
        cal = HierarchicalRelationCalibrator().fit(val)
        validation = cal.transform_val_with_loo(val)
        calibrated = cal.calibrate(test)
        aggregator = NodeScoreAggregator()
        val_nodes = aggregator.aggregate_events(validation, topk=5, include_dst=True)
        nodes = aggregator.aggregate_events(calibrated, topk=5, include_dst=True)
        threshold = compute_node_threshold(val_nodes, quantile=0.999)
        outputs.append((validation, calibrated, nodes, threshold, apply_node_threshold(nodes, threshold)))
        assert cal.global_scores.size == len(val)  # test never enters reference
    assert outputs[0] == outputs[1]
    calibrated = outputs[0][1]
    assert calibrated[0]["empirical_p"] == 183 / 370  # >= includes all 153 ties
    assert calibrated[1]["empirical_p"] == 30 / 370
    assert calibrated[2]["empirical_p"] == 1 / 370
    assert calibrated[0]["score_calibrated"] == -np.log(183 / 370)
    # Query perturbation reproduces the known step; perturbing references flips it back.
    perturbed = [dict(row, score_raw=row["score_raw"] + 0.000001) for row in references]
    assert HierarchicalRelationCalibrator().fit(perturbed).calibrate(queries)[1]["empirical_p"] == 183 / 370


def test_fallback_ties_n_plus_one_and_leave_one_out():
    val = [event(2.0, i) for i in range(100)]
    val += [event(1.0, 100 + i, edge=8) for i in range(100)]
    cal = HierarchicalRelationCalibrator().fit(val)
    queries = [event(2.0, 1000), event(2.0, 1001, edge=99),
               event(2.0, 1002, src_type=99), event(3.0, 1003, src_type=99)]
    actual = cal.calibrate(queries)
    assert [r["calibration_level"] for r in actual] == ["triplet", "type_pair", "global", "global"]
    assert [r["empirical_p"] for r in actual] == [1.0, 101 / 201, 101 / 201, 1 / 201]
    # Removing one makes 99/199 supports fall back to global, removing exactly one tie.
    loo = cal.transform_val_with_loo(val)
    assert loo[0]["calibration_level"] == "global"
    assert loo[0]["empirical_p"] == 100 / 200


def test_policy_rng_boundaries_keep_six_checkpoint_canonical_sequence(tmp_path, monkeypatch):
    # Real policy controller with model/IO replaced; these are NOT GPU score evidence.
    from test_checkpoint_policy import make_cfg, patch_common_runtime, FakeData, testing

    outputs = []
    for policy in ("all_checkpoints", "val_select_then_test_best"):
        c = make_cfg(tmp_path / policy, policy=policy)
        c._seed = 17
        c.detection.evaluation.deterministic_inference = True
        checkpoints = [f"model_epoch_{i}" for i in range(1, 7)]
        train, val, test = [FakeData("train")], [FakeData("val")], [FakeData("test")]
        calls, samples, cleanup = [], [], []
        patch_common_runtime(monkeypatch, c, checkpoints=checkpoints, train_data=train,
                             val_data=val, test_data=test, calls=calls,
                             build_samples=samples, cleanup_calls=cleanup)
        monkeypatch.setattr(testing, "load_all_datasets", lambda c: (train, val, test, object(), 10))
        monkeypatch.setattr(testing, "load_train_val_datasets", lambda c: (train, val, object(), 10))
        monkeypatch.setattr(testing, "compute_validation_objective", lambda p, c: (-int(p[-1]), "score_raw"))
        build = testing.build_model
        def consume_construction(**kwargs):
            draws()
            return build(**kwargs)
        monkeypatch.setattr(testing, "build_model", consume_construction)
        seen = []
        def replay(model, *args):
            calls.append((model.epoch, "replay"))
            seen.append((model.epoch, "replay", draws()))
        def split(*args, split, model, **kwargs):
            calls.append((model.epoch, split))
            seen.append((model.epoch, split, draws()))
            return {"processed_event_count": 1}
        monkeypatch.setattr(testing, "_replay_train_history", replay)
        monkeypatch.setattr(testing, "test", split)
        draws()  # policy-independent boundaries must ignore different incoming state
        testing.main(c)
        count = 6 if policy == "all_checkpoints" else 7
        assert sum(s == "replay" for _, s in calls) == count
        assert sum(s == "val" for _, s in calls) == count
        assert sum(s == "test" for _, s in calls) == (6 if count == 6 else 1)
        outputs.append([r for r in seen if r[0] == "model_epoch_6"][-3:])
    assert outputs[0] == outputs[1]


def test_config_opt_in_defaults_to_false():
    from config import get_default_cfg
    c = get_default_cfg(NS(dataset="THEIA_E5", cpu=True, from_weights=False, seed=0,
                           max_windows_per_split=None, perf_profile=False,
                           perf_profile_sample_every_n_batches=None, skip_tracing=False))
    assert c.detection.evaluation.deterministic_inference is False
    c.merge_from_list(["detection.evaluation.deterministic_inference", "True"])
    assert c.detection.evaluation.deterministic_inference is True


def test_event_comparator_rejects_bit_difference_reordering_and_missing_raw(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p4_ns_probe", Path(__file__).resolve().parents[1] / "scripts/check_numeric_stability.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    def write(name, rows):
        path = tmp_path / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path
    # Signed zero is numerically equal but fails the bitwise gate.
    rows = [dict(event(0.0, 0), time=0), dict(event(1.0, 1), time=1)]
    left = write("left.csv", rows)
    right = write("right.csv", [dict(rows[0], score_raw=-0.0), rows[1]])
    result = probe.compare(NS(left=left, right=right))
    assert not result["exactly_equal"]
    assert result["fields"]["score_raw"]["changed"] == 1
    right = write("reordered.csv", rows[::-1])
    result = probe.compare(NS(left=left, right=right))
    assert result["fields"]["score_raw"]["bitwise_equal"]
    assert not result["structure_and_order_equal"]
    assert not result["exactly_equal"]
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "events.csv").write_bytes(right.read_bytes())
    assert not probe.compare(NS(left=left, right=directory))["exactly_equal"]
    right = write("missing.csv", [{k: v for k, v in row.items() if k != "score_raw"} for row in rows])
    with pytest.raises(ValueError, match="score_raw"):
        probe.compare(NS(left=left, right=right))
