"""Read-only P4-NS probes. Every output path must be new; never train models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def records(path, *, preserve_order=False):
    from mstc.calibration_runner import load_event_records_from_csv, load_event_records_from_csv_directory
    path = Path(path)
    if path.is_dir() and preserve_order:
        return [record for csv_path in sorted(path.glob("*.csv"))
                for record in load_event_records_from_csv(csv_path)]
    return load_event_records_from_csv_directory(path) if path.is_dir() else load_event_records_from_csv(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recalibrate(args):
    from mstc.calibration import HierarchicalRelationCalibrator
    from mstc.aggregation import NodeScoreAggregator
    from mstc.thresholding import compute_node_threshold, apply_node_threshold
    val, test = records(args.val), records(args.test)
    outputs = []
    for _ in range(2):
        calibrator = HierarchicalRelationCalibrator().fit(val)
        calibrated_val = calibrator.transform_val_with_loo(val)
        calibrated = calibrator.calibrate(test)
        aggregator = NodeScoreAggregator()
        val_nodes = aggregator.aggregate_events(calibrated_val)
        nodes = aggregator.aggregate_events(calibrated)
        threshold = compute_node_threshold(val_nodes)
        outputs.append(dict(val=calibrated_val, test=calibrated, nodes=nodes,
                            threshold=threshold, y_hat=apply_node_threshold(nodes, threshold)))
    evidence = {}
    for record in outputs[0]["test"]:
        if record["event_index"] in (363334, 370794):
            event = calibrator._normalise_event(record)
            reference, level = calibrator._select_reference(event, leave_one_out=False)
            evidence[str(record["event_index"])] = dict(
                score_raw=record["score_raw"], empirical_p=record["empirical_p"],
                score_calibrated=record["score_calibrated"], level=level,
                reference_count=len(reference),
                count_ge=calibrator._count_greater_equal(reference, record["score_raw"]),
            )
    result = dict(mode="recalibrate", val=str(args.val), test=str(args.test),
                  val_events=len(val), test_events=len(test),
                  exactly_equal=outputs[0] == outputs[1],
                  output_sha256=[fingerprint(o) for o in outputs], evidence=evidence)
    return result


def compare(args):
    import numpy as np
    left, right = records(args.left, preserve_order=True), records(args.right, preserve_order=True)
    fields = ("event_index", "time", "srcnode", "dstnode", "src_type", "dst_type", "edge_type_index")
    if not left or not right:
        raise ValueError("Cannot compare empty event records")
    for record in left + right:
        for field in fields + ("score_raw",):
            if field not in record:
                raise ValueError(f"Missing comparison field: {field}")
    keys_left = [tuple(r.get(f) for f in fields) for r in left]
    keys_right = [tuple(r.get(f) for f in fields) for r in right]
    same_order = keys_left == keys_right
    if len(set(keys_left)) != len(keys_left) or len(set(keys_right)) != len(keys_right):
        raise ValueError("Duplicate event keys")
    mapping = dict(zip(keys_right, right))
    same_keys = set(keys_left) == set(keys_right)
    result = dict(mode="compare", left=str(args.left), right=str(args.right),
                  rows=[len(left), len(right)], structure_and_order_equal=same_order,
                  key_sets_equal=same_keys, fields={})
    if not same_keys:
        result["exactly_equal"] = False
        return result
    for field in ("score_raw", "loss_type", "loss_time_src", "loss_time_dst", "loss_time",
                  "empirical_p", "score_calibrated"):
        if not all(field in r for r in left + right):
            continue
        a = np.array([r[field] for r in left], dtype=np.float64)
        b = np.array([mapping[k][field] for k in keys_left], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Non-finite values in {field}")
        changed = np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))
        result["fields"][field] = dict(bitwise_equal=not len(changed), changed=len(changed),
            max_abs_delta=float(np.max(np.abs(a-b), initial=0)),
            examples=[dict(event_key=keys_left[i], left=float(a[i]), right=float(b[i])) for i in changed[:5]])
    result["exactly_equal"] = same_order and bool(result["fields"]) and all(
        f["bitwise_equal"] for f in result["fields"].values())
    return result


def operators(args):
    import torch
    from torch_geometric.nn import TransformerConv
    from detection.inference_stability import deterministic_inference, backend_metadata
    device = torch.device(args.device)
    torch.manual_seed(17)
    x = torch.randn(512, 32).to(device)
    edge = torch.randint(512, (2, 8192)).to(device)
    values = torch.randn(8192, 32).to(device)
    conv = TransformerConv(32, 16, heads=2, dropout=0).to(device).eval()
    inputs = {name: hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
              for name, t in (("x", x), ("edge", edge), ("values", values))}
    operations = {
        "scatter_add": lambda: torch.zeros_like(x).scatter_add_(0, edge[1, :, None].expand_as(values), values),
        "transformer_conv": lambda: conv(x, edge),
    }
    result = dict(mode="operators", device=str(device), torch_version=torch.__version__,
                  input_sha256=inputs, repeats=args.repeats, cases=[])
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False)
        for deterministic in (False, True):
            cfg = NS(detection=NS(evaluation=NS(deterministic_inference=deterministic)))
            with deterministic_inference(cfg, device), torch.no_grad():
                for name, operation in operations.items():
                    case = dict(operator=name, deterministic=deterministic, backend=backend_metadata())
                    try:
                        reference = operation()
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        start = time.perf_counter()
                        changed_runs, max_delta = 0, 0.0
                        for _ in range(args.repeats):
                            actual = operation()
                            changed_runs += int(not torch.equal(reference.view(torch.uint8), actual.view(torch.uint8)))
                            max_delta = max(max_delta, float((reference-actual).abs().max()))
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        case.update(changed_runs=changed_runs, max_abs_delta=max_delta,
                                    seconds=time.perf_counter()-start, bitwise_equal=changed_runs == 0)
                    except RuntimeError as exc:
                        case["error"] = str(exc)
                        result["cases"].append(case)
                        result["error"] = f"Fail-fast: {name}: {exc}"
                        return result
                    result["cases"].append(case)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    return result


def loader(args):
    import torch
    import yaml
    from yacs.config import CfgNode
    import data_utils
    cfg = CfgNode(yaml.safe_load(Path(args.config).read_text()))
    cfg._test_mode = False
    cfg._max_windows_per_split = 2
    cfg._shared_artifact_readonly = True
    cfg.edge_featurization.embed_edges._edge_embeds_dir = str(Path(args.embeddings).resolve())
    scan = data_utils._load_datasets(cfg, ("train", "val", "test"), allow_full_sidecar=False)
    sidecar = data_utils.load_all_datasets(cfg)
    train_val = data_utils.load_train_val_datasets(cfg)
    checked, digest = [], hashlib.sha256()
    def check(a, b, label):
        if isinstance(a, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
                raise AssertionError(f"Loader tensor mismatch: {label}")
            digest.update(label.encode())
            digest.update(str((a.dtype, tuple(a.shape))).encode())
            digest.update(a.contiguous().numpy().tobytes())
            checked.append(label)
        elif a != b:
            raise AssertionError(f"Loader metadata mismatch: {label}")
    for split, a, b in zip(("train", "val", "test"), scan[:3], sidecar[:3]):
        assert len(a) == len(b) == 2
        for i, (ga, gb) in enumerate(zip(a, b)):
            assert set(ga.keys()) == set(gb.keys())
            for field in sorted(ga.keys()):
                check(ga[field], gb[field], f"{split}/{i}/{field}")
            if split != "test":
                gc = train_val[0 if split == "train" else 1][i]
                for field in sorted(ga.keys()):
                    check(ga[field], gc[field], f"train_val/{split}/{i}/{field}")
    for field in ("src", "dst", "t", "msg", "edge_type", "x_src", "x_dst", "src_type", "dst_type", "edge_type_index"):
        ids = torch.arange(scan[3].num_events)
        check(scan[3].get_event_values(field, ids), sidecar[3].get_event_values(field, ids), f"history/{field}")
        ids = torch.arange(train_val[2].num_events)
        check(scan[3].get_event_values(field, ids), train_val[2].get_event_values(field, ids), f"train_val_history/{field}")
    return dict(mode="loader", exactly_equal=True, checked_tensors=len(checked),
                tensor_sha256=digest.hexdigest(), sidecar=sidecar[3].loader_telemetry,
                note="Full tensor comparison, not IDs/counts only; historical assets read-only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON output path")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("recalibrate")
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.set_defaults(run=recalibrate)
    p = sub.add_parser("compare")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.set_defaults(run=compare)
    p = sub.add_parser("operators")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--repeats", type=int, default=30)
    p.set_defaults(run=operators)
    p = sub.add_parser("loader")
    p.add_argument("--config", required=True)
    p.add_argument("--embeddings", required=True)
    p.set_defaults(run=loader)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output exists; choose a fresh path")
    result = args.run(args)
    save(args.output, result)
    print(json.dumps(result, indent=2))
    if result.get("error") or result.get("exactly_equal") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
