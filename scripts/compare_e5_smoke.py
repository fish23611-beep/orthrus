"""Stream saved Windows/G3 CSVs; report equality without changing tolerances."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FLOATS = {"loss", "loss_type", "loss_time_src", "loss_time_dst", "loss_time", "score_raw",
          "empirical_p", "score_calibrated", "score"}
IDENTITY = {"event_index", "time", "srcnode", "dstnode", "src_type", "dst_type", "edge_type_index", "node_id"}


def csv_rows(path):
    paths = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No CSV input at " + str(path))
    for file in paths:
        with file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Missing or duplicate CSV header")
            yield from reader


def compare_csv(left, right, *, atol=1e-6, rtol=1e-5):
    result = {"left": str(left), "right": str(right), "rows_left": 0, "rows_right": 0,
              "header_equal": True, "fields": {}, "tolerance": {"atol": atol, "rtol": rtol},
              "bitwise_scope": "IEEE-754 binary64 after parsing saved CSV values; not raw GPU tensor bytes"}
    signatures = [hashlib.sha256(), hashlib.sha256()]
    missing = object()
    for index, (a, b) in enumerate(zip_longest(csv_rows(left), csv_rows(right), fillvalue=missing)):
        for side, row in enumerate((a, b)):
            if row is not missing:
                result["rows_left" if side == 0 else "rows_right"] += 1
                key = [(k, row[k]) for k in sorted(IDENTITY & row.keys())]
                signatures[side].update(json.dumps(key, separators=(",", ":")).encode())
        if a is missing or b is missing:
            continue
        if a.keys() != b.keys():
            result["header_equal"] = False
            continue
        for field in a:
            entry = result["fields"].setdefault(field, {"kind": "float" if field in FLOATS else "discrete",
                "changed": 0, "outside_tolerance": 0, "max_abs_error": 0.0, "max_rel_error": 0.0, "first_changed_rows": []})
            if field in FLOATS:
                if a[field] == b[field] == "":
                    changed = False
                elif not a[field] or not b[field]:
                    changed = True
                    entry["outside_tolerance"] += 1
                else:
                    x, y = float(a[field]), float(b[field])
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError(f"Nonfinite scientific CSV value: {field} row {index}")
                    changed = struct.pack(">d", x) != struct.pack(">d", y)
                    absolute = abs(x-y)
                    relative = absolute/max(abs(x), abs(y), 1e-300)
                    entry["max_abs_error"] = max(entry["max_abs_error"], absolute)
                    entry["max_rel_error"] = max(entry["max_rel_error"], relative)
                    entry["outside_tolerance"] += int(not math.isclose(x,y,abs_tol=atol,rel_tol=rtol))
            else:
                changed = a[field] != b[field]
            entry["changed"] += int(changed)
            if changed and len(entry["first_changed_rows"]) < 5:
                entry["first_changed_rows"].append(index)
    result["identity_order_sha256"] = [item.hexdigest() for item in signatures]
    result["row_counts_equal"] = result["rows_left"] == result["rows_right"]
    if not result["rows_left"] or not result["rows_right"]:
        raise ValueError("Cannot compare empty scientific outputs")
    result["identity_order_equal"] = result["identity_order_sha256"][0] == result["identity_order_sha256"][1]
    result["all_discrete_equal"] = all(v["changed"] == 0 for v in result["fields"].values() if v["kind"] == "discrete")
    result["all_float_bitwise_equal"] = all(v["changed"] == 0 for v in result["fields"].values() if v["kind"] == "float")
    result["all_float_close"] = all(v["outside_tolerance"] == 0 for v in result["fields"].values() if v["kind"] == "float")
    if not result["row_counts_equal"] or not result["header_equal"] or not result["identity_order_equal"]:
        result["classification"] = "STRUCTURAL_MISMATCH"
    elif result["all_discrete_equal"] and result["all_float_bitwise_equal"]:
        result["classification"] = "BITWISE_EQUAL"
    elif result["all_discrete_equal"] and result["all_float_close"]:
        result["classification"] = "NUMERICALLY_CLOSE_NOT_BITWISE"
    else:
        result["classification"] = "DIFFERENT"
    return result


def compare_metrics(left, right):
    result = {}
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            result[key] = {"equal": False, "missing_side": "left" if key not in left else "right"}
            continue
        a, b = left[key], right[key]
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            result[key] = {"equal": True, "matching_undefined_nan": True}
        elif isinstance(a, (float,int)) and isinstance(b,(float,int)):
            if not math.isfinite(a) or not math.isfinite(b):
                result[key] = {"equal": a == b, "nonfinite": True}
            else:
                result[key] = {"equal": struct.pack(">d",float(a)) == struct.pack(">d",float(b)),
                               "left": a, "right": b, "abs_error": abs(a-b)}
        else:
            result[key] = {"equal": a == b, "left": a, "right": b}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new comparison report path")
    run_result = json.loads((args.run_root / "smoke_result.json").read_text())
    if run_result.get("real_two_window_inference") != "PASS":
        raise ValueError("A completed genuine two-window run is required")
    reference = args.data_root / "artifacts/THEIA_E5/linux_reference"
    pairs = {
        "raw_val": (reference / "edge_scores/val/model_epoch_6", args.run_root / "edge_scores/val/model_epoch_6"),
        "raw_test": (reference / "edge_scores/test/model_epoch_6", args.run_root / "edge_scores/test/model_epoch_6"),
        "calibrated_val": (reference / "calibration/model_epoch_6/validation_calibrated.csv", args.run_root / "evaluation_results/calibration/model_epoch_6/validation_calibrated.csv"),
        "calibrated_test": (reference / "calibration/model_epoch_6/test_calibrated.csv", args.run_root / "evaluation_results/calibration/model_epoch_6/test_calibrated.csv"),
        "node_predictions": (reference / "calibration/model_epoch_6/node_predictions.csv", args.run_root / "evaluation_results/calibration/model_epoch_6/node_predictions.csv"),
    }
    comparisons = {name: compare_csv(a,b) for name,(a,b) in pairs.items()}
    old_metrics = json.loads((reference / "node_scores/metrics.json").read_text())
    new_metrics = json.loads((args.run_root / "node_scores/metrics.json").read_text())
    from mstc.calibration_runner import load_event_records_from_csv
    from mstc.aggregation import NodeScoreAggregator
    from mstc.thresholding import compute_node_threshold
    import yaml
    cfg = yaml.safe_load((args.run_root / "config_resolved.windows.yml").read_text())
    records = load_event_records_from_csv(pairs["calibrated_val"][1])
    aggregation = cfg["node_aggregation"]
    nodes = NodeScoreAggregator().aggregate_events(records, method=aggregation["method"], topk=aggregation["topk"],
                include_dst=aggregation["include_dst"], score_field=aggregation["score_field"])
    threshold = compute_node_threshold(nodes, method=cfg["node_threshold"]["method"], quantile=cfg["node_threshold"]["quantile"])
    linux_threshold = json.loads((reference / "all_checkpoints_derived_threshold.json").read_text())
    same_batch = run_result["batch_sizes"]["Linux"] == run_result["batch_sizes"]["Windows_full"]
    result = {
        "status": "PASS_COMPARISON_EXECUTED", "csv": comparisons,
        "metrics": compare_metrics(old_metrics,new_metrics),
        "threshold": compare_metrics({"threshold":linux_threshold["threshold"]},{"threshold":threshold}),
        "num_validation_nodes": len(nodes), "same_full_batch_size": same_batch,
        "strict_numerical_equivalence": "PASS" if same_batch and all(v["classification"] == "BITWISE_EQUAL" for v in comparisons.values())
            and all(v["equal"] for v in compare_metrics(old_metrics,new_metrics).values())
            and struct.pack(">d", threshold) == struct.pack(">d",linux_threshold["threshold"]) else "NOT_ESTABLISHED",
        "selected_checkpoint": "model_epoch_6 (preselected on Linux; no six-candidate Windows selection claim)",
        "checkpoint_sha256": run_result["checkpoint_sha256"],
        "windows_versions": run_result["versions"], "linux_environment": run_result["linux_environment"],
        "qualification": "Current server source files transferred with SHA256 verification; the original G3 protected snapshot did not list each window individually. CSV equality does not prove raw intermediate GPU tensor equality.",
    }
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"classifications":{k:v["classification"] for k,v in comparisons.items()},
                      "strict_numerical_equivalence":result["strict_numerical_equivalence"],
                      "threshold":result["threshold"]},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
