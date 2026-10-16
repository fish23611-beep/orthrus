#!/usr/bin/env python3
"""Choose a deterministic representative contiguous window block."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "batch_index",
    "epoch",
    "window_index",
    "batch_events",
    "history_valid_slots",
    "total_seconds",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_recorded_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=False)


def _validate_inventory_result(path: Path, source: Path, dataset: str) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"inventory result cannot be parsed: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError("inventory result must be a JSON object")
    evidence = result.get("execution_evidence")
    if (
        result.get("result_schema_version") != 3
        or result.get("status") != "completed"
        or result.get("returncode") != 0
        or result.get("run_protocol") != "selection_inventory_diagnostic"
        or result.get("selection_inventory") is not True
        or result.get("formal_baseline_eligible") is not False
        or result.get("statistics_published") is not False
        or result.get("diagnostic_statistics_published") is not True
        or result.get("dataset") != dataset
        or result.get("requested_profile_mode") != "off"
        or result.get("requested_basic_timing") != "on"
        or not isinstance(evidence, dict)
        or evidence.get("valid") is not True
        or evidence.get("measurement_source") != "basic_timing"
    ):
        raise ValueError("inventory result is not a completed trusted selection inventory")
    recorded_source = evidence.get("measurement_path")
    if not isinstance(recorded_source, str) or not recorded_source:
        raise ValueError("inventory result lacks a measurement path")
    if _resolve_recorded_path(recorded_source, relative_to=path.parent) != source:
        raise ValueError("inventory result measurement path does not match batches CSV")
    if evidence.get("measurement_sha256") != _sha256(source):
        raise ValueError("inventory result measurement SHA256 does not match batches CSV")
    return result


def _load_rows(source: Path, *, expected_epochs: int) -> list[dict]:
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"inventory CSV missing required fields: {missing}")
            raw_rows = list(reader)
    except OSError as exc:
        raise ValueError(f"inventory CSV cannot be read: {source}") from exc
    if not raw_rows:
        raise ValueError("inventory CSV has no records")

    rows = []
    for position, row in enumerate(raw_rows, start=2):
        try:
            parsed = {
                "batch_index": int(row["batch_index"]),
                "epoch": int(row["epoch"]),
                "window_index": int(row["window_index"]),
                "batch_events": int(row["batch_events"]),
                "history_valid_slots": int(row["history_valid_slots"]),
                "total_seconds": float(row["total_seconds"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"inventory CSV has an invalid record at row {position}") from exc
        if (
            parsed["batch_index"] <= 0
            or parsed["epoch"] <= 0
            or parsed["window_index"] < 0
            or parsed["batch_events"] <= 0
            or parsed["history_valid_slots"] < 0
            or not math.isfinite(parsed["total_seconds"])
            or parsed["total_seconds"] <= 0
        ):
            raise ValueError(f"inventory CSV has an invalid record at row {position}")
        rows.append(parsed)

    batch_indices = [row["batch_index"] for row in rows]
    if len(batch_indices) != len(set(batch_indices)):
        raise ValueError("inventory CSV contains duplicate batch_index records")
    if sorted(batch_indices) != list(range(1, len(rows) + 1)):
        raise ValueError("inventory CSV batch_index sequence is incomplete")
    observed_epochs = sorted({row["epoch"] for row in rows})
    if observed_epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("inventory CSV does not cover every expected training epoch")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("THEIA_E3", "THEIA_E5"))
    parser.add_argument("--batches-csv", required=True)
    parser.add_argument("--inventory-result", required=True)
    parser.add_argument("--window-count", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--expected-epochs", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.window_count <= 0:
        raise ValueError("window-count must be positive")
    if args.expected_epochs <= 0:
        raise ValueError("expected-epochs must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.expected_epochs:
        raise ValueError("warmup-epochs must be within the training epoch range")

    source = Path(args.batches_csv).expanduser().resolve(strict=True)
    inventory_result_path = Path(args.inventory_result).expanduser().resolve(strict=True)
    inventory_result = _validate_inventory_result(
        inventory_result_path, source, args.dataset
    )
    if inventory_result.get("warmup_epochs") != args.warmup_epochs:
        raise ValueError("inventory result warmup rule does not match planner request")
    rows = _load_rows(source, expected_epochs=args.expected_epochs)
    if inventory_result["execution_evidence"].get("batch_row_count") != len(rows):
        raise ValueError("inventory result Batch row count does not match batches CSV")

    eligible = [row for row in rows if row["epoch"] > args.warmup_epochs]
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["window_index"]].append(row)
    indices = sorted(grouped)
    if len(indices) < args.window_count:
        raise ValueError("inventory has fewer windows than requested")
    if indices != list(range(0, indices[-1] + 1)):
        raise ValueError("inventory window indices must be contiguous from zero")

    eligible_epochs = list(range(args.warmup_epochs + 1, args.expected_epochs + 1))
    expected_epoch_set = set(eligible_epochs)
    for index in indices:
        observed_epochs = {row["epoch"] for row in grouped[index]}
        if observed_epochs != expected_epoch_set:
            raise ValueError(
                "inventory does not cover every eligible epoch for window "
                f"{index}"
            )

    features = {}
    for index in indices:
        per_epoch = defaultdict(list)
        for row in grouped[index]:
            per_epoch[row["epoch"]].append(row)
        event_totals = [
            sum(row["batch_events"] for row in per_epoch[epoch])
            for epoch in eligible_epochs
        ]
        history_maxima = [
            max(row["history_valid_slots"] for row in per_epoch[epoch])
            for epoch in eligible_epochs
        ]
        features[index] = {
            "events": statistics.median(event_totals),
            "history": statistics.median(history_maxima),
            "batch_seconds": statistics.median(
                row["total_seconds"] for row in grouped[index]
            ),
        }

    centers = {
        key: statistics.median(value[key] for value in features.values())
        for key in ("events", "history", "batch_seconds")
    }
    scales = {
        key: max(
            1e-12,
            statistics.median(
                abs(value[key] - centers[key]) for value in features.values()
            ),
        )
        for key in centers
    }
    candidates = []
    index_set = set(indices)
    for start in indices:
        block = list(range(start, start + args.window_count))
        if not set(block) <= index_set:
            continue
        means = {
            key: sum(features[index][key] for index in block) / len(block)
            for key in centers
        }
        score = sum(abs(means[key] - centers[key]) / scales[key] for key in centers)
        candidates.append((score, start, means))
    if not candidates:
        raise ValueError("no contiguous candidate block")

    score, start, means = min(
        candidates, key=lambda candidate: (candidate[0], candidate[1])
    )
    by_time = sorted(indices, key=lambda index: (features[index]["batch_seconds"], index))
    result = {
        "schema_version": 2,
        "dataset": args.dataset,
        "selection_rule": (
            "closest-to-median contiguous block over per-epoch events/history "
            "and post-warmup batch time; ties choose the lowest start index"
        ),
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_sha256_scope": "content_sha256_of_batches_csv_file",
        "inventory_result": str(inventory_result_path),
        "inventory_result_sha256": _sha256(inventory_result_path),
        "inventory_result_sha256_scope": "content_sha256_of_benchmark_result_json_file",
        "warmup_epochs_excluded": args.warmup_epochs,
        "eligible_epochs": eligible_epochs,
        "inventory_window_start": indices[0],
        "inventory_window_end_exclusive": indices[-1] + 1,
        "inventory_window_count": len(indices),
        "inventory_window_indices_contiguous": True,
        "selected_start": start,
        "selected_end_exclusive": start + args.window_count,
        "window_count": args.window_count,
        "score": score,
        "selected_feature_means": means,
        "fast_windows": by_time[: min(3, len(by_time))],
        "slow_windows": by_time[-min(3, len(by_time)) :],
        "window_features": features,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
