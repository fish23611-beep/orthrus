#!/usr/bin/env python3
"""Choose a deterministic representative contiguous window block."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("THEIA_E3", "THEIA_E5"))
    parser.add_argument("--batches-csv", required=True)
    parser.add_argument("--window-count", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.window_count <= 0:
        raise ValueError("window-count must be positive")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs must be non-negative")

    source = Path(args.batches_csv)
    rows = list(csv.DictReader(source.open()))
    eligible = [
        row
        for row in rows
        if row.get("epoch") not in ("", None)
        and int(row["epoch"]) > args.warmup_epochs
    ]
    if not eligible:
        raise ValueError("inventory has no rows after warm-up exclusion")

    grouped = defaultdict(list)
    for row in eligible:
        grouped[int(row["window_index"])].append(row)
    indices = sorted(grouped)
    if len(indices) < args.window_count:
        raise ValueError("inventory has fewer windows than requested")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("inventory window indices must be contiguous")

    eligible_epochs = sorted({int(row["epoch"]) for row in eligible})
    expected_epochs = set(eligible_epochs)
    for index in indices:
        observed_epochs = {int(row["epoch"]) for row in grouped[index]}
        if observed_epochs != expected_epochs:
            raise ValueError(
                "inventory does not cover every eligible epoch for window "
                f"{index}"
            )

    features = {}
    for index in indices:
        per_epoch = defaultdict(list)
        for row in grouped[index]:
            per_epoch[int(row["epoch"])].append(row)
        event_totals = [
            sum(int(row.get("batch_events", 0) or 0) for row in epoch_rows)
            for epoch_rows in per_epoch.values()
        ]
        history_maxima = [
            max(
                [int(float(row.get("history_valid_slots", 0) or 0)) for row in epoch_rows]
                or [0]
            )
            for epoch_rows in per_epoch.values()
        ]
        batch_seconds = [
            float(row["total_seconds"])
            for row in grouped[index]
            if row.get("total_seconds") not in ("", None)
        ]
        features[index] = {
            "events": median(event_totals),
            "history": median(history_maxima),
            "batch_seconds": median(batch_seconds),
        }

    centers = {
        key: median([value[key] for value in features.values()])
        for key in ("events", "history", "batch_seconds")
    }
    scales = {
        key: max(
            1e-12,
            median([abs(value[key] - centers[key]) for value in features.values()]),
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

    score, start, means = min(candidates)
    by_time = sorted(indices, key=lambda index: (features[index]["batch_seconds"], index))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "schema_version": 2,
        "dataset": args.dataset,
        "selection_rule": (
            "closest-to-median contiguous block over per-epoch events/history "
            "and post-warmup batch time"
        ),
        "source": str(source.resolve()),
        "source_sha256": digest,
        "source_sha256_scope": "content_sha256_of_batches_csv_file",
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
