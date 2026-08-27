"""Post-processing runner for hierarchical empirical event-score calibration.

This module is responsible for:
- Loading validation and test raw event records
- Fitting a HierarchicalRelationCalibrator on validation data only
- Applying LOO transformation to validation events
- Applying fixed reference calibration to test events
- Persisting the fitted calibrator, calibrated records, and summary statistics

All mathematical logic is delegated to HierarchicalRelationCalibrator in calibration.py.
"""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np

def _import_calibrator():
    """Import the calibrator without creating duplicate module identities."""
    import sys
    from pathlib import Path

    try:
        from .calibration import HierarchicalRelationCalibrator
        return HierarchicalRelationCalibrator
    except (ModuleNotFoundError, ImportError):
        pass

    # Support entry points that expose the repository root as a package.
    try:
        from src.mstc.calibration import HierarchicalRelationCalibrator
        return HierarchicalRelationCalibrator
    except (ModuleNotFoundError, ImportError):
        pass

    # Try direct file import for testing
    CALIBRATION_PATH = Path(__file__).parent / "calibration.py"
    if CALIBRATION_PATH.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("calibration_dynamic", CALIBRATION_PATH)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["calibration_dynamic"] = module
            spec.loader.exec_module(module)
            return module.HierarchicalRelationCalibrator

    # Fallback: try from the calling test module
    for key in list(sys.modules.keys()):
        if "calibration_for_tests" in key:
            return sys.modules[key].HierarchicalRelationCalibrator

    raise ImportError("Cannot import HierarchicalRelationCalibrator")


HierarchicalRelationCalibrator = _import_calibrator()


CalibrationLevel = Literal["triplet", "type_pair", "global"]


def _validate_records(records: list[dict[str, Any]], split_name: str) -> None:
    """Validate that records contain required fields and have stable identities."""
    if not records:
        raise ValueError(f"{split_name} records cannot be empty")

    # Check required fields exist
    required_fields = ["score_raw", "src_type", "dst_type"]
    for record in records:
        for field in required_fields:
            if field not in record:
                raise KeyError(f"{split_name} record missing required field: {field}")

        # Check edge_type_index or edge_type
        if "edge_type_index" not in record and "edge_type" not in record:
            raise KeyError(
                f"{split_name} record missing edge_type_index or edge_type field"
            )

        # Validate score_raw is finite
        try:
            score = float(record["score_raw"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{split_name} record has invalid score_raw: {record['score_raw']}"
            ) from exc
        if not np.isfinite(score):
            raise ValueError(
                f"{split_name} record has non-finite score_raw: {score}"
            )

    # Check event identity field (prefer event_index, fallback to index in list)
    has_event_index = "event_index" in records[0]
    if has_event_index:
        event_ids = []
        for record in records:
            try:
                event_ids.append(int(record["event_index"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{split_name} record has non-integer event_index: {record['event_index']}"
                ) from exc

        # Check for duplicates
        if len(event_ids) != len(set(event_ids)):
            seen = set()
            for eid in event_ids:
                if eid in seen:
                    raise ValueError(
                        f"{split_name} records contain duplicate event_index: {eid}"
                    )
                seen.add(eid)


def _sort_records_by_event_index(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort records stably by event_index if present, preserving original order otherwise."""
    if not records:
        return records

    if "event_index" in records[0]:
        return sorted(records, key=lambda r: int(r["event_index"]))
    return records


def load_event_records_from_csv(
    csv_path: Path | str,
) -> list[dict[str, Any]]:
    """Load event records from a single CSV file.

    Returns a list of dictionaries with all CSV columns preserved.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            record: dict[str, Any] = {}
            for key, value in row.items():
                # Skip empty values
                if value == "" or value is None:
                    record[key] = None
                    continue

                # Try numeric conversion
                try:
                    if "." in value:
                        record[key] = float(value)
                    else:
                        record[key] = int(value)
                except (ValueError, TypeError):
                    record[key] = value

            records.append(record)
    return records


def load_event_records_from_csv_directory(
    csv_dir: Path | str,
) -> list[dict[str, Any]]:
    """Load and concatenate all CSV files from a directory.

    Files are processed in sorted order to ensure deterministic results.
    Each file's records are sorted by event_index internally.
    """
    csv_dir = Path(csv_dir)
    all_records = []

    # Process files in sorted order for determinism
    csv_files = sorted(csv_dir.glob("*.csv"))
    for csv_path in csv_files:
        records = load_event_records_from_csv(csv_path)
        records = _sort_records_by_event_index(records)
        all_records.extend(records)

    return all_records


def _compute_quantiles(scores: np.ndarray) -> dict[str, float]:
    """Compute summary quantiles for a sorted score array."""
    if scores.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}

    return {
        "min": float(scores[0]),
        "p50": float(np.quantile(scores, 0.50)),
        "p90": float(np.quantile(scores, 0.90)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(scores[-1]),
    }


def _triplet_key_to_str(key: tuple[int, int, int]) -> str:
    """Convert triplet key to a stable string representation."""
    src, edge, dst = key
    return f"src={src}|edge={edge}|dst={dst}"


def _type_pair_key_to_str(key: tuple[int, int]) -> str:
    """Convert type_pair key to a stable string representation."""
    src, dst = key
    return f"src={src}|dst={dst}"


def run_calibration(
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    output_dir: Path | str,
    *,
    min_triplet_samples: int = 100,
    min_type_pair_samples: int = 200,
    epsilon: float = 1e-12,
    method: str = "hierarchical_relation",
) -> dict[str, Any]:
    """Run calibration on validation and test records.

    Parameters
    ----------
    validation_records : list[dict[str, Any]]
        List of validation event records. Must contain score_raw, src_type, dst_type,
        and either edge_type_index or edge_type. Records will be sorted by event_index
        if present.
    test_records : list[dict[str, Any]]
        List of test event records with the same required fields.
    output_dir : Path | str
        Directory to save all output artifacts.
    min_triplet_samples : int
        Minimum samples required for triplet-level calibration.
    min_type_pair_samples : int
        Minimum samples required for type_pair-level calibration.
    epsilon : float
        Epsilon value for log transformation to prevent log(0).
    method : str
        Reference-selection ablation: global_empirical, relation_triplet, or
        hierarchical_relation.

    Returns
    -------
    dict[str, Any]
        Summary of the calibration run including event counts, reference statistics,
        and calibration level distributions.

    Raises
    ------
    ValueError
        If validation_records is empty or contains invalid data.
    KeyError
        If required fields are missing from records.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate validation records (will raise if empty)
    _validate_records(validation_records, "validation")

    # Sort validation records by event_index for stable LOO
    validation_records = _sort_records_by_event_index(list(validation_records))

    # Validate test records if non-empty
    if test_records:
        test_records = _sort_records_by_event_index(list(test_records))
        _validate_records(test_records, "test")

    # Fit calibrator on validation data only
    calibrator = HierarchicalRelationCalibrator(
        min_triplet_samples=min_triplet_samples,
        min_type_pair_samples=min_type_pair_samples,
        epsilon=epsilon,
        method=method,
    )
    calibrator.fit(validation_records)

    # Transform validation with LOO
    val_calibrated = calibrator.transform_val_with_loo(validation_records)

    # Transform test with fixed reference (no LOO)
    test_calibrated = calibrator.calibrate(test_records) if test_records else []

    # Save calibrator
    calibrator_path = output_dir / "calibrator.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)

    # Save calibrated records
    val_output_path = output_dir / "validation_calibrated.csv"
    test_output_path = output_dir / "test_calibrated.csv"

    _save_calibrated_csv(val_calibrated, val_output_path)
    _save_calibrated_csv(test_calibrated, test_output_path)

    # Build summary
    summary = _build_summary(
        calibrator,
        validation_records,
        test_records,
        val_calibrated,
        test_calibrated,
        min_triplet_samples=min_triplet_samples,
        min_type_pair_samples=min_type_pair_samples,
        epsilon=epsilon,
    )

    # Save summary
    summary_path = output_dir / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _save_calibrated_csv(
    records: list[dict[str, Any]], output_path: Path
) -> None:
    """Save calibrated records to CSV, preserving all original fields."""
    if not records:
        # Create empty file with header
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
        return

    # Get all keys from first record for consistent ordering
    fieldnames = list(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            # Convert values to strings for CSV
            row = {}
            for key, value in record.items():
                if value is None:
                    row[key] = ""
                elif isinstance(value, (int, float)):
                    row[key] = str(value)
                else:
                    row[key] = str(value)
            writer.writerow(row)


def _build_summary(
    calibrator: HierarchicalRelationCalibrator,
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    val_calibrated: list[dict[str, Any]],
    test_calibrated: list[dict[str, Any]],
    *,
    min_triplet_samples: int,
    min_type_pair_samples: int,
    epsilon: float,
) -> dict[str, Any]:
    """Build the calibration summary dictionary."""
    num_val = len(validation_records)
    num_test = len(test_records)

    # Reference statistics
    global_scores = calibrator.global_scores
    triplet_scores = calibrator.triplet_scores
    type_pair_scores = calibrator.type_pair_scores

    # Count calibration levels in validation (LOO)
    val_levels = Counter(r["calibration_level"] for r in val_calibrated)
    val_triplet_count = val_levels.get("triplet", 0)
    val_type_pair_count = val_levels.get("type_pair", 0)
    val_global_count = val_levels.get("global", 0)

    # Count calibration levels in test
    test_levels = Counter(r["calibration_level"] for r in test_calibrated)
    test_triplet_count = test_levels.get("triplet", 0)
    test_type_pair_count = test_levels.get("type_pair", 0)
    test_global_count = test_levels.get("global", 0)

    summary: dict[str, Any] = {
        "calibration_method": calibrator.method,
        "calibration_parameters": {
            "min_triplet_samples": min_triplet_samples,
            "min_type_pair_samples": min_type_pair_samples,
            "epsilon": epsilon,
        },
        "event_counts": {
            "num_validation_events": num_val,
            "num_test_events": num_test,
        },
        "reference": {
            "global": {
                "sample_count": int(global_scores.size),
                "quantiles": _compute_quantiles(global_scores),
            },
            "triplet_groups": {},
            "type_pair_groups": {},
        },
        "validation_loo": {
            "triplet_count": val_triplet_count,
            "type_pair_count": val_type_pair_count,
            "global_count": val_global_count,
            "triplet_ratio": (
                float(val_triplet_count) / num_val if num_val > 0 else 0.0
            ),
            "type_pair_ratio": (
                float(val_type_pair_count) / num_val if num_val > 0 else 0.0
            ),
            "global_ratio": (
                float(val_global_count) / num_val if num_val > 0 else 0.0
            ),
        },
        "test": {
            "triplet_count": test_triplet_count,
            "type_pair_count": test_type_pair_count,
            "global_count": test_global_count,
            "triplet_ratio": (
                float(test_triplet_count) / num_test if num_test > 0 else 0.0
            ),
            "type_pair_ratio": (
                float(test_type_pair_count) / num_test if num_test > 0 else 0.0
            ),
            "global_ratio": (
                float(test_global_count) / num_test if num_test > 0 else 0.0
            ),
        },
    }

    # Triplet group statistics
    for key, scores in sorted(triplet_scores.items()):
        key_str = _triplet_key_to_str(key)
        summary["reference"]["triplet_groups"][key_str] = {
            "sample_count": int(scores.size),
            "quantiles": _compute_quantiles(scores),
        }

    # Type pair group statistics
    for key, scores in sorted(type_pair_scores.items()):
        key_str = _type_pair_key_to_str(key)
        summary["reference"]["type_pair_groups"][key_str] = {
            "sample_count": int(scores.size),
            "quantiles": _compute_quantiles(scores),
        }

    return summary
