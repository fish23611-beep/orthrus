"""Shared checkpoint-selection semantics and manifest persistence.

This module intentionally has no dependency on detection.evaluation or model
construction so both testing and evaluation can reuse one formal objective.
"""

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd


CHECKPOINT_POLICIES = ("all_checkpoints", "val_select_then_test_best")


def validation_objective_field(cfg):
    """Return the CSV field required by the formal validation-only selector."""
    variant = getattr(getattr(cfg, "model", None), "variant", None)
    if variant == "orthrus_baseline":
        return "loss_type"

    decoder = cfg.detection.gnn_training.decoder
    type_enabled = bool(decoder.predict_edge_type.enabled)
    time_enabled = bool(decoder.time_gap.enabled)
    if type_enabled and time_enabled:
        return "score_raw"
    if type_enabled:
        return "loss_type"
    if time_enabled:
        return "loss_time"
    raise ValueError(
        "min_val_objective requires at least one enabled prediction task"
    )


def compute_validation_objective(val_tw_path, cfg):
    """Return the arithmetic mean of the formal validation event-score field."""
    field = validation_objective_field(cfg)
    values = []
    if not os.path.isdir(val_tw_path):
        raise FileNotFoundError(
            f"Validation edge-score directory not found: {val_tw_path}"
        )

    csv_files = [
        name for name in sorted(os.listdir(val_tw_path)) if name.endswith(".csv")
    ]
    if not csv_files:
        raise ValueError(f"No validation CSV files found under {val_tw_path}")

    for name in csv_files:
        path = os.path.join(val_tw_path, name)
        df = pd.read_csv(path)
        if field not in df.columns:
            raise ValueError(
                f"Validation objective field {field!r} is missing from {path}. "
                f"Available columns: {list(df.columns)}"
            )
        arr = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
        if arr.size and not np.isfinite(arr).all():
            raise ValueError(
                f"Validation objective field {field!r} contains non-finite "
                f"values in {path}"
            )
        values.extend(arr.tolist())

    if not values:
        raise ValueError(
            f"Validation objective field {field!r} contains no values"
        )
    return float(np.mean(values)), field


def select_checkpoint_from_validation(per_epoch_objective):
    """Return the minimum finite objective with insertion-order tie-breaking."""
    if not per_epoch_objective:
        raise ValueError("No checkpoint validation objectives were provided")
    invalid = {
        epoch: value
        for epoch, value in per_epoch_objective.items()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    }
    if invalid:
        raise ValueError(f"Invalid checkpoint validation objectives: {invalid}")
    return min(per_epoch_objective, key=per_epoch_objective.get)


def selection_manifest_path(cfg):
    """Return the run-scoped manifest path, never a shared preprocessing path."""
    run_dir = getattr(cfg, "_run_dir", None)
    if not isinstance(run_dir, (str, os.PathLike)) or not os.fspath(run_dir):
        raise ValueError(
            "val_select_then_test_best requires a real run-scoped cfg._run_dir"
        )
    return os.path.join(os.fspath(run_dir), "checkpoint_selection.json")


def write_selection_manifest(path, payload):
    """Atomically persist selection before test split access is permitted."""
    directory = os.path.dirname(os.fspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def read_selection_manifest(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint selection manifest: {path}")
    return payload


def validate_selection_manifest(payload, cfg):
    """Validate policy, method, field, objectives, and selected minimum."""
    expected = {
        "checkpoint_policy": "val_select_then_test_best",
        "model_selection_method": "min_val_objective",
        "val_objective_field": validation_objective_field(cfg),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Invalid checkpoint selection manifest {key}: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )

    objectives = payload.get("per_epoch_validation_objective")
    if not isinstance(objectives, dict):
        raise ValueError(
            "Invalid checkpoint selection manifest: "
            "per_epoch_validation_objective must be a mapping"
        )
    selected = select_checkpoint_from_validation(objectives)
    if payload.get("selected_epoch") != selected:
        raise ValueError(
            "Invalid checkpoint selection manifest selected_epoch: "
            f"expected {selected!r}, got {payload.get('selected_epoch')!r}"
        )
    return selected
