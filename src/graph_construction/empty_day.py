"""Verified empty-day artifact contract for graph_construction.

A legal "empty day" (e.g. THEIA_E3 day 2 where the upstream database has zero
raw events for that date) is a *first-class* preprocessing outcome. It is
persisted as an atomic JSON marker inside ``graph_<day>/`` and never as a
fake/empty NetworkX graph.

The marker is authoritative only when:

* the schema is exactly ``EMPTY_DAY_MARKER_SCHEMA``,
* the dataset / graph_name / day fields match the surrounding config,
* the ``raw_event_count`` is exactly ``0`` (raw-empty is a *necessary*
  condition for an "empty" declaration; raw > 0 with supported == 0 must
  surface as a data/relation-mapping problem, not be silently papered over),
* the file was written atomically (``.tmp`` replaced into place).

Anything else (malformed JSON, mismatched identity, raw_event_count != 0,
``.tmp`` left over) is rejected at validation time so it cannot silently
mark a corrupt artifact set as complete.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping


EMPTY_DAY_MARKER_FILENAME = ".preprocess_empty_day.json"
EMPTY_DAY_MARKER_SCHEMA = "orthrus.empty_day_marker/v1"
EMPTY_DAY_REASON_NO_RAW_EVENTS = "no_raw_events"


def marker_path(graph_day_dir: str | os.PathLike[str]) -> str:
    """Return the canonical empty-day marker path inside ``graph_<day>/``."""
    return os.path.join(str(graph_day_dir), EMPTY_DAY_MARKER_FILENAME)


def build_marker_payload(
    *,
    dataset: str,
    graph_name: str,
    day: int,
    date_start: str,
    date_stop: str,
    start_ns: int,
    end_ns: int,
    raw_event_count: int,
    reason: str = EMPTY_DAY_REASON_NO_RAW_EVENTS,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """Construct a serialized empty-day marker payload (still JSON-ready)."""
    if raw_event_count != 0:
        raise ValueError(
            "Empty-day marker requires raw_event_count == 0; got "
            f"{raw_event_count!r}. A non-zero raw event count with zero "
            "supported relations is a data/relation mapping issue and must "
            "not be silently classified as empty."
        )
    if reason != EMPTY_DAY_REASON_NO_RAW_EVENTS:
        raise ValueError(
            f"Unsupported empty-day reason: {reason!r}. Only "
            f"{EMPTY_DAY_REASON_NO_RAW_EVENTS!r} is currently accepted."
        )

    payload: dict[str, Any] = {
        "schema": EMPTY_DAY_MARKER_SCHEMA,
        "dataset": str(dataset),
        "graph_name": str(graph_name),
        "day": int(day),
        "date_start": str(date_start),
        "date_stop": str(date_stop),
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "raw_event_count": int(raw_event_count),
        "reason": str(reason),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_marker(graph_day_dir: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    """Atomically write an empty-day marker inside ``graph_<day>/``.

    The marker is written to ``<marker>.tmp`` first, then ``os.replace``'d
    into place. A crash between the create and the rename leaves the
    existing state (real graph files, prior marker, or absence) untouched,
    so the marker is never half-written.
    """
    os.makedirs(str(graph_day_dir), exist_ok=True)
    final_path = marker_path(graph_day_dir)
    tmp_path = final_path + ".tmp"
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def read_marker(graph_day_dir: str | os.PathLike[str]) -> dict | None:
    """Return the parsed marker payload or ``None`` if not present or malformed.

    A malformed JSON file is treated as "no marker" so validators and
    corpus collectors do not propagate JSON errors as runtime crashes;
    the marker is then rejected by ``validate_marker`` semantics.
    """
    path = marker_path(graph_day_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _expected_identity(*, dataset: str, graph_name: str, day: int) -> dict:
    return {
        "schema": EMPTY_DAY_MARKER_SCHEMA,
        "dataset": str(dataset),
        "graph_name": str(graph_name),
        "day": int(day),
    }


def validate_marker(
    payload: Any,
    *,
    dataset: str,
    graph_name: str,
    day: int,
) -> bool:
    """Return ``True`` only for a structurally valid, identity-matching,
    ``raw_event_count == 0`` marker. Anything else is rejected loudly."""
    if not isinstance(payload, dict):
        return False
    expected = _expected_identity(dataset=dataset, graph_name=graph_name, day=day)
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            return False
    if payload.get("raw_event_count") != 0:
        return False
    reason = payload.get("reason")
    if reason != EMPTY_DAY_REASON_NO_RAW_EVENTS:
        return False
    for required in ("date_start", "date_stop", "start_ns", "end_ns"):
        if required not in payload:
            return False
    if not isinstance(payload["start_ns"], int) or not isinstance(payload["end_ns"], int):
        return False
    return True


def is_verified_empty_day(
    graph_day_dir: str | os.PathLike[str],
    *,
    dataset: str,
    graph_name: str,
    day: int,
) -> bool:
    """Strict boolean check used by validators and corpus collection."""
    payload = read_marker(graph_day_dir)
    if payload is None:
        return False
    return validate_marker(
        payload, dataset=dataset, graph_name=graph_name, day=day
    )


def clear_marker(graph_day_dir: str | os.PathLike[str]) -> None:
    """Remove the empty-day marker if present (used when real data is found
    later, so a stale marker cannot mask subsequent normal-day data)."""
    path = marker_path(graph_day_dir)
    try:
        os.remove(path)
    except FileNotFoundError:
        return
