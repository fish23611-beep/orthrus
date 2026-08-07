"""Hierarchical empirical calibration for event anomaly scores.

The calibrator is deliberately independent of model, graph, and file-system
state.  ``fit`` accepts normal validation-event records only; subsequent
methods only query the fixed reference distributions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np


CalibrationLevel = Literal["triplet", "type_pair", "global"]
CalibrationMethod = Literal[
    "global_empirical", "relation_triplet", "hierarchical_relation"
]
_CALIBRATION_METHODS = frozenset(
    {"global_empirical", "relation_triplet", "hierarchical_relation"}
)


@dataclass(frozen=True)
class _Event:
    """Validated fields required for one event-calibration query."""

    score: float
    triplet_key: tuple[int, int, int]
    type_pair_key: tuple[int, int]


class HierarchicalRelationCalibrator:
    """Calibrate anomaly scores against normal validation-event references.

    Records must provide ``score_raw``, ``src_type``, ``dst_type``, and either
    ``edge_type_index`` (the project-standard field) or ``edge_type``.  Higher
    raw scores are more anomalous.  Validation LOO transformation must receive
    the same records, in the same stable order, that were passed to ``fit``.
    """

    def __init__(
        self,
        min_triplet_samples: int = 100,
        min_type_pair_samples: int = 200,
        epsilon: float = 1.0e-12,
        method: CalibrationMethod = "hierarchical_relation",
    ) -> None:
        self.min_triplet_samples = self._validate_min_samples(
            min_triplet_samples, "min_triplet_samples"
        )
        self.min_type_pair_samples = self._validate_min_samples(
            min_type_pair_samples, "min_type_pair_samples"
        )
        self.epsilon = self._validate_epsilon(epsilon)
        self.method = self._validate_method(method)
        self.triplet_scores: dict[tuple[int, int, int], np.ndarray] = {}
        self.type_pair_scores: dict[tuple[int, int], np.ndarray] = {}
        self.global_scores = np.array([], dtype=np.float64)
        self._fit_events: tuple[_Event, ...] = ()
        self._fitted = False

    @staticmethod
    def _validate_min_samples(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _validate_epsilon(value: float) -> float:
        try:
            epsilon = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("epsilon must be a finite float in (0, 1]") from exc
        if not np.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
            raise ValueError("epsilon must be a finite float in (0, 1]")
        return epsilon


    @staticmethod
    def _validate_method(method: str) -> CalibrationMethod:
        method_str = str(method).strip().lower()
        if method_str not in _CALIBRATION_METHODS:
            raise ValueError(
                "method must be one of: global_empirical, relation_triplet, "
                "hierarchical_relation"
            )
        return method_str  # type: ignore[return-value]


    @staticmethod
    def _required(record: Mapping[str, Any], field: str) -> Any:
        if field not in record:
            raise KeyError(f"event record is missing required field: {field}")
        return record[field]

    @classmethod
    def _normalise_event(cls, record: Mapping[str, Any]) -> _Event:
        if not isinstance(record, Mapping):
            raise TypeError("each event record must be a mapping")
        try:
            score = float(cls._required(record, "score_raw"))
        except (TypeError, ValueError) as exc:
            raise ValueError("score_raw must be a finite float") from exc
        if not np.isfinite(score):
            raise ValueError("score_raw must be a finite float")

        edge_field = "edge_type_index" if "edge_type_index" in record else "edge_type"
        values = (
            cls._required(record, "src_type"),
            cls._required(record, edge_field),
            cls._required(record, "dst_type"),
        )
        try:
            src_type, edge_type, dst_type = (int(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("src_type, edge type, and dst_type must be integer-like") from exc
        return _Event(
            score=score,
            triplet_key=(src_type, edge_type, dst_type),
            type_pair_key=(src_type, dst_type),
        )

    @classmethod
    def _normalise_events(cls, records: Iterable[Mapping[str, Any]]) -> list[_Event]:
        try:
            return [cls._normalise_event(record) for record in records]
        except TypeError as exc:
            if str(exc) == "'NoneType' object is not iterable":
                raise TypeError("event records must be an iterable of mappings") from exc
            raise

    def fit(self, val_event_records: Iterable[Mapping[str, Any]]) -> "HierarchicalRelationCalibrator":
        """Fit sorted reference distributions from normal validation records.

        The caller is responsible for supplying only normal validation events.
        This method does not inspect labels or any test-event data.
        """
        events = self._normalise_events(val_event_records)
        if not events:
            raise ValueError("cannot fit calibrator with empty validation events")

        triplets: defaultdict[tuple[int, int, int], list[float]] = defaultdict(list)
        type_pairs: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        for event in events:
            triplets[event.triplet_key].append(event.score)
            type_pairs[event.type_pair_key].append(event.score)

        self.triplet_scores = {
            key: np.sort(np.asarray(scores, dtype=np.float64))
            for key, scores in triplets.items()
        }
        self.type_pair_scores = {
            key: np.sort(np.asarray(scores, dtype=np.float64))
            for key, scores in type_pairs.items()
        }
        self.global_scores = np.sort(
            np.asarray([event.score for event in events], dtype=np.float64)
        )
        self._fit_events = tuple(events)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("calibrator must be fitted with normal validation events first")

    @staticmethod
    def _count_greater_equal(sorted_scores: np.ndarray, score: float) -> int:
        """Count reference scores >= score in O(log n), including ties."""
        left = np.searchsorted(sorted_scores, score, side="left")
        return int(sorted_scores.size - left)

    def _select_reference(
        self, event: _Event, *, leave_one_out: bool
    ) -> tuple[np.ndarray, CalibrationLevel]:
        if self.method == "global_empirical":
            return self.global_scores, "global"

        adjustment = 1 if leave_one_out else 0
        triplet = self.triplet_scores.get(event.triplet_key)
        if triplet is not None and triplet.size - adjustment >= self.min_triplet_samples:
            return triplet, "triplet"

        if self.method == "relation_triplet":
            return self.global_scores, "global"

        type_pair = self.type_pair_scores.get(event.type_pair_key)
        if type_pair is not None and type_pair.size - adjustment >= self.min_type_pair_samples:
            return type_pair, "type_pair"
        return self.global_scores, "global"

    def _empirical_p(
        self, sorted_scores: np.ndarray, score: float, *, leave_one_out: bool
    ) -> float:
        count_ge = self._count_greater_equal(sorted_scores, score)
        n_reference = int(sorted_scores.size)
        if leave_one_out:
            # The current validation event is known by its stable fit position,
            # so subtract exactly one occurrence even when scores are repeated.
            count_ge -= 1
            n_reference -= 1
        if count_ge < 0 or n_reference < 0:
            raise RuntimeError("invalid leave-one-out reference state")
        return (1.0 + count_ge) / (n_reference + 1.0)

    def _transform_event(self, event: _Event, *, leave_one_out: bool) -> tuple[float, CalibrationLevel]:
        reference, level = self._select_reference(event, leave_one_out=leave_one_out)
        p_value = self._empirical_p(reference, event.score, leave_one_out=leave_one_out)
        return p_value, level

    @staticmethod
    def _with_calibration(
        record: Mapping[str, Any], p_value: float, level: CalibrationLevel, epsilon: float
    ) -> dict[str, Any]:
        result = dict(record)
        result["empirical_p"] = p_value
        result["score_calibrated"] = float(-np.log(max(p_value, epsilon)))
        result["calibration_level"] = level
        return result

    def calibrate(self, test_event_records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Transform test events using fixed validation references only."""
        self._require_fitted()
        results: list[dict[str, Any]] = []
        for record in test_event_records:
            event = self._normalise_event(record)
            p_value, level = self._transform_event(event, leave_one_out=False)
            results.append(self._with_calibration(record, p_value, level, self.epsilon))
        return results

    def transform_val_with_loo(
        self, val_event_records: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Transform fitted validation records while excluding each event once.

        The stable sequence position is the event identity used for LOO.  This
        prevents equal scores from causing every matching reference to be
        removed.
        """
        self._require_fitted()
        records = list(val_event_records)
        events = self._normalise_events(records)
        if tuple(events) != self._fit_events:
            raise ValueError(
                "LOO records must match the records passed to fit in the same order"
            )
        results: list[dict[str, Any]] = []
        for record, event in zip(records, events):
            p_value, level = self._transform_event(event, leave_one_out=True)
            results.append(self._with_calibration(record, p_value, level, self.epsilon))
        return results
