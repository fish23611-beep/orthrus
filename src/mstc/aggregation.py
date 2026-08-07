"""Pure node-score aggregation for calibrated event anomaly scores."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from numbers import Integral
from typing import Any, Literal

import numpy as np


AggregationMethod = Literal["mean", "max", "topk_mean", "topk_sum"]


class NodeScoreAggregator:
    """Aggregate calibrated event scores into continuous node anomaly scores.

    This class does not read labels, splits, thresholds, or files.  It first
    assigns event scores to nodes, then independently reduces each node's
    score list.
    """

    _METHODS = frozenset({"mean", "max", "topk_mean", "topk_sum"})

    @staticmethod
    def _validate_method(method: str) -> AggregationMethod:
        if method not in NodeScoreAggregator._METHODS:
            raise ValueError(
                "method must be one of: mean, max, topk_mean, topk_sum"
            )
        return method  # type: ignore[return-value]

    @staticmethod
    def _validate_topk(topk: int) -> int:
        if isinstance(topk, bool) or not isinstance(topk, Integral) or topk <= 0:
            raise ValueError("topk must be a positive integer")
        return int(topk)

    @staticmethod
    def _validate_score(value: Any, field: str) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a finite float") from exc
        if not np.isfinite(score):
            raise ValueError(f"{field} must be a finite float")
        return score

    @staticmethod
    def _require_field(record: Mapping[str, Any], field: str) -> Any:
        if field not in record:
            raise KeyError(f"event record is missing required field: {field}")
        return record[field]

    @staticmethod
    def _validate_node_id(node_id: Any, field: str) -> Hashable:
        try:
            hash(node_id)
        except TypeError as exc:
            raise TypeError(f"{field} must be hashable") from exc
        return node_id

    def build_node_to_event_scores(
        self,
        event_records: Iterable[Mapping[str, Any]],
        *,
        include_dst: bool = True,
        score_field: str = "score_calibrated",
        src_field: str = "srcnode",
        dst_field: str = "dstnode",
    ) -> dict[Hashable, list[float]]:
        """Assign each event score to its source and optionally destination.

        When ``include_dst`` is true and source equals destination, the score is
        appended twice.  This preserves the existing node_evaluation behavior.
        """
        if not isinstance(include_dst, bool):
            raise TypeError("include_dst must be a bool")

        node_to_scores: defaultdict[Hashable, list[float]] = defaultdict(list)
        for record in event_records:
            if not isinstance(record, Mapping):
                raise TypeError("each event record must be a mapping")
            score = self._validate_score(
                self._require_field(record, score_field), score_field
            )
            src_node = self._validate_node_id(
                self._require_field(record, src_field), src_field
            )
            node_to_scores[src_node].append(score)
            if include_dst:
                dst_node = self._validate_node_id(
                    self._require_field(record, dst_field), dst_field
                )
                node_to_scores[dst_node].append(score)
        return dict(node_to_scores)

    def aggregate_node_scores(
        self,
        node_to_event_scores: Mapping[Hashable, Iterable[float]],
        *,
        method: AggregationMethod = "topk_mean",
        topk: int = 5,
    ) -> dict[Hashable, float]:
        """Reduce independently supplied node-to-event-score mappings."""
        method = self._validate_method(method)
        topk = self._validate_topk(topk)
        results: dict[Hashable, float] = {}

        for node_id, scores_iterable in node_to_event_scores.items():
            node_id = self._validate_node_id(node_id, "node_id")
            scores = [self._validate_score(score, "event score") for score in scores_iterable]
            if not scores:
                raise ValueError("cannot aggregate an empty event-score list")

            if method == "mean":
                aggregate = float(np.mean(scores))
            elif method == "max":
                aggregate = float(max(scores))
            else:
                highest = sorted(scores, reverse=True)[:topk]
                aggregate = (
                    float(np.mean(highest))
                    if method == "topk_mean"
                    else float(sum(highest))
                )
            if not np.isfinite(aggregate):
                raise RuntimeError("aggregation produced a non-finite node score")
            results[node_id] = aggregate
        return results

    def aggregate_events(
        self,
        event_records: Iterable[Mapping[str, Any]],
        *,
        method: AggregationMethod = "topk_mean",
        topk: int = 5,
        include_dst: bool = True,
        score_field: str = "score_calibrated",
        src_field: str = "srcnode",
        dst_field: str = "dstnode",
    ) -> dict[Hashable, float]:
        """Build node score lists from events and aggregate them."""
        node_to_scores = self.build_node_to_event_scores(
            event_records,
            include_dst=include_dst,
            score_field=score_field,
            src_field=src_field,
            dst_field=dst_field,
        )
        return self.aggregate_node_scores(
            node_to_scores,
            method=method,
            topk=topk,
        )

    def aggregate(
        self,
        event_records: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> dict[Hashable, float]:
        """Compatibility alias for :meth:`aggregate_events`."""
        return self.aggregate_events(event_records, **kwargs)
