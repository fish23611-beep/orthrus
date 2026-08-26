"""
TimeGapStatistics: per-training-session time-gap bucket statistics.

Tracks the distribution of inter-event times for each node during training,
computes bucket boundaries from training quantiles, and provides batch-level
transform that computes per-event time-gap targets without leaking future
information into past computations.

Key semantic contracts:
- scale_boundaries_seconds: raw seconds values for MultiScaleNeighborLoader
- time_bucket_boundaries_log1p: log1p(seconds) values for TimeGap classification
- transform_batch: per-event exact target with causal supervision state update
"""

import json
import math
import os
from typing import Dict, List, Literal, Optional

import torch
from torch_geometric.data import TemporalData


# Scale boundaries are stored as raw seconds
# Time bucket boundaries are stored as log1p(seconds)


NO_HISTORY = 0
VERY_SHORT = 1
SHORT = 2
MEDIUM = 3
LONG = 4
VERY_LONG = 5

BUCKET_NAMES = {
    NO_HISTORY: "NO_HISTORY",
    VERY_SHORT: "VERY_SHORT",
    SHORT: "SHORT",
    MEDIUM: "MEDIUM",
    LONG: "LONG",
    VERY_LONG: "VERY_LONG",
}

SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
SPLIT_ID_TO_NAME = {v: k for k, v in SPLIT_NAME_TO_ID.items()}


class TimeGapStatistics:
    """
    Computes and stores per-node time-gap bucket boundaries from training data.

    ``fit`` must be called once with the training windows before any
    ``transform_batch`` call.  The fitted statistics can be saved to and
    loaded from a JSON file.

    Attributes
    ----------
    time_bucket_quantiles : list[float]
        Quantiles used to define the five time-gap buckets.
    scale_quantiles : list[float]
        Quantiles used to define the three scale boundaries (Q50 / Q90 / Q99).
    time_bucket_boundaries : list[float]
        Five bucket boundaries in log1p(seconds) space.
    scale_boundaries_seconds : list[float]
        Three scale boundaries in seconds space (for MultiScaleNeighborLoader).
    _last_seen_ns : dict[int, int]
        Internal per-node timestamp cache.  Not serialised.
    """

    def __init__(
        self,
        time_bucket_quantiles: Optional[List[float]] = None,
        scale_quantiles: Optional[List[float]] = None,
    ) -> None:
        if time_bucket_quantiles is None:
            time_bucket_quantiles = [0.2, 0.4, 0.6, 0.8]
        if scale_quantiles is None:
            scale_quantiles = [0.5, 0.9, 0.99]
        self.time_bucket_quantiles = list(time_bucket_quantiles)
        self.scale_quantiles = list(scale_quantiles)

        self.time_bucket_boundaries: List[float] = []  # log1p(seconds)
        self.scale_boundaries_seconds: List[float] = []  # seconds
        self._last_seen_ns: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_data_list: List[TemporalData]) -> "TimeGapStatistics":
        """
        Computes bucket boundaries from training data only.

        Parameters
        ----------
        train_data_list
            List of training TemporalData windows in temporal order.
            Each window must have ``t`` (nanoseconds), ``src``, ``dst``,
            and ``global_event_index`` attributes.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``train_data_list`` is empty or contains no finite intervals.
        """
        if not train_data_list:
            raise ValueError(
                "train_data_list is empty; cannot fit TimeGapStatistics."
            )

        last_seen_ns: Dict[int, int] = {}
        finite_intervals_src: List[float] = []
        finite_intervals_dst: List[float] = []

        for g in train_data_list:
            t = g.t
            if not (t[1:] >= t[:-1]).all():
                raise ValueError(
                    "Timestamps within a window must be non-decreasing."
                )
            if len(g) > 1:
                prev_t = t[:-1]
                curr_t = t[1:]
                if not (curr_t >= prev_t).all():
                    raise ValueError(
                        "Timestamps within a window must be non-decreasing."
                    )

            for i in range(len(g)):
                src_i = int(g.src[i].item())
                dst_i = int(g.dst[i].item())
                t_i_ns = int(t[i].item())

                last_src = last_seen_ns.get(src_i)
                if last_src is not None:
                    delta_ns = t_i_ns - last_src
                    if delta_ns < 0:
                        raise ValueError(
                            f"Negative time delta detected: "
                            f"current t={t_i_ns}, last_seen[{src_i}]={last_src}."
                        )
                    delta_seconds = delta_ns / 1_000_000_000.0
                    finite_intervals_src.append(delta_seconds)

                last_dst = last_seen_ns.get(dst_i)
                if last_dst is not None:
                    delta_ns = t_i_ns - last_dst
                    if delta_ns < 0:
                        raise ValueError(
                            f"Negative time delta detected: "
                            f"current t={t_i_ns}, last_seen[{dst_i}]={last_dst}."
                        )
                    delta_seconds = delta_ns / 1_000_000_000.0
                    finite_intervals_dst.append(delta_seconds)

                # Self-loop double-counting note:
                # When src == dst, the same event contributes one sample to
                # finite_intervals_src AND one to finite_intervals_dst.
                # This is intentional: each target (src_time_gap / dst_time_gap)
                # is an independent prediction task, and a self-loop provides
                # two finite-interval training samples — one for each target.
                # This is NOT equivalent to updating state twice; the state
                # itself is updated once per event (see lines below).

                last_seen_ns[src_i] = t_i_ns
                last_seen_ns[dst_i] = t_i_ns

        if not finite_intervals_src and not finite_intervals_dst:
            raise ValueError(
                "No finite intervals found in training data; cannot fit "
                "TimeGapStatistics."
            )

        all_intervals = finite_intervals_src + finite_intervals_dst

        # Time bucket boundaries: computed from log1p(seconds) values
        all_intervals_log1p = [math.log1p(x) for x in all_intervals]
        self.time_bucket_boundaries = self._quantile_boundaries(
            all_intervals_log1p, self.time_bucket_quantiles
        )

        # Scale boundaries: computed from raw seconds values (for MultiScaleNeighborLoader)
        self.scale_boundaries_seconds = self._quantile_boundaries(
            all_intervals, self.scale_quantiles
        )

        self._last_seen_ns = None
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, delta_seconds: Optional[float]) -> int:
        """
        Maps a non-negative time gap in seconds to a bucket index.

        Parameters
        ----------
        delta_seconds
            Elapsed time in seconds, or ``None`` / ``NO_HISTORY`` sentinel
            to indicate no prior history.

        Returns
        -------
        int
            Bucket index in [0, 5].

        Raises
        ------
        ValueError
            If ``delta_seconds`` is negative.
        """
        if delta_seconds is None or delta_seconds == NO_HISTORY:
            return NO_HISTORY
        if not isinstance(delta_seconds, (int, float)):
            raise TypeError(
                f"delta_seconds must be a number or None; got {type(delta_seconds).__name__}"
            )
        if delta_seconds < 0:
            raise ValueError(
                f"delta_seconds must be non-negative; got {delta_seconds}."
            )

        z = math.log1p(delta_seconds)
        return self._bucket(z)

    def transform_batch(
        self, g: TemporalData, last_seen_per_node: Dict[int, int]
    ) -> tuple:
        """
        Computes event-level time-gap targets for every event in ``g``.

        Per the final specification: events are processed in order of
        (timestamp, global_event_index) for stable ordering. Each event:
        1. Reads current src/dst target_last_seen
        2. Computes src/dst target
        3. Updates target_last_seen for involved nodes
        4. Proceeds to next event

        This is the supervision state update - it does NOT affect encoder
        HistoryStore which maintains separate causal query-before-insert.

        Parameters
        ----------
        g
            TemporalData for the current batch. Timestamps must be
            non-decreasing.
        last_seen_per_node
            Dict mapping node IDs to their last-seen timestamps in
            nanoseconds. Values of ``-1`` (or absence) indicate no history.

        Returns
        -------
        tuple
            ``(src_target, dst_target, updated_last_seen)`` where each
            target is a ``torch.long`` tensor of shape ``[E]``.
            Targets are in original event order.
        """
        event_count = len(g)

        if not (g.t[1:] >= g.t[:-1]).all():
            raise ValueError("Timestamps within a batch must be non-decreasing.")

        # Create local mutable copy of target state for per-event update
        working_last_seen = dict(last_seen_per_node)

        # Build stable sort order: (timestamp, global_event_index)
        timestamps = g.t.numpy()
        if hasattr(g, "global_event_index"):
            event_indices = g.global_event_index.numpy()
        else:
            event_indices = torch.arange(event_count).numpy()

        # Stable sort indices by (timestamp, global_event_index)
        sort_indices = sorted(
            range(event_count),
            key=lambda i: (int(timestamps[i]), int(event_indices[i]))
        )

        # Create inverse mapping: original_index -> sorted_position
        inverse_sort = [0] * event_count
        for sorted_pos, orig_idx in enumerate(sort_indices):
            inverse_sort[orig_idx] = sorted_pos

        # Initialize target tensors
        src_target = torch.full((event_count,), NO_HISTORY, dtype=torch.long)
        dst_target = torch.full((event_count,), NO_HISTORY, dtype=torch.long)

        # Process events in stable sorted order
        # Note: for self-loop (src == dst), both src and dst targets use the
        # same pre-event snapshot, then state is updated once
        for orig_idx in sort_indices:
            src_i = int(g.src[orig_idx].item())
            dst_i = int(g.dst[orig_idx].item())
            t_i_ns = int(g.t[orig_idx].item())

            # Compute src target
            last_src = working_last_seen.get(src_i)
            if last_src is not None and last_src != -1:
                delta_ns = t_i_ns - last_src
                if delta_ns < 0:
                    raise ValueError(
                        f"Negative time delta detected: "
                        f"current t={t_i_ns}, last_seen[{src_i}]={last_src}."
                    )
                src_target[orig_idx] = self._bucket(
                    math.log1p(delta_ns / 1_000_000_000.0)
                )

            # Compute dst target
            last_dst = working_last_seen.get(dst_i)
            if last_dst is not None and last_dst != -1:
                delta_ns = t_i_ns - last_dst
                if delta_ns < 0:
                    raise ValueError(
                        f"Negative time delta detected: "
                        f"current t={t_i_ns}, last_seen[{dst_i}]={last_dst}."
                    )
                dst_target[orig_idx] = self._bucket(
                    math.log1p(delta_ns / 1_000_000_000.0)
                )

            # Update working state: both src and dst get current event time
            # For self-loop (src == dst), this is a single update
            working_last_seen[src_i] = t_i_ns
            if dst_i != src_i:
                working_last_seen[dst_i] = t_i_ns

        # Return post-batch state (from working copy)
        return src_target, dst_target, working_last_seen

    # ------------------------------------------------------------------
    # bucket helpers
    # ------------------------------------------------------------------
    def _bucket(self, z: float) -> int:
        """
        Maps a log1p(seconds) value to a time bucket index using the fitted
        time-bucket boundaries. Boundaries are already in log1p(seconds) space.
        """
        if z <= self.time_bucket_boundaries[0]:
            return VERY_SHORT
        elif z <= self.time_bucket_boundaries[1]:
            return SHORT
        elif z <= self.time_bucket_boundaries[2]:
            return MEDIUM
        elif z <= self.time_bucket_boundaries[3]:
            return LONG
        else:
            return VERY_LONG

    @staticmethod
    def _quantile_boundaries(
        values: List[float], quantiles: List[float]
    ) -> List[float]:
        """
        Returns ``len(quantiles)`` quantile boundaries from a list of values.
        Values are assumed to be in the desired output space already
        (e.g., raw seconds for scale_boundaries_seconds, or log1p(seconds)
        for time_bucket_boundaries_log1p).
        """
        if not values:
            raise ValueError("values list is empty.")
        t = torch.tensor(values, dtype=torch.float64)
        sorted_t, _ = torch.sort(t)
        n = len(sorted_t)
        boundaries = []
        for q in quantiles:
            idx_float = q * (n - 1)
            idx_low = int(idx_float)
            frac = idx_float - idx_low
            if idx_low >= n - 1:
                val = float(sorted_t[-1].item())
            else:
                val = float(sorted_t[idx_low].item()) + frac * float(
                    (sorted_t[idx_low + 1] - sorted_t[idx_low]).item()
                )
            boundaries.append(val)
        return boundaries

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Saves fitted statistics to a JSON file.

        Parameters
        ----------
        path
            Path to the output JSON file.  Parent directories are created
            if they do not exist.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "raw_unit": "seconds",
            "time_bucket_space": "log1p_seconds",
            "scale_quantiles": self.scale_quantiles,
            "scale_boundaries_seconds": self.scale_boundaries_seconds,
            "time_bucket_quantiles": self.time_bucket_quantiles,
            "time_bucket_boundaries_log1p": self.time_bucket_boundaries,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "TimeGapStatistics":
        """
        Loads fitted statistics from a JSON file.

        Supports both the new schema (scale_boundaries_seconds) and legacy schema
        (scale_boundaries in log1p space).

        Parameters
        ----------
        path
            Path to the input JSON file.

        Returns
        -------
        TimeGapStatistics
            A new instance with all boundaries restored.
        """
        with open(path) as fh:
            payload = json.load(fh)

        time_bucket_quantiles = payload.get("time_bucket_quantiles")
        scale_quantiles = payload.get("scale_quantiles")

        # Support both new field names and legacy field names
        if "scale_boundaries_seconds" in payload:
            # New schema: boundaries already in seconds
            scale_boundaries = payload["scale_boundaries_seconds"]
        elif "scale_boundaries" in payload:
            # Legacy schema: boundaries in log1p space - convert to seconds
            # This maintains backward compatibility with old artifacts
            scale_boundaries = [math.expm1(b) for b in payload["scale_boundaries"]]
        else:
            raise ValueError(
                f"time_statistics.json at {path} is missing scale_boundaries. "
                "This may indicate a corrupted or legacy artifact."
            )

        if "time_bucket_boundaries_log1p" in payload:
            time_bucket_boundaries = payload["time_bucket_boundaries_log1p"]
        elif "time_bucket_boundaries" in payload:
            time_bucket_boundaries = payload["time_bucket_boundaries"]
        else:
            raise ValueError(
                f"time_statistics.json at {path} is missing time_bucket_boundaries."
            )

        obj = cls(
            time_bucket_quantiles=time_bucket_quantiles,
            scale_quantiles=scale_quantiles,
        )
        obj.time_bucket_boundaries = list(time_bucket_boundaries)
        obj.scale_boundaries_seconds = list(scale_boundaries)
        return obj
