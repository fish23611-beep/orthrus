"""
TimeGapStatistics: per-training-session time-gap bucket statistics.

Tracks the distribution of inter-event times for each node during training,
computes bucket boundaries from training quantiles, and provides batch-level
transform that computes per-event time-gap targets without leaking future
information into past computations.
"""

import json
import math
import os
from typing import Dict, List, Literal, Optional

import torch
from torch_geometric.data import TemporalData


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
    scale_boundaries : list[float]
        Three scale boundaries in log1p(seconds) space.
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

        self.time_bucket_boundaries: List[float] = []
        self.scale_boundaries: List[float] = []
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

        self.time_bucket_boundaries = self._quantile_boundaries(
            all_intervals, self.time_bucket_quantiles
        )
        self.scale_boundaries = self._quantile_boundaries(
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
        Computes time-gap targets for every event in ``g`` using the
        provided historical state.

        Causal requirement: all targets are computed using the
        ``last_seen_per_node`` snapshot that existed *before* processing
        the current batch.  ``last_seen_per_node`` is updated *after*
        all targets have been computed, using the maximum timestamp seen
        per node in the current batch.

        Parameters
        ----------
        g
            TemporalData window for the current batch.  Must have ``t``,
            ``src``, ``dst``, and ``global_event_index`` attributes.
        last_seen_per_node
            Dict mapping node IDs to their last-seen timestamps in
            nanoseconds.  Values of ``-1`` (or absence) indicate
            no prior history.

        Returns
        -------
        tuple
            ``(src_target, dst_target, updated_last_seen)`` where each
            target is a ``torch.long`` tensor of shape ``[E]``.
        """
        last_seen = dict(last_seen_per_node)

        E = len(g)
        src_target = torch.full((E,), NO_HISTORY, dtype=torch.long)
        dst_target = torch.full((E,), NO_HISTORY, dtype=torch.long)

        for i in range(E):
            src_i = int(g.src[i].item())
            dst_i = int(g.dst[i].item())
            t_i_ns = int(g.t[i].item())

            last_src = last_seen.get(src_i)
            if last_src is not None and last_src != -1:
                delta_ns = t_i_ns - last_src
                if delta_ns < 0:
                    raise ValueError(
                        f"Negative time delta detected: "
                        f"current t={t_i_ns}, last_seen[{src_i}]={last_src}."
                    )
                delta_seconds = delta_ns / 1_000_000_000.0
                src_target[i] = self.transform(delta_seconds)

            last_dst = last_seen.get(dst_i)
            if last_dst is not None and last_dst != -1:
                delta_ns = t_i_ns - last_dst
                if delta_ns < 0:
                    raise ValueError(
                        f"Negative time delta detected: "
                        f"current t={t_i_ns}, last_seen[{dst_i}]={last_dst}."
                    )
                delta_seconds = delta_ns / 1_000_000_000.0
                dst_target[i] = self.transform(delta_seconds)

        for i in range(E):
            src_i = int(g.src[i].item())
            dst_i = int(g.dst[i].item())
            t_i_ns = int(g.t[i].item())

            prev_src = last_seen.get(src_i, -1)
            prev_dst = last_seen.get(dst_i, -1)

            current_src = t_i_ns if prev_src == -1 else max(prev_src, t_i_ns)
            current_dst = t_i_ns if prev_dst == -1 else max(prev_dst, t_i_ns)

            last_seen[src_i] = current_src
            last_seen[dst_i] = current_dst

        return src_target, dst_target, last_seen

    # ------------------------------------------------------------------
    # bucket helpers
    # ------------------------------------------------------------------
    def _bucket(self, z: float) -> int:
        """
        Maps a log1p(seconds) value to a bucket index using the fitted
        time-bucket boundaries.
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
        Returns ``len(quantiles)`` boundaries in log1p-space from a list
        of finite interval values.
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
            boundaries.append(math.log1p(val))
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
            "unit": "seconds",
            "transform": "log1p",
            "scale_quantiles": self.scale_quantiles,
            "scale_boundaries": self.scale_boundaries,
            "time_bucket_quantiles": self.time_bucket_quantiles,
            "time_bucket_boundaries": self.time_bucket_boundaries,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "TimeGapStatistics":
        """
        Loads fitted statistics from a JSON file.

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
        obj = cls(
            time_bucket_quantiles=payload.get("time_bucket_quantiles"),
            scale_quantiles=payload.get("scale_quantiles"),
        )
        obj.time_bucket_boundaries = list(payload["time_bucket_boundaries"])
        obj.scale_boundaries = list(payload["scale_boundaries"])
        return obj
