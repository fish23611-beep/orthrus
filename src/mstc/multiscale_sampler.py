"""
MultiScaleNeighborLoader: temporal multi-scale neighbor sampling.

Groups historical neighbors into short / medium / long scales based on
their time distance from the current reference timestamp.

Final semantic contract:
- short:  0 < delta <= tau_short_ns
- medium: tau_short_ns < delta <= tau_medium_ns
- long:   delta > tau_medium_ns (NO upper bound)

Q99 (tau_extreme) is for diagnostics only, not a hard cutoff for long scale.

Separates query (read-only, before current batch) from insert
(post-encode, after current batch) to maintain causal ordering.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Sequence

import torch
from torch import Tensor

from .history_store import HistoryStore


def _recency_order(timestamps: Tensor, event_ids: Tensor) -> Tensor:
    """Order by timestamp descending, then event id descending, without overflow."""
    event_order = torch.argsort(event_ids, descending=True, stable=True)
    timestamp_order = torch.argsort(
        timestamps[event_order], descending=True, stable=True
    )
    return event_order[timestamp_order]


# ------------------------------------------------------------------
# Boundary conversion utility (for legacy compatibility)
# ------------------------------------------------------------------
def seconds_boundaries_to_ns(
    boundaries: Sequence[float],
) -> tuple[int, int, Optional[int]]:
    """
    Convert three seconds boundaries to nanoseconds.

    Parameters
    ----------
    boundaries
        Three values in seconds space, e.g.
        from TimeGapStatistics.scale_boundaries_seconds.

    Returns
    -------
    tuple[int, int, Optional[int]]
        (tau_short_ns, tau_medium_ns, tau_extreme_ns) as integers in nanoseconds.
        tau_extreme_ns is Q99 for diagnostics only.

    Raises
    ------
    ValueError
        If boundaries does not contain exactly 3 elements,
        values are negative, or boundaries are not non-decreasing.
    """
    if len(boundaries) != 3:
        raise ValueError(
            f"boundaries must contain exactly 3 values, got {len(boundaries)}"
        )

    b0, b1, b2 = float(boundaries[0]), float(boundaries[1]), float(boundaries[2])

    if b0 < 0 or b1 < 0 or b2 < 0:
        raise ValueError(
            f"boundaries must be non-negative, got [{b0}, {b1}, {b2}]"
        )

    if not (b0 <= b1 <= b2):
        raise ValueError(
            f"boundaries must be non-decreasing, got [{b0}, {b1}, {b2}]"
        )

    tau_short_ns = round(b0 * 1_000_000_000)
    tau_medium_ns = round(b1 * 1_000_000_000)
    tau_extreme_ns = round(b2 * 1_000_000_000)  # Q99 for diagnostics

    return tau_short_ns, tau_medium_ns, tau_extreme_ns


# ------------------------------------------------------------------
# MultiScaleNeighborLoader
# ------------------------------------------------------------------
class MultiScaleNeighborLoader:
    """
    Multi-scale temporal neighbor sampler.

    Queries historical neighbors and groups them into three temporal scales:

    - short:  0 < delta <= tau_short_ns
    - medium: tau_short_ns < delta <= tau_medium_ns
    - long:   delta > tau_medium_ns  (NO upper bound)

    where delta = reference_time_ns - historical_timestamp_ns.

    Q99 (tau_extreme_ns) is stored for diagnostics but NOT used as a hard
    cutoff for the long scale.

    This class maintains causal ordering: query reads history BEFORE the
    current batch is inserted. The caller must explicitly call insert()
    after encoding.

    Parameters
    ----------
    history_store
        HistoryStore instance containing per-node history.
    tau_short_ns
        Upper bound (inclusive) for the short scale in nanoseconds.
    tau_medium_ns
        Upper bound (inclusive) for the medium scale in nanoseconds.
        Must satisfy tau_short_ns <= tau_medium_ns.
    short_budget
        Maximum number of neighbors to return per node for the short scale.
        Default: 8.
    medium_budget
        Maximum number of neighbors to return per node for the medium scale.
        Default: 8.
    long_budget
        Maximum number of neighbors to return per node for the long scale.
        Default: 8.
    """

    def __init__(
        self,
        history_store: HistoryStore,
        tau_short_ns: int,
        tau_medium_ns: int,
        short_budget: int = 8,
        medium_budget: int = 8,
        long_budget: int = 8,
        tau_max_ns: Optional[int] = None,
    ) -> None:
        if tau_short_ns < 0:
            raise ValueError(
                f"tau_short_ns must be non-negative, got {tau_short_ns}"
            )
        if tau_medium_ns < tau_short_ns:
            raise ValueError(
                f"tau_medium_ns ({tau_medium_ns}) must be >= "
                f"tau_short_ns ({tau_short_ns})"
            )
        if short_budget < 0:
            raise ValueError(
                f"short_budget must be non-negative, got {short_budget}"
            )
        if medium_budget < 0:
            raise ValueError(
                f"medium_budget must be non-negative, got {medium_budget}"
            )
        if long_budget < 0:
            raise ValueError(
                f"long_budget must be non-negative, got {long_budget}"
            )

        self.history_store = history_store
        self.tau_short_ns = int(tau_short_ns)
        self.tau_medium_ns = int(tau_medium_ns)
        # tau_max_ns is Q99 for diagnostics only - NOT used as hard cutoff for long scale
        self.tau_max_ns = int(tau_max_ns) if tau_max_ns is not None else None
        self.short_budget = int(short_budget)
        self.medium_budget = int(medium_budget)
        self.long_budget = int(long_budget)

    # ------------------------------------------------------------------
    # __call__ (query)
    # ------------------------------------------------------------------
    def __call__(
        self,
        src: Tensor,
        dst: Tensor,
        timestamp_ns: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Query historical neighbors BEFORE inserting the current batch.

        Parameters
        ----------
        src
            Source node IDs for current batch, shape [batch_size].
        dst
            Destination node IDs for current batch, shape [batch_size].
        timestamp_ns
            Event timestamps in nanoseconds, shape [batch_size].

        Returns
        -------
        dict
            Dictionary containing:

            query_nodes: Tensor, shape [Q]
                Unique node IDs involved in this batch (src ∪ dst).
            reference_time_ns: Tensor, shape [Q]
                For each query node: minimum timestamp of that node in current batch.
            src_query_index: Tensor, shape [batch_size]
                Maps each src event to its index in query_nodes.
            dst_query_index: Tensor, shape [batch_size]
                Maps each dst event to its index in query_nodes.

            short_neighbor_id: Tensor, shape [Q, short_budget], int64, -1 = invalid
            short_event_id: Tensor, shape [Q, short_budget], int64, -1 = invalid
            short_timestamp_ns: Tensor, shape [Q, short_budget], int64, -1 = invalid
            short_direction: Tensor, shape [Q, short_budget], int8, -1 = invalid
            short_mask: Tensor, shape [Q, short_budget], bool

            medium_*: same layout with medium_budget
            long_*: same layout with long_budget
        """
        batch_size = src.shape[0]
        device = src.device

        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        t_cpu = timestamp_ns.cpu().long()

        all_nodes_list = torch.cat([src_cpu, dst_cpu]).unique().tolist()
        Q = len(all_nodes_list)
        query_nodes = torch.tensor(all_nodes_list, dtype=torch.long)

        ref_time_per_node = {}
        for i in range(batch_size):
            node = int(src_cpu[i])
            t = int(t_cpu[i])
            if node not in ref_time_per_node:
                ref_time_per_node[node] = t
            else:
                ref_time_per_node[node] = min(ref_time_per_node[node], t)

            node = int(dst_cpu[i])
            if node not in ref_time_per_node:
                ref_time_per_node[node] = t
            else:
                ref_time_per_node[node] = min(ref_time_per_node[node], t)

        reference_time_ns = torch.zeros(Q, dtype=torch.int64)
        for idx, node in enumerate(all_nodes_list):
            reference_time_ns[idx] = ref_time_per_node[node]

        history = self.history_store.query(query_nodes, reference_time_ns)

        src_query_index = torch.zeros(batch_size, dtype=torch.long)
        node_to_qidx = {node: idx for idx, node in enumerate(all_nodes_list)}
        for i in range(batch_size):
            src_query_index[i] = node_to_qidx[int(src_cpu[i])]

        dst_query_index = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            dst_query_index[i] = node_to_qidx[int(dst_cpu[i])]

        result: Dict[str, Tensor] = {
            "query_nodes": query_nodes.to(device),
            "reference_time_ns": reference_time_ns.to(device),
            "src_query_index": src_query_index.to(device),
            "dst_query_index": dst_query_index.to(device),
        }

        for scale_name, tau_low, tau_high, budget, has_upper in [
            ("short", 0, self.tau_short_ns, self.short_budget, True),
            ("medium", self.tau_short_ns, self.tau_medium_ns, self.medium_budget, True),
            ("long", self.tau_medium_ns, None, self.long_budget, False),  # No upper bound
        ]:
            neighbor_ids = history["neighbor_id"]
            event_ids = history["event_id"]
            timestamps = history["timestamp_ns"]
            directions = history["direction"]

            scale_neighbor = torch.full(
                (Q, budget), fill_value=-1, dtype=torch.int64, device=device
            )
            scale_event_id = torch.full(
                (Q, budget), fill_value=-1, dtype=torch.int64, device=device
            )
            scale_timestamp = torch.full(
                (Q, budget), fill_value=-1, dtype=torch.int64, device=device
            )
            scale_direction = torch.full(
                (Q, budget), fill_value=-1, dtype=torch.int8, device=device
            )
            scale_mask = torch.zeros(
                (Q, budget), dtype=torch.bool, device=device
            )

            ref_cpu = reference_time_ns.cpu()
            ts_cpu = timestamps.cpu()
            ev_cpu = event_ids.cpu()
            nb_cpu = neighbor_ids.cpu()
            dir_cpu = directions.cpu()

            for q_idx in range(Q):
                # First filter out invalid timestamps (marked as -1)
                valid_ts_mask = ts_cpu[q_idx] >= 0
                delta = ref_cpu[q_idx] - ts_cpu[q_idx]
                # Short: 0 < delta <= tau_short
                # Medium: tau_short < delta <= tau_medium
                # Long: delta > tau_medium (no upper bound)
                if scale_name == "short":
                    in_range = valid_ts_mask & (delta > 0) & (delta <= tau_high)
                elif scale_name == "medium":
                    in_range = valid_ts_mask & (delta > tau_low) & (delta <= tau_high)
                else:  # long
                    in_range = valid_ts_mask & (delta > tau_low)

                valid_ts = ts_cpu[q_idx][in_range]
                valid_ev = ev_cpu[q_idx][in_range]
                valid_nb = nb_cpu[q_idx][in_range]
                valid_dir = dir_cpu[q_idx][in_range]

                sorted_idx = _recency_order(valid_ts, valid_ev)
                sorted_ts = valid_ts[sorted_idx]
                sorted_ev = valid_ev[sorted_idx]
                sorted_nb = valid_nb[sorted_idx]
                sorted_dir = valid_dir[sorted_idx]

                num_valid = min(len(sorted_ts), budget)
                if num_valid > 0:
                    scale_neighbor[q_idx, :num_valid] = sorted_nb[:num_valid].to(device)
                    scale_event_id[q_idx, :num_valid] = sorted_ev[:num_valid].to(device)
                    scale_timestamp[q_idx, :num_valid] = sorted_ts[:num_valid].to(device)
                    scale_direction[q_idx, :num_valid] = sorted_dir[:num_valid].to(device)
                    scale_mask[q_idx, :num_valid] = True

            result[f"{scale_name}_neighbor_id"] = scale_neighbor
            result[f"{scale_name}_event_id"] = scale_event_id
            result[f"{scale_name}_timestamp_ns"] = scale_timestamp
            result[f"{scale_name}_direction"] = scale_direction
            result[f"{scale_name}_mask"] = scale_mask

        return result

    # ------------------------------------------------------------------
    # insert
    # ------------------------------------------------------------------
    def insert(
        self,
        src: Tensor,
        dst: Tensor,
        timestamp_ns: Tensor,
        global_event_index: Tensor,
    ) -> None:
        """
        Insert the current batch into history AFTER encoding.

        Parameters
        ----------
        src
            Source node IDs, shape [batch_size].
        dst
            Destination node IDs, shape [batch_size].
        timestamp_ns
            Event timestamps in nanoseconds, shape [batch_size].
        global_event_index
            Global event indices, shape [batch_size].
            These correspond to full_data positions.
        """
        batch_size = src.shape[0]
        direction = torch.zeros(batch_size, dtype=torch.int8)

        self.history_store.insert(
            src=src,
            dst=dst,
            event_id=global_event_index,
            timestamp_ns=timestamp_ns,
            direction=direction,
        )

    # ------------------------------------------------------------------
    # reset_state
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Clear all history in the underlying store."""
        self.history_store.reset_state()

    # ------------------------------------------------------------------
    # history_state_dict / load_history_state_dict
    # ------------------------------------------------------------------
    def history_state_dict(self) -> Dict[str, Tensor]:
        """
        Return a CPU-cloned copy of the history state.

        Returns
        -------
        dict
            The history store's state_dict.
        """
        return self.history_store.state_dict()

    def load_history_state_dict(self, state: Dict[str, Tensor]) -> None:
        """
        Restore history state from a previously saved state_dict.

        Parameters
        ----------
        state
            Dictionary produced by history_state_dict().
        """
        self.history_store.load_state_dict(state)


# ------------------------------------------------------------------
# SingleWindowNeighborLoader
# ------------------------------------------------------------------
class SingleWindowNeighborLoader:
    """
    Single-window temporal neighbor sampler for ablation experiments.

    Queries historical neighbors within a fixed time window (0, tau_window_ns]
    and returns at most ``budget`` most-recent neighbors per node.

    This is distinct from MultiScaleNeighborLoader because:
    - It uses a single time window bounded by tau_window_ns (Q99), not three scales
    - It returns up to ``budget`` neighbors total, not per-scale

    Final semantic contract:
    - window: 0 < delta <= tau_window_ns (inclusive)
    - budget: max neighbors to return per node
    - selection: most recent ``budget`` events within the window

    Parameters
    ----------
    history_store
        HistoryStore instance containing per-node history.
    tau_window_ns
        Upper bound (inclusive) for the time window in nanoseconds.
        This is typically Q99 from TimeGapStatistics.
    budget
        Maximum number of neighbors to return per node.
        Default: 24.
    """

    def __init__(
        self,
        history_store: HistoryStore,
        tau_window_ns: int,
        budget: int = 24,
    ) -> None:
        if tau_window_ns < 0:
            raise ValueError(
                f"tau_window_ns must be non-negative, got {tau_window_ns}"
            )
        if budget < 0:
            raise ValueError(
                f"budget must be non-negative, got {budget}"
            )

        self.history_store = history_store
        self.tau_window_ns = int(tau_window_ns)
        self.budget = int(budget)

    def __call__(
        self,
        src: Tensor,
        dst: Tensor,
        timestamp_ns: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Query historical neighbors BEFORE inserting the current batch.

        Returns
        -------
        dict
            Dictionary containing:

            query_nodes: Tensor, shape [Q]
            reference_time_ns: Tensor, shape [Q]
            src_query_index: Tensor, shape [batch_size]
            dst_query_index: Tensor, shape [batch_size]

            window_neighbor_id: Tensor, shape [Q, budget], int64, -1 = invalid
            window_event_id: Tensor, shape [Q, budget], int64, -1 = invalid
            window_timestamp_ns: Tensor, shape [Q, budget], int64, -1 = invalid
            window_direction: Tensor, shape [Q, budget], int8, -1 = invalid
            window_mask: Tensor, shape [Q, budget], bool
        """
        batch_size = src.shape[0]
        device = src.device

        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        t_cpu = timestamp_ns.cpu().long()

        all_nodes_list = torch.cat([src_cpu, dst_cpu]).unique().tolist()
        Q = len(all_nodes_list)
        query_nodes = torch.tensor(all_nodes_list, dtype=torch.long)

        ref_time_per_node = {}
        for i in range(batch_size):
            node = int(src_cpu[i])
            t = int(t_cpu[i])
            if node not in ref_time_per_node:
                ref_time_per_node[node] = t
            else:
                ref_time_per_node[node] = min(ref_time_per_node[node], t)

            node = int(dst_cpu[i])
            if node not in ref_time_per_node:
                ref_time_per_node[node] = t
            else:
                ref_time_per_node[node] = min(ref_time_per_node[node], t)

        reference_time_ns = torch.zeros(Q, dtype=torch.int64)
        for idx, node in enumerate(all_nodes_list):
            reference_time_ns[idx] = ref_time_per_node[node]

        history = self.history_store.query(query_nodes, reference_time_ns)

        src_query_index = torch.zeros(batch_size, dtype=torch.long)
        node_to_qidx = {node: idx for idx, node in enumerate(all_nodes_list)}
        for i in range(batch_size):
            src_query_index[i] = node_to_qidx[int(src_cpu[i])]

        dst_query_index = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            dst_query_index[i] = node_to_qidx[int(dst_cpu[i])]

        result: Dict[str, Tensor] = {
            "query_nodes": query_nodes.to(device),
            "reference_time_ns": reference_time_ns.to(device),
            "src_query_index": src_query_index.to(device),
            "dst_query_index": dst_query_index.to(device),
        }

        tau_window = self.tau_window_ns
        budget = self.budget

        neighbor_ids = history["neighbor_id"]
        event_ids = history["event_id"]
        timestamps = history["timestamp_ns"]
        directions = history["direction"]

        window_neighbor = torch.full(
            (Q, budget), fill_value=-1, dtype=torch.int64, device=device
        )
        window_event_id = torch.full(
            (Q, budget), fill_value=-1, dtype=torch.int64, device=device
        )
        window_timestamp = torch.full(
            (Q, budget), fill_value=-1, dtype=torch.int64, device=device
        )
        window_direction = torch.full(
            (Q, budget), fill_value=-1, dtype=torch.int8, device=device
        )
        window_mask = torch.zeros(
            (Q, budget), dtype=torch.bool, device=device
        )

        ref_cpu = reference_time_ns.cpu()
        ts_cpu = timestamps.cpu()
        ev_cpu = event_ids.cpu()
        nb_cpu = neighbor_ids.cpu()
        dir_cpu = directions.cpu()

        for q_idx in range(Q):
            delta = ref_cpu[q_idx] - ts_cpu[q_idx]
            # Window: 0 < delta <= tau_window_ns (inclusive)
            in_window = (delta > 0) & (delta <= tau_window)

            valid_ts = ts_cpu[q_idx][in_window]
            valid_ev = ev_cpu[q_idx][in_window]
            valid_nb = nb_cpu[q_idx][in_window]
            valid_dir = dir_cpu[q_idx][in_window]

            sorted_idx = _recency_order(valid_ts, valid_ev)
            sorted_ts = valid_ts[sorted_idx]
            sorted_ev = valid_ev[sorted_idx]
            sorted_nb = valid_nb[sorted_idx]
            sorted_dir = valid_dir[sorted_idx]

            num_valid = min(len(sorted_ts), budget)
            if num_valid > 0:
                window_neighbor[q_idx, :num_valid] = sorted_nb[:num_valid].to(device)
                window_event_id[q_idx, :num_valid] = sorted_ev[:num_valid].to(device)
                window_timestamp[q_idx, :num_valid] = sorted_ts[:num_valid].to(device)
                window_direction[q_idx, :num_valid] = sorted_dir[:num_valid].to(device)
                window_mask[q_idx, :num_valid] = True

        result["window_neighbor_id"] = window_neighbor
        result["window_event_id"] = window_event_id
        result["window_timestamp_ns"] = window_timestamp
        result["window_direction"] = window_direction
        result["window_mask"] = window_mask

        return result

    def insert(
        self,
        src: Tensor,
        dst: Tensor,
        timestamp_ns: Tensor,
        global_event_index: Tensor,
    ) -> None:
        """
        Insert the current batch into history AFTER encoding.
        """
        batch_size = src.shape[0]
        direction = torch.zeros(batch_size, dtype=torch.int8)

        self.history_store.insert(
            src=src,
            dst=dst,
            event_id=global_event_index,
            timestamp_ns=timestamp_ns,
            direction=direction,
        )

    def reset_state(self) -> None:
        """Clear all history in the underlying store."""
        self.history_store.reset_state()

    def history_state_dict(self) -> Dict[str, Tensor]:
        """Return a CPU-cloned copy of the history state."""
        return self.history_store.state_dict()

    def load_history_state_dict(self, state: Dict[str, Tensor]) -> None:
        """Restore history state from a previously saved state_dict."""
        self.history_store.load_state_dict(state)
