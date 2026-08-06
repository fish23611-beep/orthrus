"""
MultiScaleNeighborLoader: temporal multi-scale neighbor sampling.

Groups historical neighbors into short / medium / long scales based on
their time distance from the current reference timestamp.

Separates query (read-only, before current batch) from insert
(post-encode, after current batch) to maintain causal ordering.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Sequence

import torch
from torch import Tensor

from .history_store import HistoryStore


# ------------------------------------------------------------------
# Boundary conversion utility
# ------------------------------------------------------------------
def log_seconds_boundaries_to_ns(
    boundaries: Sequence[float],
) -> tuple[int, int, int]:
    """
    Convert three log1p(seconds) boundaries to nanoseconds.

    Parameters
    ----------
    boundaries
        Exactly three values in log1p(seconds) space, e.g.
        from TimeGapStatistics.scale_boundaries.

    Returns
    -------
    tuple[int, int, int]
        (tau_short_ns, tau_medium_ns, tau_max_ns) as integers in nanoseconds.

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

    tau_short_ns = round(math.expm1(b0) * 1_000_000_000)
    tau_medium_ns = round(math.expm1(b1) * 1_000_000_000)
    tau_max_ns = round(math.expm1(b2) * 1_000_000_000)

    return tau_short_ns, tau_medium_ns, tau_max_ns


# ------------------------------------------------------------------
# MultiScaleNeighborLoader
# ------------------------------------------------------------------
class MultiScaleNeighborLoader:
    """
    Multi-scale temporal neighbor sampler.

    Queries historical neighbors and groups them into three temporal scales:

    - short:  0 < delta <= tau_short
    - medium: tau_short < delta <= tau_medium
    - long:   tau_medium < delta <= tau_max

    where delta = reference_time_ns - historical_timestamp_ns.

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
    tau_max_ns
        Upper bound (inclusive) for the long scale in nanoseconds.
        Must satisfy tau_medium_ns <= tau_max_ns.
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
        tau_max_ns: int,
        short_budget: int = 8,
        medium_budget: int = 8,
        long_budget: int = 8,
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
        if tau_max_ns < tau_medium_ns:
            raise ValueError(
                f"tau_max_ns ({tau_max_ns}) must be >= "
                f"tau_medium_ns ({tau_medium_ns})"
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
        self.tau_max_ns = int(tau_max_ns)
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

        for scale_name, tau_low, tau_high, budget in [
            ("short", 0, self.tau_short_ns, self.short_budget),
            ("medium", self.tau_short_ns, self.tau_medium_ns, self.medium_budget),
            ("long", self.tau_medium_ns, self.tau_max_ns, self.long_budget),
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
                delta = ref_cpu[q_idx] - ts_cpu[q_idx]
                in_range = (delta > 0) & (delta <= tau_high)
                if scale_name == "medium":
                    in_range = (delta > tau_low) & (delta <= tau_high)
                elif scale_name == "long":
                    in_range = (delta > tau_low) & (delta <= tau_high)

                valid_ts = ts_cpu[q_idx][in_range]
                valid_ev = ev_cpu[q_idx][in_range]
                valid_nb = nb_cpu[q_idx][in_range]
                valid_dir = dir_cpu[q_idx][in_range]

                sort_keys = valid_ts * (10**10) + valid_ev
                sorted_idx = torch.argsort(sort_keys, descending=True)
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
