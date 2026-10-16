"""
HistoryStore: per-node temporal history with minimal footprint.

Stores only the minimal fields needed for multi-scale neighbor sampling:
    - neighbor_id  (int64)
    - event_id    (int64) — global event index for full_data lookup
    - timestamp_ns (int64)
    - direction   (int8)

This is NOT an nn.Module to avoid polluting model state_dict.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence
from time import perf_counter

import torch
from torch import Tensor
try:
    from runtime.runtime_profiler import get_profiler
except ModuleNotFoundError:
    from src.runtime.runtime_profiler import get_profiler


def _stable_lexsort_desc(timestamp_ns: Tensor, event_id: Tensor) -> Tensor:
    """Return a stable descending lexicographic order for timestamp/event ID."""
    event_order = torch.argsort(event_id, descending=True, stable=True)
    timestamp_order = torch.argsort(
        timestamp_ns[event_order], descending=True, stable=True
    )
    return event_order[timestamp_order]


class HistoryStore:
    """
    Stores per-node temporal history for multi-scale sampling.

    Each node keeps at most ``candidate_capacity`` most-recent events,
    sorted by timestamp descending (most recent first).

    Storage is dense: each node has exactly ``candidate_capacity`` slots,
    with invalid entries marked by ``neighbor_id == -1``.

    Parameters
    ----------
    num_nodes
        Total number of nodes in the graph.
    candidate_capacity
        Maximum number of historical events stored per node.
    device
        Device to store the history tensors on. Default is "cpu".
    store_direction
        If True, store direction (0=outgoing, 1=incoming) as int8.
        If False, direction field is omitted.
    """

    def __init__(
        self,
        num_nodes: int,
        candidate_capacity: int,
        device: str | torch.device = "cpu",
        store_direction: bool = True,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")
        if candidate_capacity <= 0:
            raise ValueError(
                f"candidate_capacity must be positive, got {candidate_capacity}"
            )

        self.num_nodes = num_nodes
        self.candidate_capacity = candidate_capacity
        self.device = torch.device(device)
        self.store_direction = store_direction

        self._neighbor_id = torch.full(
            (num_nodes, candidate_capacity),
            fill_value=-1,
            dtype=torch.int64,
            device=self.device,
        )
        self._event_id = torch.full(
            (num_nodes, candidate_capacity),
            fill_value=-1,
            dtype=torch.int64,
            device=self.device,
        )
        self._timestamp_ns = torch.full(
            (num_nodes, candidate_capacity),
            fill_value=-1,
            dtype=torch.int64,
            device=self.device,
        )
        self._count = torch.zeros(
            num_nodes,
            dtype=torch.int32,
            device=self.device,
        )
        if store_direction:
            self._direction = torch.full(
                (num_nodes, candidate_capacity),
                fill_value=-1,
                dtype=torch.int8,
                device=self.device,
            )
        else:
            self._direction = None

    # ------------------------------------------------------------------
    # reset_state
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Clear all history and reset per-node counters."""
        self._neighbor_id.fill_(-1)
        self._event_id.fill_(-1)
        self._timestamp_ns.fill_(-1)
        self._count.zero_()
        if self._direction is not None:
            self._direction.fill_(-1)

    # ------------------------------------------------------------------
    # insert
    # ------------------------------------------------------------------
    def insert(
        self,
        src: Tensor,
        dst: Tensor,
        event_id: Tensor,
        timestamp_ns: Tensor,
        direction: Optional[Tensor] = None,
    ) -> None:
        """
        Insert events into the history store (bidirectional).

        Parameters
        ----------
        src
            Source node IDs, shape [batch_size].
        dst
            Destination node IDs, shape [batch_size].
        event_id
            Global event indices, shape [batch_size].
            These are used to index into full_data for feature lookup.
        timestamp_ns
            Event timestamps in nanoseconds, shape [batch_size].
            Must be int64.
        direction
            Optional direction per event, shape [batch_size].
            0 = src→dst (neighbor = dst for src node)
            1 = dst→src (neighbor = src for dst node)
            If None, direction is derived from the insertion pass.
        """
        if src.shape != dst.shape:
            raise ValueError(
                f"src and dst must have same shape, got {src.shape} and {dst.shape}"
            )
        if src.shape != event_id.shape:
            raise ValueError(
                f"src and event_id must have same shape, got {src.shape} and {event_id.shape}"
            )
        if src.shape != timestamp_ns.shape:
            raise ValueError(
                f"src and timestamp_ns must have same shape, got {src.shape} and {timestamp_ns.shape}"
            )
        if timestamp_ns.dtype not in (torch.int64, torch.long):
            raise ValueError(
                f"timestamp_ns must be int64, got {timestamp_ns.dtype}"
            )
        if direction is not None:
            if src.shape != direction.shape:
                raise ValueError(
                    f"src and direction must have same shape, got {src.shape} and {direction.shape}"
                )
            if direction.dtype != torch.int8:
                raise ValueError(
                    f"direction must be int8, got {direction.dtype}"
                )
            if self.store_direction:
                unique_vals = set(direction.unique().tolist())
                if not unique_vals.issubset({0, 1}):
                    raise ValueError(
                        f"direction values must be 0 or 1, got {unique_vals}"
                    )
            derived_direction: Optional[Tensor] = None
            if self.store_direction:
                derived_direction = direction.to(self.device)
        elif self.store_direction:
            derived_direction = torch.zeros(
                src.shape,
                dtype=torch.int8,
                device=self.device,
            )
        else:
            derived_direction = None

        B = src.shape[0]
        src_flat = src.long()
        dst_flat = dst.long()
        event_flat = event_id.long()
        ts_flat = timestamp_ns.long()

        nodes_a = torch.cat([src_flat, dst_flat], dim=0)
        neighbors_b = torch.cat([dst_flat, src_flat], dim=0)
        events = torch.cat([event_flat, event_flat], dim=0)
        times = torch.cat([ts_flat, ts_flat], dim=0)
        if derived_direction is not None:
            derived_long = derived_direction.long()
            dirs = torch.cat([derived_long, 1 - derived_long], dim=0)
        else:
            dirs = None

        self._insert_vectorized(
            nodes=nodes_a,
            neighbors=neighbors_b,
            event_ids=events,
            timestamps=times,
            directions=dirs,
        )

    def _insert_vectorized(
        self,
        nodes: Tensor,
        neighbors: Tensor,
        event_ids: Tensor,
        timestamps: Tensor,
        directions: Optional[Tensor],
    ) -> None:
        """
        Vectorized batch insert using scatter operations.

        Algorithm:
        1. Sort events by (node, timestamp desc, event_id desc)
        2. Use bincount to find node boundaries
        3. Scatter into per-node slots
        4. Merge with existing history and keep top-K
        """
        # History storage may intentionally remain on CPU while the encoder
        # and its input batch live on CUDA.
        device = self._count.device
        cap = self.candidate_capacity
        profiler = get_profiler()
        detail = profiler.detail_active
        sort_started = perf_counter() if detail else None

        nodes_cpu = nodes.cpu()
        neighbors_cpu = neighbors.cpu()
        event_ids_cpu = event_ids.cpu()
        timestamps_cpu = timestamps.cpu()
        directions_cpu = directions.cpu() if directions is not None else None

        valid_node_mask = (nodes_cpu >= 0) & (nodes_cpu < self.num_nodes)
        touched_nodes = nodes_cpu[valid_node_mask].unique(sorted=True)
        if sort_started is not None:
            profiler.record(
                "training.history_store.insert_sort_prepare",
                perf_counter() - sort_started,
                parent="training.T_history_insert",
            )
        num_touched = int(touched_nodes.numel())
        if num_touched == 0:
            return

        merge_started = perf_counter() if detail else None
        existing_counts = self._count[touched_nodes.to(device)].cpu().long()
        existing_mask = (
            torch.arange(cap, dtype=torch.long).unsqueeze(0)
            < existing_counts.unsqueeze(1)
        )
        existing_nodes = touched_nodes.unsqueeze(1).expand(-1, cap)[existing_mask]
        touched_on_store = touched_nodes.to(device)
        existing_neighbors = self._neighbor_id[touched_on_store].cpu()[existing_mask]
        existing_events = self._event_id[touched_on_store].cpu()[existing_mask]
        existing_times = self._timestamp_ns[touched_on_store].cpu()[existing_mask]

        new_nodes = nodes_cpu[valid_node_mask]
        all_nodes = torch.cat([existing_nodes, new_nodes])
        all_neighbors = torch.cat([
            existing_neighbors,
            neighbors_cpu[valid_node_mask],
        ])
        all_events = torch.cat([
            existing_events,
            event_ids_cpu[valid_node_mask],
        ])
        all_times = torch.cat([
            existing_times,
            timestamps_cpu[valid_node_mask],
        ])
        all_dirs = None
        if self._direction is not None:
            if directions_cpu is None:
                raise ValueError("directions are required when store_direction is enabled")
            existing_dirs = self._direction[touched_on_store].cpu()[existing_mask]
            all_dirs = torch.cat([existing_dirs, directions_cpu[valid_node_mask]])

        # Stable least-significant-key-first sorting is overflow safe and
        # preserves the old tie semantics: existing equal-key records precede
        # new records, and equal-key new records retain expanded input order.
        order = torch.argsort(all_events, descending=True, stable=True)
        order = order[
            torch.argsort(all_times[order], descending=True, stable=True)
        ]
        order = order[torch.argsort(all_nodes[order], stable=True)]

        sorted_nodes = all_nodes[order]
        _, records_per_node = torch.unique_consecutive(
            sorted_nodes, return_counts=True
        )
        group_starts = records_per_node.cumsum(0) - records_per_node
        rank_within_node = torch.arange(sorted_nodes.numel()) - torch.repeat_interleave(
            group_starts, records_per_node
        )
        keep_mask = rank_within_node < cap
        kept_order = order[keep_mask]
        kept_nodes = all_nodes[kept_order]
        kept_columns = rank_within_node[keep_mask]
        kept_rows = torch.searchsorted(touched_nodes, kept_nodes)

        merged_neighbors = torch.full((num_touched, cap), -1, dtype=torch.int64)
        merged_events = torch.full_like(merged_neighbors, -1)
        merged_times = torch.full_like(merged_neighbors, -1)
        merged_neighbors[kept_rows, kept_columns] = all_neighbors[kept_order]
        merged_events[kept_rows, kept_columns] = all_events[kept_order]
        merged_times[kept_rows, kept_columns] = all_times[kept_order]

        self._neighbor_id[touched_on_store] = merged_neighbors.to(device)
        self._event_id[touched_on_store] = merged_events.to(device)
        self._timestamp_ns[touched_on_store] = merged_times.to(device)
        if self._direction is not None and all_dirs is not None:
            merged_dirs = torch.full(
                (num_touched, cap), -1, dtype=self._direction.dtype
            )
            merged_dirs[kept_rows, kept_columns] = all_dirs[kept_order].to(
                merged_dirs.dtype
            )
            self._direction[touched_on_store] = merged_dirs.to(device)
        self._count[touched_on_store] = records_per_node.clamp_max(cap).to(
            device=device, dtype=self._count.dtype
        )
        if merge_started is not None:
            profiler.record(
                "training.history_store.insert_node_merge",
                perf_counter() - merge_started,
                parent="training.T_history_insert",
            )
        profiler.record_batch_stat("history_insert_nodes", num_touched)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------
    def query(
        self,
        node_ids: Tensor,
        reference_time_ns: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Return all history for the given nodes.

        Parameters
        ----------
        node_ids
            Node IDs to query, shape [num_query_nodes].
        reference_time_ns
            Reference timestamps for each node, shape [num_query_nodes].
            Used by MultiScaleNeighborLoader for delta computation.

        Returns
        -------
        dict
            Dictionary with keys:
            - "neighbor_id": [num_query_nodes, candidate_capacity], int64, -1 = invalid
            - "event_id":    [num_query_nodes, candidate_capacity], int64, -1 = invalid
            - "timestamp_ns": [num_query_nodes, candidate_capacity], int64, -1 = invalid
            - "direction":    [num_query_nodes, candidate_capacity], int8, -1 = invalid
            - "count":       [num_query_nodes], int32 — actual number of valid entries
            - "node_ids":    [num_query_nodes], int64 — the queried node IDs
            - "reference_time_ns": [num_query_nodes], int64 — reference timestamps
        """
        if node_ids.shape != reference_time_ns.shape:
            raise ValueError(
                f"node_ids and reference_time_ns must have same shape, "
                f"got {node_ids.shape} and {reference_time_ns.shape}"
            )

        profiler = get_profiler()
        clone_started = perf_counter() if profiler.detail_active else None
        if profiler.detail_active and self._count.device.type == "cpu":
            profiler.record_batch_stat(
                "history_total_slots", int(self._count.sum().item())
            )
        node_ids_long = node_ids.long()
        reference_time_ns_long = reference_time_ns.long()

        result = {
            "neighbor_id": self._neighbor_id[node_ids_long].clone(),
            "event_id": self._event_id[node_ids_long].clone(),
            "timestamp_ns": self._timestamp_ns[node_ids_long].clone(),
            "direction": (
                self._direction[node_ids_long].clone()
                if self._direction is not None
                else torch.full(
                    (len(node_ids), self.candidate_capacity),
                    fill_value=-1,
                    dtype=torch.int8,
                )
            ),
            "count": self._count[node_ids_long].clone(),
            "node_ids": node_ids_long.clone(),
            "reference_time_ns": reference_time_ns_long.clone(),
        }
        if clone_started is not None:
            profiler.record(
                "training.history_store.query_clone",
                perf_counter() - clone_started,
                parent="training.history_store.query",
            )
        return result

    # ------------------------------------------------------------------
    # state_dict / load_state_dict
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Tensor]:
        """
        Return a CPU-cloned copy of the internal state.

        Returns
        -------
        dict
            Keys: neighbor_id, event_id, timestamp_ns, direction (if enabled),
            count, plus metadata (num_nodes, candidate_capacity, store_direction).
        """
        state: Dict[str, Tensor] = {
            "neighbor_id": self._neighbor_id.cpu().clone(),
            "event_id": self._event_id.cpu().clone(),
            "timestamp_ns": self._timestamp_ns.cpu().clone(),
            "count": self._count.cpu().clone(),
            "_num_nodes": self.num_nodes,
            "_candidate_capacity": self.candidate_capacity,
            "_store_direction": self.store_direction,
        }
        if self._direction is not None:
            state["direction"] = self._direction.cpu().clone()
        return state

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        """
        Restore state from a previously saved state_dict.

        Parameters
        ----------
        state
            Dictionary produced by state_dict().

        Raises
        ------
        ValueError
            If required keys are missing, shapes mismatch, dtypes are wrong,
            or capacity/node counts are inconsistent.
        """
        required_keys = {"neighbor_id", "event_id", "timestamp_ns", "count"}
        missing = required_keys - set(state.keys())
        if missing:
            raise ValueError(f"state_dict missing required keys: {missing}")

        if state["neighbor_id"].shape != (self.num_nodes, self.candidate_capacity):
            raise ValueError(
                f"neighbor_id shape mismatch: expected "
                f"({self.num_nodes}, {self.candidate_capacity}), "
                f"got {state['neighbor_id'].shape}"
            )
        if state["event_id"].shape != (self.num_nodes, self.candidate_capacity):
            raise ValueError(
                f"event_id shape mismatch: expected "
                f"({self.num_nodes}, {self.candidate_capacity}), "
                f"got {state['event_id'].shape}"
            )
        if state["timestamp_ns"].shape != (self.num_nodes, self.candidate_capacity):
            raise ValueError(
                f"timestamp_ns shape mismatch: expected "
                f"({self.num_nodes}, {self.candidate_capacity}), "
                f"got {state['timestamp_ns'].shape}"
            )
        if state["count"].shape != (self.num_nodes,):
            raise ValueError(
                f"count shape mismatch: expected ({self.num_nodes},), "
                f"got {state['count'].shape}"
            )

        if state["neighbor_id"].dtype != torch.int64:
            raise ValueError(
                f"neighbor_id dtype must be int64, got {state['neighbor_id'].dtype}"
            )
        if state["event_id"].dtype != torch.int64:
            raise ValueError(
                f"event_id dtype must be int64, got {state['event_id'].dtype}"
            )
        if state["timestamp_ns"].dtype != torch.int64:
            raise ValueError(
                f"timestamp_ns dtype must be int64, got {state['timestamp_ns'].dtype}"
            )
        if state["count"].dtype not in (torch.int32, torch.long, torch.int64):
            raise ValueError(
                f"count dtype must be int-like, got {state['count'].dtype}"
            )

        if self.store_direction and "direction" not in state:
            raise ValueError(
                "state_dict is missing 'direction' but HistoryStore was created with "
                "store_direction=True"
            )

        loaded_num_nodes = state.get("_num_nodes")
        loaded_capacity = state.get("_candidate_capacity")
        if loaded_num_nodes is not None and loaded_num_nodes != self.num_nodes:
            raise ValueError(
                f"num_nodes mismatch: HistoryStore has {self.num_nodes}, "
                f"state_dict has {loaded_num_nodes}"
            )
        if loaded_capacity is not None and loaded_capacity != self.candidate_capacity:
            raise ValueError(
                f"candidate_capacity mismatch: HistoryStore has {self.candidate_capacity}, "
                f"state_dict has {loaded_capacity}"
            )

        self._neighbor_id.copy_(state["neighbor_id"].to(self.device))
        self._event_id.copy_(state["event_id"].to(self.device))
        self._timestamp_ns.copy_(state["timestamp_ns"].to(self.device))
        self._count.copy_(state["count"].to(self.device).long())

        if self._direction is not None:
            if "direction" in state:
                self._direction.copy_(state["direction"].to(self.device))
            else:
                self._direction.fill_(-1)
        elif "direction" in state:
            raise ValueError(
                "state_dict contains direction but HistoryStore was created with "
                "store_direction=False"
            )
