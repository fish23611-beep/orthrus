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

import torch
from torch import Tensor


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
        device = nodes.device
        cap = self.candidate_capacity

        nodes_cpu = nodes.cpu()
        neighbors_cpu = neighbors.cpu()
        event_ids_cpu = event_ids.cpu()
        timestamps_cpu = timestamps.cpu()
        directions_cpu = directions.cpu() if directions is not None else None

        sort_keys = timestamps_cpu * (10**10) + event_ids_cpu
        sorted_idx = torch.argsort(sort_keys, descending=True)

        sorted_nodes = nodes_cpu[sorted_idx]
        sorted_neighbors = neighbors_cpu[sorted_idx]
        sorted_events = event_ids_cpu[sorted_idx]
        sorted_times = timestamps_cpu[sorted_idx]
        sorted_dirs = directions_cpu[sorted_idx] if directions_cpu is not None else None

        node_list = sorted_nodes.unique().tolist()
        if not node_list:
            return

        for n in node_list:
            if n < 0 or n >= self.num_nodes:
                continue

            mask = sorted_nodes == n
            n_events = mask.sum().item()

            n_neighbors = sorted_neighbors[mask]
            n_events_ids = sorted_events[mask]
            n_times = sorted_times[mask]
            n_dirs = sorted_dirs[mask] if sorted_dirs is not None else None

            existing_count = int(self._count[n].item())

            all_neighbors = torch.cat([
                self._neighbor_id[n, :existing_count].cpu(),
                n_neighbors,
            ])
            all_events = torch.cat([
                self._event_id[n, :existing_count].cpu(),
                n_events_ids,
            ])
            all_times = torch.cat([
                self._timestamp_ns[n, :existing_count].cpu(),
                n_times,
            ])
            all_dirs = (
                torch.cat([
                    self._direction[n, :existing_count].cpu(),
                    n_dirs,
                ])
                if self._direction is not None and n_dirs is not None
                else (
                    self._direction[n, :existing_count].cpu()
                    if self._direction is not None
                    else None
                )
            )

            sort_keys_all = all_times * (10**10) + all_events
            sorted_idx_all = torch.argsort(sort_keys_all, descending=True)

            all_neighbors = all_neighbors[sorted_idx_all]
            all_events = all_events[sorted_idx_all]
            all_times = all_times[sorted_idx_all]
            if all_dirs is not None:
                all_dirs = all_dirs[sorted_idx_all]

            keep = min(len(all_neighbors), cap)

            self._neighbor_id[n, :keep] = all_neighbors[:keep].to(device)
            self._event_id[n, :keep] = all_events[:keep].to(device)
            self._timestamp_ns[n, :keep] = all_times[:keep].to(device)
            if self._direction is not None and all_dirs is not None:
                self._direction[n, :keep] = all_dirs[:keep].to(device)
            if keep < cap:
                self._neighbor_id[n, keep:].fill_(-1)
                self._event_id[n, keep:].fill_(-1)
                self._timestamp_ns[n, keep:].fill_(-1)
                if self._direction is not None:
                    self._direction[n, keep:].fill_(-1)

            self._count[n] = keep

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

        node_ids_long = node_ids.long()
        reference_time_ns_long = reference_time_ns.long()

        return {
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
