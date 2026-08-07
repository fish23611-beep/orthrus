from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn


_SCALE_NAMES = ("short", "medium", "long")


def _masked_softmax_with_fallback(logits: Tensor, mask: Tensor) -> Tensor:
    """Normalize only valid scales and return zero for an all-empty row."""
    if logits.shape != mask.shape:
        raise ValueError(f"logits and mask must have the same shape, got {logits.shape} and {mask.shape}")
    if logits.ndim != 2:
        raise ValueError(f"logits and mask must be rank 2, got {logits.ndim}")

    safe_logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
    weights = torch.softmax(safe_logits, dim=-1).masked_fill(~mask, 0.0)
    has_scale = mask.any(dim=-1, keepdim=True)
    normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return torch.where(has_scale, weights / normalizer, torch.zeros_like(weights))


class MultiScaleOrthrusEncoder(nn.Module):
    """Causal three-scale Orthrus encoder with event-wise gated fusion."""

    def __init__(
        self,
        shared_graph_encoder: Optional[nn.Module] = None,
        neighbor_loader: Any = None,
        in_dim: Optional[int] = None,
        temporal_dim: Optional[int] = None,
        node_out_dim: Optional[int] = None,
        use_node_feats_in_gnn: bool = True,
        edge_features: Sequence[str] | str = ("edge_type", "msg"),
        device: str | torch.device = "cpu",
        gate_hidden_dim: int = 64,
        use_scale_embedding: bool = False,
        encoder: Optional[nn.Module] = None,
        fusion: Optional[str] = None,
    ) -> None:
        super().__init__()
        if shared_graph_encoder is None:
            shared_graph_encoder = encoder
        if shared_graph_encoder is None:
            raise ValueError("shared_graph_encoder is required")
        if neighbor_loader is None:
            raise ValueError("neighbor_loader is required")
        if in_dim is None or temporal_dim is None or node_out_dim is None:
            raise ValueError("in_dim, temporal_dim, and node_out_dim are required")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")

        self.shared_graph_encoder = shared_graph_encoder
        self.neighbor_loader = neighbor_loader
        self.device = torch.device(device)
        self.in_dim = int(in_dim)
        self.temporal_dim = int(temporal_dim)
        self.node_out_dim = int(node_out_dim)
        self.use_node_feats_in_gnn = bool(use_node_feats_in_gnn)
        self.requires_global_event_index = True  # Capability flag for factory/model
        if isinstance(edge_features, str):
            edge_features = edge_features.split(",")
        self.edge_features = tuple(feature.strip() for feature in edge_features)

        self.src_linear = nn.Linear(self.in_dim, self.temporal_dim)
        self.dst_linear = nn.Linear(self.in_dim, self.temporal_dim)
        self.current_src_out_proj = nn.Linear(self.temporal_dim, self.node_out_dim)
        self.current_dst_out_proj = nn.Linear(self.temporal_dim, self.node_out_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(8 * self.node_out_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )
        self.use_scale_embedding = bool(use_scale_embedding)
        self.fusion = str(fusion).strip().lower() if fusion is not None else "gated"
        if self.fusion not in ("gated", "equal"):
            raise ValueError(
                f"Invalid fusion={fusion!r}. Expected 'gated' or 'equal'."
            )
        if self.use_scale_embedding:
            self.scale_embedding = nn.Embedding(3, self.temporal_dim)

        self.last_gate_weights: Optional[Tensor] = None
        self.last_scale_mask: Optional[Tensor] = None

    @property
    def gate_input_dim(self) -> int:
        return 8 * self.node_out_dim

    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _full_values(full_data: Any, field: str, event_ids: Tensor, device: torch.device) -> Tensor:
        if not hasattr(full_data, field):
            raise ValueError(f"full_data is missing required field '{field}'")
        values = getattr(full_data, field)
        return values.cpu()[event_ids.cpu()].to(device)

    def _edge_features(self, full_data: Any, event_ids: Tensor, device: torch.device) -> Optional[Tensor]:
        parts = []
        for feature in self.edge_features:
            if feature in ("", "none"):
                continue
            if feature not in ("edge_type", "msg"):
                raise ValueError(f"Unsupported edge feature '{feature}'")
            parts.append(self._full_values(full_data, feature, event_ids, device))
        return torch.cat(parts, dim=-1) if parts else None

    def _historical_node_features(
        self, full_data: Any, event_ids: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        if hasattr(full_data, "x_src") and hasattr(full_data, "x_dst"):
            return (
                self._full_values(full_data, "x_src", event_ids, device),
                self._full_values(full_data, "x_dst", event_ids, device),
            )
        if not hasattr(full_data, "msg"):
            raise ValueError("full_data must provide x_src/x_dst or msg for historical nodes")
        message = self._full_values(full_data, "msg", event_ids, device)
        if message.ndim != 2 or message.size(-1) < 2 * self.in_dim:
            raise ValueError(
                "full_data.msg must contain source and destination node features "
                f"with at least {2 * self.in_dim} columns"
            )
        return message[:, :self.in_dim], message[:, self.in_dim:2 * self.in_dim]

    def _scan_scale(
        self,
        scale: str,
        sampled: Dict[str, Tensor],
        full_data: Any,
        device: torch.device,
    ) -> list[tuple[int, int, int, int]]:
        event_ids = sampled[f"{scale}_event_id"].cpu()
        neighbors = sampled[f"{scale}_neighbor_id"].cpu()
        directions = sampled[f"{scale}_direction"].cpu()
        masks = sampled[f"{scale}_mask"].cpu()
        query_nodes = sampled["query_nodes"].cpu()
        records: list[tuple[int, int, int, int]] = []
        seen_event_ids: set[int] = set()

        for q_idx in range(event_ids.size(0)):
            for slot in range(event_ids.size(1)):
                if not bool(masks[q_idx, slot]):
                    continue
                event_id = int(event_ids[q_idx, slot])
                if event_id == -1 or event_id in seen_event_ids:
                    continue
                direction = int(directions[q_idx, slot])
                if direction not in (0, 1):
                    raise ValueError(f"history direction must be 0 or 1, got {direction}")
                query = int(query_nodes[q_idx])
                neighbor = int(neighbors[q_idx, slot])
                full_src = int(self._full_values(full_data, "src", torch.tensor([event_id]), device)[0])
                full_dst = int(self._full_values(full_data, "dst", torch.tensor([event_id]), device)[0])
                full_timestamp = int(self._full_values(full_data, "t", torch.tensor([event_id]), device)[0])
                sampled_timestamp = int(sampled[f"{scale}_timestamp_ns"].cpu()[q_idx, slot])
                expected_src = query if direction == 0 else neighbor
                expected_dst = neighbor if direction == 0 else query
                if (full_src, full_dst) != (expected_src, expected_dst):
                    raise ValueError("history direction and full_data edge endpoints disagree")
                if full_timestamp != sampled_timestamp:
                    raise ValueError("history timestamp and full_data timestamp disagree")
                seen_event_ids.add(event_id)
                records.append((event_id, query, neighbor, direction))
        return records

    def _encode_scale(
        self,
        scale_index: int,
        sampled: Dict[str, Tensor],
        current_src_hidden: Tensor,
        current_dst_hidden: Tensor,
        current_src_out: Tensor,
        current_dst_out: Tensor,
        full_data: Any,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scale = _SCALE_NAMES[scale_index]
        records = self._scan_scale(scale, sampled, full_data, device)
        query_nodes = sampled["query_nodes"].to(device).long()
        src_indices = sampled["src_query_index"].to(device).long()
        dst_indices = sampled["dst_query_index"].to(device).long()
        src_query_nodes = query_nodes[src_indices]
        dst_query_nodes = query_nodes[dst_indices]
        batch_size = current_src_hidden.size(0)
        src_has_history = torch.zeros(batch_size, dtype=torch.bool, device=device)
        dst_has_history = torch.zeros_like(src_has_history)

        for _, query, neighbor, direction in records:
            if direction == 0:
                src_has_history |= src_query_nodes == query
            else:
                dst_has_history |= dst_query_nodes == query

        if not records:
            return current_src_out, current_dst_out, src_has_history | dst_has_history

        event_ids = torch.tensor([record[0] for record in records], dtype=torch.long, device=device)
        edge_src_global = torch.tensor(
            [query if direction == 0 else neighbor for _, query, neighbor, direction in records],
            dtype=torch.long,
            device=device,
        )
        edge_dst_global = torch.tensor(
            [neighbor if direction == 0 else query for _, query, neighbor, direction in records],
            dtype=torch.long,
            device=device,
        )
        node_ids = torch.cat([query_nodes, edge_src_global, edge_dst_global]).unique(sorted=True)
        node_to_local = {int(node): index for index, node in enumerate(node_ids.cpu().tolist())}
        local_src = torch.tensor([node_to_local[int(node)] for node in edge_src_global.cpu()], dtype=torch.long, device=device)
        local_dst = torch.tensor([node_to_local[int(node)] for node in edge_dst_global.cpu()], dtype=torch.long, device=device)
        edge_index = torch.stack([local_src, local_dst])

        node_src_hidden = torch.zeros((node_ids.numel(), self.temporal_dim), dtype=current_src_hidden.dtype, device=device)
        node_dst_hidden = torch.zeros_like(node_src_hidden)
        historical_src, historical_dst = self._historical_node_features(full_data, event_ids, device)
        for row, (edge_src, edge_dst) in enumerate(zip(edge_src_global.tolist(), edge_dst_global.tolist())):
            node_src_hidden[node_to_local[edge_src]] = self.src_linear(historical_src[row])
            node_dst_hidden[node_to_local[edge_dst]] = self.dst_linear(historical_dst[row])

        for batch_index in range(batch_size):
            node_src_hidden[node_to_local[int(src_query_nodes[batch_index])]] = current_src_hidden[batch_index]
            node_dst_hidden[node_to_local[int(dst_query_nodes[batch_index])]] = current_dst_hidden[batch_index]
        node_features = node_src_hidden + node_dst_hidden
        if self.use_scale_embedding:
            scale_ids = torch.full((node_ids.numel(),), scale_index, dtype=torch.long, device=device)
            node_features = node_features + self.scale_embedding(scale_ids)

        encoded = self.shared_graph_encoder(
            node_features,
            edge_index,
            edge_feats=self._edge_features(full_data, event_ids, device),
        )
        src_local = torch.tensor([node_to_local[int(node)] for node in src_query_nodes], dtype=torch.long, device=device)
        dst_local = torch.tensor([node_to_local[int(node)] for node in dst_query_nodes], dtype=torch.long, device=device)
        encoded_src = encoded[src_local]
        encoded_dst = encoded[dst_local]
        encoded_src = torch.where(src_has_history.unsqueeze(-1), encoded_src, current_src_out)
        encoded_dst = torch.where(dst_has_history.unsqueeze(-1), encoded_dst, current_dst_out)
        return encoded_src, encoded_dst, src_has_history | dst_has_history

    def forward(self, edge_index, t, msg, x, full_data, inference=False, **kwargs) -> tuple[Tensor, Tensor]:
        del msg, inference
        global_event_index = kwargs.get("global_event_index")
        if global_event_index is None:
            raise ValueError("global_event_index is required")
        src, dst = edge_index
        batch_size = src.numel()
        if not isinstance(global_event_index, Tensor):
            raise ValueError("global_event_index must be a Tensor")
        if global_event_index.dtype != torch.long:
            raise ValueError("global_event_index must have dtype torch.long")
        if global_event_index.ndim != 1:
            raise ValueError("global_event_index must be one-dimensional")
        if global_event_index.numel() != batch_size:
            raise ValueError("global_event_index length must equal batch size")

        device = self._module_device()
        src, dst, t = src.to(device), dst.to(device), t.to(device)
        x_src, x_dst = x
        if x_src.size(0) != batch_size:
            x_src, x_dst = x_src[edge_index[0]], x_dst[edge_index[1]]
        x_src, x_dst = x_src.to(device), x_dst.to(device)
        current_src_hidden = self.src_linear(x_src) if self.use_node_feats_in_gnn else self.src_linear(torch.zeros_like(x_src))
        current_dst_hidden = self.dst_linear(x_dst) if self.use_node_feats_in_gnn else self.dst_linear(torch.zeros_like(x_dst))
        current_src_out = self.current_src_out_proj(current_src_hidden)
        current_dst_out = self.current_dst_out_proj(current_dst_hidden)

        sampled = self.neighbor_loader(src, dst, timestamp_ns=t)
        scale_src, scale_dst, scale_masks = [], [], []
        for scale_index in range(3):
            encoded_src, encoded_dst, scale_mask = self._encode_scale(
                scale_index, sampled, current_src_hidden, current_dst_hidden,
                current_src_out, current_dst_out, full_data, device,
            )
            scale_src.append(encoded_src)
            scale_dst.append(encoded_dst)
            scale_masks.append(scale_mask)
        h_src_stack = torch.stack(scale_src, dim=1)
        h_dst_stack = torch.stack(scale_dst, dim=1)
        non_empty_scale_mask = torch.stack(scale_masks, dim=1)
        all_empty = ~non_empty_scale_mask.any(dim=-1)
        if self.fusion == "gated":
            gate_input = torch.cat(
                [h_src_stack[:, 0], h_dst_stack[:, 0], h_src_stack[:, 1], h_dst_stack[:, 1],
                 h_src_stack[:, 2], h_dst_stack[:, 2], current_src_out, current_dst_out], dim=-1
            )
            gate_weights = _masked_softmax_with_fallback(self.gate_mlp(gate_input), non_empty_scale_mask)
            h_src = (gate_weights.unsqueeze(-1) * h_src_stack).sum(dim=1)
            h_dst = (gate_weights.unsqueeze(-1) * h_dst_stack).sum(dim=1)
        elif self.fusion == "equal":
            weights = non_empty_scale_mask.float()
            denom = weights.sum(dim=-1, keepdim=True)
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            weights = torch.where(denom > 0, weights / safe_denom, torch.zeros_like(weights))
            gate_weights = weights
            h_src = (weights.unsqueeze(-1) * h_src_stack).sum(dim=1)
            h_dst = (weights.unsqueeze(-1) * h_dst_stack).sum(dim=1)
        h_src = torch.where(all_empty.unsqueeze(-1), current_src_out, h_src)
        h_dst = torch.where(all_empty.unsqueeze(-1), current_dst_out, h_dst)
        self.last_gate_weights = gate_weights.detach().clone()
        self.last_scale_mask = non_empty_scale_mask.detach().clone()
        self.neighbor_loader.insert(src, dst, timestamp_ns=t, global_event_index=global_event_index)
        return h_src, h_dst

    def reset_state(self) -> None:
        self.neighbor_loader.reset_state()
        self.last_gate_weights = None
        self.last_scale_mask = None

    def history_state_dict(self) -> Dict[str, Tensor]:
        return self.neighbor_loader.history_state_dict()

    def load_history_state_dict(self, state: Dict[str, Tensor]) -> None:
        self.neighbor_loader.load_history_state_dict(state)

    def get_last_gate_weights(self) -> Optional[Tensor]:
        return None if self.last_gate_weights is None else self.last_gate_weights.clone()

    def get_last_scale_mask(self) -> Optional[Tensor]:
        return None if self.last_scale_mask is None else self.last_scale_mask.clone()
