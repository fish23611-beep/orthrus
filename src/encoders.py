from provnet_utils import *
from config import *
from typing import Tuple
import torch.nn as nn
from torch_geometric.nn import SAGEConv, TransformerConv


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_dim, dropout, activation, num_heads):
        super(GraphTransformer, self).__init__()

        self.conv = TransformerConv(in_dim, hid_dim, heads=num_heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hid_dim * num_heads, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        x = self.activation(self.conv(x, edge_index, edge_feats))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_feats)
        return x


class GraphSAGEBackbone(nn.Module):
    """C7-B1 GraphSAGE backbone.

    A two-layer GraphSAGE that mirrors the constructor shape and forward
    signature of ``GraphTransformer`` closely enough to be plugged into the
    existing encoder factory wiring. ``GraphSAGE`` does not consume edge
    features; the ``edge_feats`` keyword argument is therefore accepted and
    explicitly ignored, so the same ``OrthrusEncoder`` / ``MultiScaleOrthrusEncoder``
    call sites can keep passing ``edge_feats`` without behavioural change.

    Note:
        ``num_heads`` is **not** part of the API: GraphSAGE does not have
        multi-head attention, so it has no comparable hyperparameter.
    """

    def __init__(self, in_dim, hid_dim, out_dim, dropout, activation):
        super(GraphSAGEBackbone, self).__init__()

        self.conv1 = SAGEConv(in_dim, hid_dim, aggr="mean")
        self.conv2 = SAGEConv(hid_dim, out_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_feats=None, **kwargs):
        # GraphSAGE does not consume edge features; ignore them safely.
        del edge_feats
        x = self.activation(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class SemanticMLPBackbone(nn.Module):
    """Two-layer, event-local semantic baseline.

    The backbone deliberately operates on the current event's source and
    destination semantic features rather than graph nodes. It returns a pair
    of endpoint representations because Orthrus decoders consume
    ``(h_src, h_dst)`` for each event.
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float, activation: nn.Module) -> None:
        super().__init__()
        self.lin1 = nn.Linear(2 * in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.lin2 = nn.Linear(hid_dim, 2 * out_dim)
        self.out_dim = out_dim

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index=None,
        edge_feats=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode current event endpoints without reading graph inputs."""
        del edge_index, edge_feats, kwargs
        if x_src.ndim != 2 or x_dst.ndim != 2:
            raise ValueError("x_src and x_dst must be rank-2 event feature tensors")
        if x_src.shape != x_dst.shape:
            raise ValueError(f"x_src and x_dst must have matching shapes, got {x_src.shape} and {x_dst.shape}")
        hidden = self.activation(self.lin1(torch.cat([x_src, x_dst], dim=-1)))
        outputs = self.lin2(self.dropout(hidden))
        return outputs.split(self.out_dim, dim=-1)


class SemanticMLPEncoder(nn.Module):
    """Orthrus-compatible, stateless wrapper for :class:`SemanticMLPBackbone`."""

    def __init__(self, backbone: SemanticMLPBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, edge_index, t, msg, x, full_data, inference=False, edge_feats=None, **kwargs):
        del edge_index, t, msg, full_data, inference, edge_feats, kwargs
        x_src, x_dst = x
        return self.backbone(x_src, x_dst)

    def reset_state(self) -> None:
        """Compatibility no-op: semantic features have no replay state."""


class OrthrusEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        neighbor_loader,
        in_dim,
        temporal_dim,
        use_node_feats_in_gnn,
        graph_reindexer,
        edge_features,
        device,
        num_nodes,
        edge_dim,
    ):
        super(OrthrusEncoder, self).__init__()
        self.encoder = encoder
        self.neighbor_loader = neighbor_loader
        self.device = device
        self.assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        
        self.edge_features = edge_features

        self.use_node_feats_in_gnn = use_node_feats_in_gnn
        if self.use_node_feats_in_gnn:
            self.src_linear = nn.Linear(in_dim, temporal_dim)
            self.dst_linear = nn.Linear(in_dim, temporal_dim)
            
        self.graph_reindexer = graph_reindexer

    def forward(self, edge_index, t, msg, x, full_data, inference=False, **kwargs):
        # NOTE: full_data is the full list of all edges in the entire dataset (train/val/test)
        
        src, dst = edge_index
        x_src, x_dst = x
        batch_edge_index = edge_index.clone()
        
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        x_proj = None
        x_src, x_dst = self.graph_reindexer.node_features_reshape(batch_edge_index, x_src, x_dst, max_num_node=n_id.max())
        x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])

        h = x_proj
        
        # Edge features
        edge_feats = []

        def full_values(field):
            if hasattr(full_data, "get_event_values"):
                values = full_data.get_event_values(field, e_id.cpu())
            else:
                values = getattr(full_data, field)[e_id.cpu()]
            return values.to(self.device)

        if "edge_type" in self.edge_features:
            edge_feats.append(full_values("edge_type"))
        if "msg" in self.edge_features:
            edge_feats.append(full_values("msg"))
        edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
        
        h = self.encoder(h, edge_index, edge_feats=edge_feats)

        h_src = h[self.assoc[src]]
        h_dst = h[self.assoc[dst]]

        self.neighbor_loader.insert(src, dst)
        
        return h_src, h_dst

    def reset_state(self):
        self.neighbor_loader.reset_state()
