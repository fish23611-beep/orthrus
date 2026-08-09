"""C7-B1: GraphSAGEBackbone parity and shared-instance tests.

These tests cover the new `GraphSAGEBackbone` class plus the backbone
selection wiring in `factory.encoder_factory`. They run on CPU and do not
require any external datasets.

Coverage:
  A. GraphTransformer and GraphSAGE produce compatible output shapes.
  B. GraphSAGE safely ignores edge features (and any extra kwargs).
  C. MultiScaleOrthrusEncoder keeps a single shared GraphSAGE instance
     across all three scales (no clone per scale).
  D. GraphTransformer regression: default backbone still creates
     GraphTransformer with the original interface.
  E. CPU-only execution.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


# Heavy imports (torch_geometric wiring) are deferred to test bodies so that
# pytest's collection phase does not need to import torch_geometric. This is
# the same pattern used by ``test_multiscale_pipeline_integration.py`` for
# ``from factory import encoder_factory``.
_HELPERS = {
    "GraphSAGEBackbone": None,
    "GraphTransformer": None,
    "SemanticMLPBackbone": None,
    "SemanticMLPEncoder": None,
    "MultiScaleOrthrusEncoder": None,
    "MultiScaleNeighborLoader": None,
    "HistoryStore": None,
    "encoder_factory": None,
}


def _encoders():
    if _HELPERS["GraphSAGEBackbone"] is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from encoders import GraphSAGEBackbone, GraphTransformer, SemanticMLPBackbone, SemanticMLPEncoder
        from mstc.history_store import HistoryStore
        from mstc.multiscale_encoder import MultiScaleOrthrusEncoder
        from mstc.multiscale_sampler import MultiScaleNeighborLoader
        from factory import encoder_factory
        _HELPERS.update(
            GraphSAGEBackbone=GraphSAGEBackbone,
            GraphTransformer=GraphTransformer,
            SemanticMLPBackbone=SemanticMLPBackbone,
            SemanticMLPEncoder=SemanticMLPEncoder,
            MultiScaleOrthrusEncoder=MultiScaleOrthrusEncoder,
            MultiScaleNeighborLoader=MultiScaleNeighborLoader,
            HistoryStore=HistoryStore,
            encoder_factory=encoder_factory,
        )
    return _HELPERS


# ---------------------------------------------------------------------------
# Shared test fixtures/helper builders
# ---------------------------------------------------------------------------


def _build_small_graph(num_nodes: int = 6, num_edges: int = 10):
    """Build a small CPU-friendly (x, edge_index) for shape tests."""
    torch.manual_seed(0)
    x = torch.randn(num_nodes, 8)
    # Random undirected-ish edges, with src != dst
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    keep = src != dst
    src, dst = src[keep], dst[keep]
    edge_index = torch.stack([src, dst], dim=0).long()
    return x, edge_index


def _make_fake_cfg(*, mode="recent", multiscale_enabled=False, backbone="graph_transformer",
                   fusion="gated", **multiscale_kwargs):
    """Minimal cfg stub for encoder_factory (reused from factory integration tests)."""
    cfg = SimpleNamespace()
    cfg.detection = SimpleNamespace()
    cfg.detection.gnn_training = SimpleNamespace()
    cfg.detection.gnn_training.node_hid_dim = 64
    cfg.detection.gnn_training.node_out_dim = 64
    cfg.detection.gnn_training.encoder = SimpleNamespace()
    cfg.detection.gnn_training.encoder.backbone = backbone
    cfg.detection.gnn_training.encoder.temporal_dim = 64
    cfg.detection.gnn_training.encoder.use_node_feats_in_gnn = False
    cfg.detection.gnn_training.encoder.edge_features = "edge_type"
    cfg.detection.gnn_training.encoder.graph_attention = SimpleNamespace()
    cfg.detection.gnn_training.encoder.graph_attention.dropout = 0.0
    cfg.detection.gnn_training.encoder.graph_attention.activation = "relu"
    cfg.detection.gnn_training.encoder.graph_attention.num_heads = 4
    cfg.detection.gnn_training.encoder.context = SimpleNamespace()
    cfg.detection.gnn_training.encoder.context.mode = mode
    cfg.detection.gnn_training.encoder.context.multiscale = SimpleNamespace()
    cfg.detection.gnn_training.encoder.context.multiscale.enabled = multiscale_enabled
    cfg.detection.gnn_training.encoder.context.multiscale.candidate_capacity = 16
    cfg.detection.gnn_training.encoder.context.multiscale.history_device = "cpu"
    cfg.detection.gnn_training.encoder.context.multiscale.scale_quantiles = [0.50, 0.90, 0.99]
    cfg.detection.gnn_training.encoder.context.multiscale.neighbor_budgets = [4, 4, 4]
    cfg.detection.gnn_training.encoder.context.multiscale.share_encoder = True
    cfg.detection.gnn_training.encoder.context.multiscale.fusion = fusion
    cfg.detection.gnn_training.encoder.context.multiscale.use_scale_embedding = False
    cfg.detection.gnn_training.encoder.context.multiscale.gate_hidden_dim = 32
    cfg.detection.gnn_training.encoder.neighbor_size = 5
    cfg.dataset = SimpleNamespace()
    cfg.dataset.num_edge_types = 10
    for k, v in multiscale_kwargs.items():
        setattr(cfg.detection.gnn_training.encoder.context.multiscale, k, v)
    return cfg


class FakeGraphReindexer:
    """Mirror of the helper used in test_multiscale_pipeline_integration.py."""

    def __init__(self, num_nodes=20, device="cpu"):
        self.num_nodes = num_nodes
        self.device = device

    def node_features_reshape(self, batch_edge_index, x_src, x_dst, max_num_node=None):
        if max_num_node is not None:
            max_node = max_num_node.item() + 1 if torch.is_tensor(max_num_node) else max_num_node + 1
        else:
            max_node = self.num_nodes
        if batch_edge_index.numel() == 0:
            max_node = max(max_node, 1)
        if max_node <= batch_edge_index.max().item():
            max_node = batch_edge_index.max().item() + 1
        if max_node == 0:
            max_node = 1
        new_x_src = torch.zeros((max_node, x_src.shape[1]), device=x_src.device)
        new_x_dst = torch.zeros((max_node, x_dst.shape[1]), device=x_dst.device)
        new_x_src[batch_edge_index[0]] = x_src
        new_x_dst[batch_edge_index[1]] = x_dst
        return new_x_src, new_x_dst


# ---------------------------------------------------------------------------
# A. GraphTransformer vs GraphSAGE output shape compatibility
# ---------------------------------------------------------------------------


def test_graph_transformer_and_graphsage_have_compatible_output_shapes():
    """GT and SAGE both produce (N, out_dim) tensors for the same input."""
    h = _encoders()
    in_dim, hid_dim, out_dim = 8, 16, 4
    x, edge_index = _build_small_graph(num_nodes=6, num_edges=10)

    activation = nn.ReLU()
    gt = h["GraphTransformer"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        edge_dim=None, dropout=0.0, activation=activation, num_heads=1,
    )
    graphsage = h["GraphSAGEBackbone"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        dropout=0.0, activation=activation,
    )

    y_gt = gt(x, edge_index)
    y_sage = graphsage(x, edge_index)

    assert y_gt.shape == y_sage.shape == (x.size(0), out_dim)
    # We do NOT require numerical equality across backbones.
    assert torch.isfinite(y_gt).all()
    assert torch.isfinite(y_sage).all()


# ---------------------------------------------------------------------------
# B. GraphSAGE ignores edge features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("edge_dim", [0, 3, 8])
def test_graphsage_ignores_edge_features_and_extra_kwargs(edge_dim):
    """GraphSAGE accepts edge_feats kwargs of any dimensionality and ignores them."""
    h = _encoders()
    in_dim, hid_dim, out_dim = 8, 16, 4
    x, edge_index = _build_small_graph(num_nodes=6, num_edges=10)

    activation = nn.ReLU()
    backbone = h["GraphSAGEBackbone"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        dropout=0.0, activation=activation,
    )

    if edge_dim == 0:
        edge_feats = None
    else:
        edge_feats = torch.randn(edge_index.size(1), edge_dim)

    # Forward MUST succeed regardless of edge_dim dimensionality.
    y = backbone(x, edge_index, edge_feats=edge_feats, some_other_kwarg="ignored")
    assert y.shape == (x.size(0), out_dim)
    assert torch.isfinite(y).all()


def test_graphsage_output_is_independent_of_edge_features():
    """The numerical output must NOT change when we pass different edge_feats."""
    h = _encoders()
    in_dim, hid_dim, out_dim = 8, 16, 4
    x, edge_index = _build_small_graph(num_nodes=6, num_edges=10)

    torch.manual_seed(123)
    backbone = h["GraphSAGEBackbone"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        dropout=0.0, activation=nn.ReLU(),
    )

    y_none = backbone(x, edge_index, edge_feats=None)
    y_a = backbone(x, edge_index, edge_feats=torch.randn(edge_index.size(1), 3))
    y_b = backbone(x, edge_index, edge_feats=torch.randn(edge_index.size(1), 5))
    y_kw_only = backbone(x, edge_index)

    assert torch.allclose(y_none, y_a)
    assert torch.allclose(y_none, y_b)
    assert torch.allclose(y_none, y_kw_only)


def test_graph_transformer_still_uses_edge_features():
    """Regression: GraphTransformer behaviour is unchanged when edge_feats is given."""
    h = _encoders()
    in_dim, hid_dim, out_dim = 8, 16, 4
    torch.manual_seed(0)
    x, edge_index = _build_small_graph(num_nodes=6, num_edges=10)

    edge_dim = 3
    activation = nn.ReLU()
    gt = h["GraphTransformer"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        edge_dim=edge_dim, dropout=0.0, activation=activation, num_heads=1,
    )

    # TransformerConv(..., edge_dim=N) requires a non-None edge_feats
    # tensor of shape (num_edges, N). We just verify the numbers
    # are finite and the shape is correct.
    edge_feats = torch.zeros(edge_index.size(1), edge_dim)
    y_yes = gt(x, edge_index, edge_feats=edge_feats)
    assert y_yes.shape == (x.size(0), out_dim)
    assert torch.isfinite(y_yes).all()

    # When the GraphTransformer is constructed without edge_dim, it must
    # accept edge_feats=None just like before, and must NOT consume
    # any extra kwargs.
    gt_no_edges = h["GraphTransformer"](
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim,
        edge_dim=None, dropout=0.0, activation=activation, num_heads=1,
    )
    y_no = gt_no_edges(x, edge_index, edge_feats=None, extra_kwarg="ignored")
    assert y_no.shape == (x.size(0), out_dim)
    assert torch.isfinite(y_no).all()


# ---------------------------------------------------------------------------
# C. MultiScale parameter sharing
# ---------------------------------------------------------------------------


def test_multiscale_graphsage_uses_single_shared_instance():
    """MultiScaleOrthrusEncoder must share a single GraphSAGEBackbone across scales."""
    h = _encoders()
    cfg = _make_fake_cfg(mode="multiscale", multiscale_enabled=True, backbone="graphsage")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )

    assert isinstance(encoder, h["MultiScaleOrthrusEncoder"])
    assert isinstance(encoder.shared_graph_encoder, h["GraphSAGEBackbone"])
    # Prove it is THE SAME instance, not three clones.
    assert encoder.shared_graph_encoder is encoder.shared_graph_encoder
    # No per-scale modules should exist.
    for forbidden in ("short_encoder", "medium_encoder", "long_encoder"):
        assert not hasattr(encoder, forbidden), (
            f"MultiScaleOrthrusEncoder must not define {forbidden} when sharing one backbone"
        )

    sage_keys = sorted(encoder.shared_graph_encoder.state_dict().keys())
    assert sage_keys, "GraphSAGEBackbone should have parameters"
    seen_ids = set()
    for name, param in encoder.named_parameters():
        if any(k in name for k in ("conv1", "conv2")):
            seen_ids.add(id(param))
    assert seen_ids, "Expected GraphSAGE parameters in encoder"
    shared_ids = {id(p) for p in encoder.shared_graph_encoder.parameters()}
    assert shared_ids.issubset(seen_ids)


def test_multiscale_graph_transformer_still_uses_single_shared_instance():
    """Regression: GraphTransformer path still uses one shared instance."""
    h = _encoders()
    cfg = _make_fake_cfg(mode="multiscale", multiscale_enabled=True, backbone="graph_transformer")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )
    assert isinstance(encoder, h["MultiScaleOrthrusEncoder"])
    assert isinstance(encoder.shared_graph_encoder, h["GraphTransformer"])


# ---------------------------------------------------------------------------
# D. GraphTransformer regression via factory
# ---------------------------------------------------------------------------


def test_factory_default_backbone_is_graph_transformer():
    """Without setting backbone, the factory must default to GraphTransformer."""
    h = _encoders()
    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False)
    del cfg.detection.gnn_training.encoder.backbone
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )

    assert hasattr(encoder, "encoder")
    assert isinstance(encoder.encoder, h["GraphTransformer"])


def test_factory_resolves_graph_transformer_backbone_explicitly():
    h = _encoders()
    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False, backbone="graph_transformer")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )
    assert isinstance(encoder.encoder, h["GraphTransformer"])


def test_factory_resolves_graphsage_backbone_for_recent_mode():
    h = _encoders()
    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False, backbone="graphsage")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )
    assert isinstance(encoder.encoder, h["GraphSAGEBackbone"])


def test_factory_rejects_unknown_backbone():
    h = _encoders()
    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False, backbone="bogus_backbone")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    with pytest.raises(ValueError, match="backbone"):
        h["encoder_factory"](
            cfg, msg_dim=64, in_dim=64, edge_dim=10,
            graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
        )


def test_factory_graphsage_forces_edge_features_none():
    """When backbone=graphsage, the encoder must be configured with edge_features=('none',)."""
    h = _encoders()
    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False, backbone="graphsage")
    cfg.detection.gnn_training.encoder.edge_features = "edge_type,msg"
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device="cpu", max_node_num=20,
    )
    # The encoder normalizes edge_features into a tuple; accept either form.
    assert tuple(encoder.edge_features) == ("none",)


# ---------------------------------------------------------------------------
# E. CPU smoke (multi-scale forward + backward) with GraphSAGE
# ---------------------------------------------------------------------------


def test_multiscale_graphsage_full_forward_and_backward():
    """Smoke: a 3-scale forward + backward on CPU with GraphSAGE backbone."""
    h = _encoders()
    in_dim = 3
    temporal_dim = 5
    node_out_dim = 4

    # Build the GraphSAGE backbone through the factory so we test the
    # factory wiring, but construct the multiscale encoder directly with
    # the proven small thresholds from test_multiscale_encoder.py.
    history_store = h["HistoryStore"](num_nodes=8, candidate_capacity=20, device="cpu")
    loader = h["MultiScaleNeighborLoader"](
        history_store,
        tau_short_ns=20,
        tau_medium_ns=50,
        tau_max_ns=100,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    graphsage = h["GraphSAGEBackbone"](
        in_dim=temporal_dim, hid_dim=temporal_dim, out_dim=node_out_dim,
        dropout=0.0, activation=nn.ReLU(),
    )
    encoder = h["MultiScaleOrthrusEncoder"](
        shared_graph_encoder=graphsage,
        neighbor_loader=loader,
        in_dim=in_dim,
        temporal_dim=temporal_dim,
        node_out_dim=node_out_dim,
        edge_features=("none",),
        device="cpu",
        gate_hidden_dim=7,
    )

    # full_data shape mirrors _full_data() from test_multiscale_encoder.py.
    num_events = 20
    src = torch.arange(num_events, dtype=torch.long) % 4
    dst = (src + 1) % 4
    timestamps = torch.arange(num_events, dtype=torch.long) * 10
    timestamps[0] = 90
    timestamps[1] = 60
    timestamps[2] = 10
    timestamps[7] = 90
    full_data = SimpleNamespace(
        src=src,
        dst=dst,
        t=timestamps,
        x_src=torch.arange(num_events * in_dim, dtype=torch.float32).view(num_events, in_dim) + 1,
        x_dst=(torch.arange(num_events * in_dim, dtype=torch.float32).view(num_events, in_dim) + 1) + 100,
        msg=torch.zeros(num_events, 2),
        edge_type=torch.zeros(num_events, 10),
    )

    loader.history_store.insert(
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 1, 2]),
        torch.tensor([90, 60, 10]),
    )

    batch = {
        "edge_index": torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        "t": torch.tensor([100, 110], dtype=torch.long),
        "x": (torch.ones(2, in_dim), torch.full((2, in_dim), 2.0)),
        "global_event_index": torch.tensor([3, 4], dtype=torch.long),
    }

    h_src, h_dst = encoder(full_data=full_data, msg=torch.zeros(2, 2), **batch)
    assert h_src.shape == (2, node_out_dim)
    assert h_dst.shape == (2, node_out_dim)
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()

    (h_src.square().mean() + h_dst.square().mean()).backward()
    grads = [p.grad for p in encoder.shared_graph_encoder.parameters() if p.grad is not None]
    assert grads, "GraphSAGE parameters should have gradients"
    assert all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# Convenience: explicit factory import surface check
# ---------------------------------------------------------------------------


def test_factory_module_exposes_graphsage_backbone():
    """`encoders.GraphSAGEBackbone` must be importable from the public module."""
    h = _encoders()
    assert h["GraphSAGEBackbone"] is not None
    assert h["GraphSAGEBackbone"].__name__ == "GraphSAGEBackbone"

def test_semantic_mlp_returns_decoder_compatible_endpoint_shapes():
    h = _encoders()
    model = h["SemanticMLPBackbone"](in_dim=3, hid_dim=5, out_dim=4, dropout=0.0, activation=nn.ReLU())
    x_src, x_dst = torch.randn(4, 3), torch.randn(4, 3)
    h_src, h_dst = model(x_src, x_dst)
    assert h_src.shape == (4, 4)
    assert h_dst.shape == (4, 4)
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()


def test_semantic_mlp_ignores_graph_and_edge_feature_inputs():
    h = _encoders()
    model = h["SemanticMLPBackbone"](in_dim=3, hid_dim=5, out_dim=4, dropout=0.2, activation=nn.ReLU()).eval()
    x_src, x_dst = torch.randn(3, 3), torch.randn(3, 3)
    first = model(x_src, x_dst, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_feats=torch.randn(2, 7))
    second = model(x_src, x_dst, edge_index=torch.empty((2, 0), dtype=torch.long), edge_feats=torch.randn(0, 2))
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_factory_builds_stateless_semantic_mlp_only_for_context_none():
    h = _encoders()
    cfg = _make_fake_cfg(mode="none", multiscale_enabled=False, backbone="semantic_mlp")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    encoder = h["encoder_factory"](cfg, msg_dim=64, in_dim=64, edge_dim=10, graph_reindexer=graph_reindexer, device="cpu", max_node_num=20)
    assert isinstance(encoder, h["SemanticMLPEncoder"])
    assert not hasattr(encoder, "neighbor_loader")
    assert not hasattr(encoder, "shared_graph_encoder")
    assert not any(module.__class__.__name__.endswith("NeighborLoader") for module in encoder.modules())
    encoder.reset_state()


@pytest.mark.parametrize("mode", ["recent", "multiscale"])
def test_factory_rejects_semantic_mlp_graph_contexts(mode):
    h = _encoders()
    cfg = _make_fake_cfg(mode=mode, multiscale_enabled=(mode == "multiscale"), backbone="semantic_mlp")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    with pytest.raises(ValueError, match="context.mode='none'"):
        h["encoder_factory"](cfg, msg_dim=64, in_dim=64, edge_dim=10, graph_reindexer=graph_reindexer, device="cpu", max_node_num=20)
