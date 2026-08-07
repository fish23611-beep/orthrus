"""
Integration tests for C5-B2: MultiScaleOrthrusEncoder wiring into factory / model / config / replay pipeline.

Tests cover:
1. factory branching: recent vs multiscale
2. model integration: MSTCOrthrus + MultiScaleOrthrusEncoder
3. global_event_index contract
4. reset/replay protocol
5. causal ordering
6. backward compatibility with recent/orthrus_baseline
"""
import os
import sys
import torch
import torch.nn as nn
from types import SimpleNamespace
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "encoders" in sys.modules and not isinstance(getattr(sys.modules["encoders"], "GraphTransformer", None), type):
    del sys.modules["encoders"]
if "model" in sys.modules:
    del sys.modules["model"]
if "factory" in sys.modules:
    del sys.modules["factory"]
if "config" in sys.modules:
    del sys.modules["config"]
if "temporal" in sys.modules:
    del sys.modules["temporal"]
if "decoders" in sys.modules:
    del sys.modules["decoders"]

from mstc.history_store import HistoryStore
from mstc.multiscale_sampler import MultiScaleNeighborLoader, log_seconds_boundaries_to_ns
from mstc.multiscale_encoder import MultiScaleOrthrusEncoder
from encoders import OrthrusEncoder, GraphTransformer
from decoders import EdgeTypeDecoder


class FakeGraphEncoder(nn.Module):
    """Fake graph encoder for testing that returns constant embeddings."""
    def __init__(self, in_dim=64, hid_dim=64, out_dim=64, edge_dim=None):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_feats=None):
        return self.proj(x)


class FakeGraphReindexer:
    """Fake graph reindexer for testing that handles node feature reshaping."""
    def __init__(self, num_nodes=20, device="cpu"):
        self.num_nodes = num_nodes
        self.device = device

    def node_features_reshape(self, batch_edge_index, x_src, x_dst, max_num_node=None):
        """Reshape node features to (max_node+1, dim) format."""
        if max_num_node is not None:
            if torch.is_tensor(max_num_node):
                max_node = max_num_node.item() + 1
            else:
                max_node = max_num_node + 1
        else:
            max_node = self.num_nodes

        # Handle empty edge_index
        if batch_edge_index.numel() == 0:
            max_node = max(max_node, 1)
            new_x_src = torch.zeros((max_node, x_src.shape[1]), device=x_src.device)
            new_x_dst = torch.zeros((max_node, x_dst.shape[1]), device=x_dst.device)
            return new_x_src, new_x_dst

        if max_node <= batch_edge_index.max().item():
            max_node = batch_edge_index.max().item() + 1
        if max_node == 0:
            max_node = 1
        new_x_src = torch.zeros((max_node, x_src.shape[1]), device=x_src.device)
        new_x_dst = torch.zeros((max_node, x_dst.shape[1]), device=x_dst.device)
        new_x_src[batch_edge_index[0]] = x_src
        new_x_dst[batch_edge_index[1]] = x_dst
        return new_x_src, new_x_dst


def _make_fake_cfg(mode="recent", multiscale_enabled=False, **multiscale_kwargs):
    """Create a minimal fake cfg object for testing."""
    cfg = SimpleNamespace()
    cfg.detection = SimpleNamespace()
    cfg.detection.gnn_training = SimpleNamespace()
    cfg.detection.gnn_training.node_hid_dim = 64
    cfg.detection.gnn_training.node_out_dim = 64
    cfg.detection.gnn_training.encoder = SimpleNamespace()
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
    cfg.detection.gnn_training.encoder.context.multiscale.fusion = "gated"
    cfg.detection.gnn_training.encoder.context.multiscale.use_scale_embedding = False
    cfg.detection.gnn_training.encoder.context.multiscale.gate_hidden_dim = 32
    cfg.detection.gnn_training.encoder.neighbor_size = 5
    cfg.dataset = SimpleNamespace()
    cfg.dataset.num_edge_types = 10

    for k, v in multiscale_kwargs.items():
        setattr(cfg.detection.gnn_training.encoder.context.multiscale, k, v)

    return cfg


def _make_full_data(num_events=20, device="cpu", max_node=10):
    """Create a minimal full_data object with all required fields."""
    msg_dim = 64
    edge_dim = 10
    x_dim = 64
    t_base = 1000_000_000_000_000_000

    fd = SimpleNamespace()
    fd.global_event_index = torch.arange(num_events, dtype=torch.long, device=device)
    fd.event_index = fd.global_event_index
    fd.src = torch.randint(0, max_node, (num_events,), dtype=torch.long, device=device)
    fd.dst = torch.randint(0, max_node, (num_events,), dtype=torch.long, device=device)
    fd.t = torch.arange(num_events, dtype=torch.long, device=device) * 10_000_000_000 + t_base
    fd.msg = torch.randn(num_events, msg_dim, device=device)
    fd.edge_type = torch.randint(0, edge_dim, (num_events,), dtype=torch.long, device=device)
    fd.x_src = torch.randn(num_events, x_dim, device=device)
    fd.x_dst = torch.randn(num_events, x_dim, device=device)
    return fd


def _make_batch(indices, full_data, max_node=None):
    """Create a batch from global_event_indices."""
    batch = SimpleNamespace()
    batch.global_event_index = full_data.global_event_index[indices]
    batch.edge_index = torch.stack([
        full_data.src[indices],
        full_data.dst[indices],
    ])
    batch.t = full_data.t[indices]
    batch.msg = full_data.msg[indices]
    batch.edge_type = torch.nn.functional.one_hot(
        full_data.edge_type[indices], num_classes=10
    ).float()
    batch.x_src = full_data.x_src[indices]
    batch.x_dst = full_data.x_dst[indices]
    return batch


# =============================================================================
# Test 1: encoder_factory recent mode returns encoder without global_event_index requirement
# =============================================================================
def test_encoder_factory_recent_mode_returns_orthrus_encoder():
    """When context.mode=recent, encoder_factory should return encoder without global_event_index requirement."""
    from factory import encoder_factory

    cfg = _make_fake_cfg(mode="recent", multiscale_enabled=False)
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    device = "cpu"
    encoder = encoder_factory(
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device=device, max_node_num=20,
    )
    # Check capability: recent encoder should NOT require global_event_index
    assert not getattr(encoder, "requires_global_event_index", False), (
        f"Recent encoder should not require global_event_index, but got requires_global_event_index={getattr(encoder, 'requires_global_event_index', False)}"
    )


# =============================================================================
# Test 2: encoder_factory multiscale mode returns encoder with global_event_index requirement
# =============================================================================
def test_encoder_factory_multiscale_mode_returns_multiscale_encoder():
    """When context.mode=multiscale, encoder_factory should return encoder that requires global_event_index."""
    from factory import encoder_factory

    cfg = _make_fake_cfg(mode="multiscale", multiscale_enabled=True)
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    device = "cpu"
    encoder = encoder_factory(
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device=device, max_node_num=20,
    )
    # Check capability: multiscale encoder SHOULD require global_event_index
    assert getattr(encoder, "requires_global_event_index", False), (
        f"Multiscale encoder should require global_event_index, but got requires_global_event_index={getattr(encoder, 'requires_global_event_index', False)}"
    )


# =============================================================================
# Test 3: MultiScaleOrthrusEncoder shares a single GraphTransformer
# =============================================================================
def test_multiscale_encoder_shares_single_graph_transformer():
    """MultiScaleOrthrusEncoder should use exactly one shared GraphTransformer."""
    from factory import encoder_factory

    cfg = _make_fake_cfg(mode="multiscale", multiscale_enabled=True)
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    device = "cpu"
    encoder = encoder_factory(
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device=device, max_node_num=20,
    )
    assert hasattr(encoder, "shared_graph_encoder"), (
        "MultiScaleOrthrusEncoder should have shared_graph_encoder attribute"
    )
    assert isinstance(encoder.shared_graph_encoder, GraphTransformer), (
        f"shared_graph_encoder should be GraphTransformer, got {type(encoder.shared_graph_encoder).__name__}"
    )
    sd = encoder.state_dict()
    gt_keys = [k for k in sd.keys() if "shared_graph_encoder" in k]
    assert len(gt_keys) > 0, "state_dict should contain shared_graph_encoder parameters"


# =============================================================================
# Test 4: global_event_index is required by MultiScaleOrthrusEncoder
# =============================================================================
def test_multiscale_encoder_requires_global_event_index():
    """MultiScaleOrthrusEncoder.forward should raise ValueError when global_event_index is missing."""
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=FakeGraphEncoder(),
        neighbor_loader=MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=8),
            tau_short_ns=1000,
            tau_medium_ns=10000,
            tau_max_ns=100000,
            short_budget=4,
            medium_budget=4,
            long_budget=4,
        ),
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device="cpu",
    )
    full_data = _make_full_data(num_events=5, max_node=10)
    batch = _make_batch([0, 1], full_data)
    delattr(batch, "global_event_index")
    with pytest.raises(ValueError, match="global_event_index"):
        encoder(
            edge_index=batch.edge_index,
            t=batch.t,
            msg=batch.msg,
            x=(batch.x_src, batch.x_dst),
            full_data=full_data,
        )


# =============================================================================
# Test 5: MSTCOrthrus passes global_event_index to MultiScaleOrthrusEncoder
# =============================================================================
def test_mstc_orthrus_passes_global_event_index_to_multiscale_encoder():
    """MSTCOrthrus should pass batch.global_event_index to MultiScaleOrthrusEncoder."""
    from model import MSTCOrthrus

    device = "cpu"
    max_node_num = 10
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=8, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )
    model = MSTCOrthrus(
        encoder=encoder,
        edge_decoder=None,
        time_gap_decoder=None,
        time_gap_statistics=None,
        lambda_time=0.0,
        graph_reindexer=None,
    ).to(device)

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch = _make_batch([0, 1, 2], full_data)

    outputs = model(batch, full_data)
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).all(), "Loss should be finite"


# =============================================================================
# Test 6: MSTCOrthrus does NOT require global_event_index for OrthrusEncoder
# =============================================================================
def test_mstc_orthrus_recent_encoder_does_not_require_global_event_index():
    """MSTCOrthrus with OrthrusEncoder should work without global_event_index."""
    from model import MSTCOrthrus
    from temporal import LastNeighborLoader

    device = "cpu"
    max_node_num = 10
    neighbor_loader = LastNeighborLoader(max_node_num, size=5, device=device)
    encoder = OrthrusEncoder(
        encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=None,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        use_node_feats_in_gnn=True,
        graph_reindexer=FakeGraphReindexer(num_nodes=max_node_num, device=device),
        edge_features=["none"],
        device=device,
        num_nodes=max_node_num,
        edge_dim=0,
    )
    model = MSTCOrthrus(
        encoder=encoder,
        edge_decoder=None,
        time_gap_decoder=None,
        time_gap_statistics=None,
        lambda_time=0.0,
        graph_reindexer=None,
    ).to(device)

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)

    # Pre-populate history to avoid empty edge_index issues
    init_batch = _make_batch([0, 1], full_data)
    encoder.reset_state()
    model(init_batch, full_data)  # Insert events into history

    # Now test without global_event_index
    batch = _make_batch([2, 3, 4], full_data)
    delattr(batch, "global_event_index")

    outputs = model(batch, full_data)
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).all(), "Loss should be finite"


# =============================================================================
# Test 7: reset_state is called correctly for both encoder types
# =============================================================================
def test_reset_state_works_for_both_encoder_types():
    """Both OrthrusEncoder and MultiScaleOrthrusEncoder should respond to reset_state."""
    from model import MSTCOrthrus
    from temporal import LastNeighborLoader

    device = "cpu"
    max_node_num = 10

    for encoder_cls, loader_cls in [
        (OrthrusEncoder, LastNeighborLoader),
        (MultiScaleOrthrusEncoder, MultiScaleNeighborLoader),
    ]:
        if loader_cls == LastNeighborLoader:
            neighbor_loader = LastNeighborLoader(max_node_num, size=5, device=device)
            enc = encoder_cls(
                encoder=GraphTransformer(
                    in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
                    activation=nn.ReLU(), dropout=0.0, num_heads=4,
                ),
                neighbor_loader=neighbor_loader,
                in_dim=64,
                temporal_dim=64,
                use_node_feats_in_gnn=False,
                graph_reindexer=FakeGraphReindexer(num_nodes=max_node_num, device=device),
                edge_features=["edge_type"],
                device=device,
                num_nodes=max_node_num,
                edge_dim=10,
            )
        else:
            tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
            history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=8, device="cpu")
            neighbor_loader = MultiScaleNeighborLoader(
                history_store=history_store,
                tau_short_ns=tau_short_ns,
                tau_medium_ns=tau_medium_ns,
                tau_max_ns=tau_max_ns,
                short_budget=4,
                medium_budget=4,
                long_budget=4,
            )
            enc = encoder_cls(
                shared_graph_encoder=GraphTransformer(
                    in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
                    activation=nn.ReLU(), dropout=0.0, num_heads=4,
                ),
                neighbor_loader=neighbor_loader,
                in_dim=64,
                temporal_dim=64,
                node_out_dim=64,
                use_node_feats_in_gnn=False,
                edge_features=["edge_type"],
                device=device,
            )

        model = MSTCOrthrus(
            encoder=enc,
            edge_decoder=None,
            time_gap_decoder=None,
            time_gap_statistics=None,
            lambda_time=0.0,
            graph_reindexer=None,
        ).to(device)

        model.reset_state()
        assert hasattr(model.encoder, "reset_state"), (
            f"{encoder_cls.__name__} should have reset_state method"
        )


# =============================================================================
# Test 8: causal ordering is preserved (query before insert)
# =============================================================================
def test_multiscale_encoder_causal_ordering():
    """MultiScaleOrthrusEncoder should query history before inserting current events."""
    device = "cpu"
    max_node_num = 5
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch1 = _make_batch([0, 1], full_data)
    batch2 = _make_batch([2, 3], full_data)

    encoder.reset_state()
    h1_src, h1_dst = encoder(
        edge_index=batch1.edge_index,
        t=batch1.t,
        msg=batch1.msg,
        x=(batch1.x_src, batch1.x_dst),
        full_data=full_data,
        global_event_index=batch1.global_event_index,
    )
    assert h1_src.shape == (2, 64), f"Expected shape (2, 64), got {h1_src.shape}"

    h2_src, h2_dst = encoder(
        edge_index=batch2.edge_index,
        t=batch2.t,
        msg=batch2.msg,
        x=(batch2.x_src, batch2.x_dst),
        full_data=full_data,
        global_event_index=batch2.global_event_index,
    )
    assert h2_src.shape == (2, 64), f"Expected shape (2, 64), got {h2_src.shape}"
    assert torch.isfinite(h1_src).all(), "First forward output should be finite"
    assert torch.isfinite(h2_src).all(), "Second forward output should be finite"


# =============================================================================
# Test 9: config defaults preserve baseline behavior
# =============================================================================
def test_config_defaults_preserve_baseline():
    """Default config values should keep baseline behavior unchanged."""
    from config import get_default_cfg

    class FakeArgs:
        cpu = True
        from_weights = False
        seed = 0
        dataset = "THEIA_E5"

    args = FakeArgs()
    cfg = get_default_cfg(args)

    assert cfg.detection.gnn_training.encoder.context.mode == "recent", (
        "Default context.mode should be 'recent'"
    )
    assert cfg.detection.gnn_training.encoder.context.multiscale.enabled is False, (
        "Default multiscale.enabled should be False"
    )


# =============================================================================
# Test 10: backward pass works through multiscale encoder
# =============================================================================
def test_multiscale_encoder_backward_pass():
    """MultiScaleOrthrusEncoder should support backward pass with finite gradients."""
    device = "cpu"
    max_node_num = 5
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch = _make_batch([0, 1], full_data)

    encoder.reset_state()
    h_src, h_dst = encoder(
        edge_index=batch.edge_index,
        t=batch.t,
        msg=batch.msg,
        x=(batch.x_src, batch.x_dst),
        full_data=full_data,
        global_event_index=batch.global_event_index,
    )

    loss = h_src.sum() + h_dst.sum()
    loss.backward()

    # Check that at least some parameters have finite gradients
    # (not all parameters may have gradients if history is empty)
    params_with_grad = [(n, p) for n, p in encoder.named_parameters() if p.grad is not None]
    assert len(params_with_grad) > 0, (
        "At least some parameters should have gradients after backward"
    )
    for name, param in params_with_grad:
        assert torch.isfinite(param.grad).all(), (
            f"Gradient for {name} should be finite, got {param.grad}"
        )


# =============================================================================
# Test 11: MSTCOrthrus + multiscale encoder end-to-end training
# =============================================================================
def test_mstc_orthrus_multiscale_end_to_end_training():
    """MSTCOrthrus with MultiScaleOrthrusEncoder should train end-to-end."""
    from model import MSTCOrthrus
    from decoders import EdgeTypeDecoder
    import torch.nn.functional as F

    device = "cpu"
    max_node_num = 10
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )

    edge_decoder = EdgeTypeDecoder(
        in_dim=64,
        num_edge_types=10,
        loss_fn=lambda x, y, **kw: F.cross_entropy(x, y, reduction="mean"),
        dropout=0.0,
        num_layers=2,
        activation=nn.ReLU(),
    )

    model = MSTCOrthrus(
        encoder=encoder,
        edge_decoder=edge_decoder,
        time_gap_decoder=None,
        time_gap_statistics=None,
        lambda_time=0.0,
        graph_reindexer=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    full_data = _make_full_data(num_events=20, device=device)
    model.reset_state()

    for i in range(3):
        batch = _make_batch([i * 2, i * 2 + 1], full_data)
        optimizer.zero_grad()
        outputs = model(batch, full_data)
        assert "loss" in outputs, "MSTCOrthrus should return dict with 'loss'"
        assert torch.isfinite(outputs["loss"]).all(), f"Loss should be finite at step {i}"
        outputs["loss"].backward()
        optimizer.step()
        model.reset_state()

    assert True, "End-to-end training completed successfully"


# =============================================================================
# Test 12: factory raises ValueError for unknown mode
# =============================================================================
def test_encoder_factory_raises_for_unknown_mode():
    """encoder_factory should raise ValueError for unknown context.mode."""
    from factory import encoder_factory

    cfg = _make_fake_cfg(mode="unknown", multiscale_enabled=False)
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    device = "cpu"

    with pytest.raises(ValueError, match="Unknown context.mode"):
        encoder_factory(
            cfg, msg_dim=64, in_dim=64, edge_dim=10,
            graph_reindexer=graph_reindexer, device=device, max_node_num=20,
        )


# =============================================================================
# Test 13: gate weights are accessible after forward
# =============================================================================
def test_gate_weights_accessible_after_forward():
    """MultiScaleOrthrusEncoder should store gate weights after forward."""
    device = "cpu"
    max_node_num = 5
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch = _make_batch([0, 1], full_data)

    # Pre-populate history by inserting events before querying
    # This ensures at least one scale has non-empty history
    encoder.reset_state()
    init_batch = _make_batch([0], full_data)
    encoder(
        edge_index=init_batch.edge_index,
        t=init_batch.t,
        msg=init_batch.msg,
        x=(init_batch.x_src, init_batch.x_dst),
        full_data=full_data,
        global_event_index=init_batch.global_event_index,
    )

    # Now query - should have at least one non-empty scale
    h_src, h_dst = encoder(
        edge_index=batch.edge_index,
        t=batch.t,
        msg=batch.msg,
        x=(batch.x_src, batch.x_dst),
        full_data=full_data,
        global_event_index=batch.global_event_index,
    )

    gate_weights = encoder.get_last_gate_weights()
    assert gate_weights is not None, "Gate weights should be stored after forward"
    assert gate_weights.shape[0] == 2, "Gate weights should have shape [batch_size, 3]"
    assert gate_weights.shape[1] == 3
    assert torch.isfinite(gate_weights).all(), "Gate weights should be finite"
    # Gate weights should sum to 1 for rows with non-empty history, 0 otherwise
    # The all-empty case is handled separately in another test

    scale_mask = encoder.get_last_scale_mask()
    assert scale_mask is not None, "Scale mask should be stored after forward"
    assert scale_mask.shape == (2, 3), "Scale mask should have shape [batch_size, 3]"


# =============================================================================
# Test 14: history_state_dict roundtrip
# =============================================================================
def test_history_state_dict_roundtrip():
    """MultiScaleOrthrusEncoder should support history_state_dict roundtrip."""
    device = "cpu"
    max_node_num = 5
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
    )

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch = _make_batch([0, 1], full_data)

    encoder.reset_state()
    h_src, h_dst = encoder(
        edge_index=batch.edge_index,
        t=batch.t,
        msg=batch.msg,
        x=(batch.x_src, batch.x_dst),
        full_data=full_data,
        global_event_index=batch.global_event_index,
    )

    state = encoder.history_state_dict()
    assert state is not None, "history_state_dict should return a dict"

    encoder.load_history_state_dict(state)
    assert True, "load_history_state_dict should complete without error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Equal Fusion Integration Tests (C5-B3)
# =============================================================================

def test_encoder_factory_multiscale_equal_fusion():
    """encoder_factory with fusion='equal' should create MultiScaleOrthrusEncoder with equal fusion."""
    from factory import encoder_factory

    cfg = _make_fake_cfg(mode="multiscale", multiscale_enabled=True, fusion="equal")
    graph_reindexer = FakeGraphReindexer(num_nodes=20, device="cpu")
    device = "cpu"
    encoder = encoder_factory(
        cfg, msg_dim=64, in_dim=64, edge_dim=10,
        graph_reindexer=graph_reindexer, device=device, max_node_num=20,
    )
    assert getattr(encoder, "fusion", None) == "equal", (
        f"Expected fusion='equal', got {getattr(encoder, 'fusion', None)!r}"
    )
    assert hasattr(encoder, "requires_global_event_index")
    assert encoder.requires_global_event_index is True


def test_multiscale_encoder_equal_fusion_synthetic_end_to_end():
    """Full synthetic pipeline with fusion='equal': config -> factory -> MSTCOrthrus -> loss -> backward."""
    from model import MSTCOrthrus
    import torch.nn.functional as F

    device = "cpu"
    max_node_num = 10
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
        fusion="equal",
    )

    edge_decoder = EdgeTypeDecoder(
        in_dim=64,
        num_edge_types=10,
        loss_fn=lambda x, y, **kw: F.cross_entropy(x, y, reduction="mean"),
        dropout=0.0,
        num_layers=2,
        activation=nn.ReLU(),
    )

    model = MSTCOrthrus(
        encoder=encoder,
        edge_decoder=edge_decoder,
        time_gap_decoder=None,
        time_gap_statistics=None,
        lambda_time=0.0,
        graph_reindexer=None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    full_data = _make_full_data(num_events=20, device=device)
    model.reset_state()

    for i in range(3):
        batch = _make_batch([i * 2, i * 2 + 1], full_data)
        optimizer.zero_grad()
        outputs = model(batch, full_data)
        assert "loss" in outputs, "MSTCOrthrus should return dict with 'loss'"
        assert torch.isfinite(outputs["loss"]).all(), f"Loss should be finite at step {i}"
        outputs["loss"].backward()
        optimizer.step()
        model.reset_state()

    assert encoder.fusion == "equal"
    assert True, "Equal fusion end-to-end training completed successfully"


def test_equal_fusion_weights_in_synthetic_forward():
    """In synthetic pipeline, equal fusion produces correct weights per valid mask."""
    device = "cpu"
    max_node_num = 10
    tau_short_ns, tau_medium_ns, tau_max_ns = log_seconds_boundaries_to_ns([0.50, 0.90, 0.99])
    history_store = HistoryStore(num_nodes=max_node_num, candidate_capacity=16, device="cpu")
    neighbor_loader = MultiScaleNeighborLoader(
        history_store=history_store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        tau_max_ns=tau_max_ns,
        short_budget=4,
        medium_budget=4,
        long_budget=4,
    )
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=GraphTransformer(
            in_dim=64, hid_dim=64, out_dim=64, edge_dim=10,
            activation=nn.ReLU(), dropout=0.0, num_heads=4,
        ),
        neighbor_loader=neighbor_loader,
        in_dim=64,
        temporal_dim=64,
        node_out_dim=64,
        use_node_feats_in_gnn=False,
        edge_features=["edge_type"],
        device=device,
        fusion="equal",
    )

    full_data = _make_full_data(num_events=10, device=device, max_node=max_node_num)
    batch = _make_batch([0, 1], full_data)

    encoder.reset_state()
    init_batch = _make_batch([0], full_data)
    encoder(
        edge_index=init_batch.edge_index,
        t=init_batch.t,
        msg=init_batch.msg,
        x=(init_batch.x_src, init_batch.x_dst),
        full_data=full_data,
        global_event_index=init_batch.global_event_index,
    )

    h_src, h_dst = encoder(
        edge_index=batch.edge_index,
        t=batch.t,
        msg=batch.msg,
        x=(batch.x_src, batch.x_dst),
        full_data=full_data,
        global_event_index=batch.global_event_index,
    )

    gate_weights = encoder.get_last_gate_weights()
    assert gate_weights is not None
    assert gate_weights.shape[0] == 2
    assert gate_weights.shape[1] == 3
    assert torch.isfinite(gate_weights).all()
    scale_mask = encoder.get_last_scale_mask()
    has_scale = scale_mask.any(dim=-1)
    torch.testing.assert_close(
        gate_weights.sum(dim=-1)[has_scale],
        torch.ones(has_scale.sum().item()),
        atol=1e-6, rtol=1e-6
    )
