from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "encoders" in sys.modules and not isinstance(getattr(sys.modules["encoders"], "GraphTransformer", None), type):
    del sys.modules["encoders"]
from encoders import GraphTransformer
from mstc.history_store import HistoryStore
from mstc.multiscale_encoder import MultiScaleOrthrusEncoder
from mstc.multiscale_sampler import MultiScaleNeighborLoader


class FakeGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.edge_projection = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        self.call_count = 0
        self.calls: list[dict[str, Tensor | None]] = []

    def forward(self, x: Tensor, edge_index: Tensor, edge_feats: Tensor | None = None, **kwargs) -> Tensor:
        self.call_count += 1
        self.calls.append({"x": x.detach().clone(), "edge_index": edge_index.detach().clone(), "edge_feats": None if edge_feats is None else edge_feats.detach().clone()})
        output = self.projection(x)
        if self.edge_projection is not None and edge_feats is not None and edge_index.numel():
            messages = self.edge_projection(edge_feats)
            output = output.clone()
            output.index_add_(0, edge_index[1], messages)
        return output


def _full_data(num_events: int = 20, in_dim: int = 3, edge_dim: int = 2) -> SimpleNamespace:
    src = torch.arange(num_events, dtype=torch.long) % 4
    dst = (src + 1) % 4
    timestamps = torch.arange(num_events, dtype=torch.long) * 10
    timestamps[0] = 90
    timestamps[1] = 60
    timestamps[2] = 10
    timestamps[7] = 90
    timestamps[10] = 90
    timestamps[11] = 90
    x_src = torch.arange(num_events * in_dim, dtype=torch.float32).view(num_events, in_dim) + 1
    x_dst = x_src + 100
    msg = torch.arange(num_events * 2, dtype=torch.float32).view(num_events, 2) + 200
    edge_type = torch.arange(num_events * edge_dim, dtype=torch.float32).view(num_events, edge_dim) + 400
    return SimpleNamespace(src=src, dst=dst, t=timestamps, x_src=x_src, x_dst=x_dst, msg=msg, edge_type=edge_type)


def _make_encoder(
    *,
    in_dim: int = 3,
    temporal_dim: int = 5,
    node_out_dim: int = 4,
    capacity: int = 20,
    budgets: tuple[int, int, int] = (4, 4, 4),
    use_scale_embedding: bool = False,
    edge_features: tuple[str, ...] = ("edge_type", "msg"),
    fake: FakeGraphEncoder | None = None,
    fusion: str | None = "gated",
) -> tuple[MultiScaleOrthrusEncoder, MultiScaleNeighborLoader, FakeGraphEncoder]:
    store = HistoryStore(num_nodes=8, candidate_capacity=capacity, device="cpu")
    loader = MultiScaleNeighborLoader(
        store,
        tau_short_ns=20,
        tau_medium_ns=50,
        tau_max_ns=100,
        short_budget=budgets[0],
        medium_budget=budgets[1],
        long_budget=budgets[2],
    )
    graph = fake if fake is not None else FakeGraphEncoder(temporal_dim, node_out_dim, edge_dim=4)
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=graph,
        neighbor_loader=loader,
        in_dim=in_dim,
        temporal_dim=temporal_dim,
        node_out_dim=node_out_dim,
        edge_features=edge_features,
        gate_hidden_dim=7,
        use_scale_embedding=use_scale_embedding,
        fusion=fusion,
    )
    return encoder, loader, graph


def _batch(src=(0,), dst=(1,), t=(100,), in_dim=3, event_ids=(0,)):
    src_tensor = torch.tensor(src, dtype=torch.long)
    dst_tensor = torch.tensor(dst, dtype=torch.long)
    return {
        "edge_index": torch.stack([src_tensor, dst_tensor]),
        "t": torch.tensor(t, dtype=torch.long),
        "x": (torch.ones(len(src), in_dim), torch.full((len(src), in_dim), 2.0)),
        "global_event_index": torch.tensor(event_ids, dtype=torch.long),
    }


def _forward(encoder, full_data, batch):
    return encoder(full_data=full_data, msg=torch.zeros(len(batch["t"]), 2), **batch)


def test_forward_shape_and_temporal_projection_dimension():
    encoder, _, _ = _make_encoder(temporal_dim=6, node_out_dim=3)
    h_src, h_dst = _forward(encoder, _full_data(), _batch())
    assert h_src.shape == (1, 3)
    assert h_dst.shape == (1, 3)
    assert encoder.gate_input_dim == 24
    assert encoder.get_last_gate_weights().shape == (1, 3)


def test_global_event_index_contract():
    encoder, _, _ = _make_encoder()
    full = _full_data()
    batch = _batch()
    with pytest.raises(ValueError):
        encoder(edge_index=batch["edge_index"], t=batch["t"], msg=batch["x"][0], x=batch["x"], full_data=full)
    with pytest.raises(ValueError):
        _forward(encoder, full, {**batch, "global_event_index": torch.tensor([0], dtype=torch.int32)})
    with pytest.raises(ValueError):
        _forward(encoder, full, {**batch, "global_event_index": torch.tensor([[0]])})
    with pytest.raises(ValueError):
        _forward(encoder, full, {**batch, "global_event_index": torch.tensor([0, 1])})


def test_shared_encoder_is_called_once_per_non_empty_scale_and_is_single_object():
    encoder, loader, graph = _make_encoder()
    full = _full_data()
    loader.history_store.insert(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    _forward(encoder, full, _batch(src=(0,), dst=(2,), t=(100,), event_ids=(3,)))
    assert graph.call_count == 3
    assert encoder.shared_graph_encoder is graph
    assert not hasattr(encoder, "short_encoder")
    assert not hasattr(encoder, "medium_encoder")
    assert not hasattr(encoder, "long_encoder")
    assert [key for key in encoder.state_dict() if "shared_graph_encoder" in key]
    assert len({key.split("shared_graph_encoder.", 1)[1] for key in encoder.state_dict() if "shared_graph_encoder" in key}) == len(list(graph.state_dict()))


    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
    _forward(encoder, full, _batch(src=(0, 3), dst=(1, 2), t=(100, 100), event_ids=(1, 2)))
    weights = encoder.get_last_gate_weights()
    mask = encoder.get_last_scale_mask()
    assert weights.shape == (2, 3)
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))
    assert torch.allclose(weights.sum(-1)[mask.any(-1)], torch.ones(1))
    assert not torch.allclose(weights[0], weights[1])


def test_all_empty_falls_back_to_current_projection_without_nan():
    encoder, _, _ = _make_encoder()
    batch = _batch()
    h_src, h_dst = _forward(encoder, _full_data(), batch)
    expected_src = encoder.current_src_out_proj(encoder.src_linear(batch["x"][0]))
    expected_dst = encoder.current_dst_out_proj(encoder.dst_linear(batch["x"][1]))
    assert torch.allclose(h_src, expected_src)
    assert torch.allclose(h_dst, expected_dst)
    assert torch.equal(encoder.get_last_gate_weights(), torch.zeros(1, 3))
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()


def test_src_only_history_uses_current_dst_exactly():
    encoder, loader, _ = _make_encoder()
    full = _full_data()
    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
    batch = _batch(src=(0,), dst=(3,), t=(100,), event_ids=(3,))
    h_src, h_dst = _forward(encoder, full, batch)
    current_dst = encoder.current_dst_out_proj(encoder.dst_linear(batch["x"][1]))
    current_src = encoder.current_src_out_proj(encoder.src_linear(batch["x"][0]))
    assert torch.allclose(h_dst, current_dst)
    assert not torch.allclose(h_src, current_src)


def test_dst_only_history_uses_current_src_exactly():
    encoder, loader, _ = _make_encoder()
    full = _full_data()
    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
    batch = _batch(src=(3,), dst=(1,), t=(100,), event_ids=(3,))
    h_src, h_dst = _forward(encoder, full, batch)
    current_src = encoder.current_src_out_proj(encoder.src_linear(batch["x"][0]))
    current_dst = encoder.current_dst_out_proj(encoder.dst_linear(batch["x"][1]))
    assert torch.allclose(h_src, current_src)
    assert not torch.allclose(h_dst, current_dst)


def test_direction_reconstructs_original_edge_for_both_query_sides():
    full = _full_data()
    for query_src, query_dst in [(0, 2), (1, 0)]:
        encoder, loader, graph = _make_encoder()
        loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
        _forward(encoder, full, _batch(src=(query_src,), dst=(query_dst,), t=(100,), event_ids=(3,)))
        if graph.call_count:
            assert any(torch.equal(call["edge_index"], torch.tensor([[0], [1]])) for call in graph.calls)


def test_dedup_by_event_id_preserves_reverse_events_and_features():
    full = _full_data()
    full.src[10], full.dst[10], full.t[10] = 0, 1, 90
    full.src[11], full.dst[11], full.t[11] = 1, 0, 90
    encoder, loader, graph = _make_encoder()
    loader.history_store.insert(torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([10, 11]), torch.tensor([90, 90]))
    _forward(encoder, full, _batch(src=(0,), dst=(1,), t=(100,), event_ids=(3,)))
    non_empty_calls = [call for call in graph.calls if call["edge_index"].size(1)]
    assert any(call["edge_index"].size(1) == 2 for call in non_empty_calls)
    call = next(call for call in non_empty_calls if call["edge_index"].size(1) == 2)
    edges = {tuple(edge) for edge in call["edge_index"].t().tolist()}
    assert edges == {(0, 1), (1, 0)}
    assert call["edge_feats"].size(0) == 2


def test_causal_query_happens_before_insert_and_history_is_visible_next_forward():
    encoder, _, graph = _make_encoder()
    full = _full_data()
    full.t[8], full.t[9] = 100, 110
    first = _batch(src=(0,), dst=(1,), t=(100,), event_ids=(8,))
    _forward(encoder, full, first)
    assert graph.call_count == 0
    second = _batch(src=(0,), dst=(1,), t=(110,), event_ids=(9,))
    _forward(encoder, full, second)
    assert graph.call_count > 0
    history_events = encoder.history_state_dict()["event_id"][0]
    assert 8 in history_events.tolist()
    assert 9 in history_events.tolist()


def test_padding_is_ignored_and_history_features_use_historical_event():
    full = _full_data()
    full.src[7], full.dst[7], full.t[7] = 0, 1, 90
    encoder, loader, graph = _make_encoder(capacity=4, budgets=(4, 0, 0))
    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([7]), torch.tensor([90]))
    _forward(encoder, full, _batch(src=(0,), dst=(1,), t=(100,), event_ids=(8,)))
    call = graph.calls[0]
    assert call["edge_feats"].size(0) == 1
    assert torch.equal(call["edge_feats"][0], torch.cat([full.edge_type[7], full.msg[7]]))
    assert not torch.equal(call["edge_feats"][0], torch.cat([full.edge_type[8], full.msg[8]]))


def test_scale_embedding_is_optional_and_scale_identity_is_observable():
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[1], full.dst[1], full.t[1] = 0, 2, 60
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, graph = _make_encoder(use_scale_embedding=True, edge_features=("none",))
    loader.history_store.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    assert graph.call_count == 3
    assert not torch.equal(graph.calls[0]["x"], graph.calls[1]["x"])


def test_state_helpers_reset_roundtrip_and_clone():
    encoder, _, _ = _make_encoder()
    _forward(encoder, _full_data(), _batch(event_ids=(3,)))
    state = encoder.history_state_dict()
    weights = encoder.get_last_gate_weights()
    weights[0, 0] = 99
    assert encoder.get_last_gate_weights()[0, 0] != 99
    encoder.reset_state()
    assert encoder.get_last_gate_weights() is None
    assert encoder.get_last_scale_mask() is None
    encoder.load_history_state_dict(state)
    assert torch.equal(encoder.history_state_dict()["event_id"], state["event_id"])


def test_backward_has_finite_gate_projection_and_shared_gradients():
    encoder, loader, graph = _make_encoder()
    full = _full_data()
    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    (h_src.square().mean() + h_dst.square().mean()).backward()
    for parameter in list(encoder.gate_mlp.parameters()) + list(encoder.current_src_out_proj.parameters()) + list(graph.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_real_graph_transformer_integration_backward():
    full = _full_data(edge_dim=2)
    graph = GraphTransformer(in_dim=5, hid_dim=3, out_dim=4, edge_dim=4, dropout=0.0, activation=nn.ReLU(), num_heads=1)
    encoder, loader, _ = _make_encoder(temporal_dim=5, node_out_dim=4, edge_features=("edge_type", "msg"), fake=graph)
    loader.history_store.insert(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), torch.tensor([90]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    (h_src.square().mean() + h_dst.square().mean()).backward()
    assert h_src.shape == (1, 4)
    assert h_dst.shape == (1, 4)
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()
    grads = [parameter.grad for parameter in graph.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


# =============================================================================
# Equal Fusion Tests (C5-B3)
# =============================================================================

def test_equal_fusion_three_scales_all_valid():
    """All three scales non-empty → equal weights [1/3, 1/3, 1/3]."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[1], full.dst[1], full.t[1] = 0, 2, 60
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="equal")
    loader.history_store.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    assert weights is not None
    assert weights.shape == (1, 3)
    torch.testing.assert_close(weights[0], torch.tensor([1/3, 1/3, 1/3]), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(weights).all()


def test_equal_fusion_two_scales_valid_short_and_long():
    """Short and long valid, medium empty → weights [1/2, 0, 1/2]."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="equal", budgets=(4, 0, 4))
    loader.history_store.insert(torch.tensor([0, 0]), torch.tensor([1, 3]), torch.tensor([0, 2]), torch.tensor([90, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    assert weights is not None
    assert weights.shape == (1, 3)
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.0, 0.5]), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(weights).all()


def test_equal_fusion_only_long_scale_valid():
    """Only long scale valid → weights [0, 0, 1]."""
    full = _full_data()
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="equal", budgets=(0, 0, 4))
    loader.history_store.insert(torch.tensor([0]), torch.tensor([3]), torch.tensor([2]), torch.tensor([10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    assert weights is not None
    assert weights.shape == (1, 3)
    torch.testing.assert_close(weights[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(weights).all()


def test_equal_fusion_all_empty_no_nan_inf():
    """All three scales empty → fallback to current projection, no NaN/Inf."""
    encoder, _, _ = _make_encoder(fusion="equal")
    batch = _batch()
    h_src, h_dst = _forward(encoder, _full_data(), batch)
    expected_src = encoder.current_src_out_proj(encoder.src_linear(batch["x"][0]))
    expected_dst = encoder.current_dst_out_proj(encoder.dst_linear(batch["x"][1]))
    assert torch.allclose(h_src, expected_src)
    assert torch.allclose(h_dst, expected_dst)
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()
    weights = encoder.get_last_gate_weights()
    assert torch.equal(weights, torch.zeros(1, 3))


def test_equal_fusion_weights_normalize_to_one():
    """For any case with at least one valid scale, weights sum to 1."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[1], full.dst[1], full.t[1] = 0, 2, 60
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="equal")
    loader.history_store.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    scale_mask = encoder.get_last_scale_mask()
    has_scale = scale_mask.any(dim=-1)
    torch.testing.assert_close(weights.sum(dim=-1)[has_scale], torch.ones(has_scale.sum()), atol=1e-6, rtol=1e-6)


def test_equal_fusion_empty_scale_weights_strictly_zero():
    """Empty scales get exactly 0 weight, not a tiny epsilon."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="equal", budgets=(4, 0, 4))
    loader.history_store.insert(torch.tensor([0, 0]), torch.tensor([1, 3]), torch.tensor([0, 2]), torch.tensor([90, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    scale_mask = encoder.get_last_scale_mask()
    empty_mask = ~scale_mask
    assert (weights[empty_mask] == 0.0).all(), "Empty scales should have exactly 0 weight"


def test_equal_fusion_regression_gated_still_uses_gate():
    """fusion='gated' still uses gate MLP (not equal) and outputs finite weights."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[1], full.dst[1], full.t[1] = 0, 2, 60
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, _ = _make_encoder(fusion="gated")
    loader.history_store.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    weights = encoder.get_last_gate_weights()
    assert weights is not None
    assert torch.isfinite(weights).all()
    assert weights.shape == (1, 3)
    has_scale = encoder.get_last_scale_mask().any(dim=-1)
    torch.testing.assert_close(weights.sum(dim=-1)[has_scale], torch.ones(1), atol=1e-6, rtol=1e-6)
    assert list(encoder.gate_mlp.parameters()), "gate_mlp should have parameters (not replaced by equal)"


def test_equal_fusion_invalid_fusion_raises():
    """fusion='abc' must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid fusion"):
        MultiScaleOrthrusEncoder(
            shared_graph_encoder=FakeGraphEncoder(5, 4),
            neighbor_loader=MultiScaleNeighborLoader(
                history_store=HistoryStore(num_nodes=8, candidate_capacity=20, device="cpu"),
                tau_short_ns=20, tau_medium_ns=50, tau_max_ns=100,
                short_budget=4, medium_budget=4, long_budget=4,
            ),
            in_dim=3, temporal_dim=5, node_out_dim=4,
            edge_features=("edge_type", "msg"),
            gate_hidden_dim=7,
            fusion="abc",
        )


def test_equal_fusion_backward_pass():
    """Equal fusion backward pass produces finite gradients."""
    full = _full_data()
    full.src[0], full.dst[0], full.t[0] = 0, 1, 90
    full.src[1], full.dst[1], full.t[1] = 0, 2, 60
    full.src[2], full.dst[2], full.t[2] = 0, 3, 10
    encoder, loader, graph = _make_encoder(fusion="equal")
    loader.history_store.insert(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), torch.tensor([90, 60, 10]))
    h_src, h_dst = _forward(encoder, full, _batch(t=(100,), event_ids=(3,)))
    (h_src.square().mean() + h_dst.square().mean()).backward()
    assert torch.isfinite(torch.cat([h_src, h_dst])).all()
    for name, param in encoder.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
