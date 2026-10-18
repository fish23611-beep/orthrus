"""P1-4A differential tests: current-node assignment last-write-wins semantics.

The reference implementation below is a verbatim transliteration of the
pre-P1-4A loop in ``MultiScaleOrthrusEncoder._encode_scale``:

    for batch_index in range(batch_size):
        node_src_hidden[node_to_local[int(src_query_nodes[batch_index])]] = current_src_hidden[batch_index]
        node_dst_hidden[node_to_local[int(dst_query_nodes[batch_index])]] = current_dst_hidden[batch_index]

The optimized implementation must reproduce its values, duplicate handling and
gradient routing exactly.
"""

from __future__ import annotations

import copy
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mstc.history_store import HistoryStore
from mstc.multiscale_encoder import MultiScaleOrthrusEncoder
from mstc.multiscale_sampler import MultiScaleNeighborLoader


# ---------------------------------------------------------------------------
# Reference implementation (original Python loop)
# ---------------------------------------------------------------------------
def _reference_assign_branch(
    node_hidden: Tensor,
    node_ids: Tensor,
    query_nodes: Tensor,
    current_hidden: Tensor,
) -> None:
    node_to_local = {int(node): index for index, node in enumerate(node_ids.cpu().tolist())}
    for batch_index in range(query_nodes.numel()):
        node_hidden[node_to_local[int(query_nodes[batch_index])]] = current_hidden[batch_index]


def _reference_assign_current(
    self,
    node_hidden: Tensor,
    node_ids: Tensor,
    query_nodes: Tensor,
    current_hidden: Tensor,
) -> None:
    _reference_assign_branch(node_hidden, node_ids, query_nodes, current_hidden)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
class FakeGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.edge_projection = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_feats: Tensor | None = None, **kwargs) -> Tensor:
        output = self.projection(x)
        if self.edge_projection is not None and edge_feats is not None and edge_index.numel():
            messages = self.edge_projection(edge_feats)
            output = output.clone()
            output.index_add_(0, edge_index[1], messages)
        return output


def _full_data(num_events: int = 32, in_dim: int = 3, edge_dim: int = 2) -> SimpleNamespace:
    src = torch.arange(num_events, dtype=torch.long) % 6
    dst = (src + 1) % 6
    # History events 0..19 are strictly before the current batch (t=1000).
    timestamps = torch.full((num_events,), 1000, dtype=torch.long)
    timestamps[:20] = 990 - torch.arange(20, dtype=torch.long)
    x_src = (
        torch.arange(num_events * in_dim, dtype=torch.float32).view(num_events, in_dim) % 17
    ) + 1
    x_dst = x_src + 100
    msg = (torch.arange(num_events * 2, dtype=torch.float32).view(num_events, 2) % 13) + 200
    edge_type = (torch.arange(num_events * edge_dim, dtype=torch.float32).view(num_events, edge_dim) % 11) + 400
    return SimpleNamespace(src=src, dst=dst, t=timestamps, x_src=x_src, x_dst=x_dst, msg=msg, edge_type=edge_type)


def _make_encoder(
    device: torch.device,
    *,
    in_dim: int = 3,
    temporal_dim: int = 5,
    node_out_dim: int = 4,
    capacity: int = 64,
    budgets: tuple[int, int, int] = (4, 4, 4),
) -> tuple[MultiScaleOrthrusEncoder, MultiScaleNeighborLoader]:
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
    graph = FakeGraphEncoder(temporal_dim, node_out_dim, edge_dim=4)
    encoder = MultiScaleOrthrusEncoder(
        shared_graph_encoder=graph,
        neighbor_loader=loader,
        in_dim=in_dim,
        temporal_dim=temporal_dim,
        node_out_dim=node_out_dim,
        edge_features=("edge_type", "msg"),
        gate_hidden_dim=7,
        use_scale_embedding=True,
        fusion="gated",
        device=device,
    ).to(device)
    return encoder, loader


def _insert_history(loader: MultiScaleNeighborLoader, full: SimpleNamespace) -> None:
    loader.history_store.insert(
        full.src[:20].clone(),
        full.dst[:20].clone(),
        torch.arange(20, dtype=torch.long),
        full.t[:20].clone(),
    )


def _duplicate_heavy_batch(device: torch.device) -> dict[str, object]:
    # Repeated src (0, 0, 2, 2), repeated dst (2, 2, 4, 4) and src/dst overlap.
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5], dtype=torch.long)
    dst = torch.tensor([1, 2, 2, 3, 4, 4, 5, 6], dtype=torch.long)
    size = src.numel()
    x_src = (torch.arange(size * 3, dtype=torch.float32).view(size, 3) % 7) + 1
    x_dst = (torch.arange(size * 3, dtype=torch.float32).view(size, 3) % 5) + 20
    return {
        "edge_index": torch.stack([src, dst]),
        "t": torch.full((size,), 1000, dtype=torch.long),
        "x": (x_src.to(device), x_dst.to(device)),
        "global_event_index": torch.arange(20, 20 + size, dtype=torch.long),
    }


def _forward(encoder: MultiScaleOrthrusEncoder, full: SimpleNamespace, batch: dict[str, object]):
    return encoder(
        full_data=full,
        msg=torch.zeros(len(batch["t"]), 2),
        **batch,
    )


def _loss(h_src: Tensor, h_dst: Tensor) -> Tensor:
    w_src = torch.arange(1, h_src.numel() + 1, dtype=h_src.dtype, device=h_src.device).view_as(h_src)
    w_dst = torch.arange(1, h_dst.numel() + 1, dtype=h_dst.dtype, device=h_dst.device).view_as(h_dst)
    return (h_src * w_src).sum() + (h_dst * w_dst).sum()


def _tolerance(device: torch.device, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float64:
        return 0.0, 0.0
    if device.type == "cuda":
        return 1e-6, 1e-5
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Unit tests: optimized helper vs original loop
# ---------------------------------------------------------------------------
def _query_pattern(name: str, num_nodes: int, batch_size: int) -> Tensor:
    if batch_size == 0:
        return torch.empty(0, dtype=torch.long)
    if name == "unique":
        return torch.randperm(num_nodes, generator=torch.Generator().manual_seed(1))[:batch_size]
    if name == "all_same":
        return torch.zeros(batch_size, dtype=torch.long)
    if name == "pairs":
        half = torch.arange((batch_size + 1) // 2, dtype=torch.long) % num_nodes
        return torch.cat([half, half])[:batch_size]
    if name == "random_dups":
        generator = torch.Generator().manual_seed(11 + batch_size)
        return torch.randint(0, max(num_nodes // 2, 1), (batch_size,), generator=generator)
    raise AssertionError(f"unknown pattern {name}")


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch_size", [0, 1, 2, 7, 33, 128])
@pytest.mark.parametrize("pattern", ["unique", "all_same", "pairs", "random_dups"])
def test_vectorized_assignment_matches_reference(
    device_name: str, dtype: torch.dtype, batch_size: int, pattern: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name if device_name == "cpu" else "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

    # Sorted-unique node ids with gaps, mimicking torch.unique(sorted=True).
    num_ids = max(16, batch_size)
    node_ids = torch.arange(0, 2 * num_ids, 2, dtype=torch.long, device=device)
    query_nodes = _query_pattern(pattern, num_ids, batch_size).to(device) * 2

    generator = torch.Generator().manual_seed(100 + batch_size)
    current = torch.randn(batch_size, 5, dtype=dtype, generator=generator).to(device)
    base = torch.arange(node_ids.numel() * 5, dtype=dtype).view(node_ids.numel(), 5).to(device)

    hidden_ref = base.clone()
    hidden_opt = base.clone()
    current_ref = current.detach().clone().requires_grad_(True)
    current_opt = current.detach().clone().requires_grad_(True)

    _reference_assign_branch(hidden_ref, node_ids, query_nodes, current_ref)
    MultiScaleOrthrusEncoder._assign_current_hidden_last_write(
        hidden_opt, node_ids, query_nodes, current_opt
    )
    assert torch.equal(hidden_opt, hidden_ref)

    if batch_size == 0:
        assert current_ref.grad is None and current_opt.grad is None
        return

    weight = torch.arange(1, hidden_ref.numel() + 1, dtype=dtype, device=device).view_as(hidden_ref)
    (hidden_ref * weight).sum().backward()
    (hidden_opt * weight).sum().backward()
    assert torch.equal(current_opt.grad, current_ref.grad)
    assert torch.equal(hidden_opt, hidden_ref)


def test_last_write_wins_and_history_is_overwritten() -> None:
    device = torch.device("cpu")
    node_ids = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    # Node 4 appears at rows 0, 2, 4 -> row 4 must win; node 8 at rows 1, 5 -> row 5.
    query_nodes = torch.tensor([4, 8, 4, 0, 4, 8], dtype=torch.long)
    current = torch.arange(6 * 3, dtype=torch.float64).view(6, 3)
    base = torch.full((5, 3), -7.0, dtype=torch.float64)

    hidden = base.clone()
    MultiScaleOrthrusEncoder._assign_current_hidden_last_write(hidden, node_ids, query_nodes, current)

    assert torch.equal(hidden[2], current[4])  # node 4, last row
    assert torch.equal(hidden[4], current[5])  # node 8, last row
    assert torch.equal(hidden[0], current[3])  # node 0
    assert torch.equal(hidden[1], base[1])  # node 2 untouched
    assert torch.equal(hidden[3], base[3])  # node 6 untouched


def test_empty_batch_is_a_no_op() -> None:
    node_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    hidden = torch.ones(3, 4, dtype=torch.float64)
    current = torch.empty(0, 4, dtype=torch.float64)
    MultiScaleOrthrusEncoder._assign_current_hidden_last_write(
        hidden, node_ids, torch.empty(0, dtype=torch.long), current
    )
    assert torch.equal(hidden, torch.ones(3, 4, dtype=torch.float64))


def test_src_and_dst_branches_are_independent() -> None:
    node_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    src_q = torch.tensor([0, 1, 1], dtype=torch.long)
    dst_q = torch.tensor([2, 2, 3], dtype=torch.long)
    src_cur = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    dst_cur = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float64)

    src_hidden = torch.zeros(4, 1, dtype=torch.float64)
    dst_hidden = torch.zeros(4, 1, dtype=torch.float64)
    MultiScaleOrthrusEncoder._assign_current_hidden_last_write(src_hidden, node_ids, src_q, src_cur)
    MultiScaleOrthrusEncoder._assign_current_hidden_last_write(dst_hidden, node_ids, dst_q, dst_cur)

    assert torch.equal(src_hidden[:, 0], torch.tensor([1.0, 3.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(dst_hidden[:, 0], torch.tensor([0.0, 0.0, 20.0, 30.0], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Full-model differential tests (optimized helper vs monkeypatched original)
# ---------------------------------------------------------------------------
def _build_pair(device: torch.device) -> tuple[MultiScaleOrthrusEncoder, MultiScaleNeighborLoader,
                                               MultiScaleOrthrusEncoder, MultiScaleNeighborLoader]:
    encoder_opt, loader_opt = _make_encoder(device)
    encoder_ref, loader_ref = _make_encoder(device)
    encoder_ref.load_state_dict(copy.deepcopy(encoder_opt.state_dict()))
    encoder_ref._assign_current_hidden_last_write = types.MethodType(
        _reference_assign_current, encoder_ref
    )
    return encoder_opt, loader_opt, encoder_ref, loader_ref


def _run_once(
    encoder: MultiScaleOrthrusEncoder,
    loader: MultiScaleNeighborLoader,
    full: SimpleNamespace,
    batch: dict[str, object],
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    h_src, h_dst = _forward(encoder, full, batch)
    loss = _loss(h_src, h_dst)
    loss.backward()
    grads = {
        name: (None if parameter.grad is None else parameter.grad.detach().clone())
        for name, parameter in encoder.named_parameters()
    }
    optimizer.step()
    params = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    return {
        "h_src": h_src.detach().clone(),
        "h_dst": h_dst.detach().clone(),
        "loss": loss.detach().clone(),
        "gate": encoder.get_last_gate_weights(),
        "mask": encoder.get_last_scale_mask(),
        "grads": grads,
        "params": params,
        "history": {
            key: value.detach().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in encoder.history_state_dict().items()
        },
    }


def _assert_run_equal(
    result_opt: dict[str, object], result_ref: dict[str, object], device: torch.device
) -> None:
    for key in ("h_src", "h_dst"):
        atol, rtol = _tolerance(device, result_opt[key].dtype)
        torch.testing.assert_close(result_opt[key], result_ref[key], atol=atol, rtol=rtol, msg=key)
    torch.testing.assert_close(result_opt["loss"], result_ref["loss"], atol=0.0, rtol=0.0, msg="loss")
    for key in ("gate", "mask"):
        torch.testing.assert_close(result_opt[key], result_ref[key], atol=0.0, rtol=0.0, msg=key)
    for name, grad_opt in result_opt["grads"].items():
        grad_ref = result_ref["grads"][name]
        assert (grad_opt is None) == (grad_ref is None), name
        if grad_opt is not None:
            atol, rtol = _tolerance(device, grad_opt.dtype)
            torch.testing.assert_close(grad_opt, grad_ref, atol=atol, rtol=rtol, msg=f"grad:{name}")
    for name, param_opt in result_opt["params"].items():
        atol, rtol = _tolerance(device, param_opt.dtype)
        torch.testing.assert_close(param_opt, result_ref["params"][name], atol=atol, rtol=rtol, msg=f"param:{name}")
    assert result_opt["history"].keys() == result_ref["history"].keys()
    for key, value_opt in result_opt["history"].items():
        value_ref = result_ref["history"][key]
        if isinstance(value_opt, Tensor):
            assert torch.equal(value_opt, value_ref), key
        else:
            assert value_opt == value_ref, key


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_full_model_differential_duplicate_heavy(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name if device_name == "cpu" else "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    full = _full_data()
    batch = _duplicate_heavy_batch(device)
    encoder_opt, loader_opt, encoder_ref, loader_ref = _build_pair(device)
    _insert_history(loader_opt, full)
    _insert_history(loader_ref, full)

    rng_before = torch.random.get_rng_state().clone()
    result_opt = _run_once(encoder_opt, loader_opt, full, batch, seed=1234)
    rng_opt = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before)
    result_ref = _run_once(encoder_ref, loader_ref, full, batch, seed=1234)
    rng_ref = torch.random.get_rng_state().clone()

    assert torch.equal(rng_opt, rng_ref)
    _assert_run_equal(result_opt, result_ref, device)

    # Duplicate-heavy batch must actually exercise last-write-wins.
    scale_mask = result_opt["mask"]
    assert scale_mask.any()
    assert torch.equal(result_opt["mask"], result_ref["mask"])


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 16])
def test_full_model_differential_various_batch_sizes(batch_size: int) -> None:
    device = torch.device("cpu")
    full = _full_data()
    src = (torch.arange(batch_size) % 4).long()
    dst = ((torch.arange(batch_size) + 1) % 4).long()
    size = src.numel()
    x_src = (torch.arange(size * 3, dtype=torch.float32).view(size, 3) % 7) + 1
    x_dst = (torch.arange(size * 3, dtype=torch.float32).view(size, 3) % 5) + 20
    batch = {
        "edge_index": torch.stack([src, dst]),
        "t": torch.full((size,), 1000, dtype=torch.long),
        "x": (x_src, x_dst),
        "global_event_index": torch.arange(20, 20 + size, dtype=torch.long),
    }
    encoder_opt, loader_opt, encoder_ref, loader_ref = _build_pair(device)
    _insert_history(loader_opt, full)
    _insert_history(loader_ref, full)
    result_opt = _run_once(encoder_opt, loader_opt, full, batch, seed=99)
    result_ref = _run_once(encoder_ref, loader_ref, full, batch, seed=99)
    _assert_run_equal(result_opt, result_ref, device)


def test_temporal_causality_and_history_visibility_match_reference() -> None:
    device = torch.device("cpu")
    full = _full_data()
    # Endpoints must match full_data for events 20..23. Event 20 is changed to
    # 4 -> 5 so that node 4 carries src-side history into the second batch.
    full.src[20] = 4
    full.dst[20] = 5
    first_batch = {
        "edge_index": torch.tensor([[4, 3], [5, 4]], dtype=torch.long),
        "t": torch.tensor([1000, 1000], dtype=torch.long),
        "x": (torch.ones(2, 3), torch.full((2, 3), 2.0)),
        "global_event_index": torch.tensor([20, 21], dtype=torch.long),
    }
    second_batch = {
        "edge_index": torch.tensor([[4, 5], [5, 0]], dtype=torch.long),
        "t": torch.tensor([1010, 1010], dtype=torch.long),
        "x": (torch.full((2, 3), 3.0), torch.full((2, 3), 4.0)),
        "global_event_index": torch.tensor([22, 23], dtype=torch.long),
    }
    # full_data timestamps must match the current batch for the scan validation.
    full.t[22] = 1010
    full.t[23] = 1010

    encoder_opt, loader_opt, encoder_ref, loader_ref = _build_pair(device)

    first_opt = _run_once(encoder_opt, loader_opt, full, first_batch, seed=7)
    first_ref = _run_once(encoder_ref, loader_ref, full, first_batch, seed=7)
    _assert_run_equal(first_opt, first_ref, device)
    assert not first_opt["mask"].any(), "empty store must not produce history"

    second_opt = _run_once(encoder_opt, loader_opt, full, second_batch, seed=8)
    second_ref = _run_once(encoder_ref, loader_ref, full, second_batch, seed=8)
    _assert_run_equal(second_opt, second_ref, device)
    assert second_opt["mask"].any(), "history inserted by the first forward must be visible"

    # Causal ordering: the second forward must not see its own current batch.
    history_events = encoder_opt.history_state_dict()["event_id"].flatten().tolist()
    assert 22 in history_events and 23 in history_events
