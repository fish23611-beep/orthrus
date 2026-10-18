"""P1-4C differential tests for vectorized ``_scan_scale`` candidate collection.

The reference implementation below is a verbatim transliteration of the
P1-4B scalar loop.  The optimized implementation must reproduce:

* ordered records (query-major, slot-major),
* first-occurrence event-id deduplication,
* all record fields taken from the first occurrence,
* direction validation order and messages,
* full_data validation order and messages,
* profiler keys and scan-record counts.
"""

from __future__ import annotations

import importlib
import os
import random
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
# Reference implementation (P1-4B scalar loop)
# ---------------------------------------------------------------------------
def _reference_scan_scale(
    encoder: MultiScaleOrthrusEncoder,
    scale: str,
    sampled: dict[str, Tensor],
    full_data,
    device: torch.device,
) -> list[tuple[int, int, int, int]]:
    event_ids = sampled[f"{scale}_event_id"].cpu()
    neighbors = sampled[f"{scale}_neighbor_id"].cpu()
    directions = sampled[f"{scale}_direction"].cpu()
    masks = sampled[f"{scale}_mask"].cpu()
    sampled_timestamps = sampled[f"{scale}_timestamp_ns"].cpu()
    query_nodes = sampled["query_nodes"].cpu()
    candidates: list[tuple[int, int, int, int, int]] = []
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
            sampled_timestamp = int(sampled_timestamps[q_idx, slot])
            seen_event_ids.add(event_id)
            candidates.append(
                (event_id, query, neighbor, direction, sampled_timestamp)
            )

    if candidates:
        candidate_values = torch.tensor(candidates, dtype=torch.long)
        lookup_event_ids = candidate_values[:, 0]
        cpu = lookup_event_ids.device
        full_src_values = encoder._full_values(full_data, "src", lookup_event_ids, cpu)
        full_dst_values = encoder._full_values(full_data, "dst", lookup_event_ids, cpu)
        full_timestamp_values = encoder._full_values(full_data, "t", lookup_event_ids, cpu)

        full_src_values = full_src_values.reshape(-1)
        full_dst_values = full_dst_values.reshape(-1)
        full_timestamp_values = full_timestamp_values.reshape(-1)
        expected_size = len(candidates)
        if not (
            full_src_values.numel()
            == full_dst_values.numel()
            == full_timestamp_values.numel()
            == expected_size
        ):
            raise ValueError("full_data validation fields have inconsistent sizes")

        queries = candidate_values[:, 1]
        neighbors = candidate_values[:, 2]
        forward = candidate_values[:, 3] == 0
        expected_src = torch.where(forward, queries, neighbors)
        expected_dst = torch.where(forward, neighbors, queries)
        if not (
            torch.equal(full_src_values.long(), expected_src)
            and torch.equal(full_dst_values.long(), expected_dst)
        ):
            raise ValueError("history direction and full_data edge endpoints disagree")
        if not torch.equal(full_timestamp_values.long(), candidate_values[:, 4]):
            raise ValueError("history timestamp and full_data timestamp disagree")

    return [record[:4] for record in candidates]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class CountingFullData:
    def __init__(self, src: Tensor, dst: Tensor, t: Tensor) -> None:
        self.src = src
        self.dst = dst
        self.t = t
        self.calls: list[tuple[str, Tensor]] = []

    def get_event_values(self, field: str, event_ids: Tensor) -> Tensor:
        self.calls.append((field, event_ids.clone()))
        return getattr(self, field)[event_ids]


class FakeGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.edge_projection = (
            nn.Linear(edge_dim, out_dim, bias=False) if edge_dim else None
        )
        self.calls: list[dict[str, Tensor | None]] = []

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_feats: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        self.calls.append(
            {
                "x": x.detach().clone(),
                "edge_index": edge_index.detach().clone(),
                "edge_feats": None if edge_feats is None else edge_feats.detach().clone(),
            }
        )
        output = self.projection(x)
        if (
            self.edge_projection is not None
            and edge_feats is not None
            and edge_index.numel()
        ):
            messages = self.edge_projection(edge_feats)
            output = output.clone()
            output.index_add_(0, edge_index[1], messages)
        return output


def _make_encoder(
    *,
    device: torch.device = torch.device("cpu"),
    budgets: tuple[int, int, int] = (4, 4, 4),
    capacity: int = 32,
    temporal_dim: int = 5,
    node_out_dim: int = 4,
    use_scale_embedding: bool = True,
) -> tuple[MultiScaleOrthrusEncoder, MultiScaleNeighborLoader, FakeGraphEncoder]:
    store = HistoryStore(num_nodes=12, candidate_capacity=capacity, device="cpu")
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
        in_dim=3,
        temporal_dim=temporal_dim,
        node_out_dim=node_out_dim,
        edge_features=("edge_type", "msg"),
        gate_hidden_dim=7,
        use_scale_embedding=use_scale_embedding,
        fusion="gated",
        device=device,
    ).to(device)
    return encoder, loader, graph


def _sample_dict(
    scale: str,
    event_ids: Tensor,
    neighbor_ids: Tensor,
    directions: Tensor,
    masks: Tensor,
    timestamps: Tensor,
    query_nodes: Tensor,
) -> dict[str, Tensor]:
    return {
        "query_nodes": query_nodes,
        f"{scale}_event_id": event_ids,
        f"{scale}_neighbor_id": neighbor_ids,
        f"{scale}_direction": directions,
        f"{scale}_mask": masks,
        f"{scale}_timestamp_ns": timestamps,
    }


def _full_data_from_sampled(
    event_ids: Tensor,
    neighbor_ids: Tensor,
    directions: Tensor,
    masks: Tensor,
    timestamps: Tensor,
    query_nodes: Tensor,
    num_events: int,
) -> CountingFullData:
    src = torch.zeros(num_events, dtype=torch.long)
    dst = torch.zeros_like(src)
    t = torch.zeros_like(src)
    seen: set[int] = set()
    for q_idx in range(event_ids.size(0)):
        for slot in range(event_ids.size(1)):
            if not bool(masks[q_idx, slot]):
                continue
            event_id = int(event_ids[q_idx, slot])
            if event_id == -1 or event_id in seen:
                continue
            direction = int(directions[q_idx, slot])
            query = int(query_nodes[q_idx])
            neighbor = int(neighbor_ids[q_idx, slot])
            if direction == 0:
                src[event_id] = query
                dst[event_id] = neighbor
            else:
                src[event_id] = neighbor
                dst[event_id] = query
            t[event_id] = int(timestamps[q_idx, slot])
            seen.add(event_id)
    return CountingFullData(src, dst, t)


def _random_case(
    rng: random.Random, q_count: int, budget: int, num_events: int
) -> tuple[dict[str, Tensor], CountingFullData]:
    query_nodes = torch.tensor(
        rng.sample(range(num_events), min(q_count, num_events)), dtype=torch.long
    )
    if query_nodes.numel() != q_count:
        query_nodes = torch.randint(0, num_events, (q_count,), dtype=torch.long)
        query_nodes = torch.unique(query_nodes, sorted=True)
    event_ids = torch.full((query_nodes.numel(), budget), -1, dtype=torch.long)
    neighbor_ids = torch.randint(0, num_events, event_ids.shape, dtype=torch.long)
    directions = torch.randint(0, 2, event_ids.shape, dtype=torch.int8)
    masks = torch.zeros(event_ids.shape, dtype=torch.bool)
    timestamps = torch.randint(0, 1000, event_ids.shape, dtype=torch.long)

    for q_idx in range(event_ids.size(0)):
        for slot in range(event_ids.size(1)):
            draw = rng.random()
            if draw < 0.55:
                event_ids[q_idx, slot] = rng.randrange(num_events)
                masks[q_idx, slot] = True
            elif draw < 0.70:
                event_ids[q_idx, slot] = -1
                masks[q_idx, slot] = rng.random() < 0.5
            # else leave mask=False / event_id=-1

    sampled = _sample_dict(
        "long",
        event_ids,
        neighbor_ids,
        directions,
        masks,
        timestamps,
        query_nodes,
    )
    full = _full_data_from_sampled(
        event_ids,
        neighbor_ids,
        directions,
        masks,
        timestamps,
        query_nodes,
        num_events,
    )
    return sampled, full


def _assert_same_records(
    encoder: MultiScaleOrthrusEncoder,
    sampled: dict[str, Tensor],
    full_data,
) -> list[tuple[int, int, int, int]]:
    expected = _reference_scan_scale(
        encoder, "long", sampled, full_data, torch.device("cpu")
    )
    actual = encoder._scan_scale(
        "long", sampled, full_data, torch.device("cpu")
    )
    assert actual == expected
    return actual


# ---------------------------------------------------------------------------
# Random differential
# ---------------------------------------------------------------------------
def test_random_differential_preserves_records() -> None:
    rng = random.Random(20261018)
    encoder, _, _ = _make_encoder()
    for trial in range(160):
        q_count = rng.choice([0, 1, 2, 5, 17, 50])
        budget = rng.choice([0, 1, 2, 8, 24])
        num_events = rng.choice([1, 4, 16, 64])
        sampled, full = _random_case(rng, q_count, budget, num_events)

        sampled_before = {key: value.clone() for key, value in sampled.items()}
        full_before = (full.src.clone(), full.dst.clone(), full.t.clone())

        records = _assert_same_records(encoder, sampled, full)

        for key, value in sampled.items():
            assert torch.equal(value, sampled_before[key]), (trial, key)
        assert torch.equal(full.src, full_before[0])
        assert torch.equal(full.dst, full_before[1])
        assert torch.equal(full.t, full_before[2])
        assert len(records) == len(set(record[0] for record in records))


# ---------------------------------------------------------------------------
# Duplicate / shared-event semantics
# ---------------------------------------------------------------------------
def test_duplicate_event_same_row_and_across_query_nodes() -> None:
    encoder, _, _ = _make_encoder()
    # Event 10 appears first in row 0 / slot 1; later occurrences must be
    # ignored even though their query/neighbor/direction fields differ.
    event_ids = torch.tensor(
        [[-1, 10, 10, 11], [10, 11, -1, -1], [11, -1, -1, -1]],
        dtype=torch.long,
    )
    neighbor_ids = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 0, 0], [6, 0, 0, 0]], dtype=torch.long
    )
    directions = torch.tensor(
        [[0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int8
    )
    masks = torch.tensor(
        [[False, True, True, True], [True, True, False, False], [True, False, False, False]],
        dtype=torch.bool,
    )
    timestamps = torch.tensor(
        [[0, 100, 200, 300], [400, 500, 0, 0], [600, 0, 0, 0]], dtype=torch.long
    )
    query_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = _full_data_from_sampled(
        event_ids, neighbor_ids, directions, masks, timestamps, query_nodes, 20
    )

    expected = [(10, 0, 1, 0), (11, 0, 3, 1)]
    assert _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu")) == expected
    assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == expected


def test_first_occurrence_fields_win_even_with_same_timestamp() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 10, 11, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    directions = torch.tensor([[0, 1, 1, 0]], dtype=torch.int8)
    masks = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    timestamps = torch.tensor([[50, 50, 50, 50]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = _full_data_from_sampled(
        event_ids, neighbor_ids, directions, masks, timestamps, query_nodes, 20
    )

    expected = [(10, 0, 1, 0), (11, 0, 3, 1)]
    assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == expected


# ---------------------------------------------------------------------------
# Mask / sentinel handling
# ---------------------------------------------------------------------------
def test_mask_false_minus_one_and_minus_two_semantics() -> None:
    encoder, _, _ = _make_encoder()
    # -1 is the only filtered sentinel.  -2 must remain a valid event id and
    # be looked up with Python/PyTorch negative indexing exactly as before.
    event_ids = torch.tensor(
        [[-1, 5, -2, 6], [5, 6, -1, -2]], dtype=torch.long
    )
    neighbor_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    directions = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 1]], dtype=torch.int8)
    masks = torch.tensor(
        [[True, False, True, True], [True, True, True, True]], dtype=torch.bool
    )
    timestamps = torch.tensor([[90, 91, 92, 93], [94, 95, 96, 97]], dtype=torch.long)
    query_nodes = torch.tensor([0, 1], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )

    src = torch.zeros(20, dtype=torch.long)
    dst = torch.zeros(20, dtype=torch.long)
    t = torch.zeros(20, dtype=torch.long)
    # event 5 first valid occurrence is row 1 / slot 0, direction 1
    src[5], dst[5], t[5] = 4, 1, 94
    # event -2 first valid occurrence is row 0 / slot 2, direction 1
    src[-2], dst[-2], t[-2] = 2, 0, 92
    # event 6 first valid occurrence is row 0 / slot 3, direction 0
    src[6], dst[6], t[6] = 0, 3, 93
    full = CountingFullData(src, dst, t)

    expected = [(-2, 0, 2, 1), (6, 0, 3, 0), (5, 1, 4, 1)]
    assert _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu")) == expected
    assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == expected


@pytest.mark.parametrize(
    "q_count,budget",
    [(0, 8), (2, 0), (0, 0), (1, 1), (5, 8)],
)
def test_empty_and_zero_budget(q_count: int, budget: int) -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.full((q_count, budget), -1, dtype=torch.long)
    neighbor_ids = torch.zeros_like(event_ids)
    directions = torch.zeros((q_count, budget), dtype=torch.int8)
    masks = torch.zeros((q_count, budget), dtype=torch.bool)
    timestamps = torch.zeros_like(event_ids)
    query_nodes = torch.arange(q_count, dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = CountingFullData(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    )
    assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == []
    assert _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu")) == []


def test_all_masks_false_and_all_event_ids_minus_one() -> None:
    encoder, _, _ = _make_encoder()
    query_nodes = torch.tensor([0, 1], dtype=torch.long)
    event_ids = torch.full((2, 4), -1, dtype=torch.long)
    neighbor_ids = torch.arange(8, dtype=torch.long).view(2, 4)
    directions = torch.zeros((2, 4), dtype=torch.int8)
    timestamps = torch.arange(8, dtype=torch.long).view(2, 4)
    full = CountingFullData(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
    )

    for masks in (
        torch.zeros((2, 4), dtype=torch.bool),
        torch.ones((2, 4), dtype=torch.bool),
    ):
        sampled = _sample_dict(
            "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
        )
        assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == []


# ---------------------------------------------------------------------------
# Direction errors and full_data error precedence
# ---------------------------------------------------------------------------
def test_invalid_direction_first_error_matches_reference() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    directions = torch.tensor([[2, 3, 0, 1]], dtype=torch.int8)
    masks = torch.ones((1, 4), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = CountingFullData(
        torch.zeros(20, dtype=torch.long),
        torch.zeros(20, dtype=torch.long),
        torch.zeros(20, dtype=torch.long),
    )

    with pytest.raises(ValueError) as ref_error:
        _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu"))
    with pytest.raises(ValueError) as opt_error:
        encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    assert str(opt_error.value) == str(ref_error.value) == "history direction must be 0 or 1, got 2"


def test_duplicate_invalid_direction_is_skipped() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 10, 11, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    directions = torch.tensor([[0, 2, 1, 3]], dtype=torch.int8)
    masks = torch.ones((1, 4), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = _full_data_from_sampled(
        event_ids, neighbor_ids, directions, masks, timestamps, query_nodes, 20
    )
    expected = [(10, 0, 1, 0), (11, 0, 3, 1)]
    assert encoder._scan_scale("long", sampled, full, torch.device("cpu")) == expected


def test_direction_error_precedes_full_data_error() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2]], dtype=torch.long)
    directions = torch.tensor([[2, 0]], dtype=torch.int8)
    masks = torch.ones((1, 2), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    # full_data is deliberately inconsistent; direction validation must win.
    full = CountingFullData(
        torch.full((20,), 99, dtype=torch.long),
        torch.full((20,), 99, dtype=torch.long),
        torch.full((20,), 99, dtype=torch.long),
    )
    with pytest.raises(ValueError) as ref_error:
        _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu"))
    with pytest.raises(ValueError) as opt_error:
        encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    assert str(opt_error.value) == str(ref_error.value) == "history direction must be 0 or 1, got 2"


def test_full_data_endpoint_and_timestamp_errors_match_reference() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2]], dtype=torch.long)
    directions = torch.tensor([[0, 1]], dtype=torch.int8)
    masks = torch.ones((1, 2), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )

    cases = []
    # endpoint mismatch
    src = torch.zeros(20, dtype=torch.long)
    dst = torch.zeros(20, dtype=torch.long)
    t = torch.zeros(20, dtype=torch.long)
    src[10], dst[10], t[10] = 0, 1, 10
    src[11], dst[11], t[11] = 0, 0, 20  # expected (2, 0)
    cases.append((CountingFullData(src, dst, t), "history direction and full_data edge endpoints disagree"))
    # timestamp mismatch
    src2 = torch.zeros(20, dtype=torch.long)
    dst2 = torch.zeros(20, dtype=torch.long)
    t2 = torch.zeros(20, dtype=torch.long)
    src2[10], dst2[10], t2[10] = 0, 1, 10
    src2[11], dst2[11], t2[11] = 2, 0, 999
    cases.append((CountingFullData(src2, dst2, t2), "history timestamp and full_data timestamp disagree"))

    for full, message in cases:
        with pytest.raises(ValueError, match=message) as ref_error:
            _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu"))
        with pytest.raises(ValueError, match=message) as opt_error:
            encoder._scan_scale("long", sampled, full, torch.device("cpu"))
        assert str(opt_error.value) == str(ref_error.value) == message


def test_full_data_size_mismatch_matches_reference() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2]], dtype=torch.long)
    directions = torch.tensor([[0, 1]], dtype=torch.int8)
    masks = torch.ones((1, 2), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )

    class ShortFullData(CountingFullData):
        def get_event_values(self, field: str, event_ids: Tensor) -> Tensor:
            values = super().get_event_values(field, event_ids)
            return values[:-1]

    full = ShortFullData(
        torch.zeros(20, dtype=torch.long),
        torch.zeros(20, dtype=torch.long),
        torch.zeros(20, dtype=torch.long),
    )
    message = "full_data validation fields have inconsistent sizes"
    with pytest.raises(ValueError, match=message):
        _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu"))
    with pytest.raises(ValueError, match=message):
        encoder._scan_scale("long", sampled, full, torch.device("cpu"))


def test_full_data_lookup_order_is_src_dst_t() -> None:
    encoder, _, _ = _make_encoder()
    event_ids = torch.tensor([[10, 11]], dtype=torch.long)
    neighbor_ids = torch.tensor([[1, 2]], dtype=torch.long)
    directions = torch.tensor([[0, 1]], dtype=torch.int8)
    masks = torch.ones((1, 2), dtype=torch.bool)
    timestamps = torch.tensor([[10, 20]], dtype=torch.long)
    query_nodes = torch.tensor([0], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = _full_data_from_sampled(
        event_ids, neighbor_ids, directions, masks, timestamps, query_nodes, 20
    )
    encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    assert [field for field, _ in full.calls] == ["src", "dst", "t"]
    for _, lookup in full.calls:
        assert lookup.tolist() == [10, 11]


# ---------------------------------------------------------------------------
# Non-contiguous inputs
# ---------------------------------------------------------------------------
def test_non_contiguous_inputs_match_reference() -> None:
    encoder, _, _ = _make_encoder()
    q_count, budget = 3, 4
    event_ids = torch.arange(q_count * budget * 2).reshape(q_count, budget, 2)[..., 0]
    neighbor_ids = torch.arange(q_count * budget * 3).reshape(q_count, budget, 3)[..., 0]
    directions = torch.zeros((q_count, budget, 2), dtype=torch.int8)[..., 0]
    masks = torch.zeros((q_count, budget, 2), dtype=torch.bool)[..., 0]
    timestamps = torch.arange(q_count * budget * 2).reshape(q_count, budget, 2)[..., 0]

    for q_idx in range(q_count):
        for slot in range(budget):
            event_ids[q_idx, slot] = 10 + q_idx * budget + slot
            neighbor_ids[q_idx, slot] = slot
            directions[q_idx, slot] = (q_idx + slot) % 2
            masks[q_idx, slot] = True
            timestamps[q_idx, slot] = 100 + q_idx * 10 + slot

    assert not event_ids.is_contiguous()
    assert not masks.is_contiguous()
    query_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    sampled = _sample_dict(
        "long", event_ids, neighbor_ids, directions, masks, timestamps, query_nodes
    )
    full = _full_data_from_sampled(
        event_ids, neighbor_ids, directions, masks, timestamps, query_nodes, 40
    )
    expected = _reference_scan_scale(encoder, "long", sampled, full, torch.device("cpu"))
    actual = encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    assert actual == expected


# ---------------------------------------------------------------------------
# RNG / HistoryStore / profiling
# ---------------------------------------------------------------------------
def test_rng_state_is_unchanged() -> None:
    encoder, _, _ = _make_encoder()
    sampled, full = _random_case(random.Random(101), 5, 8, 32)
    torch.manual_seed(1234)
    py_state = random.getstate()
    rng_before = torch.random.get_rng_state()
    encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after)
    assert random.getstate() == py_state


def test_history_store_state_is_unchanged_by_scan() -> None:
    encoder, loader, _ = _make_encoder()
    loader.history_store.insert(
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        torch.arange(5, dtype=torch.long),
        torch.tensor([990, 980, 970, 960, 950], dtype=torch.long),
    )
    before = {
        key: value.clone() for key, value in loader.history_state_dict().items()
        if isinstance(value, Tensor)
    }
    sampled, full = _random_case(random.Random(102), 4, 8, 32)
    encoder._scan_scale("long", sampled, full, torch.device("cpu"))
    after = {
        key: value.clone() for key, value in loader.history_state_dict().items()
        if isinstance(value, Tensor)
    }
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key


def test_profile_metrics_and_scan_records(tmp_path) -> None:
    # ``multiscale_encoder`` may resolve the profiler through either
    # ``runtime`` or ``src.runtime`` depending on the test session's import
    # order.  Configure the exact module instance used by the encoder.
    encoder_module = importlib.import_module(MultiScaleOrthrusEncoder.__module__)
    profiler_module = importlib.import_module(encoder_module.get_profiler.__module__)
    profiler = profiler_module.configure_profiler(
        enabled=True,
        run_root=tmp_path,
        sample_every_n_batches=1,
    )
    try:
        encoder, _, _ = _make_encoder()
        sampled, full = _random_case(random.Random(103), 5, 8, 32)

        token = profiler.begin_batch()
        records = encoder._scan_scale("long", sampled, full, torch.device("cpu"))
        profiler.end_batch(token, "training.T_total_batch")
        row = profiler._batch_rows[-1]
        assert len(records) > 0
        assert row["long_scan_records"] == len(records)
        for key in (
            "time.training.scan_scale.long.cpu_materialize",
            "time.training.scan_scale.long.full_data_query",
            "time.training.scan_scale.long.candidate_loop",
        ):
            assert key in row

        empty_ids = torch.full((2, 4), -1, dtype=torch.long)
        empty_sampled = _sample_dict(
            "long",
            empty_ids,
            torch.zeros_like(empty_ids),
            torch.zeros_like(empty_ids, dtype=torch.int8),
            torch.zeros_like(empty_ids, dtype=torch.bool),
            torch.zeros_like(empty_ids),
            torch.tensor([0, 1], dtype=torch.long),
        )
        empty_full = CountingFullData(
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
        )
        token = profiler.begin_batch()
        assert encoder._scan_scale("long", empty_sampled, empty_full, torch.device("cpu")) == []
        profiler.end_batch(token, "training.T_total_batch")
        empty_row = profiler._batch_rows[-1]
        assert empty_row["long_scan_records"] == 0
        for key in (
            "time.training.scan_scale.long.cpu_materialize",
            "time.training.scan_scale.long.full_data_query",
            "time.training.scan_scale.long.candidate_loop",
        ):
            assert key in empty_row
    finally:
        profiler_module.configure_profiler(enabled=False, run_root=None)


# ---------------------------------------------------------------------------
# Full model differential: ordered edge_index / features / masks / gradients
# ---------------------------------------------------------------------------
def _full_data(num_events: int = 48) -> SimpleNamespace:
    src = torch.arange(num_events, dtype=torch.long) % 8
    dst = (src + 1) % 8
    timestamps = torch.full((num_events,), 1000, dtype=torch.long)
    # History events 0..23 have mixed short/medium/long deltas from t=1000.
    timestamps[:24] = torch.tensor(
        [990, 970, 900, 995, 960, 800, 985, 930, 700, 992, 955, 600,
         980, 940, 500, 975, 920, 400, 965, 910, 300, 950, 890, 200],
        dtype=torch.long,
    )
    x_src = (torch.arange(num_events * 3, dtype=torch.float32).view(num_events, 3) % 17) + 1
    x_dst = x_src + 100
    msg = (torch.arange(num_events * 2, dtype=torch.float32).view(num_events, 2) % 13) + 200
    edge_type = (torch.arange(num_events * 2, dtype=torch.float32).view(num_events, 2) % 11) + 400
    return SimpleNamespace(
        src=src,
        dst=dst,
        t=timestamps,
        x_src=x_src,
        x_dst=x_dst,
        msg=msg,
        edge_type=edge_type,
    )


def _insert_history(loader: MultiScaleNeighborLoader, full: SimpleNamespace) -> None:
    loader.history_store.insert(
        full.src[:24].clone(),
        full.dst[:24].clone(),
        torch.arange(24, dtype=torch.long),
        full.t[:24].clone(),
    )


def _batch(full: SimpleNamespace, device: torch.device) -> dict[str, object]:
    src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    size = src.numel()
    return {
        "edge_index": torch.stack([src, dst]),
        "t": torch.full((size,), 1000, dtype=torch.long, device=device),
        "x": (
            full.x_src[24:24 + size].to(device),
            full.x_dst[24:24 + size].to(device),
        ),
        "global_event_index": torch.arange(24, 24 + size, dtype=torch.long),
    }


def test_full_model_edge_index_features_masks_and_backward_match_reference() -> None:
    full = _full_data()
    encoder_ref, loader_ref, graph_ref = _make_encoder()
    encoder_opt, loader_opt, graph_opt = _make_encoder()
    encoder_opt.load_state_dict(encoder_ref.state_dict())
    _insert_history(loader_ref, full)
    _insert_history(loader_opt, full)

    # Replace only the scan implementation on the reference encoder.
    encoder_ref._scan_scale = types.MethodType(_reference_scan_scale, encoder_ref)

    batch = _batch(full, torch.device("cpu"))
    torch.manual_seed(777)
    h_src_ref, h_dst_ref = encoder_ref(
        full_data=full,
        msg=torch.zeros(4, 2),
        **batch,
    )
    torch.manual_seed(777)
    h_src_opt, h_dst_opt = encoder_opt(
        full_data=full,
        msg=torch.zeros(4, 2),
        **batch,
    )

    assert torch.equal(h_src_ref, h_src_opt)
    assert torch.equal(h_dst_ref, h_dst_opt)
    assert torch.equal(encoder_ref.get_last_scale_mask(), encoder_opt.get_last_scale_mask())
    assert torch.equal(encoder_ref.get_last_gate_weights(), encoder_opt.get_last_gate_weights())
    assert len(graph_ref.calls) == len(graph_opt.calls)
    for ref_call, opt_call in zip(graph_ref.calls, graph_opt.calls):
        assert torch.equal(ref_call["x"], opt_call["x"])
        assert torch.equal(ref_call["edge_index"], opt_call["edge_index"])
        assert torch.equal(ref_call["edge_feats"], opt_call["edge_feats"])

    ref_state = loader_ref.history_state_dict()
    opt_state = loader_opt.history_state_dict()
    assert ref_state.keys() == opt_state.keys()
    for key in ref_state:
        if isinstance(ref_state[key], Tensor):
            assert torch.equal(ref_state[key], opt_state[key]), key

    loss_ref = h_src_ref.sum() + h_dst_ref.sum()
    loss_opt = h_src_opt.sum() + h_dst_opt.sum()
    loss_ref.backward()
    loss_opt.backward()
    ref_params = dict(encoder_ref.named_parameters())
    opt_params = dict(encoder_opt.named_parameters())
    assert ref_params.keys() == opt_params.keys()
    for name, ref_param in ref_params.items():
        opt_param = opt_params[name]
        assert opt_param.grad is not None, name
        assert ref_param.grad is not None, name
        assert torch.equal(ref_param.grad, opt_param.grad), name


# ---------------------------------------------------------------------------
# CUDA validation is opt-in; this phase only records it as pending.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("ORTHRUS_ALLOW_CUDA_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="CUDA scan tests require ORTHRUS_ALLOW_CUDA_TESTS=1 and a GPU",
)
def test_cuda_scan_records_match_cpu_reference() -> None:
    encoder, _, _ = _make_encoder()
    sampled, full = _random_case(random.Random(104), 5, 8, 32)
    expected = _reference_scan_scale(encoder, "long", sampled, full, torch.device("cuda"))
    sampled_cuda = {key: value.to("cuda") for key, value in sampled.items()}
    actual = encoder._scan_scale("long", sampled_cuda, full, torch.device("cuda"))
    assert actual == expected
