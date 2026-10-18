"""P1-4B differential tests: vectorized MultiScaleNeighborLoader.

The reference implementation below is a verbatim transliteration of the
pre-P1-4B scalar sampling loop. The optimized implementation must reproduce
its integer/bool outputs bit-for-bit, including:

- query node order and per-node reference timestamps;
- short/medium/long boundaries (long has no upper bound);
- timestamp-descending / event-id-descending stable recency order;
- top-k budget truncation and -1/False padding;
- query-before-insert causal ordering and HistoryStore state;
- random-number-generator state (the sampler itself must not consume RNG).
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mstc.history_store import HistoryStore
from src.mstc.multiscale_sampler import (
    MultiScaleNeighborLoader,
    _recency_order,
)


# ---------------------------------------------------------------------------
# Frozen reference implementation (pre-P1-4B scalar behavior)
# ---------------------------------------------------------------------------
def _reference_query_prepare(src, dst, timestamp_ns):
    batch_size = src.shape[0]
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

    node_to_qidx = {node: idx for idx, node in enumerate(all_nodes_list)}
    src_query_index = torch.zeros(batch_size, dtype=torch.long)
    dst_query_index = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        src_query_index[i] = node_to_qidx[int(src_cpu[i])]
        dst_query_index[i] = node_to_qidx[int(dst_cpu[i])]

    return query_nodes, reference_time_ns, src_query_index, dst_query_index


def _reference_scale(
    history,
    reference_time_ns,
    scale_name,
    tau_low,
    tau_high,
    budget,
    device,
):
    neighbor_ids = history["neighbor_id"]
    event_ids = history["event_id"]
    timestamps = history["timestamp_ns"]
    directions = history["direction"]
    Q = reference_time_ns.numel()

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
    scale_mask = torch.zeros((Q, budget), dtype=torch.bool, device=device)

    selected_count = 0
    ref_cpu = reference_time_ns.cpu()
    ts_cpu = timestamps.cpu()
    ev_cpu = event_ids.cpu()
    nb_cpu = neighbor_ids.cpu()
    dir_cpu = directions.cpu()

    for q_idx in range(Q):
        valid_ts_mask = ts_cpu[q_idx] >= 0
        delta = ref_cpu[q_idx] - ts_cpu[q_idx]
        if scale_name == "short":
            in_range = valid_ts_mask & (delta > 0) & (delta <= tau_high)
        elif scale_name == "medium":
            in_range = valid_ts_mask & (delta > tau_low) & (delta <= tau_high)
        else:  # long: no upper bound
            in_range = valid_ts_mask & (delta > tau_low)

        valid_ts = ts_cpu[q_idx][in_range]
        valid_ev = ev_cpu[q_idx][in_range]
        valid_nb = nb_cpu[q_idx][in_range]
        valid_dir = dir_cpu[q_idx][in_range]

        sorted_idx = _recency_order(valid_ts, valid_ev)
        sorted_ts = valid_ts[sorted_idx]
        sorted_ev = valid_ev[sorted_idx]
        sorted_nb = valid_nb[sorted_idx]
        sorted_dir = valid_dir[sorted_idx]

        num_valid = min(len(sorted_ts), budget)
        selected_count += num_valid
        if num_valid > 0:
            scale_neighbor[q_idx, :num_valid] = sorted_nb[:num_valid].to(device)
            scale_event_id[q_idx, :num_valid] = sorted_ev[:num_valid].to(device)
            scale_timestamp[q_idx, :num_valid] = sorted_ts[:num_valid].to(device)
            scale_direction[q_idx, :num_valid] = sorted_dir[:num_valid].to(device)
            scale_mask[q_idx, :num_valid] = True

    return {
        f"{scale_name}_neighbor_id": scale_neighbor,
        f"{scale_name}_event_id": scale_event_id,
        f"{scale_name}_timestamp_ns": scale_timestamp,
        f"{scale_name}_direction": scale_direction,
        f"{scale_name}_mask": scale_mask,
    }, selected_count


def _reference_loader_call(loader, src, dst, timestamp_ns):
    device = src.device
    query_nodes, reference_time_ns, src_query_index, dst_query_index = (
        _reference_query_prepare(src, dst, timestamp_ns)
    )
    history = loader.history_store.query(query_nodes, reference_time_ns)

    result = {
        "query_nodes": query_nodes.to(device),
        "reference_time_ns": reference_time_ns.to(device),
        "src_query_index": src_query_index.to(device),
        "dst_query_index": dst_query_index.to(device),
    }
    for scale_name, tau_low, tau_high, budget in [
        ("short", 0, loader.tau_short_ns, loader.short_budget),
        ("medium", loader.tau_short_ns, loader.tau_medium_ns, loader.medium_budget),
        ("long", loader.tau_medium_ns, None, loader.long_budget),
    ]:
        scale_result, _ = _reference_scale(
            history,
            reference_time_ns,
            scale_name,
            tau_low,
            tau_high,
            budget,
            device,
        )
        result.update(scale_result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_loader(
    *,
    num_nodes=20,
    capacity=8,
    budgets=(3, 3, 3),
    tau_short_ns=10,
    tau_medium_ns=30,
    device="cpu",
):
    store = HistoryStore(
        num_nodes=num_nodes,
        candidate_capacity=capacity,
        device=device,
    )
    loader = MultiScaleNeighborLoader(
        store,
        tau_short_ns=tau_short_ns,
        tau_medium_ns=tau_medium_ns,
        short_budget=budgets[0],
        medium_budget=budgets[1],
        long_budget=budgets[2],
    )
    return loader


def _assert_result_equal(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        actual_tensor = actual[key]
        expected_tensor = expected[key]
        assert actual_tensor.shape == expected_tensor.shape, key
        assert actual_tensor.dtype == expected_tensor.dtype, key
        assert actual_tensor.device == expected_tensor.device, key
        assert torch.equal(actual_tensor, expected_tensor), key


def _clone_state(state):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def _assert_state_equal(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected:
        if torch.is_tensor(expected[key]):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def _random_batch(rng, num_nodes, batch_size, *, allow_empty=True):
    if batch_size == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone()
    src = torch.tensor(
        [rng.randrange(num_nodes) for _ in range(batch_size)], dtype=torch.long
    )
    dst = torch.tensor(
        [rng.randrange(num_nodes) for _ in range(batch_size)], dtype=torch.long
    )
    timestamps = torch.tensor(
        [rng.randrange(0, 1000) for _ in range(batch_size)], dtype=torch.long
    )
    if batch_size > 0 and rng.random() < 0.4:
        dst = src.clone()
    if batch_size > 1 and rng.random() < 0.4:
        src[batch_size // 2 :] = src[0]
    return src, dst, timestamps


def _random_history(rng, Q, capacity):
    if Q == 0:
        timestamps = torch.empty((0, capacity), dtype=torch.long)
        event_ids = torch.empty((0, capacity), dtype=torch.long)
        neighbor_ids = torch.empty((0, capacity), dtype=torch.long)
        directions = torch.empty((0, capacity), dtype=torch.int8)
        invalid = torch.zeros((0, capacity), dtype=torch.bool)
        reference_time_ns = torch.empty(0, dtype=torch.long)
        return {
            "timestamp_ns": timestamps,
            "event_id": event_ids,
            "neighbor_id": neighbor_ids,
            "direction": directions,
        }, reference_time_ns

    timestamps = torch.tensor(
        [
            [rng.randrange(-1, 60) for _ in range(capacity)]
            for _ in range(Q)
        ],
        dtype=torch.long,
    )
    event_ids = torch.tensor(
        [[rng.randrange(-1, 20) for _ in range(capacity)] for _ in range(Q)],
        dtype=torch.long,
    )
    neighbor_ids = torch.tensor(
        [[rng.randrange(-1, 30) for _ in range(capacity)] for _ in range(Q)],
        dtype=torch.long,
    )
    directions = torch.tensor(
        [[rng.randrange(-1, 2) for _ in range(capacity)] for _ in range(Q)],
        dtype=torch.long,
    ).to(torch.int8)

    invalid = torch.rand(Q, capacity) < 0.25
    timestamps[invalid] = -1
    event_ids[invalid] = -1
    neighbor_ids[invalid] = -1
    directions[invalid] = -1

    # Force stable-tie cases.
    if Q > 0 and capacity > 0 and rng.random() < 0.5:
        timestamps[0, :] = timestamps[0, 0]
        event_ids[0, :] = event_ids[0, 0]

    reference_time_ns = torch.tensor(
        [rng.randrange(0, 70) for _ in range(Q)], dtype=torch.long
    )
    return {
        "timestamp_ns": timestamps,
        "event_id": event_ids,
        "neighbor_id": neighbor_ids,
        "direction": directions,
    }, reference_time_ns


# ---------------------------------------------------------------------------
# Query prepare
# ---------------------------------------------------------------------------
def test_query_prepare_random_differential():
    rng = random.Random(1001)
    for trial in range(120):
        num_nodes = rng.choice([1, 2, 10, 200])
        capacity = rng.choice([1, 2, 8, 64])
        batch_size = rng.choice([0, 1, 2, 5, 64, 512])
        loader = _make_loader(
            num_nodes=num_nodes,
            capacity=capacity,
            budgets=(3, 3, 3),
        )
        src, dst, timestamps = _random_batch(rng, num_nodes, batch_size)

        actual = loader(src, dst, timestamps)
        expected = _reference_loader_call(loader, src, dst, timestamps)
        _assert_result_equal(actual, expected)

        expected_nodes, expected_ref, expected_src, expected_dst = (
            _reference_query_prepare(src, dst, timestamps)
        )
        for key, expected_tensor in (
            ("query_nodes", expected_nodes),
            ("reference_time_ns", expected_ref),
            ("src_query_index", expected_src),
            ("dst_query_index", expected_dst),
        ):
            assert torch.equal(actual[key], expected_tensor), (trial, key)


# ---------------------------------------------------------------------------
# Per-scale selection / sorting / top-k
# ---------------------------------------------------------------------------
def test_scale_selection_random_differential():
    rng = random.Random(2002)
    scale_specs = [
        ("short", 0, 10),
        ("medium", 10, 30),
        ("long", 30, None),
    ]
    for trial in range(240):
        Q = rng.choice([0, 1, 2, 5, 17, 50])
        capacity = rng.choice([1, 2, 8, 16, 64])
        budgets = {
            name: rng.choice([0, 1, 3, 8, 24, 80])
            for name, _, _ in scale_specs
        }
        history, reference_time_ns = _random_history(rng, Q, capacity)
        device = torch.device("cpu")

        for scale_name, tau_low, tau_high in scale_specs:
            budget = budgets[scale_name]
            expected, expected_count = _reference_scale(
                history,
                reference_time_ns,
                scale_name,
                tau_low,
                tau_high,
                budget,
                device,
            )
            actual, actual_count, _, _, _ = (
                MultiScaleNeighborLoader._sample_scale_vectorized(
                    history,
                    reference_time_ns,
                    scale_name,
                    tau_low,
                    tau_high,
                    budget,
                    device,
                    detail=False,
                )
            )
            _assert_result_equal(actual, expected)
            if budget > 0:
                assert actual_count == 0  # detail=False does not count

            actual_detail, actual_count_detail, f, s, h = (
                MultiScaleNeighborLoader._sample_scale_vectorized(
                    history,
                    reference_time_ns,
                    scale_name,
                    tau_low,
                    tau_high,
                    budget,
                    device,
                    detail=True,
                )
            )
            _assert_result_equal(actual_detail, expected)
            assert actual_count_detail == expected_count
            assert f >= 0.0 and s >= 0.0 and h >= 0.0


def test_timestamp_boundaries_and_long_upper_bound():
    reference_time = 100
    loader = _make_loader(
        num_nodes=2,
        capacity=16,
        budgets=(8, 8, 8),
        tau_short_ns=10,
        tau_medium_ns=30,
    )
    event_ids = torch.arange(7, dtype=torch.long)
    # delta: 10(short upper), 11(medium lower), 30(medium upper),
    #        31(long lower), 0(current), -1(future), 101(invalid)
    timestamps = torch.tensor(
        [
            reference_time - 10,
            reference_time - 11,
            reference_time - 30,
            reference_time - 31,
            reference_time,
            reference_time + 1,
            -1,
        ],
        dtype=torch.long,
    )
    loader.history_store.insert(
        src=torch.zeros(7, dtype=torch.long),
        dst=torch.ones(7, dtype=torch.long),
        event_id=event_ids,
        timestamp_ns=timestamps,
    )
    src = torch.tensor([0], dtype=torch.long)
    dst = torch.tensor([0], dtype=torch.long)
    t = torch.tensor([reference_time], dtype=torch.long)

    actual = loader(src, dst, t)
    expected = _reference_loader_call(loader, src, dst, t)
    _assert_result_equal(actual, expected)

    short_ids = actual["short_event_id"][actual["short_mask"]].tolist()
    medium_ids = actual["medium_event_id"][actual["medium_mask"]].tolist()
    long_ids = actual["long_event_id"][actual["long_mask"]].tolist()
    assert short_ids == [0]
    assert medium_ids == [1, 2]
    assert long_ids == [3]


def test_event_id_tie_breaks_by_timestamp_and_preserves_stability():
    history = {
        "timestamp_ns": torch.tensor([[90, 90, 90]], dtype=torch.long),
        "event_id": torch.tensor([[5, 5, 7]], dtype=torch.long),
        "neighbor_id": torch.tensor([[100, 101, 102]], dtype=torch.long),
        "direction": torch.tensor([[0, 0, 0]], dtype=torch.int8),
    }
    reference_time_ns = torch.tensor([100], dtype=torch.long)
    expected, _ = _reference_scale(
        history,
        reference_time_ns,
        "short",
        0,
        10,
        3,
        torch.device("cpu"),
    )
    actual, _, _, _, _ = MultiScaleNeighborLoader._sample_scale_vectorized(
        history,
        reference_time_ns,
        "short",
        0,
        10,
        3,
        torch.device("cpu"),
    )
    _assert_result_equal(actual, expected)
    # event_id 7 first, then the two stable event_id 5 records in input order.
    assert actual["short_event_id"].tolist() == [[7, 5, 5]]
    assert actual["short_neighbor_id"].tolist() == [[102, 100, 101]]


# ---------------------------------------------------------------------------
# Empty / padding / budget edge cases
# ---------------------------------------------------------------------------
def test_empty_batch_and_empty_history():
    rng = random.Random(3003)

    empty_loader = _make_loader(num_nodes=10, capacity=8, budgets=(3, 3, 3))
    empty = torch.empty(0, dtype=torch.long)
    actual = empty_loader(empty, empty, empty)
    expected = _reference_loader_call(empty_loader, empty, empty, empty)
    _assert_result_equal(actual, expected)
    assert actual["query_nodes"].numel() == 0
    for scale in ("short", "medium", "long"):
        assert actual[f"{scale}_mask"].shape == (0, 3)

    populated_loader = _make_loader(num_nodes=10, capacity=8, budgets=(3, 3, 3))
    populated_loader.history_store.insert(
        src=torch.zeros(4, dtype=torch.long),
        dst=torch.ones(4, dtype=torch.long),
        event_id=torch.arange(4, dtype=torch.long),
        timestamp_ns=torch.tensor([90, 80, 70, 60], dtype=torch.long),
    )
    src, dst, timestamps = _random_batch(rng, 10, 0)
    actual = populated_loader(src, dst, timestamps)
    expected = _reference_loader_call(populated_loader, src, dst, timestamps)
    _assert_result_equal(actual, expected)

    src = torch.tensor([5, 6], dtype=torch.long)
    dst = torch.tensor([7, 8], dtype=torch.long)
    timestamps = torch.tensor([100, 100], dtype=torch.long)
    actual = populated_loader(src, dst, timestamps)
    expected = _reference_loader_call(populated_loader, src, dst, timestamps)
    _assert_result_equal(actual, expected)
    for scale in ("short", "medium", "long"):
        assert not actual[f"{scale}_mask"].any()


def test_budget_zero_and_budget_over_capacity_padding():
    capacity = 4
    budget_cases = (0, 10)
    history = {
        "timestamp_ns": torch.tensor(
            [[99, 98, 97, 96], [95, 94, 93, 92], [91, 92, 93, 94]],
            dtype=torch.long,
        ),
        "event_id": torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=torch.long,
        ),
        "neighbor_id": torch.tensor(
            [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21, 22]],
            dtype=torch.long,
        ),
        "direction": torch.zeros((3, 4), dtype=torch.int8),
    }
    reference_time_ns = torch.tensor([100, 100, 100], dtype=torch.long)

    for budget in budget_cases:
        expected, expected_count = _reference_scale(
            history,
            reference_time_ns,
            "short",
            0,
            10,
            budget,
            torch.device("cpu"),
        )
        actual, actual_count, _, _, _ = (
            MultiScaleNeighborLoader._sample_scale_vectorized(
                history,
                reference_time_ns,
                "short",
                0,
                10,
                budget,
                torch.device("cpu"),
                detail=True,
            )
        )
        _assert_result_equal(actual, expected)
        assert actual_count == expected_count
        assert actual["short_event_id"].shape == (3, budget)
        assert actual["short_event_id"].dtype == torch.int64
        assert actual["short_direction"].dtype == torch.int8
        assert actual["short_mask"].dtype == torch.bool
        if budget > capacity:
            assert torch.equal(
                actual["short_event_id"][:, capacity:],
                torch.full((3, budget - capacity), -1, dtype=torch.long),
            )
            assert not actual["short_mask"][:, capacity:].any()


# ---------------------------------------------------------------------------
# Full loader, causal ordering, history state
# ---------------------------------------------------------------------------
def test_full_loader_random_differential_causal_and_state():
    rng = random.Random(4004)
    for trial in range(30):
        num_nodes = rng.choice([2, 10, 50])
        capacity = rng.choice([1, 2, 8, 64])
        budget_choices = [0, 1, 3, 8, 24]
        budgets = (
            rng.choice(budget_choices),
            rng.choice(budget_choices),
            rng.choice(budget_choices),
        )
        loader_opt = _make_loader(
            num_nodes=num_nodes, capacity=capacity, budgets=budgets
        )

        # Seed the same initial history into the reference loader.
        init_batches = []
        for _ in range(rng.choice([0, 1, 2])):
            batch_size = rng.choice([1, 4, 16])
            init_batches.append(
                (
                    _random_batch(rng, num_nodes, batch_size),
                    torch.arange(batch_size, dtype=torch.long),
                )
            )
        for (src, dst, timestamps), event_ids in init_batches:
            loader_opt.history_store.insert(
                src=src,
                dst=dst,
                event_id=event_ids,
                timestamp_ns=timestamps,
            )
        initial_state = _clone_state(loader_opt.history_state_dict())
        loader_ref = _make_loader(
            num_nodes=num_nodes, capacity=capacity, budgets=budgets
        )
        loader_ref.history_store.load_state_dict(initial_state)

        for step in range(3):
            batch_size = rng.choice([0, 1, 3, 16, 64])
            src, dst, timestamps = _random_batch(
                rng, num_nodes, batch_size
            )
            event_ids = torch.arange(
                trial * 1000 + step * 100,
                trial * 1000 + step * 100 + batch_size,
                dtype=torch.long,
            )

            # Reference query is read-only and sees the pre-insert state.
            expected = _reference_loader_call(
                loader_ref, src, dst, timestamps
            )
            actual = loader_opt(src, dst, timestamps)
            _assert_result_equal(actual, expected)

            loader_opt.insert(src, dst, timestamps, event_ids)
            loader_ref.insert(src, dst, timestamps, event_ids)
            _assert_state_equal(
                loader_opt.history_state_dict(),
                loader_ref.history_state_dict(),
            )


def test_query_before_insert_is_visible_only_in_next_forward():
    loader = _make_loader(
        num_nodes=4,
        capacity=8,
        budgets=(3, 3, 3),
        tau_short_ns=10,
        tau_medium_ns=30,
    )
    full_timestamps = torch.tensor([100, 110], dtype=torch.long)
    first_src = torch.tensor([0], dtype=torch.long)
    first_dst = torch.tensor([1], dtype=torch.long)
    first_t = torch.tensor([100], dtype=torch.long)

    first = loader(first_src, first_dst, first_t)
    assert not first["short_mask"].any()
    assert not first["medium_mask"].any()
    assert not first["long_mask"].any()

    loader.insert(
        src=first_src,
        dst=first_dst,
        timestamp_ns=first_t,
        global_event_index=torch.tensor([0], dtype=torch.long),
    )

    second_src = torch.tensor([0], dtype=torch.long)
    second_dst = torch.tensor([1], dtype=torch.long)
    second_t = torch.tensor([110], dtype=torch.long)
    second = loader(second_src, second_dst, second_t)
    expected = _reference_loader_call(
        loader, second_src, second_dst, second_t
    )
    _assert_result_equal(second, expected)
    assert second["short_mask"].any()
    src_query_idx = int(second["src_query_index"][0])
    assert second["short_event_id"][src_query_idx][
        second["short_mask"][src_query_idx]
    ].tolist() == [0]


# ---------------------------------------------------------------------------
# RNG and output protocol
# ---------------------------------------------------------------------------
def test_rng_state_is_unchanged():
    rng = random.Random(5005)
    loader = _make_loader(num_nodes=20, capacity=16, budgets=(3, 3, 3))
    src, dst, timestamps = _random_batch(rng, 20, 64)
    torch.manual_seed(12345)
    before = torch.random.get_rng_state().clone()
    loader(src, dst, timestamps)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_output_keys_shapes_and_dtypes():
    loader = _make_loader(num_nodes=8, capacity=4, budgets=(2, 3, 4))
    src = torch.tensor([0, 1, 2], dtype=torch.long)
    dst = torch.tensor([3, 4, 5], dtype=torch.long)
    timestamps = torch.tensor([100, 101, 102], dtype=torch.long)
    result = loader(src, dst, timestamps)
    assert set(result.keys()) == {
        "query_nodes",
        "reference_time_ns",
        "src_query_index",
        "dst_query_index",
        "short_neighbor_id",
        "short_event_id",
        "short_timestamp_ns",
        "short_direction",
        "short_mask",
        "medium_neighbor_id",
        "medium_event_id",
        "medium_timestamp_ns",
        "medium_direction",
        "medium_mask",
        "long_neighbor_id",
        "long_event_id",
        "long_timestamp_ns",
        "long_direction",
        "long_mask",
    }
    short_budget = 2
    medium_budget = 3
    long_budget = 4
    for name, budget in (
        ("short", short_budget),
        ("medium", medium_budget),
        ("long", long_budget),
    ):
        assert result[f"{name}_neighbor_id"].shape == (6, budget)
        assert result[f"{name}_event_id"].shape == (6, budget)
        assert result[f"{name}_timestamp_ns"].shape == (6, budget)
        assert result[f"{name}_direction"].shape == (6, budget)
        assert result[f"{name}_mask"].shape == (6, budget)
        assert result[f"{name}_neighbor_id"].dtype == torch.int64
        assert result[f"{name}_event_id"].dtype == torch.int64
        assert result[f"{name}_timestamp_ns"].dtype == torch.int64
        assert result[f"{name}_direction"].dtype == torch.int8
        assert result[f"{name}_mask"].dtype == torch.bool


def test_profiling_metric_names_and_outputs_preserved():
    from src.runtime.runtime_profiler import configure_profiler, get_profiler

    rng = random.Random(7007)
    loader = _make_loader(num_nodes=20, capacity=16, budgets=(3, 3, 3))
    src, dst, timestamps = _random_batch(rng, 20, 64)
    loader.history_store.insert(
        src=src,
        dst=dst,
        event_id=torch.arange(64, dtype=torch.long),
        timestamp_ns=timestamps - 50,
    )

    try:
        configure_profiler(enabled=False, run_root=None)
        expected = loader(src, dst, timestamps)

        configure_profiler(
            enabled=True,
            run_root=None,
            sample_every_n_batches=1,
        )
        profiler = get_profiler()
        with profiler.batch("training.T_total_batch"):
            actual = loader(src, dst, timestamps)
        batch_row = dict(profiler._batch_rows[0]) if profiler._batch_rows else {}
    finally:
        configure_profiler(enabled=False, run_root=None)

    _assert_result_equal(actual, expected)
    required_timings = {
        f"time.training.sampler.{scale}.{phase}"
        for scale in ("short", "medium", "long")
        for phase in ("time_filter", "sort_select", "h2d_enqueue")
    }
    assert required_timings <= set(batch_row)
    assert "query_nodes" in batch_row
    assert batch_row["query_nodes"] == int(expected["query_nodes"].numel())
    for scale in ("short", "medium", "long"):
        assert f"{scale}_sampled_edges" in batch_row


# ---------------------------------------------------------------------------
# CUDA validation is intentionally opt-in; do not consume a GPU by default.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("ORTHRUS_ALLOW_CUDA_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="CUDA sampler tests require ORTHRUS_ALLOW_CUDA_TESTS=1 and a GPU",
)
def test_cuda_device_equivalence():
    device = torch.device("cuda")
    rng = random.Random(6006)
    loader = _make_loader(
        num_nodes=20,
        capacity=16,
        budgets=(3, 3, 3),
    )
    src, dst, timestamps = _random_batch(rng, 20, 64)
    loader.history_store.insert(
        src=src,
        dst=dst,
        event_id=torch.arange(64, dtype=torch.long),
        timestamp_ns=timestamps - 50,
    )
    src_cuda = src.to(device)
    dst_cuda = dst.to(device)
    timestamps_cuda = timestamps.to(device)

    before = torch.cuda.get_rng_state_all()
    actual = loader(src_cuda, dst_cuda, timestamps_cuda)
    after = torch.cuda.get_rng_state_all()
    expected = _reference_loader_call(
        loader, src_cuda, dst_cuda, timestamps_cuda
    )
    _assert_result_equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
