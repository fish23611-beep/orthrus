"""Correctness tests for HistoryStore's per-node lexicographic ordering."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Tuple

import pytest
import torch

from src.mstc.history_store import HistoryStore


Record = Tuple[int, int, int, int]


def _python_reference_insert(
    history: Dict[int, List[Record]],
    *,
    src: Iterable[int],
    dst: Iterable[int],
    event_id: Iterable[int],
    timestamp_ns: Iterable[int],
    direction: Optional[Iterable[int]],
    capacity: int,
    store_direction: bool,
) -> None:
    """Update ``history`` using Python integers and Python's stable sort."""
    src_values = list(src)
    dst_values = list(dst)
    event_values = list(event_id)
    timestamp_values = list(timestamp_ns)
    direction_values = list(direction) if direction is not None else None

    expanded: List[Tuple[int, Record]] = []
    # HistoryStore concatenates all source-side records before destination-side
    # records. This order is observable when both sort fields are identical.
    for index in range(len(src_values)):
        value = direction_values[index] if direction_values is not None else 0
        expanded.append(
            (
                src_values[index],
                (
                    dst_values[index],
                    event_values[index],
                    timestamp_values[index],
                    value if store_direction else -1,
                ),
            )
        )
    for index in range(len(src_values)):
        value = direction_values[index] if direction_values is not None else 0
        expanded.append(
            (
                dst_values[index],
                (
                    src_values[index],
                    event_values[index],
                    timestamp_values[index],
                    (1 - value) if store_direction else -1,
                ),
            )
        )

    nodes = {node for node, _ in expanded}
    for node in nodes:
        new_records = [record for record_node, record in expanded if record_node == node]
        # Python's sort is stable and its integers do not overflow.
        new_records.sort(key=lambda record: (record[2], record[1]), reverse=True)
        merged = history.get(node, []) + new_records
        merged.sort(key=lambda record: (record[2], record[1]), reverse=True)
        history[node] = merged[:capacity]


def _assert_store_matches_reference(
    store: HistoryStore,
    history: Dict[int, List[Record]],
) -> None:
    capacity = store.candidate_capacity
    for node in range(store.num_nodes):
        expected = history.get(node, [])
        assert int(store._count[node]) == len(expected)

        expected_neighbors = [record[0] for record in expected]
        expected_events = [record[1] for record in expected]
        expected_times = [record[2] for record in expected]

        assert store._neighbor_id[node].tolist() == expected_neighbors + [-1] * (
            capacity - len(expected)
        )
        assert store._event_id[node].tolist() == expected_events + [-1] * (
            capacity - len(expected)
        )
        assert store._timestamp_ns[node].tolist() == expected_times + [-1] * (
            capacity - len(expected)
        )
        if store.store_direction:
            expected_directions = [record[3] for record in expected]
            assert store._direction[node].tolist() == expected_directions + [-1] * (
                capacity - len(expected)
            )


def _insert_and_reference(
    store: HistoryStore,
    history: Dict[int, List[Record]],
    *,
    src: List[int],
    dst: List[int],
    event_id: List[int],
    timestamp_ns: List[int],
    direction: Optional[List[int]] = None,
) -> None:
    direction_tensor = (
        torch.tensor(direction, dtype=torch.int8) if direction is not None else None
    )
    store.insert(
        src=torch.tensor(src, dtype=torch.int64),
        dst=torch.tensor(dst, dtype=torch.int64),
        event_id=torch.tensor(event_id, dtype=torch.int64),
        timestamp_ns=torch.tensor(timestamp_ns, dtype=torch.int64),
        direction=direction_tensor,
    )
    _python_reference_insert(
        history,
        src=src,
        dst=dst,
        event_id=event_id,
        timestamp_ns=timestamp_ns,
        direction=direction,
        capacity=store.candidate_capacity,
        store_direction=store.store_direction,
    )
    _assert_store_matches_reference(store, history)


def test_2019_nanoseconds_single_batch_keeps_latest_events() -> None:
    """A single batch reproduces overflow in the initial event sort."""
    base = 1_550_000_000_000_000_000
    store = HistoryStore(num_nodes=5, candidate_capacity=2, device="cpu")
    history: Dict[int, List[Record]] = {}

    _insert_and_reference(
        store,
        history,
        src=[0, 0, 0, 0],
        dst=[1, 2, 3, 4],
        event_id=[10, 11, 12, 13],
        timestamp_ns=[
            base,
            base + 1_000_000_000,
            base + 2_000_000_000,
            base + 3_000_000_000,
        ],
        direction=[0, 0, 0, 0],
    )

    assert store._event_id[0].tolist() == [13, 12]


def test_2019_nanoseconds_multi_batch_merge_is_descending() -> None:
    """Multiple inserts reproduce overflow in the existing-history merge."""
    base = 1_550_000_000_000_000_000
    store = HistoryStore(num_nodes=3, candidate_capacity=4, device="cpu")
    history: Dict[int, List[Record]] = {}

    for event, offset in [(10, 0), (12, 2_000_000_000), (11, 1_000_000_000)]:
        _insert_and_reference(
            store,
            history,
            src=[0],
            dst=[1],
            event_id=[event],
            timestamp_ns=[base + offset],
            direction=[0],
        )

    assert store._timestamp_ns[0, :3].tolist() == [
        base + 2_000_000_000,
        base + 1_000_000_000,
        base,
    ]


def test_int64_extremes_and_event_id_tiebreaker() -> None:
    minimum = torch.iinfo(torch.int64).min
    maximum = torch.iinfo(torch.int64).max
    store = HistoryStore(num_nodes=6, candidate_capacity=8, device="cpu")
    history: Dict[int, List[Record]] = {}

    _insert_and_reference(
        store,
        history,
        src=[0, 0, 0, 0, 0, 0],
        dst=[1, 2, 3, 4, 5, 1],
        event_id=[5, 90, 10, 30, 20, 40],
        timestamp_ns=[minimum, maximum, 0, maximum, maximum, minimum],
        direction=[0, 1, 0, 1, 0, 1],
    )

    assert store._event_id[0, :6].tolist() == [90, 30, 20, 10, 40, 5]
    assert store._timestamp_ns[0, :6].tolist() == [
        maximum,
        maximum,
        maximum,
        0,
        minimum,
        minimum,
    ]


def test_identical_keys_are_stable_across_batches_and_self_loop() -> None:
    timestamp = 1_550_000_000_000_000_000
    store = HistoryStore(num_nodes=5, candidate_capacity=8, device="cpu")
    history: Dict[int, List[Record]] = {}

    _insert_and_reference(
        store,
        history,
        src=[0, 0],
        dst=[1, 2],
        event_id=[7, 7],
        timestamp_ns=[timestamp, timestamp],
        direction=[0, 1],
    )
    _insert_and_reference(
        store,
        history,
        src=[0, 0],
        dst=[3, 0],
        event_id=[7, 7],
        timestamp_ns=[timestamp, timestamp],
        direction=[1, 0],
    )

    # Existing equal-key records remain before new ones. Within a batch, all
    # source-side records precede the self-loop's destination-side record.
    assert store._neighbor_id[0, :5].tolist() == [1, 2, 3, 0, 0]
    assert store._direction[0, :5].tolist() == [0, 1, 1, 0, 1]


@pytest.mark.parametrize("capacity", [1, 2, 8, 64])
@pytest.mark.parametrize("store_direction", [True, False])
def test_random_batches_match_python_reference(
    capacity: int, store_direction: bool
) -> None:
    rng = random.Random(20250301 + capacity + int(store_direction))
    store = HistoryStore(
        num_nodes=9,
        candidate_capacity=capacity,
        device="cpu",
        store_direction=store_direction,
    )
    history: Dict[int, List[Record]] = {}
    timestamp_pool = [
        torch.iinfo(torch.int64).min,
        -1,
        0,
        1,
        1_550_000_000_000_000_000,
        1_550_000_001_000_000_000,
        torch.iinfo(torch.int64).max,
    ]

    for _ in range(7):
        batch_size = rng.randint(1, 19)
        src = [rng.randrange(9) for _ in range(batch_size)]
        dst = [rng.randrange(9) for _ in range(batch_size)]
        # A small range deliberately creates duplicate event IDs and complete
        # duplicate sort keys, while neighbors/directions expose stability.
        event_id = [rng.randrange(6) for _ in range(batch_size)]
        timestamp_ns = [rng.choice(timestamp_pool) for _ in range(batch_size)]
        direction = [rng.randrange(2) for _ in range(batch_size)]
        _insert_and_reference(
            store,
            history,
            src=src,
            dst=dst,
            event_id=event_id,
            timestamp_ns=timestamp_ns,
            direction=direction,
        )

    queried = store.query(
        node_ids=torch.arange(store.num_nodes),
        reference_time_ns=torch.zeros(store.num_nodes, dtype=torch.int64),
    )
    assert torch.equal(queried["neighbor_id"], store._neighbor_id)
    assert torch.equal(queried["event_id"], store._event_id)
    assert torch.equal(queried["timestamp_ns"], store._timestamp_ns)
    assert torch.equal(queried["count"], store._count)
    if store_direction:
        assert torch.equal(queried["direction"], store._direction)
    else:
        assert store._direction is None
        assert (queried["direction"] == -1).all()


def test_empty_insert_preserves_state_and_state_dict_roundtrip() -> None:
    store = HistoryStore(num_nodes=4, candidate_capacity=2, device="cpu")
    history: Dict[int, List[Record]] = {}
    _insert_and_reference(
        store,
        history,
        src=[0, 0, 1],
        dst=[1, 2, 3],
        event_id=[1, 2, 3],
        timestamp_ns=[
            1_550_000_000_000_000_000,
            1_550_000_001_000_000_000,
            1_550_000_002_000_000_000,
        ],
        direction=[0, 1, 0],
    )
    before = store.state_dict()

    store.insert(
        src=torch.empty(0, dtype=torch.int64),
        dst=torch.empty(0, dtype=torch.int64),
        event_id=torch.empty(0, dtype=torch.int64),
        timestamp_ns=torch.empty(0, dtype=torch.int64),
        direction=torch.empty(0, dtype=torch.int8),
    )
    after = store.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        if isinstance(before[key], torch.Tensor):
            assert torch.equal(before[key], after[key])
        else:
            assert before[key] == after[key]

    restored = HistoryStore(num_nodes=4, candidate_capacity=2, device="cpu")
    restored.load_state_dict(after)
    _assert_store_matches_reference(restored, history)
    assert restored.state_dict().keys() == after.keys()
    for key in after:
        if isinstance(after[key], torch.Tensor):
            assert torch.equal(restored.state_dict()[key], after[key])
        else:
            assert restored.state_dict()[key] == after[key]
