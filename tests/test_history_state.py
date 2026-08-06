"""
Tests for src/mstc/history_store.py — HistoryStore.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mstc.history_store import HistoryStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _store(num_nodes=10, capacity=5, device="cpu", store_direction=True):
    return HistoryStore(
        num_nodes=num_nodes,
        candidate_capacity=capacity,
        device=device,
        store_direction=store_direction,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_constructor_creates_correct_dtypes():
    store = _store(num_nodes=20, capacity=8)
    assert store._neighbor_id.dtype == torch.int64
    assert store._event_id.dtype == torch.int64
    assert store._timestamp_ns.dtype == torch.int64
    assert store._direction.dtype == torch.int8
    assert store._count.dtype == torch.int32


def test_constructor_negative_num_nodes_raises():
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        HistoryStore(num_nodes=0, candidate_capacity=5)


def test_constructor_negative_capacity_raises():
    with pytest.raises(ValueError, match="candidate_capacity must be positive"):
        HistoryStore(num_nodes=10, candidate_capacity=0)


def test_constructor_no_direction():
    store = HistoryStore(num_nodes=10, candidate_capacity=5, store_direction=False)
    assert store._direction is None


# --------------------------------------------------------------------------- #
# reset_state
# --------------------------------------------------------------------------- #
def test_reset_state_clears_all():
    store = _store(num_nodes=10, capacity=5)
    store.insert(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        event_id=torch.tensor([0, 1]),
        timestamp_ns=torch.tensor([100, 200], dtype=torch.int64),
        direction=torch.tensor([0, 0], dtype=torch.int8),
    )
    store.reset_state()

    for node in range(10):
        assert int(store._count[node].item()) == 0
        assert (store._neighbor_id[node] == -1).all()
        assert (store._event_id[node] == -1).all()
        assert (store._timestamp_ns[node] == -1).all()


# --------------------------------------------------------------------------- #
# Insert — bidirectional
# --------------------------------------------------------------------------- #
def test_insert_bidirectional():
    store = _store(num_nodes=5, capacity=5)
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([99]),
        timestamp_ns=torch.tensor([1_000_000_000], dtype=torch.int64),
        direction=torch.tensor([0], dtype=torch.int8),
    )

    assert int(store._count[0].item()) == 1
    assert int(store._count[1].item()) == 1
    assert int(store._count[2].item()) == 0

    assert int(store._neighbor_id[0, 0].item()) == 1
    assert int(store._neighbor_id[1, 0].item()) == 0
    assert int(store._event_id[0, 0].item()) == 99
    assert int(store._event_id[1, 0].item()) == 99
    assert int(store._timestamp_ns[0, 0].item()) == 1_000_000_000
    assert int(store._direction[0, 0].item()) == 0
    assert int(store._direction[1, 0].item()) == 1


def test_insert_direction_override():
    store = _store(num_nodes=5, capacity=5)
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([5]),
        timestamp_ns=torch.tensor([500], dtype=torch.int64),
        direction=torch.tensor([0], dtype=torch.int8),
    )

    assert int(store._direction[0, 0].item()) == 0
    assert int(store._direction[1, 0].item()) == 1


# --------------------------------------------------------------------------- #
# Insert — capacity limit
# --------------------------------------------------------------------------- #
def test_capacity_respected():
    store = _store(num_nodes=3, capacity=3)

    for i in range(5):
        store.insert(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([i]),
            timestamp_ns=torch.tensor([i + 1], dtype=torch.int64),
            direction=torch.tensor([0], dtype=torch.int8),
        )

    assert int(store._count[0].item()) == 3
    assert int(store._count[1].item()) == 3


def test_capacity_keeps_most_recent():
    store = _store(num_nodes=3, capacity=3)

    for i in range(5):
        store.insert(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([i]),
            timestamp_ns=torch.tensor([i * 100], dtype=torch.int64),
            direction=torch.tensor([0], dtype=torch.int8),
        )

    node0_events = store._event_id[0].cpu().tolist()
    node0_ts = store._timestamp_ns[0].cpu().tolist()

    assert -1 not in node0_events[:3]
    assert node0_ts[0] >= node0_ts[1] >= node0_ts[2]


def test_capacity_same_timestamp_stable_by_event_id():
    store = _store(num_nodes=3, capacity=3)

    for eid in [5, 3, 7, 1, 4]:
        store.insert(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([eid]),
            timestamp_ns=torch.tensor([100], dtype=torch.int64),
            direction=torch.tensor([0], dtype=torch.int8),
        )

    node0_events = store._event_id[0].cpu().tolist()
    assert node0_events[0] == 7
    assert node0_events[1] == 5
    assert node0_events[2] == 4


# --------------------------------------------------------------------------- #
# Insert — same node multiple times in batch
# --------------------------------------------------------------------------- #
def test_same_node_multiple_times_in_batch():
    store = _store(num_nodes=5, capacity=10)

    store.insert(
        src=torch.tensor([0, 0, 1]),
        dst=torch.tensor([1, 2, 0]),
        event_id=torch.tensor([10, 11, 12]),
        timestamp_ns=torch.tensor([100, 110, 120], dtype=torch.int64),
        direction=torch.tensor([0, 0, 0], dtype=torch.int8),
    )

    # Node 0: as src twice + as dst once = 3 entries
    assert int(store._count[0].item()) == 3
    # Node 1: as dst (event 0) + as src (event 2) = 2 entries
    assert int(store._count[1].item()) == 2
    # Node 2: as dst (event 1) = 1 entry
    assert int(store._count[2].item()) == 1

    node0_events = store._event_id[0].cpu().tolist()
    assert 12 in node0_events[:3]
    assert 10 in node0_events[:3]
    assert 11 in node0_events[:3]


# --------------------------------------------------------------------------- #
# Insert — dtype validation
# --------------------------------------------------------------------------- #
def test_insert_int32_timestamp_raises():
    store = _store(num_nodes=5, capacity=5)
    with pytest.raises(ValueError, match="timestamp_ns must be int64"):
        store.insert(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([0]),
            timestamp_ns=torch.tensor([100], dtype=torch.int32),
        )


def test_insert_mismatched_shapes_raises():
    store = _store(num_nodes=5, capacity=5)
    with pytest.raises(ValueError, match="same shape"):
        store.insert(
            src=torch.tensor([0, 1]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([0]),
            timestamp_ns=torch.tensor([100], dtype=torch.int64),
        )


def test_insert_direction_required_when_enabled():
    """When store_direction=True and direction is None, it is auto-derived
    as 0 (src records dst as neighbor)."""
    store = _store(num_nodes=5, capacity=5)
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100], dtype=torch.int64),
    )
    # src node 0 sees neighbor 1 with direction 0
    assert int(store._direction[0, 0].item()) == 0
    # dst node 1 sees neighbor 0 with direction 1
    assert int(store._direction[1, 0].item()) == 1


def test_insert_direction_dtype_validation():
    """direction must be int8 when provided."""
    store = _store(num_nodes=5, capacity=5)
    with pytest.raises(ValueError, match="direction must be int8"):
        store.insert(
            src=torch.tensor([0]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([0]),
            timestamp_ns=torch.tensor([100], dtype=torch.int64),
            direction=torch.tensor([0], dtype=torch.int64),
        )


# --------------------------------------------------------------------------- #
# Insert — large timestamp (int32 overflow test)
# --------------------------------------------------------------------------- #
def test_large_timestamp_no_overflow():
    store = _store(num_nodes=5, capacity=5)

    large_ts = 2**40
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([large_ts], dtype=torch.int64),
        direction=torch.tensor([0], dtype=torch.int8),
    )

    assert int(store._timestamp_ns[0, 0].item()) == large_ts


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
def test_query_returns_correct_dtypes():
    store = _store(num_nodes=10, capacity=5)
    store.insert(
        src=torch.tensor([0, 2]),
        dst=torch.tensor([1, 3]),
        event_id=torch.tensor([5, 6]),
        timestamp_ns=torch.tensor([100, 200], dtype=torch.int64),
        direction=torch.tensor([0, 0], dtype=torch.int8),
    )

    result = store.query(
        node_ids=torch.tensor([0, 2]),
        reference_time_ns=torch.tensor([1000, 2000], dtype=torch.int64),
    )

    assert result["neighbor_id"].dtype == torch.int64
    assert result["event_id"].dtype == torch.int64
    assert result["timestamp_ns"].dtype == torch.int64
    assert result["direction"].dtype == torch.int8


def test_query_returns_correct_data():
    store = _store(num_nodes=10, capacity=5)
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([42]),
        timestamp_ns=torch.tensor([1000], dtype=torch.int64),
        direction=torch.tensor([0], dtype=torch.int8),
    )

    result = store.query(
        node_ids=torch.tensor([0, 1]),
        reference_time_ns=torch.tensor([2000, 2000], dtype=torch.int64),
    )

    assert int(result["event_id"][0, 0].item()) == 42
    assert int(result["neighbor_id"][0, 0].item()) == 1
    assert int(result["event_id"][1, 0].item()) == 42
    assert int(result["neighbor_id"][1, 0].item()) == 0


def test_query_empty_history():
    store = _store(num_nodes=10, capacity=5)
    result = store.query(
        node_ids=torch.tensor([5, 6]),
        reference_time_ns=torch.tensor([1000, 2000], dtype=torch.int64),
    )

    assert int(result["count"][0].item()) == 0
    assert int(result["count"][1].item()) == 0
    assert (result["event_id"] == -1).all()


def test_query_mismatched_shapes_raises():
    store = _store(num_nodes=10, capacity=5)
    with pytest.raises(ValueError, match="same shape"):
        store.query(
            node_ids=torch.tensor([0, 1]),
            reference_time_ns=torch.tensor([1000], dtype=torch.int64),
        )


# --------------------------------------------------------------------------- #
# state_dict / load_state_dict
# --------------------------------------------------------------------------- #
def test_state_dict_returns_clone():
    store = _store(num_nodes=5, capacity=4)
    store.insert(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        event_id=torch.tensor([10, 11]),
        timestamp_ns=torch.tensor([100, 200], dtype=torch.int64),
        direction=torch.tensor([0, 0], dtype=torch.int8),
    )

    state = store.state_dict()

    assert state["neighbor_id"].device.type == "cpu"
    assert state["event_id"].device.type == "cpu"
    assert state["timestamp_ns"].device.type == "cpu"

    state["neighbor_id"][0, 0] = -999
    assert int(store._neighbor_id[0, 0].item()) != -999


def test_state_dict_roundtrip():
    store = _store(num_nodes=5, capacity=4)
    store.insert(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 3]),
        event_id=torch.tensor([10, 11, 12]),
        timestamp_ns=torch.tensor([100, 200, 300], dtype=torch.int64),
        direction=torch.tensor([0, 0, 0], dtype=torch.int8),
    )

    state = store.state_dict()
    store2 = _store(num_nodes=5, capacity=4)
    store2.load_state_dict(state)

    for node in range(5):
        assert store._count[node].item() == store2._count[node].item()

    assert torch.equal(
        store._neighbor_id.cpu(), store2._neighbor_id.cpu()
    )
    assert torch.equal(
        store._event_id.cpu(), store2._event_id.cpu()
    )
    assert torch.equal(
        store._timestamp_ns.cpu(), store2._timestamp_ns.cpu()
    )
    assert torch.equal(
        store._direction.cpu(), store2._direction.cpu()
    )


def test_load_state_dict_validates_required_keys():
    store = _store(num_nodes=5, capacity=4)
    with pytest.raises(ValueError, match="missing required keys"):
        store.load_state_dict({})


def test_load_state_dict_validates_dtype():
    store = _store(num_nodes=5, capacity=4)
    state = store.state_dict()
    state["neighbor_id"] = state["neighbor_id"].float()
    with pytest.raises(ValueError, match="int64"):
        store.load_state_dict(state)


def test_load_state_dict_validates_shape():
    store = _store(num_nodes=5, capacity=4)
    state = store.state_dict()
    state["neighbor_id"] = torch.zeros(3, 4, dtype=torch.int64)
    state["event_id"] = torch.zeros(3, 4, dtype=torch.int64)
    state["timestamp_ns"] = torch.zeros(3, 4, dtype=torch.int64)
    state["count"] = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(ValueError, match="shape mismatch"):
        store.load_state_dict(state)


def test_load_state_dict_validates_capacity_consistency():
    store = _store(num_nodes=5, capacity=4)
    state = store.state_dict()
    state["_candidate_capacity"] = 10
    with pytest.raises(ValueError, match="candidate_capacity mismatch"):
        store.load_state_dict(state)


def test_load_state_dict_validates_num_nodes_consistency():
    store = _store(num_nodes=5, capacity=4)
    state = store.state_dict()
    state["_num_nodes"] = 20
    with pytest.raises(ValueError, match="num_nodes mismatch"):
        store.load_state_dict(state)


def test_load_state_dict_direction_required_when_enabled():
    store = _store(num_nodes=5, capacity=4)
    state = store.state_dict()
    del state["direction"]
    store2 = HistoryStore(num_nodes=5, candidate_capacity=4, store_direction=True)
    with pytest.raises(ValueError, match="direction"):
        store2.load_state_dict(state)


def test_load_state_dict_direction_forbidden_when_disabled():
    store_no_dir = HistoryStore(num_nodes=5, candidate_capacity=4, store_direction=False)
    store_with_dir = _store(num_nodes=5, capacity=4)
    state = store_with_dir.state_dict()
    with pytest.raises(ValueError, match="store_direction=False"):
        store_no_dir.load_state_dict(state)


def test_load_state_dict_rejects_direction_when_not_stored():
    store = HistoryStore(num_nodes=5, candidate_capacity=4, store_direction=False)
    state = {
        "neighbor_id": torch.full((5, 4), -1, dtype=torch.int64),
        "event_id": torch.full((5, 4), -1, dtype=torch.int64),
        "timestamp_ns": torch.full((5, 4), -1, dtype=torch.int64),
        "count": torch.zeros(5, dtype=torch.long),
        "_num_nodes": 5,
        "_candidate_capacity": 4,
        "_store_direction": False,
    }
    store.load_state_dict(state)


# --------------------------------------------------------------------------- #
# Reset after load
# --------------------------------------------------------------------------- #
def test_reset_after_load():
    store = _store(num_nodes=5, capacity=4)
    store.insert(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([7]),
        timestamp_ns=torch.tensor([700], dtype=torch.int64),
        direction=torch.tensor([0], dtype=torch.int8),
    )

    state = store.state_dict()
    store2 = _store(num_nodes=5, capacity=4)
    store2.load_state_dict(state)
    store2.reset_state()

    for node in range(5):
        assert int(store2._count[node].item()) == 0
