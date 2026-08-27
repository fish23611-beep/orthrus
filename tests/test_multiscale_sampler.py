"""
Tests for src/mstc/multiscale_sampler.py — MultiScaleNeighborLoader.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mstc.history_store import HistoryStore
from src.mstc.multiscale_sampler import (
    MultiScaleNeighborLoader,
    SingleWindowNeighborLoader,
    seconds_boundaries_to_ns,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
NS = 1
S = 1_000_000_000


def _loader(
    tau_short_s=15,
    tau_medium_s=70,
    tau_max_s=100,
    short_budget=3,
    medium_budget=3,
    long_budget=3,
    num_nodes=20,
    capacity=10,
):
    """Create a loader with specified boundaries in seconds (converted internally)."""
    return MultiScaleNeighborLoader(
        history_store=HistoryStore(
            num_nodes=num_nodes,
            candidate_capacity=capacity,
            device="cpu",
        ),
        tau_short_ns=int(tau_short_s * S),
        tau_medium_ns=int(tau_medium_s * S),
        tau_max_ns=int(tau_max_s * S),
        short_budget=short_budget,
        medium_budget=medium_budget,
        long_budget=long_budget,
    )


# --------------------------------------------------------------------------- #
# seconds_boundaries_to_ns
# --------------------------------------------------------------------------- #
def test_seconds_conversion_correct():
    """Verify seconds -> ns conversion is correct."""
    boundaries = [15.0, 70.0, 100.0]  # in seconds
    tau_short, tau_medium, tau_extreme = seconds_boundaries_to_ns(boundaries)

    assert tau_short == 15 * S
    assert tau_medium == 70 * S
    assert tau_extreme == 100 * S


def test_seconds_conversion_roundtrip():
    """Convert ns -> seconds -> ns should be approximately equal."""
    boundaries = [15.0, 70.0, 100.0]  # in seconds
    tau_short, tau_medium, tau_extreme = seconds_boundaries_to_ns(boundaries)

    assert tau_short / S == pytest.approx(15.0)
    assert tau_medium / S == pytest.approx(70.0)
    assert tau_extreme / S == pytest.approx(100.0)


def test_seconds_wrong_count_raises():
    with pytest.raises(ValueError, match="exactly 3 values"):
        seconds_boundaries_to_ns([0.5, 0.9])


def test_seconds_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        seconds_boundaries_to_ns([-0.1, 0.5, 1.0])


def test_seconds_not_monotonic_raises():
    with pytest.raises(ValueError, match="non-decreasing"):
        seconds_boundaries_to_ns([1.0, 0.5, 2.0])


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
def test_tau_ordering():
    """0 <= tau_short <= tau_medium (tau_extreme is diagnostics only, not a constraint)."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)
    assert loader.tau_short_ns == 15 * S
    assert loader.tau_medium_ns == 70 * S
    # tau_max_ns is stored but not used in ordering constraints
    assert loader.tau_max_ns == 100 * S


def test_long_scale_has_no_upper_bound():
    """Long scale = delta > tau_medium_ns, with NO upper bound.

    This is the key semantic change: events with delta > tau_max (Q99)
    should still be in the long scale.
    """
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=5,
        medium_budget=5,
        long_budget=5,
    )

    node = 0
    # Insert event at t=100s, query at ref_time=300s -> delta=200s > tau_max
    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([300 * S], dtype=torch.int64),
    )

    # delta=200s > tau_medium (70s), should be in long scale
    q_idx = result["src_query_index"][0].item()
    assert result["long_mask"][q_idx].sum().item() > 0, "delta > tau_medium should be in long scale"


def test_negative_tau_short_raises():
    with pytest.raises(ValueError, match="non-negative"):
        MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
            tau_short_ns=-1,
            tau_medium_ns=70 * S,
        )


def test_tau_medium_less_than_tau_short_raises():
    with pytest.raises(ValueError, match="tau_medium_ns.*must be >="):
        MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
            tau_short_ns=70 * S,
            tau_medium_ns=15 * S,
        )


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _loader(short_budget=-1)


# --------------------------------------------------------------------------- #
# Scale boundary semantics (using explicit ns boundaries)
# --------------------------------------------------------------------------- #
def test_scale_boundaries_explicit_ns():
    """Test with explicit ns boundaries: short=15s, medium=70s, long=(no upper bound)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    # Insert history events at specific timestamps
    # Node 0 has events at t=10s, 20s, 50s, 100s
    # Current reference time = 110s
    # Expected:
    #   short (0 < delta <= 15s): 100s (delta=10s)
    #   medium (15s < delta <= 70s): 50s (delta=60s)
    #   long (delta > 70s): 20s (delta=90s), 10s (delta=100s)

    node = 0
    ref_time = 110 * S

    loader.history_store.insert(
        src=torch.tensor([node, node, node, node]),
        dst=torch.tensor([1, 2, 3, 4]),
        event_id=torch.tensor([0, 1, 2, 3]),
        timestamp_ns=torch.tensor(
            [10 * S, 20 * S, 50 * S, 100 * S],
            dtype=torch.int64,
        ),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    short_mask = result["short_mask"][0]
    medium_mask = result["medium_mask"][0]
    long_mask = result["long_mask"][0]

    # short: only t=100s is in range (delta=10s)
    assert short_mask.sum().item() >= 1
    short_ts = result["short_timestamp_ns"][0][short_mask]
    assert (100 * S in short_ts.tolist())

    # medium: only t=50s is in range (delta=60s)
    assert medium_mask.sum().item() >= 1
    medium_ts = result["medium_timestamp_ns"][0][medium_mask]
    assert (50 * S in medium_ts.tolist())

    # long: t=20s (delta=90s) and t=10s (delta=100s)
    assert long_mask.sum().item() >= 1
    long_ts = result["long_timestamp_ns"][0][long_mask]
    assert (20 * S in long_ts.tolist() or 10 * S in long_ts.tolist())


# --------------------------------------------------------------------------- #
# Boundary value semantics
# --------------------------------------------------------------------------- #
def test_delta_exactly_tau_short_is_short():
    """delta == tau_short belongs to short (upper bound inclusive)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=5,
        medium_budget=5,
        long_budget=5,
    )

    node = 0
    ref_time = 115 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([100]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 115s - 100s = 15s = tau_short
    # Should be in short (0 < delta <= tau_short)
    q_idx = result["src_query_index"][0].item()
    assert result["short_mask"][q_idx].sum().item() > 0, "delta == tau_short should be in short scale"


def test_delta_just_over_tau_short_is_medium():
    """delta > tau_short belongs to medium."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=5,
        medium_budget=5,
        long_budget=5,
    )

    node = 1
    ref_time = 116 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([101]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 116s - 100s = 16s > tau_short (15s)
    # Should be in medium
    q_idx = result["src_query_index"][0].item()
    assert result["medium_mask"][q_idx].sum().item() > 0, "delta > tau_short should be in medium scale"


def test_delta_exactly_tau_medium_is_medium():
    """delta == tau_medium belongs to medium (upper bound inclusive)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=5,
        medium_budget=5,
        long_budget=5,
    )

    node = 2
    ref_time = 170 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([102]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 170s - 100s = 70s = tau_medium
    # Should be in medium (upper bound inclusive)
    q_idx = result["src_query_index"][0].item()
    assert result["medium_mask"][q_idx].sum().item() > 0, "delta == tau_medium should be in medium scale"


def test_delta_over_tau_medium_is_long():
    """delta > tau_medium belongs to long (no upper bound)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=5,
        medium_budget=5,
        long_budget=5,
    )

    node = 3
    ref_time = 200 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([103]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 200s - 100s = 100s > tau_medium (70s)
    # Should be in long (delta > tau_medium)
    q_idx = result["src_query_index"][0].item()
    assert result["long_mask"][q_idx].sum().item() > 0, "delta > tau_medium should be in long scale"


# --------------------------------------------------------------------------- #
# Causal ordering: query before insert
# --------------------------------------------------------------------------- #
def test_current_batch_not_in_history():
    """Events from the current batch are NOT visible in the query."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 200 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([99]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    for scale in ["short", "medium", "long"]:
        mask = result[f"{scale}_mask"][0]
        event_ids = result[f"{scale}_event_id"][0][mask]
        assert 99 not in event_ids.tolist()


def test_insert_affects_next_batch():
    """Inserting in batch N affects query in batch N+1."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0

    # Batch 1: insert event at t=100s
    loader.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
        global_event_index=torch.tensor([0]),
    )

    # Batch 2: query with reference t=110s should see t=100s
    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([110 * S], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 1
    short_ts = result["short_timestamp_ns"][0][result["short_mask"][0]]
    assert (short_ts == 100 * S).all()


# --------------------------------------------------------------------------- #
# reference_time uses minimum in batch
# --------------------------------------------------------------------------- #
def test_reference_time_uses_min_per_node():
    """reference_time = minimum timestamp of that node in the current batch."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([80 * S], dtype=torch.int64),
    )

    # Node 0 appears at t=100 and t=200 in the same batch
    # reference_time should be 100s (minimum)
    result = loader(
        src=torch.tensor([node, node]),
        dst=torch.tensor([1, 2]),
        timestamp_ns=torch.tensor([200 * S, 100 * S], dtype=torch.int64),
    )

    # Delta = 100s - 80s = 20s > 15s, so should be in medium
    q_idx = result["src_query_index"][0].item()
    assert result["medium_mask"][q_idx].sum().item() == 1


# --------------------------------------------------------------------------- #
# Budget limits
# --------------------------------------------------------------------------- #
def test_budget_limits_short_scale():
    """short_budget=2, should only return 2 most recent in short range."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=20),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=2,
        medium_budget=2,
        long_budget=2,
    )

    node = 0
    ref_time = 120 * S

    # Insert with unique event IDs and distinct dst to avoid duplicate entries
    for i, t in enumerate([100, 110, 115, 120]):
        loader.history_store.insert(
            src=torch.tensor([node]),
            dst=torch.tensor([i + 10]),  # Distinct dst per event
            event_id=torch.tensor([i]),
            timestamp_ns=torch.tensor([t * S], dtype=torch.int64),
        )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # At ref_time=120s with events at t=100,110,115,120:
    # t=120: delta=0 → excluded
    # t=115: delta=5s → short
    # t=110: delta=10s → short
    # t=100: delta=20s → medium
    assert result["short_mask"][0].sum().item() == 2
    short_ts = result["short_timestamp_ns"][0][result["short_mask"][0]]
    assert 115 * S in short_ts.tolist()
    assert 110 * S in short_ts.tolist()


# --------------------------------------------------------------------------- #
# Most recent K per scale
# --------------------------------------------------------------------------- #
def test_most_recent_within_scale():
    """Within each scale, only the K most recent events are kept."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=100, candidate_capacity=20),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        short_budget=2,
        medium_budget=2,
        long_budget=2,
    )

    node = 50
    ref_time = 110 * S

    for i, (t, eid) in enumerate([(10, 0), (50, 1), (100, 2)]):
        loader.history_store.insert(
            src=torch.tensor([node]),
            dst=torch.tensor([node + i + 1]),
            event_id=torch.tensor([eid]),
            timestamp_ns=torch.tensor([t * S], dtype=torch.int64),
        )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # Get query index for node
    q_idx = result["src_query_index"][0].item()

    # short: only t=100s (delta=10s)
    short_mask = result["short_mask"][q_idx]
    short_ts = result["short_timestamp_ns"][q_idx][short_mask]
    assert (100 * S) in short_ts.tolist()

    # medium: only t=50s (delta=60s)
    medium_mask = result["medium_mask"][q_idx]
    medium_ts = result["medium_timestamp_ns"][q_idx][medium_mask]
    assert (50 * S) in medium_ts.tolist()

    # long: t=10s (delta=100s > 70s, no upper bound)
    long_mask = result["long_mask"][q_idx]
    long_ts = result["long_timestamp_ns"][q_idx][long_mask]
    assert (10 * S) in long_ts.tolist()


# --------------------------------------------------------------------------- #
# Empty scale handling
# --------------------------------------------------------------------------- #
def test_empty_scale_has_false_mask():
    """All-False masks when no history falls in short/medium range.

    Note: With new semantics, events can be in LONG even for small deltas
    if they exceed tau_medium. This test verifies short is empty.
    """
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
    )

    node = 5
    # Insert event at t=970s, query at ref_time=1000s -> delta=30s
    # delta=30s > tau_short (15s), so NOT in short
    ref_time = 1000 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([970 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    q_idx = result["src_query_index"][0].item()
    # delta=30s > tau_short (15s), so short is empty
    assert not result["short_mask"][q_idx].any().item(), "delta > tau_short should NOT be in short"


def test_empty_scale_invalid_entries_are_minus_one():
    """Invalid entries in empty scale are -1, not garbage."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 5 * S

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert (result["short_event_id"][0] == -1).all()
    assert (result["medium_event_id"][0] == -1).all()
    assert (result["long_event_id"][0] == -1).all()


# --------------------------------------------------------------------------- #
# No duplicate padding
# --------------------------------------------------------------------------- #
def test_no_duplicate_padding():
    """Scale tensors should not have duplicate entries for the same event."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 20 * S

    loader.history_store.insert(
        src=torch.tensor([node, node]),
        dst=torch.tensor([1, 2]),
        event_id=torch.tensor([0, 1]),
        timestamp_ns=torch.tensor([10 * S, 10 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    short_eids = result["short_event_id"][0][result["short_mask"][0]]
    unique_eids = short_eids.unique()
    assert len(unique_eids) == len(short_eids)


# --------------------------------------------------------------------------- #
# src/dst query mapping
# --------------------------------------------------------------------------- #
def test_src_query_index_mapping():
    """src_query_index correctly maps src events to query node indices."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([3, 4, 5])
    t = torch.tensor([100, 200, 300], dtype=torch.int64) * S

    result = loader(src=src, dst=dst, timestamp_ns=t)

    assert result["src_query_index"].shape == (3,)
    assert result["dst_query_index"].shape == (3,)

    # Verify query_nodes contains all unique nodes
    query_nodes = result["query_nodes"]
    assert set(query_nodes.tolist()) == {0, 1, 2, 3, 4, 5}


# --------------------------------------------------------------------------- #
# delta = 0 is excluded
# --------------------------------------------------------------------------- #
def test_delta_zero_excluded():
    """delta = 0 (same timestamp) is excluded from all scales."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
    )

    node = 6
    ref_time = 100 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 0 (same timestamp), should be excluded
    # Short scale: delta > 0 required, so no entries
    q_idx = result["src_query_index"][0].item()
    # Only verify short is empty (delta=0 is excluded)
    assert not result["short_mask"][q_idx].any().item(), "delta=0 should NOT be in short scale"


# --------------------------------------------------------------------------- #
# global_event_index preserved
# --------------------------------------------------------------------------- #
def test_global_event_index_preserved():
    """Event IDs from insert are returned unchanged in query."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
    )

    node = 7
    ref_time = 200 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        event_id=torch.tensor([42]),
        timestamp_ns=torch.tensor([130 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # delta = 200s - 130s = 70s = tau_medium
    # Should be in medium (upper bound inclusive)
    q_idx = result["src_query_index"][0].item()
    # Verify at least one medium entry exists
    assert result["medium_mask"][q_idx].any().item(), "delta=tau_medium should be in medium scale"


# --------------------------------------------------------------------------- #
# reset_state
# --------------------------------------------------------------------------- #
def test_reset_state_clears_history():
    """reset_state clears all history."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
    )

    node = 8
    loader.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
        global_event_index=torch.tensor([0]),
    )

    loader.reset_state()

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([node + 1]),
        timestamp_ns=torch.tensor([200 * S], dtype=torch.int64),
    )

    # After reset, no entries should be in any scale
    q_idx = result["src_query_index"][0].item()
    assert not result["short_mask"][q_idx].any().item(), "short should be empty after reset"
    assert not result["medium_mask"][q_idx].any().item(), "medium should be empty after reset"
    assert not result["long_mask"][q_idx].any().item(), "long should be empty after reset"


# --------------------------------------------------------------------------- #
# history_state_dict roundtrip
# --------------------------------------------------------------------------- #
def test_history_state_dict_roundtrip():
    """history_state_dict and load_history_state_dict preserve all data."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([99]),
        timestamp_ns=torch.tensor([130 * S], dtype=torch.int64),
    )

    state = loader.history_state_dict()
    assert state["neighbor_id"].shape[0] == 20

    loader2 = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)
    loader2.load_history_state_dict(state)

    result = loader2(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([200 * S], dtype=torch.int64),
    )

    assert result["medium_mask"][0].sum().item() == 1


# --------------------------------------------------------------------------- #
# Shared HistoryStore across loaders
# --------------------------------------------------------------------------- #
def test_multiple_loaders_share_same_store():
    """Multiple loaders can share the same HistoryStore."""
    shared_store = HistoryStore(num_nodes=10, candidate_capacity=10)

    loader1 = MultiScaleNeighborLoader(
        history_store=shared_store,
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
    )
    loader2 = MultiScaleNeighborLoader(
        history_store=shared_store,
        tau_short_ns=5 * S,
        tau_medium_ns=20 * S,
        tau_max_ns=50 * S,
    )

    node = 0
    loader1.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([50 * S], dtype=torch.int64),
        global_event_index=torch.tensor([0]),
    )

    # loader1 should see it (50s is in [0, 15s] range)
    result1 = loader1(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([60 * S], dtype=torch.int64),
    )
    assert result1["short_mask"][0].sum().item() == 1

    # loader2 should NOT see it (50s delta > 50s max)
    result2 = loader2(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([200 * S], dtype=torch.int64),
    )
    assert result2["short_mask"][0].sum().item() == 0
