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
    log_seconds_boundaries_to_ns,
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
# log_seconds_boundaries_to_ns
# --------------------------------------------------------------------------- #
def test_log_seconds_conversion_correct():
    """Verify log1p(seconds) -> ns conversion is correct."""
    boundaries = [math.log1p(15), math.log1p(70), math.log1p(100)]
    tau_short, tau_medium, tau_max = log_seconds_boundaries_to_ns(boundaries)

    assert tau_short == 15 * S
    assert tau_medium == 70 * S
    assert tau_max == 100 * S


def test_log_seconds_conversion_roundtrip():
    """Convert ns -> log1p(seconds) -> ns should be approximately equal."""
    boundaries = [math.log1p(15), math.log1p(70), math.log1p(100)]
    tau_short, tau_medium, tau_max = log_seconds_boundaries_to_ns(boundaries)

    assert math.log1p(tau_short / S) == pytest.approx(math.log1p(15))
    assert math.log1p(tau_medium / S) == pytest.approx(math.log1p(70))
    assert math.log1p(tau_max / S) == pytest.approx(math.log1p(100))


def test_log_seconds_wrong_count_raises():
    with pytest.raises(ValueError, match="exactly 3 values"):
        log_seconds_boundaries_to_ns([0.5, 0.9])


def test_log_seconds_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        log_seconds_boundaries_to_ns([-0.1, 0.5, 1.0])


def test_log_seconds_not_monotonic_raises():
    with pytest.raises(ValueError, match="non-decreasing"):
        log_seconds_boundaries_to_ns([1.0, 0.5, 2.0])


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
def test_tau_ordering():
    """0 <= tau_short <= tau_medium <= tau_max."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)
    assert loader.tau_short_ns == 15 * S
    assert loader.tau_medium_ns == 70 * S
    assert loader.tau_max_ns == 100 * S


def test_negative_tau_short_raises():
    with pytest.raises(ValueError, match="non-negative"):
        MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
            tau_short_ns=-1,
            tau_medium_ns=70 * S,
            tau_max_ns=100 * S,
        )


def test_tau_medium_less_than_tau_short_raises():
    with pytest.raises(ValueError, match="tau_medium_ns.*must be >="):
        MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
            tau_short_ns=70 * S,
            tau_medium_ns=15 * S,
            tau_max_ns=100 * S,
        )


def test_tau_max_less_than_tau_medium_raises():
    with pytest.raises(ValueError, match="tau_max_ns.*must be >="):
        MultiScaleNeighborLoader(
            history_store=HistoryStore(num_nodes=10, candidate_capacity=5),
            tau_short_ns=15 * S,
            tau_medium_ns=100 * S,
            tau_max_ns=70 * S,
        )


def test_negative_budget_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _loader(short_budget=-1)


# --------------------------------------------------------------------------- #
# Scale boundary semantics (using explicit ns boundaries)
# --------------------------------------------------------------------------- #
def test_scale_boundaries_explicit_ns():
    """Test with explicit ns boundaries: short=15s, medium=70s, long=100s."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=10, candidate_capacity=10),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
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
    #   long (70s < delta <= 100s): 20s (delta=90s), 10s (delta=100s)

    node = 0
    ref_time = 110 * S

    loader.history_store.insert(
        src=torch.tensor([node, node, node, node]),
        dst=torch.tensor([1, 1, 1, 1]),
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
    assert short_mask.sum().item() == 1
    short_ts = result["short_timestamp_ns"][0][short_mask]
    assert (short_ts == 100 * S).all()

    # medium: only t=50s is in range (delta=60s)
    assert medium_mask.sum().item() == 1
    medium_ts = result["medium_timestamp_ns"][0][medium_mask]
    assert (medium_ts == 50 * S).all()

    # long: t=20s (delta=90s) and t=10s (delta=100s)
    assert long_mask.sum().item() == 2
    long_ts = result["long_timestamp_ns"][0][long_mask]
    assert 20 * S in long_ts.tolist()
    assert 10 * S in long_ts.tolist()


# --------------------------------------------------------------------------- #
# Boundary value semantics
# --------------------------------------------------------------------------- #
def test_delta_exactly_tau_short_is_short():
    """delta == tau_short belongs to short (upper bound inclusive)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    node = 0
    ref_time = 115 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 1
    assert result["medium_mask"][0].sum().item() == 0
    assert result["long_mask"][0].sum().item() == 0


def test_delta_just_over_tau_short_is_medium():
    """delta > tau_short belongs to medium."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    node = 0
    ref_time = 116 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 0
    assert result["medium_mask"][0].sum().item() == 1
    assert result["long_mask"][0].sum().item() == 0


def test_delta_exactly_tau_medium_is_medium():
    """delta == tau_medium belongs to medium (upper bound inclusive)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    node = 0
    ref_time = 170 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["medium_mask"][0].sum().item() == 1
    assert result["long_mask"][0].sum().item() == 0


def test_delta_exactly_tau_max_is_long():
    """delta == tau_max belongs to long (upper bound inclusive)."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    node = 0
    ref_time = 200 * S
    inserted_event_id = 0

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([inserted_event_id]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # short: 0 < delta <= 15s   → delta=100s, not in short
    assert result["short_mask"][0].sum().item() == 0
    # medium: 15s < delta <= 70s → delta=100s, not in medium
    assert result["medium_mask"][0].sum().item() == 0
    # long: 70s < delta <= 100s → delta=100s, IS in long (upper bound inclusive)
    assert result["long_mask"][0].sum().item() == 1

    # Verify the long entry actually carries the inserted event id.
    long_event_ids = result["long_event_id"][0][result["long_mask"][0]]
    assert long_event_ids.numel() == 1
    assert int(long_event_ids.item()) == inserted_event_id


def test_delta_over_tau_max_excluded():
    """delta > tau_max is excluded from all scales."""
    loader = MultiScaleNeighborLoader(
        history_store=HistoryStore(num_nodes=5, candidate_capacity=5),
        tau_short_ns=15 * S,
        tau_medium_ns=70 * S,
        tau_max_ns=100 * S,
        short_budget=3,
        medium_budget=3,
        long_budget=3,
    )

    node = 0
    ref_time = 300 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 0
    assert result["medium_mask"][0].sum().item() == 0
    assert result["long_mask"][0].sum().item() == 0


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

    for t in [100, 110, 115, 120]:
        loader.history_store.insert(
            src=torch.tensor([node]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([t]),
            timestamp_ns=torch.tensor([t * S], dtype=torch.int64),
        )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 2
    short_ts = result["short_timestamp_ns"][0][result["short_mask"][0]]
    assert 115 * S in short_ts.tolist()
    assert 110 * S in short_ts.tolist()


# --------------------------------------------------------------------------- #
# Most recent K per scale
# --------------------------------------------------------------------------- #
def test_most_recent_within_scale():
    """Within each scale, only the K most recent events are kept.

    With ref_time=110s and tau_short=15s / tau_medium=70s / tau_max=100s:
        t=100s -> delta=10s  -> short
        t=50s  -> delta=60s  -> medium
        t=10s  -> delta=100s -> long  (upper bound inclusive)
    Each scale holds exactly one candidate, so budget limits do not trim.
    """
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
    ref_time = 110 * S

    for t, eid in [(10, 0), (50, 1), (100, 2)]:
        loader.history_store.insert(
            src=torch.tensor([node]),
            dst=torch.tensor([1]),
            event_id=torch.tensor([eid]),
            timestamp_ns=torch.tensor([t * S], dtype=torch.int64),
        )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    # short: only t=100s (delta=10s)
    assert result["short_mask"][0].sum().item() == 1
    short_eids = result["short_event_id"][0][result["short_mask"][0]]
    assert int(short_eids.item()) == 2

    # medium: only t=50s (delta=60s)
    assert result["medium_mask"][0].sum().item() == 1
    medium_eids = result["medium_event_id"][0][result["medium_mask"][0]]
    assert int(medium_eids.item()) == 1

    # long: only t=10s (delta=100s, upper bound inclusive)
    assert result["long_mask"][0].sum().item() == 1
    long_eids = result["long_event_id"][0][result["long_mask"][0]]
    assert int(long_eids.item()) == 0


# --------------------------------------------------------------------------- #
# Empty scale handling
# --------------------------------------------------------------------------- #
def test_empty_scale_has_false_mask():
    """All-False masks across scales when no history falls in any range."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 1000 * S  # delta > tau_max for any inserted history

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([4 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert not result["short_mask"][0].any()
    assert not result["medium_mask"][0].any()
    assert not result["long_mask"][0].any()


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
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 100 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([0]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    for scale in ["short", "medium", "long"]:
        assert result[f"{scale}_mask"][0].sum().item() == 0


# --------------------------------------------------------------------------- #
# global_event_index preserved
# --------------------------------------------------------------------------- #
def test_global_event_index_preserved():
    """Event IDs from insert are returned unchanged in query."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    ref_time = 200 * S

    loader.history_store.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        event_id=torch.tensor([42]),
        timestamp_ns=torch.tensor([130 * S], dtype=torch.int64),
    )

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([ref_time], dtype=torch.int64),
    )

    assert result["medium_event_id"][0][result["medium_mask"][0]].item() == 42


# --------------------------------------------------------------------------- #
# reset_state
# --------------------------------------------------------------------------- #
def test_reset_state_clears_history():
    """reset_state clears all history."""
    loader = _loader(tau_short_s=15, tau_medium_s=70, tau_max_s=100)

    node = 0
    loader.insert(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([100 * S], dtype=torch.int64),
        global_event_index=torch.tensor([0]),
    )

    loader.reset_state()

    result = loader(
        src=torch.tensor([node]),
        dst=torch.tensor([1]),
        timestamp_ns=torch.tensor([200 * S], dtype=torch.int64),
    )

    assert result["short_mask"][0].sum().item() == 0
    assert result["medium_mask"][0].sum().item() == 0
    assert result["long_mask"][0].sum().item() == 0


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
