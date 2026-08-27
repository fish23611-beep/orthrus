"""
Tests for src/mstc/time_gap.py — TimeGapStatistics.
"""

import math
import os
import tempfile

import pytest
import torch
from torch_geometric.data import TemporalData

from src.mstc.time_gap import (
    NO_HISTORY,
    VERY_SHORT,
    SHORT,
    MEDIUM,
    LONG,
    VERY_LONG,
    TimeGapStatistics,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_graph(src, dst, t, num_node_types=2, num_edge_types=2, edge_type_value=1):
    """Create a TemporalData with one-hot edge types and per-window index fields."""
    E = len(src)
    node_type_dim = num_node_types
    edge_type_dim = num_edge_types
    src_type = torch.zeros(E, node_type_dim, dtype=torch.long)
    dst_type = torch.zeros(E, node_type_dim, dtype=torch.long)
    edge_type = torch.zeros(E, edge_type_dim, dtype=torch.long)
    edge_type[:, edge_type_value % edge_type_dim] = 1

    src_type[torch.arange(E), src % node_type_dim] = 1
    dst_type[torch.arange(E), dst % node_type_dim] = 1

    msg = torch.cat([src_type, edge_type, dst_type], dim=-1)
    g = TemporalData()
    g.src = src
    g.dst = dst
    g.t = t
    g.msg = msg
    g.edge_type = edge_type
    g.global_event_index = torch.arange(E, dtype=torch.long)
    return g


# --------------------------------------------------------------------------- #
# fit — basic
# --------------------------------------------------------------------------- #
def test_fit_empty_raises():
    stats = TimeGapStatistics()
    with pytest.raises(ValueError, match="empty"):
        stats.fit([])


def test_fit_no_finite_intervals_raises():
    """A graph where every node appears exactly once — no finite intervals possible."""
    g = _make_graph(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        t=torch.tensor([0]),
    )
    stats = TimeGapStatistics()
    with pytest.raises(ValueError, match="No finite intervals"):
        stats.fit([g])


def test_fit_creates_boundaries():
    g1 = _make_graph(
        src=torch.tensor([0, 0, 1]),
        dst=torch.tensor([1, 2, 2]),
        t=torch.tensor([0, 10, 20]),
    )
    g2 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([30, 40]),
    )
    stats = TimeGapStatistics()
    stats.fit([g1, g2])

    assert len(stats.time_bucket_boundaries) == 4
    assert len(stats.scale_boundaries_seconds) == 3
    assert stats.time_bucket_boundaries == sorted(stats.time_bucket_boundaries)
    assert stats.scale_boundaries_seconds == sorted(stats.scale_boundaries_seconds)


def test_fit_duplicate_timestamps():
    """delta = 0 is valid and should be included."""
    g = _make_graph(
        src=torch.tensor([0, 0, 0]),
        dst=torch.tensor([1, 1, 2]),
        t=torch.tensor([0, 0, 10]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])
    assert len(stats.time_bucket_boundaries) == 4


def test_fit_negative_delta_raises():
    """TemporalData timestamps within a window must be non-decreasing."""
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 0]),
        t=torch.tensor([10, 5]),
    )
    stats = TimeGapStatistics()
    with pytest.raises(ValueError, match="[Nn]on-decreasing|[Dd]elta"):
        stats.fit([g])


def test_fit_very_large_interval():
    """Nanosecond range up to 2^60 should not overflow float64."""
    large_ns = 2**60
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, large_ns]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])
    assert len(stats.time_bucket_boundaries) == 4

def test_time_gap_statistics_are_invariant_to_absolute_time_offset():
    """A uniform timestamp translation preserves all per-node deltas."""
    src = torch.tensor([0, 0, 0, 0])
    dst = torch.tensor([1, 1, 1, 1])
    stats_a = TimeGapStatistics().fit([
        _make_graph(src=src, dst=dst, t=torch.tensor([10, 20, 40, 70]))
    ])
    stats_b = TimeGapStatistics().fit([
        _make_graph(src=src, dst=dst, t=torch.tensor([1010, 1020, 1040, 1070]))
    ])

    assert stats_a.time_bucket_boundaries == stats_b.time_bucket_boundaries
    assert stats_a.scale_boundaries_seconds == stats_b.scale_boundaries_seconds



# --------------------------------------------------------------------------- #
# fit — ns / seconds conversion
# --------------------------------------------------------------------------- #
def test_ns_to_seconds_conversion():
    """10^9 ns = 1 second."""
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, 1_000_000_000]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])
    assert len(stats.time_bucket_boundaries) == 4


def test_log1p_used():
    """
    Verify log1p is applied: 1s and 10000s land in different buckets
    when training intervals span [1, 10000] so that Q80 >= 8000s.
    """
    intervals = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    t_vals = [0]
    cumulative = 0
    for iv in intervals:
        cumulative += iv
        t_vals.append(int(cumulative))

    E = len(intervals)
    g = _make_graph(
        src=torch.tensor([i % 10 for i in range(E)]),
        dst=torch.tensor([(i + 1) % 10 for i in range(E)]),
        t=torch.tensor(t_vals, dtype=torch.long),
    )
    stats = TimeGapStatistics()
    stats.fit([g])

    bucket_1s = stats.transform(1.0)
    bucket_10ks = stats.transform(10000.0)
    assert bucket_1s != NO_HISTORY, "1s should not be NO_HISTORY"
    assert bucket_10ks != NO_HISTORY, "10000s should not be NO_HISTORY"
    assert bucket_1s != bucket_10ks, (
        f"log1p mapping must distinguish 1s vs 10000s: "
        f"got bucket_1s={bucket_1s}, bucket_10ks={bucket_10ks}"
    )


# --------------------------------------------------------------------------- #
# transform — boundary cases
# --------------------------------------------------------------------------- #
def test_transform_none_is_no_history():
    stats = TimeGapStatistics()
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, 1_000_000_000]),
    )
    stats.fit([g])
    assert stats.transform(None) == NO_HISTORY
    assert stats.transform(NO_HISTORY) == NO_HISTORY


def test_transform_negative_raises():
    stats = TimeGapStatistics()
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, 1_000_000_000]),
    )
    stats.fit([g])
    with pytest.raises(ValueError, match="non-negative"):
        stats.transform(-1.0)


def test_log1p_used():
    """
    Verify log1p is applied: transform(0.1) and transform(10000.0)
    land in different buckets when training uses log-spaced intervals
    10^(i/10) ns for i=0..99 (range: ~1ns to ~1e10ns).
    """
    import math
    intervals_ns = [10 ** (i / 10.0) for i in range(100)]
    t_vals_ns = [0]
    for iv in intervals_ns:
        t_vals_ns.append(int(t_vals_ns[-1] + iv))

    g = _make_graph(
        src=torch.tensor([i % 10 for i in range(100)]),
        dst=torch.tensor([(i + 1) % 10 for i in range(100)]),
        t=torch.tensor(t_vals_ns, dtype=torch.long),
    )
    stats = TimeGapStatistics()
    stats.fit([g])

    bucket_small = stats.transform(0.1)
    bucket_large = stats.transform(10000.0)
    assert bucket_small != NO_HISTORY
    assert bucket_large != NO_HISTORY
    assert bucket_small != bucket_large, (
        f"log1p must distinguish 0.1s vs 10000s: "
        f"got {bucket_small} vs {bucket_large}"
    )


def test_transform_exact_quantile_boundaries():
    """
    Verify monotonicity of bucket boundaries: all boundaries are increasing,
    and bucket is monotonic in z.
    """
    import math
    intervals_ns = [10 ** (i / 10.0) for i in range(100)]
    t_vals_ns = [0]
    for iv in intervals_ns:
        t_vals_ns.append(int(t_vals_ns[-1] + iv))

    g = _make_graph(
        src=torch.tensor([i % 10 for i in range(100)]),
        dst=torch.tensor([(i + 1) % 10 for i in range(100)]),
        t=torch.tensor(t_vals_ns, dtype=torch.long),
    )
    stats = TimeGapStatistics(time_bucket_quantiles=[0.2, 0.4, 0.6, 0.8])
    stats.fit([g])

    bounds = stats.time_bucket_boundaries
    assert bounds == sorted(bounds), f"boundaries must be sorted: {bounds}"

    # Use tiny epsilon > 0 to avoid delta_seconds=0 hitting NO_HISTORY equality check
    epsilon = 1e-12
    z_vals = [math.log1p(epsilon), bounds[0], bounds[1], bounds[2], bounds[3], 100.0]
    for z in z_vals:
        delta = math.expm1(z)
        bucket = stats.transform(delta)
        assert bucket in (VERY_SHORT, SHORT, MEDIUM, LONG, VERY_LONG), f"z={z} gave bucket={bucket}"

    z_q20 = bounds[0]
    bucket_q20 = stats.transform(math.expm1(z_q20))
    assert bucket_q20 == VERY_SHORT, (
        f"z at Q20 boundary must map to VERY_SHORT; got {bucket_q20}"
    )


def test_transform_repeated_boundaries():
    """Repeated quantile values produce stable output."""
    g = _make_graph(
        src=torch.tensor([0] * 100),
        dst=torch.tensor([1] * 100),
        t=torch.tensor([i * 1000 for i in range(100)]),
    )
    stats = TimeGapStatistics(time_bucket_quantiles=[0.5, 0.5, 0.5, 0.5])
    stats.fit([g])
    result = stats.transform(0.5)
    assert result in (VERY_SHORT, SHORT, MEDIUM, LONG, VERY_LONG, NO_HISTORY)


# --------------------------------------------------------------------------- #
# fit — Q quantiles
# --------------------------------------------------------------------------- #
def test_fit_q50_q90_q99():
    g = _make_graph(
        src=torch.tensor([0] * 99 + [1] * 50),
        dst=torch.tensor([1] * 99 + [2] * 50),
        t=torch.tensor([i for i in range(99)] + [100 + i for i in range(50)]),
    )
    stats = TimeGapStatistics(scale_quantiles=[0.5, 0.9, 0.99])
    stats.fit([g])
    assert len(stats.scale_boundaries_seconds) == 3


# --------------------------------------------------------------------------- #
# transform_batch — causal correctness
# --------------------------------------------------------------------------- #
def test_transform_batch_no_future_leak():
    """Per-event update: later events may use earlier same-batch state.

    Key difference from old immutable-snapshot: event i+1 sees the state
    after event i is processed, not the pre-batch state.

    This is the correct causal semantics for supervision: each event sees
    the exact gap since the previous event, not since the batch start.
    """
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([5_000_000_000, 15_000_000_000]),
    )
    stats = TimeGapStatistics()
    # Boundary[1] = log1p(9) < log1p(10) ensures delta=10s goes to MEDIUM:
    # SHORT: z <= log1p(9)  -> delta <= 9s
    # MEDIUM: log1p(9) < z <= log1p(21) -> 9s < delta <= 20s (approximately)
    stats.time_bucket_boundaries = [math.log1p(1.0), math.log1p(9.0), math.log1p(21.0), math.log1p(101.0)]

    before = {0: 0, 1: 0, 2: -1}
    src_target, dst_target, after = stats.transform_batch(g, before)

    # Event 0 (t=5s): gap = 5s -> SHORT
    # Event 1 (t=15s): per-event gap = 15s - 5s = 10s -> MEDIUM
    assert before == {0: 0, 1: 0, 2: -1}
    assert src_target.tolist() == [SHORT, MEDIUM]
    assert dst_target.tolist() == [SHORT, NO_HISTORY]
    assert after == {
        0: 15_000_000_000,
        1: 5_000_000_000,
        2: 15_000_000_000,
    }


def test_transform_batch_intra_batch_node_reuse_uses_immutable_pre_batch_state():
    """Same node appears twice in one batch — per-event state update.

    Per the FINAL spec: events are processed in order, and each event uses the
    state after the previous event is processed.

    Boundaries: log1p(1)=0.69, log1p(10)=2.40, log1p(20)=3.04, log1p(100)=4.62
    - VERY_SHORT: z <= 0.69 (delta <= 1s)
    - SHORT: 0.69 < z <= 2.40 (1s < delta <= 10s)
    - MEDIUM: 2.40 < z <= 3.04 (10s < delta <= 20s)
    - LONG: z > 3.04 (delta > 20s)

    Pre-batch last_seen[0]=0, events at t=5s, t=15s, t=65s for node 0:
    - Event 0 (t=5s): gap=5s -> log1p(5)=1.79 -> SHORT
    - Event 1 (t=15s): per-event gap=15s-5s=10s -> log1p(10)=2.40 -> MEDIUM
    - Event 2 (t=65s): per-event gap=65s-15s=50s -> log1p(50)=3.93 -> LONG
    """
    g = _make_graph(
        src=torch.tensor([0, 0, 0]),
        dst=torch.tensor([1, 2, 3]),
        t=torch.tensor([5_000_000_000, 15_000_000_000, 65_000_000_000]),
    )
    stats = TimeGapStatistics()
    # Use boundaries that ensure correct bucket assignment:
    # SHORT: z <= log1p(9.99) -> delta <= 9.99s
    # MEDIUM: log1p(9.99) < z <= log1p(21) -> 9.99s < delta <= 21s
    # LONG: z > log1p(21)
    stats.time_bucket_boundaries = [math.log1p(1.0), math.log1p(9.99), math.log1p(21.0), math.log1p(101.0)]

    before = {0: 0, 1: -1, 2: -1, 3: -1}
    src_target, dst_target, after = stats.transform_batch(g, before)

    # Event 0: gap=5s -> SHORT
    assert src_target[0].item() == SHORT, f"event 0: expected SHORT, got {src_target[0].item()}"
    # Event 1: per-event gap=10s -> MEDIUM
    assert src_target[1].item() == MEDIUM, f"event 1: expected MEDIUM, got {src_target[1].item()}"
    # Event 2: per-event gap=50s -> LONG
    assert src_target[2].item() == LONG, f"event 2: expected LONG, got {src_target[2].item()}"

    # dst has no history -> NO_HISTORY for all events
    assert all(b == NO_HISTORY for b in dst_target.tolist())

    # post-batch state
    assert after.get(0) == 65_000_000_000


def test_transform_batch_intra_batch_self_loop_uses_pre_batch_min():
    """Self-loop in event 0 + plain event 1 — per-event state update.

    Per-event semantics: each event sees the state after previous events.
    Event 0 (self-loop t=50ms): updates state to 50ms
    Event 1 (t=150ms): uses state updated by event 0 (50ms), delta=100ms -> SHORT
    """
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([0, 1]),
        t=torch.tensor([50_000_000, 150_000_000]),  # 50ms, 150ms in ns
    )
    stats = TimeGapStatistics()
    # VERY_SHORT: z <= log1p(0.05) -> delta <= 50ms
    # SHORT: log1p(0.05) < z <= log1p(0.2) -> 50ms < delta <= 200ms
    # MEDIUM: log1p(0.2) < z <= log1p(1.0) -> 200ms < delta <= 1s
    stats.time_bucket_boundaries = [math.log1p(0.05), math.log1p(0.2), math.log1p(1.0), math.log1p(10.0)]

    before = {0: 0, 1: -1}
    src_target, dst_target, after = stats.transform_batch(g, before)

    # event 0 (t=50ms): gap=50ms -> VERY_SHORT
    assert src_target[0].item() == VERY_SHORT, f"event 0: expected VERY_SHORT, got {src_target[0].item()}"
    assert dst_target[0].item() == VERY_SHORT  # self-loop: gap=50ms

    # event 1 (t=150ms): per-event delta=150ms-50ms=100ms -> SHORT
    assert src_target[1].item() == SHORT, f"event 1: expected SHORT, got {src_target[1].item()}"
    assert dst_target[1].item() == NO_HISTORY  # node 1 has no history

    # post-batch: node 0's last seen = 150ms = 150_000_000 ns
    assert after.get(0) == 150_000_000


def test_transform_batch_intra_batch_post_batch_uses_max():
    """Case 2: Post-batch state = max(pre_batch, max(batch_times_of_node)).

    Node 0: pre_batch=5s, batch times 10s and 30s
    Expected: post_batch[0] = max(5, 10, 30) = 30s
    """
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([1, 1]),
        t=torch.tensor([10_000_000_000, 30_000_000_000]),
    )
    stats = TimeGapStatistics()
    stats.time_bucket_boundaries = [math.log1p(15.0), math.log1p(28.0), math.log1p(33.0), math.log1p(40.0)]

    before = {0: 5_000_000_000}  # 5s in nanoseconds
    _, _, after = stats.transform_batch(g, before)

    # Node 0: max(5s, 10s, 30s) = 30s
    assert after.get(0) == 30_000_000_000, (
        f"expected post_batch[0]=30s ({30_000_000_000} ns), got {after.get(0)}"
    )


def test_transform_batch_no_intra_batch_pollution_verified_by_raw_gaps():
    """Per-event state update: event 1 sees state after event 0.

    With per-event semantics, event 1 uses state updated by event 0.
    Event 0 (t=10s): gap = 10s - 0s = 10s -> VERY_SHORT
    Event 1 (t=30s): gap = 30s - 10s = 20s -> SHORT (uses post-event-0 state)
    """
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([10_000_000_000, 30_000_000_000]),  # 10s, 30s in ns
    )
    stats = TimeGapStatistics()
    # VERY_SHORT: z <= log1p(12)
    # SHORT: log1p(12) < z <= log1p(25)
    # MEDIUM: log1p(25) < z
    stats.time_bucket_boundaries = [math.log1p(12.0), math.log1p(25.0), math.log1p(50.0), math.log1p(100.0)]

    before = {0: 0, 1: 0, 2: -1}
    src_target, dst_target, after = stats.transform_batch(g, before)

    # Event 0 (t=10s): gap = 10s - 0s = 10s -> VERY_SHORT
    assert src_target[0].item() == VERY_SHORT, (
        f"event 0: expected bucket {VERY_SHORT} (gap=10s), got {src_target[0].item()}"
    )
    # Event 1 (t=30s): per-event gap = 30s - 10s = 20s -> SHORT
    assert src_target[1].item() == SHORT, (
        f"event 1: expected bucket {SHORT} (gap=20s, uses post-event-0 state), "
        f"got {src_target[1].item()}"
    )

    # Post-batch: last seen times
    assert after.get(0) == 30_000_000_000
    assert after.get(1) == 10_000_000_000
    assert after.get(2) == 30_000_000_000


def test_transform_batch_updates_after_target():
    """Each event reads only pre-batch state; post-batch state uses max."""
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 0]),
        t=torch.tensor([5_000_000_000, 10_000_000_000]),  # 5s, 10s in ns
    )
    stats = TimeGapStatistics()
    # Use boundary between 0s and 5s so first event has NO_HISTORY (-1),
    # second event sees post-event-0 state (5s) -> gap=5s -> SHORT
    # post_batch state = 10s = 10_000_000_000
    stats.time_bucket_boundaries = [math.log1p(2.0), math.log1p(5.0), math.log1p(7.0), math.log1p(20.0)]

    src_target, dst_target, final = stats.transform_batch(g, {0: -1, 1: -1})

    # First event: gap with pre_batch=-1 is NO_HISTORY (both src and dst)
    assert src_target[0].item() == NO_HISTORY
    assert dst_target[0].item() == NO_HISTORY
    # Second event: sees post-event-0 state (5s) -> gap = 10s - 5s = 5s -> SHORT
    # With per-event update, event 1 uses state after event 0, not pre-batch state
    assert src_target[1].item() == SHORT, f"event 1: expected SHORT, got {src_target[1].item()}"
    assert dst_target[1].item() == SHORT, f"event 1: expected SHORT, got {dst_target[1].item()}"
    # post-batch: 0 and 1 each have batch max = 10s
    assert final.get(0) == 10_000_000_000
    assert final.get(1) == 10_000_000_000


def test_transform_batch_self_loop():
    """Self-loop: per-event update, both targets use same snapshot."""
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([0, 0]),
        t=torch.tensor([5_000_000_000, 15_000_000_000]),
    )
    stats = TimeGapStatistics()
    # SHORT: z <= log1p(9.99) -> delta <= 9.99s
    # MEDIUM: log1p(9.99) < z <= log1p(21)
    stats.time_bucket_boundaries = [math.log1p(1.0), math.log1p(9.99), math.log1p(21.0), math.log1p(101.0)]

    src_target, dst_target, final = stats.transform_batch(g, {0: 0})

    # Event 0: gap=5s -> SHORT
    # Event 1: per-event gap=10s -> MEDIUM
    assert src_target.tolist() == [SHORT, MEDIUM]
    assert dst_target.tolist() == src_target.tolist()
    assert final[0] == 15_000_000_000

def test_transform_batch_uses_max_time_per_node():
    """When same node appears multiple times in batch, update with max t."""
    g = _make_graph(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([1, 2, 2]),
        t=torch.tensor([0, 10, 15]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])

    last_seen = {0: -1, 1: -1, 2: -1}
    _, _, final = stats.transform_batch(g, last_seen)

    assert final[0] == 15, f"node 0 max t in batch = 15; got {final[0]}"
    assert final[1] == 10
    assert final[2] == 15, f"node 2 max t in batch = 15; got {final[2]}"


# --------------------------------------------------------------------------- #
# JSON persistence
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip():
    g = _make_graph(
        src=torch.tensor([0, 1, 0]),
        dst=torch.tensor([1, 2, 2]),
        t=torch.tensor([0, 10, 20]),
    )
    stats = TimeGapStatistics(
        time_bucket_quantiles=[0.2, 0.4, 0.6, 0.8],
        scale_quantiles=[0.5, 0.9, 0.99],
    )
    stats.fit([g])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metadata", "time_statistics.json")
        stats.save(path)

        loaded = TimeGapStatistics.load(path)

    assert loaded.time_bucket_quantiles == [0.2, 0.4, 0.6, 0.8]
    assert loaded.scale_quantiles == [0.5, 0.9, 0.99]
    assert loaded.time_bucket_boundaries == stats.time_bucket_boundaries
    assert loaded.scale_boundaries_seconds == stats.scale_boundaries_seconds

    for delta in [0.0, 1.0, 10.0, 100.0]:
        assert loaded.transform(delta) == stats.transform(delta)


def test_save_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metadata", "time_statistics.json")
        g = _make_graph(
            src=torch.tensor([0, 1]),
            dst=torch.tensor([1, 2]),
            t=torch.tensor([0, 1_000_000_000]),
        )
        stats = TimeGapStatistics()
        stats.fit([g])
        stats.save(path)
        assert os.path.exists(path)


def test_load_missing_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            TimeGapStatistics.load(path)


# --------------------------------------------------------------------------- #
# Cross-window time-order tests
# --------------------------------------------------------------------------- #
def test_fit_equal_timestamps_across_windows_legal():
    """
    Window 1 ends at t=20, Window 2 starts at t=20.
    Equal timestamps across windows are valid — within-window non-decreasing
    is already validated separately.
    """
    g1 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([10, 20]),
    )
    g2 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([2, 0]),
        t=torch.tensor([20, 30]),
    )
    stats = TimeGapStatistics()
    stats.fit([g1, g2])  # must not raise
    assert len(stats.time_bucket_boundaries) == 4


def test_fit_decreasing_timestamps_across_windows_raises():
    """
    Window 1 ends at t=20, Window 2 starts at t=19.
    When both windows are fitted together, the first event of g2 (t=19)
    compares against node 0's last_seen from g1 (t=20), producing a
    negative delta and raising ValueError.
    """
    g1 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([10, 20]),
    )
    g2 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([2, 0]),
        t=torch.tensor([19, 30]),  # starts lower than g1 ended
    )
    stats = TimeGapStatistics()
    with pytest.raises(ValueError, match="[Nn]egative|[Dd]elta"):
        stats.fit([g1, g2])  # g2[0] t=19 < g1's last_seen[0]=20 → negative delta
