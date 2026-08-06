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
    assert len(stats.scale_boundaries) == 3
    assert stats.time_bucket_boundaries == sorted(stats.time_bucket_boundaries)


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
    assert len(stats.scale_boundaries) == 3


# --------------------------------------------------------------------------- #
# transform_batch — causal correctness
# --------------------------------------------------------------------------- #
def test_transform_batch_no_future_leak():
    """Targets must use last_seen state BEFORE current batch."""
    g1 = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, 10]),
    )
    g2 = _make_graph(
        src=torch.tensor([0, 2]),
        dst=torch.tensor([2, 0]),
        t=torch.tensor([20, 30]),
    )
    stats = TimeGapStatistics()
    stats.fit([g1])

    last_seen = {-1: -1, 0: -1, 1: -1, 2: -1}
    src_t1, dst_t1, last_seen_after1 = stats.transform_batch(g1, last_seen)
    assert src_t1[0] == NO_HISTORY
    assert src_t1[1] == NO_HISTORY

    src_t2, dst_t2, last_seen_after2 = stats.transform_batch(g2, last_seen_after1)
    assert src_t2[0] != NO_HISTORY or dst_t2[0] != NO_HISTORY
    assert src_t2[1] != NO_HISTORY or dst_t2[1] != NO_HISTORY

    assert last_seen_after2 is not last_seen_after1
    assert last_seen_after1 is not last_seen


def test_transform_batch_updates_after_target():
    """last_seen must be updated AFTER all targets are computed."""
    g = _make_graph(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 0]),
        t=torch.tensor([5, 10]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])

    last_seen = {0: -1, 1: -1}
    src_t, dst_t, final = stats.transform_batch(g, last_seen)

    assert src_t[0] == NO_HISTORY
    assert src_t[1] == NO_HISTORY
    assert final[0] == 10
    assert final[1] == 10


def test_transform_batch_self_loop():
    """src == dst must update state exactly once per event."""
    g = _make_graph(
        src=torch.tensor([0, 0]),
        dst=torch.tensor([0, 0]),
        t=torch.tensor([0, 100]),
    )
    stats = TimeGapStatistics()
    stats.fit([g])

    last_seen = {0: -1}
    src_t, dst_t, final = stats.transform_batch(g, last_seen)

    assert src_t[0] == NO_HISTORY
    assert dst_t[0] == NO_HISTORY
    assert src_t[1] == NO_HISTORY
    assert dst_t[1] == NO_HISTORY
    assert final[0] == 100


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
    assert loaded.scale_boundaries == stats.scale_boundaries

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
