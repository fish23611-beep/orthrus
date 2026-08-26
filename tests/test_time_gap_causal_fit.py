"""Regression tests for event-level time-gap statistics and targets."""

import inspect
import json
import math
import os
import sys

import pytest
import torch
from torch_geometric.data import TemporalData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mstc.time_gap import NO_HISTORY, VERY_LONG, VERY_SHORT, TimeGapStatistics

NS = 1_000_000_000


def _graph(src, dst, seconds):
    event_count = len(src)
    return TemporalData(
        src=torch.tensor(src, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor([value * NS for value in seconds], dtype=torch.long),
        msg=torch.zeros(event_count, 1),
        global_event_index=torch.arange(event_count),
    )


def _threshold_stats(seconds=15.0):
    stats = TimeGapStatistics()
    stats.time_bucket_boundaries = [math.log1p(seconds)] * 4
    stats.scale_boundaries = [math.log1p(seconds)] * 3
    return stats


def test_fit_uses_previous_event_for_repeated_node():
    graph = _graph([0, 0, 0, 0], [0, 0, 0, 0], [10, 20, 30, 40])

    stats = TimeGapStatistics().fit([graph])

    assert stats.time_bucket_boundaries == pytest.approx([math.log1p(10.0)] * 4)
    assert stats.scale_boundaries == pytest.approx([math.log1p(10.0)] * 3)


def test_fit_has_no_batch_size_parameter_and_is_window_partition_invariant():
    assert "batch_size" not in inspect.signature(TimeGapStatistics.fit).parameters

    whole = _graph([0, 0, 0, 0], [0, 0, 0, 0], [0, 10, 30, 60])
    first = _graph([0, 0], [0, 0], [0, 10])
    second = _graph([0, 0], [0, 0], [30, 60])

    whole_stats = TimeGapStatistics().fit([whole])
    partitioned_stats = TimeGapStatistics().fit([first, second])

    assert partitioned_stats.time_bucket_boundaries == pytest.approx(
        whole_stats.time_bucket_boundaries
    )
    assert partitioned_stats.scale_boundaries == pytest.approx(
        whole_stats.scale_boundaries
    )


def test_transform_batch_progresses_sequentially_without_future_state():
    stats = _threshold_stats()
    batch = _graph([0, 0], [1, 2], [10, 30])
    before = {0: 0, 1: 0, 2: -1}

    src_target, dst_target, after = stats.transform_batch(batch, before)

    assert before == {0: 0, 1: 0, 2: -1}
    assert src_target.tolist() == [VERY_SHORT, VERY_LONG]
    assert dst_target.tolist() == [VERY_SHORT, NO_HISTORY]
    assert after == {0: 30 * NS, 1: 10 * NS, 2: 30 * NS}


def test_transform_batch_rejects_non_monotonic_input_instead_of_using_future():
    stats = _threshold_stats()
    batch = _graph([0, 0], [1, 2], [30, 10])

    with pytest.raises(ValueError, match="non-decreasing"):
        stats.transform_batch(batch, {0: 0})


def test_self_loop_targets_share_the_same_pre_event_state():
    stats = _threshold_stats()
    batch = _graph([0, 0], [0, 0], [10, 30])

    src_target, dst_target, after = stats.transform_batch(batch, {0: 0})

    assert src_target.tolist() == [VERY_SHORT, VERY_LONG]
    assert dst_target.tolist() == src_target.tolist()
    assert after == {0: 30 * NS}


def test_last_seen_progresses_across_batches_and_windows():
    stats = _threshold_stats()
    first_window = _graph([0], [1], [0])
    second_window_batch = _graph([0, 0], [2, 3], [10, 30])
    third_batch = _graph([0], [4], [35])

    first_src, _, state = stats.transform_batch(first_window, {})
    second_src, _, state = stats.transform_batch(second_window_batch, state)
    third_src, _, state = stats.transform_batch(third_batch, state)

    assert first_src.tolist() == [NO_HISTORY]
    assert second_src.tolist() == [VERY_SHORT, VERY_LONG]
    assert third_src.tolist() == [VERY_SHORT]
    assert state[0] == 35 * NS


def test_fit_rejects_non_monotonic_timestamps():
    with pytest.raises(ValueError, match="non-decreasing"):
        TimeGapStatistics().fit([_graph([0, 0], [1, 1], [20, 10])])


def test_save_load_roundtrip_contains_required_metadata(tmp_path):
    stats = TimeGapStatistics().fit([
        _graph([0, 0, 0], [0, 0, 0], [0, 10, 30])
    ])
    path = tmp_path / "metadata" / "time_statistics.json"

    stats.save(str(path))
    loaded = TimeGapStatistics.load(str(path))
    payload = json.loads(path.read_text())

    assert set(payload) >= {
        "unit",
        "transform",
        "scale_quantiles",
        "scale_boundaries",
        "time_bucket_quantiles",
        "time_bucket_boundaries",
    }
    assert payload["unit"] == "seconds"
    assert payload["transform"] == "log1p"
    assert loaded.time_bucket_boundaries == stats.time_bucket_boundaries
    assert loaded.scale_boundaries == stats.scale_boundaries
