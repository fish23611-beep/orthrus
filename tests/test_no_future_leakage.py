"""C4 integration tests for causal time targets and state progression."""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import TemporalData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decoders import TimeGapDecoder
from model import MSTCOrthrus
from mstc.time_gap import NO_HISTORY, TimeGapStatistics


def _fit_stats() -> TimeGapStatistics:
    train = TemporalData(
        src=torch.tensor([0, 0, 0]),
        dst=torch.tensor([1, 1, 1]),
        t=torch.tensor([0, 1_000_000_000, 2_000_000_000]),
        msg=torch.zeros(3, 1),
    )
    return TimeGapStatistics().fit([train])


def _graph(src, dst, t):
    return TemporalData(
        src=torch.tensor(src),
        dst=torch.tensor(dst),
        t=torch.tensor(t),
        msg=torch.zeros(len(src), 1),
    )


def test_time_targets_use_earlier_same_batch_events_but_not_future_events():
    """Within-batch event i cannot use state set by event j>i.

    Design: boundaries at log1p(25) so that:
      gap=10s -> VERY_SHORT (correct:  pre_batch_last_seen=0s, gap=10s)
      gap=20s -> SHORT      (BUG:       intra-batch pollution, gap=20s would give SHORT)
      gap=30s -> LONG       (correct:   pre_batch_last_seen=0s, gap=30s)

    Event 1's gap with the BUG would be 20s (uses event 0's timestamp as base).
    Event 1's gap CORRECT would be 30s (uses pre-batch last_seen=0 as base).
    These land in different buckets, so the test is unambiguous.
    """
    stats = _fit_stats()
    stats.time_bucket_boundaries = [math.log1p(25.0)] * 4
    batch = _graph([0, 0], [1, 2], [10_000_000_000, 30_000_000_000])
    before = {0: 0, 1: 0, 2: -1}

    src_target, dst_target, after = stats.transform_batch(batch, before)

    # Event 0: gap = 10s - 0s = 10s -> log1p(10) < log1p(25) -> VERY_SHORT = 1
    assert src_target[0].item() == 1, (
        f"event 0: expected VERY_SHORT=1 (gap=10s), got {src_target[0].item()}"
    )
    # Event 1: CORRECT gap = 30s - 0s = 30s -> log1p(30) > log1p(25) -> LONG = 5
    #           BUGGED gap = 30s - 10s = 20s -> log1p(20) < log1p(25) -> SHORT = 2
    assert src_target[1].item() == 5, (
        f"event 1: expected LONG=5 (gap=30s, base=pre_batch 0s), got {src_target[1].item()}. "
        "If you see SHORT=2, the target was computed against intra-batch state (10s) instead of pre-batch state (0s)."
    )

    # dst: node 1 has no history -> NO_HISTORY
    assert dst_target[0].item() == 1  # gap=10s-0s=10s -> VERY_SHORT
    assert dst_target[1].item() == NO_HISTORY  # node 2: no history

    # post-batch state: max(pre_batch, batch_times)
    assert after[0] == 30_000_000_000
    assert after[1] == 10_000_000_000
    assert after[2] == 30_000_000_000


def test_batch_updates_only_affect_later_batches_and_first_occurrence_is_no_history():
    stats = _fit_stats()
    first = _graph([4], [5], [10_000_000_000])
    second = _graph([4], [6], [20_000_000_000])

    first_src, first_dst, state_after_first = stats.transform_batch(first, {})
    second_src, second_dst, _ = stats.transform_batch(second, state_after_first)

    assert first_src.item() == NO_HISTORY
    assert first_dst.item() == NO_HISTORY
    assert second_src.item() != NO_HISTORY
    assert second_dst.item() == NO_HISTORY


def test_negative_time_delta_is_rejected():
    stats = _fit_stats()
    batch = _graph([0], [1], [5])

    with pytest.raises(ValueError, match="Negative time delta"):
        stats.transform_batch(batch, {0: 10, 1: 0})


class _Batch(SimpleNamespace):
    def __len__(self):
        return self.t.numel()


class _Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.ones(dim))
        self.reset_calls = 0

    def forward(self, edge_index, **kwargs):
        count = edge_index.size(1)
        h = self.embedding.expand(count, -1)
        return h, h

    def reset_state(self):
        self.reset_calls += 1


def _model_batch(timestamps):
    count = len(timestamps)
    src = torch.tensor([0] * count)
    dst = torch.tensor([1] * count)
    return _Batch(
        src=src,
        dst=dst,
        t=torch.tensor(timestamps),
        edge_index=torch.stack([src, dst]),
        x_src=torch.zeros(count, 2),
        x_dst=torch.zeros(count, 2),
        msg=torch.zeros(count, 1),
        edge_type=torch.nn.functional.one_hot(torch.zeros(count, dtype=torch.long), num_classes=2).float(),
    )


def test_mstc_reset_and_replay_progress_one_shared_time_state():
    encoder = _Encoder(dim=2)
    model = MSTCOrthrus(
        encoder=encoder,
        edge_decoder=None,
        time_gap_decoder=TimeGapDecoder(in_dim=2, hidden_dim=4),
        time_gap_statistics=_fit_stats(),
    )

    train_batch = _model_batch([10_000_000_000])
    val_batch = _model_batch([20_000_000_000])
    test_batch = _model_batch([30_000_000_000])
    model(train_batch, full_data=None, inference=True)
    train_state = dict(model.last_seen_per_node)
    model(val_batch, full_data=None, inference=True)
    val_state = dict(model.last_seen_per_node)
    model(test_batch, full_data=None, inference=True)

    assert train_state[0] == 10_000_000_000
    assert val_state[0] == 20_000_000_000
    assert model.last_seen_per_node[0] == 30_000_000_000
    assert encoder.reset_calls == 0

    model.reset_state()
    assert model.last_seen_per_node == {}
    assert encoder.reset_calls == 1