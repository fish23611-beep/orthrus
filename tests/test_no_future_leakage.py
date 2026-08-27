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
    """Per-event update: event i uses state after event i-1, not future events.

    Per-event semantics: events are processed sequentially, and each event
    sees the state updated by all previous events in the batch.

    Design: boundaries at log1p(5) and log1p(25) so that:
      delta=10s: z=log1p(10)=2.40 > log1p(5)=1.79 -> SHORT
      delta=20s: z=log1p(20)=3.04 <= log1p(25)=3.26 -> SHORT

    Event 1 uses the state after event 0 (10s), NOT future events.
    This is correct per-event causal ordering.
    """
    stats = _fit_stats()
    stats.time_bucket_boundaries = [math.log1p(5.0), math.log1p(25.0), math.log1p(50.0), math.log1p(100.0)]
    batch = _graph([0, 0], [1, 2], [10_000_000_000, 30_000_000_000])
    before = {0: 0, 1: 0, 2: -1}

    src_target, dst_target, after = stats.transform_batch(batch, before)

    # Event 0: gap = 10s - 0s = 10s -> log1p(10)=2.40 > log1p(5)=1.79 -> SHORT = 2
    assert src_target[0].item() == 2, (
        f"event 0: expected SHORT=2 (gap=10s), got {src_target[0].item()}"
    )
    # Event 1: per-event gap = 30s - 10s = 20s -> log1p(20)=3.04 <= log1p(25)=3.26 -> SHORT = 2
    assert src_target[1].item() == 2, (
        f"event 1: expected SHORT=2 (gap=20s, uses post-event-0 state), got {src_target[1].item()}"
    )

    # dst: node 1 has no history -> NO_HISTORY
    assert dst_target[0].item() == 2  # gap=10s-0s=10s -> SHORT
    assert dst_target[1].item() == NO_HISTORY  # node 2: no history

    # post-batch state: per-event update
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