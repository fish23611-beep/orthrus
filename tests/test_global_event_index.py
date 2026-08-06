"""
Tests for src/data_utils.py — event index and global alignment.

Verifies that load_data_set injects explicit event fields, and that
inject_full_data_event_fields produces globally-aligned global_event_index
and split/window_id metadata.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from torch_geometric.data import Data, TemporalData

import unittest.mock
sys.modules["encoders"] = unittest.mock.MagicMock()

from src.data_utils import _inject_event_indices, _inject_full_data_event_fields


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class FakeDataset:
    num_node_types = 3
    num_edge_types = 4


class FakeDecoder:
    used_methods = ["predict_edge_type"]


class FakeEncoder:
    use_node_type_in_node_feats = True


class FakeGnnTraining:
    decoder = FakeDecoder()
    encoder = FakeEncoder()


class FakeDetection:
    gnn_training = FakeGnnTraining()


class FakeCfg:
    dataset = FakeDataset()
    detection = FakeDetection()


class FakeDatasetOnlyType:
    num_node_types = 2
    num_edge_types = 3


class FakeDetectionOnlyType:
    gnn_training = FakeGnnTraining()


class FakeCfgOnlyType:
    dataset = FakeDatasetOnlyType()
    detection = FakeDetectionOnlyType()


def _make_window_only_type(src, dst, t):
    """only_type path: msg = [src_type | edge_type | dst_type]"""
    E = len(src)
    cfg = FakeCfgOnlyType()
    node_dim = cfg.dataset.num_node_types
    edge_dim = cfg.dataset.num_edge_types

    src_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    dst_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    edge_type_oh = torch.zeros(E, edge_dim, dtype=torch.long)
    edge_type_oh[:, 1] = 1

    src_type_oh[torch.arange(E), src % node_dim] = 1
    dst_type_oh[torch.arange(E), dst % node_dim] = 1

    msg = torch.cat([src_type_oh, edge_type_oh, dst_type_oh], dim=-1)
    x_src = src_type_oh
    x_dst = dst_type_oh

    g = TemporalData()
    g.src = src
    g.dst = dst
    g.t = t
    g.msg = msg
    g.edge_type = edge_type_oh
    g.x_src = x_src
    g.x_dst = x_dst
    return g, cfg


def _make_window_full_emb(src, dst, t):
    """Full embedding path: msg = [src_type | src_emb | edge_type | dst_type | dst_emb]"""
    E = len(src)
    cfg = FakeCfg()
    node_dim = cfg.dataset.num_node_types
    edge_dim = cfg.dataset.num_edge_types
    emb_dim = 128

    src_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    dst_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    edge_type_oh = torch.zeros(E, edge_dim, dtype=torch.long)
    edge_type_oh[:, 1] = 1

    src_emb = torch.randn(E, emb_dim)
    dst_emb = torch.randn(E, emb_dim)

    src_type_oh[torch.arange(E), src % node_dim] = 1
    dst_type_oh[torch.arange(E), dst % node_dim] = 1

    msg = torch.cat([src_type_oh, src_emb, edge_type_oh, dst_type_oh, dst_emb], dim=-1)
    x_src = torch.cat([src_emb, src_type_oh], dim=-1)
    x_dst = torch.cat([dst_emb, dst_type_oh], dim=-1)

    g = TemporalData()
    g.src = src
    g.dst = dst
    g.t = t
    g.msg = msg
    g.edge_type = edge_type_oh
    g.x_src = x_src
    g.x_dst = x_dst
    return g, cfg


def _inject_all(train, val, test, cfg):
    """Inject indices on all windows, then build full_data."""
    for g in train:
        _inject_event_indices(g, cfg)
    for g in val:
        _inject_event_indices(g, cfg)
    for g in test:
        _inject_event_indices(g, cfg)
    full = Data(
        msg=torch.cat([g.msg for g in train + val + test]),
        t=torch.cat([g.t for g in train + val + test]),
        edge_type=torch.cat([g.edge_type for g in train + val + test]),
    )
    return _inject_full_data_event_fields(train, val, test, full)


# --------------------------------------------------------------------------- #
# Per-window fields
# --------------------------------------------------------------------------- #
def test_src_type_dtype_and_shape_full_emb():
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        t=torch.tensor([0, 10, 20]),
    )
    _inject_event_indices(g, cfg)
    assert g.src_type.dtype == torch.long
    assert g.src_type.shape == (3,)
    assert (g.src_type >= 0).all()
    assert (g.src_type < cfg.dataset.num_node_types).all()


def test_src_type_dtype_and_shape_only_type():
    g, cfg = _make_window_only_type(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        t=torch.tensor([0, 10, 20]),
    )
    _inject_event_indices(g, cfg)
    assert g.src_type.dtype == torch.long
    assert g.src_type.shape == (3,)


def test_dst_type_dtype_and_shape():
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([0, 10]),
    )
    _inject_event_indices(g, cfg)
    assert g.dst_type.dtype == torch.long
    assert g.dst_type.shape == (2,)
    assert (g.dst_type >= 0).all()
    assert (g.dst_type < cfg.dataset.num_node_types).all()


def test_edge_type_index_dtype_and_shape():
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        t=torch.tensor([0]),
    )
    _inject_event_indices(g, cfg)
    assert g.edge_type_index.dtype == torch.long
    assert g.edge_type_index.shape == (1,)
    assert torch.equal(g.edge_type_index, g.edge_type.argmax(dim=-1))


def test_local_event_index():
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0, 1, 2, 3]),
        dst=torch.tensor([1, 2, 3, 0]),
        t=torch.tensor([0, 10, 20, 30]),
    )
    _inject_event_indices(g, cfg)
    expected = torch.arange(4, dtype=torch.long)
    assert torch.equal(g.local_event_index, expected)


def test_edge_type_index_one_hot_consistency():
    for edge_val in [0, 1, 2, 3]:
        g, cfg = _make_window_full_emb(
            src=torch.tensor([0, 1]),
            dst=torch.tensor([1, 2]),
            t=torch.tensor([0, 10]),
        )
        g.edge_type = torch.zeros(2, 4)
        g.edge_type[:, edge_val] = 1
        _inject_event_indices(g, cfg)
        assert torch.equal(g.edge_type_index, g.edge_type.argmax(dim=-1))


def test_only_type_path_also_generates_fields():
    g, cfg = _make_window_only_type(
        src=torch.tensor([0, 1, 2]),
        dst=torch.tensor([1, 2, 0]),
        t=torch.tensor([0, 10, 20]),
    )
    _inject_event_indices(g, cfg)
    assert hasattr(g, "src_type")
    assert hasattr(g, "dst_type")
    assert hasattr(g, "edge_type_index")
    assert hasattr(g, "local_event_index")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_non_monotonic_timestamps_raises():
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([10, 5]),
    )
    with pytest.raises(AssertionError, match="non-decreasing"):
        _inject_event_indices(g, cfg)


def test_src_type_range_is_validated():
    """
    src_type must be in [0, num_node_types).
    In the full embedding path this is always true (derived from one-hot argmax).
    We test the validation path executes by feeding a graph with 0 events.
    """
    # Zero-event graph
    g, cfg = _make_window_full_emb(
        src=torch.tensor([], dtype=torch.long),
        dst=torch.tensor([], dtype=torch.long),
        t=torch.tensor([], dtype=torch.long),
    )
    # Should not crash — empty tensors produce empty ranges
    _inject_event_indices(g, cfg)
    assert g.src_type.shape == (0,)


def test_edge_type_index_consistency_is_validated():
    """
    edge_type_index must equal edge_type.argmax(dim=-1).
    In the full embedding path this is always true (derived from one-hot argmax).
    We verify the consistency assertion passes for valid data.
    """
    g, cfg = _make_window_full_emb(
        src=torch.tensor([0]),
        dst=torch.tensor([1]),
        t=torch.tensor([0]),
    )
    _inject_event_indices(g, cfg)
    assert g.edge_type_index.shape == (1,)
    expected = g.edge_type.argmax(dim=-1)
    assert torch.equal(g.edge_type_index, expected)


# --------------------------------------------------------------------------- #
# Global event indexing
# --------------------------------------------------------------------------- #
def test_global_event_index_sequential():
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), torch.tensor([10, 11, 12]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    expected = torch.arange(6, dtype=torch.long)
    assert torch.equal(full.global_event_index, expected)
    assert torch.equal(full.event_index, full.global_event_index)


def test_global_event_index_train_val_test_continuity():
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([20, 21]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    assert train[0].global_event_index[-1].item() + 1 == val[0].global_event_index[0].item()
    assert val[0].global_event_index[-1].item() + 1 == test[0].global_event_index[0].item()


def test_split_annotation():
    train = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    assert train[0].split == "train"
    assert val[0].split == "val"
    assert test[0].split == "test"


def test_window_id_globally_unique():
    train = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([i]))[0] for i in range(3)]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([i + 100]))[0] for i in range(2)]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([i + 200]))[0] for i in range(4)]
    all_windows = train + val + test
    full = _inject_all(train, val, test, FakeCfg())
    window_ids = [g.window_id for g in all_windows]
    assert len(window_ids) == len(set(window_ids))
    assert sorted(window_ids) == list(range(len(all_windows)))


def test_global_event_index_aligns_with_full_data_msg():
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([20, 21]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    for g in train + val + test:
        for i in range(len(g)):
            k = g.global_event_index[i].item()
            assert torch.equal(full.msg[k], g.msg[i])


def test_global_event_index_aligns_with_full_data_t():
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([20, 21]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    for g in train + val + test:
        for i in range(len(g)):
            k = g.global_event_index[i].item()
            assert full.t[k] == g.t[i]


def test_event_index_is_alias_of_global_event_index():
    train = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    assert torch.equal(full.event_index, full.global_event_index)


def test_old_artifact_backfill_only_type():
    """An only_type artifact can be backfilled by _inject_event_indices."""
    cfg = FakeCfgOnlyType()
    E = 3
    g = TemporalData()
    g.src = torch.tensor([0, 1, 2])
    g.dst = torch.tensor([1, 2, 0])
    g.t = torch.tensor([0, 10, 20], dtype=torch.long)
    # only_type msg: [src_type(2) | edge_type(3) | dst_type(2)] = 7
    g.msg = torch.zeros(E, 7)
    g.edge_type = torch.zeros(E, 3)
    g.edge_type[:, 1] = 1
    g.x_src = torch.zeros(E, 2)
    g.x_dst = torch.zeros(E, 2)
    # Set src_type and dst_type one-hot in msg so extraction works
    g.msg[:, :2] = torch.tensor([[1, 0], [0, 1], [1, 0]])
    g.msg[:, -2:] = torch.tensor([[1, 0], [0, 1], [1, 0]])

    _inject_event_indices(g, cfg)

    assert hasattr(g, "src_type")
    assert hasattr(g, "dst_type")
    assert hasattr(g, "edge_type_index")
    assert hasattr(g, "local_event_index")
    assert g.src_type.shape == (E,)
    assert g.dst_type.shape == (E,)
    assert g.edge_type_index.shape == (E,)
    assert g.local_event_index.shape == (E,)
