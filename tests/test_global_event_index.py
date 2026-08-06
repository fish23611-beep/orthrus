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
from torch_geometric.loader import TemporalDataLoader

import unittest.mock
sys.modules["encoders"] = unittest.mock.MagicMock()

from src.data_utils import _inject_event_indices, _inject_full_data_event_fields, SPLIT_NAME_TO_INDEX, custom_temporal_data_loader


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
    # Store one-hot types so _inject_event_indices always uses them
    g.src_type_onehot = src_type_oh.clone()
    g.dst_type_onehot = dst_type_oh.clone()
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
    # Store one-hot types so _inject_event_indices always uses them
    g.src_type_onehot = src_type_oh.clone()
    g.dst_type_onehot = dst_type_oh.clone()
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
        src=torch.cat([g.src for g in train + val + test]),
        dst=torch.cat([g.dst for g in train + val + test]),
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
    """Per-window split is now an integer index (0=train, 1=val, 2=test)."""
    train = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    # Integer index split
    assert train[0].split == 0   # "train" -> 0
    assert val[0].split == 1     # "val" -> 1
    assert test[0].split == 2    # "test" -> 2
    # String name also available
    assert train[0].split_name == "train"
    assert val[0].split_name == "val"
    assert test[0].split_name == "test"
    # full_data split aligns
    assert full.split[:1].tolist() == [0]
    n_val = len(val[0])
    n_test = len(test[0])
    assert full.split[len(train):len(train)+n_val].tolist() == [1] * n_val
    assert full.split[len(train)+n_val:].tolist() == [2] * n_test


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
    """
    An only_type artifact without src_type_onehot/dst_type_onehot
    can still be backfilled via the msg-layout fallback path.
    """
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
    # Simulate legacy artifact without one-hot attributes
    assert not hasattr(g, "src_type_onehot")
    assert not hasattr(g, "dst_type_onehot")

    _inject_event_indices(g, cfg)

    assert hasattr(g, "src_type")
    assert hasattr(g, "dst_type")
    assert hasattr(g, "edge_type_index")
    assert hasattr(g, "local_event_index")
    assert g.src_type.shape == (E,)
    assert g.dst_type.shape == (E,)
    assert g.edge_type_index.shape == (E,)
    assert g.local_event_index.shape == (E,)


# --------------------------------------------------------------------------- #
# Blocker regression: node types must come from msg, not embedding
# --------------------------------------------------------------------------- #
def test_blocker_embedding_argmax_ignored_when_types_stored():
    """
    When src_type_onehot/dst_type_onehot are present, node types must be
    extracted from those tensors — NOT from x_src/x_dst argmax.

    This is the core Blocker fix: use_node_type_in_node_feats=False means
    x_src/x_dst contain only the embedding, so argmax on the first
    node_type_dim dimensions would be semantically wrong.
    """
    cfg = FakeCfg()
    E = 3
    node_dim = cfg.dataset.num_node_types  # 3
    emb_dim = 128

    # True types for each event
    true_src_types = torch.tensor([0, 2, 1])   # real src types
    true_dst_types = torch.tensor([1, 0, 2])   # real dst types

    src_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    dst_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    src_type_oh[torch.arange(E), true_src_types] = 1
    dst_type_oh[torch.arange(E), true_dst_types] = 1

    edge_type_oh = torch.zeros(E, 4)
    edge_type_oh[:, 1] = 1

    src_emb = torch.randn(E, emb_dim)
    dst_emb = torch.randn(E, emb_dim)

    msg = torch.cat([src_type_oh, src_emb, edge_type_oh, dst_type_oh, dst_emb], dim=-1)

    # x_src/x_dst contain ONLY embedding (use_node_type_in_node_feats=False)
    x_src = src_emb.clone()
    x_dst = dst_emb.clone()

    # Deliberately set wrong type in the embedding's "type-like" position
    # so we can verify the wrong path would fail but we take the right path
    x_src[:, 0] = 999.0  # large value at position 0 — wrong path would pick 0
    x_dst[:, 2] = 888.0  # large value at position 2 — wrong path would pick 2

    g = TemporalData()
    g.src = torch.tensor([0, 1, 2])
    g.dst = torch.tensor([1, 2, 0])
    g.t = torch.tensor([0, 10, 20], dtype=torch.long)
    g.msg = msg
    g.edge_type = edge_type_oh
    g.x_src = x_src
    g.x_dst = x_dst
    # Store correct one-hot types
    g.src_type_onehot = src_type_oh.clone()
    g.dst_type_onehot = dst_type_oh.clone()

    _inject_event_indices(g, cfg)

    # Must match the real types stored in one-hot, NOT the embedding argmax
    assert torch.equal(g.src_type, true_src_types), (
        f"src_type should be {true_src_types.tolist()} from one-hot, "
        f"got {g.src_type.tolist()}"
    )
    assert torch.equal(g.dst_type, true_dst_types), (
        f"dst_type should be {true_dst_types.tolist()} from one-hot, "
        f"got {g.dst_type.tolist()}"
    )


def test_blocker_only_type_embedding_not_used():
    """
    only_type path: x_src/x_dst are the one-hot types themselves.
    Types are extracted from the msg layout (correct), not from embedding heuristics.
    """
    cfg = FakeCfgOnlyType()
    E = 3
    node_dim = cfg.dataset.num_node_types  # 2

    true_src_types = torch.tensor([0, 1, 0])
    true_dst_types = torch.tensor([1, 0, 1])

    src_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    dst_type_oh = torch.zeros(E, node_dim, dtype=torch.long)
    src_type_oh[torch.arange(E), true_src_types] = 1
    dst_type_oh[torch.arange(E), true_dst_types] = 1

    edge_type_oh = torch.zeros(E, 3)
    edge_type_oh[:, 1] = 1

    msg = torch.cat([src_type_oh, edge_type_oh, dst_type_oh], dim=-1)

    # x_src/x_dst are type one-hot (as in only_type)
    g = TemporalData()
    g.src = torch.tensor([0, 1, 2])
    g.dst = torch.tensor([1, 2, 0])
    g.t = torch.tensor([0, 10, 20], dtype=torch.long)
    g.msg = msg
    g.edge_type = edge_type_oh
    g.x_src = src_type_oh.clone()
    g.x_dst = dst_type_oh.clone()
    g.src_type_onehot = src_type_oh.clone()
    g.dst_type_onehot = dst_type_oh.clone()

    _inject_event_indices(g, cfg)

    assert torch.equal(g.src_type, true_src_types)
    assert torch.equal(g.dst_type, true_dst_types)


# --------------------------------------------------------------------------- #
# Per-event split tests
# --------------------------------------------------------------------------- #
def test_per_event_split_dtype_shape():
    """full_data.split is torch.long [total_events]."""
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), torch.tensor([20, 21, 22]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    assert isinstance(full.split, torch.Tensor)
    assert full.split.dtype == torch.long
    assert full.split.shape == (6,)
    assert torch.equal(full.split[:2], torch.tensor([0, 0], dtype=torch.long))
    assert torch.equal(full.split[2:3], torch.tensor([1], dtype=torch.long))
    assert torch.equal(full.split[3:], torch.tensor([2, 2, 2], dtype=torch.long))
    assert torch.equal(full.split, full.event_split)


def test_per_event_split_mapping():
    """full_data.split encodes train=0, val=1, test=2."""
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), torch.tensor([20, 21, 22]))[0]]
    full = _inject_all(train, val, test, FakeCfg())
    assert (full.split[:2] == 0).all()
    assert (full.split[2:3] == 1).all()
    assert (full.split[3:] == 2).all()


def test_per_event_split_name_from_index():
    """Helper: INDEX_TO_SPLIT_NAME maps split index back to name."""
    from src.data_utils import INDEX_TO_SPLIT_NAME
    assert INDEX_TO_SPLIT_NAME[0] == "train"
    assert INDEX_TO_SPLIT_NAME[1] == "val"
    assert INDEX_TO_SPLIT_NAME[2] == "test"


def test_window_split_integer():
    """g.split on a window is now an integer index, not a string."""
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    full = _inject_all(train, [], [], FakeCfg())
    assert isinstance(train[0].split, int)
    assert train[0].split == 0


def test_full_data_split_preserved_in_save_load(tmp_path):
    """full_data.split survives torch.save / torch.load."""
    import tempfile
    import os
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([20, 21]))[0]]
    full = _inject_all(train, val, test, FakeCfg())

    path = os.path.join(str(tmp_path), "full_data.pt")
    torch.save(full, path)
    loaded = torch.load(path)

    assert torch.equal(loaded.split, full.split)
    assert torch.equal(loaded.split, loaded.event_split)
    assert torch.equal(loaded.src, full.src)
    assert torch.equal(loaded.dst, full.dst)
    assert torch.equal(loaded.msg, full.msg)
    assert torch.equal(loaded.t, full.t)
    assert torch.equal(loaded.edge_type, full.edge_type)


def test_full_data_split_in_temporal_data_loader():
    """Per-event split is preserved through TemporalDataLoader batching."""
    from torch_geometric.data import TemporalData
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 10]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    test = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([30, 40]))[0]]
    full = _inject_all(train, val, test, FakeCfg())

    # Convert Data -> TemporalData for the loader
    full_temporal = TemporalData(
        src=full.src,
        dst=full.dst,
        t=full.t,
        msg=full.msg,
        edge_type=full.edge_type,
        src_type=full.src_type,
        dst_type=full.dst_type,
        edge_type_index=full.edge_type_index,
        global_event_index=full.global_event_index,
        split=full.split,
        event_split=full.event_split,
    )
    # batch_size=5 loads all 5 events in one batch
    loader = custom_temporal_data_loader(full_temporal, batch_size=5)
    batch = next(iter(loader))
    assert hasattr(batch, "split")
    assert batch.split.dtype == torch.long
    # batch.split should match full.split for all events
    assert torch.equal(batch.split, full.split)
    assert torch.equal(batch.split, torch.tensor([0, 0, 1, 2, 2], dtype=torch.long))
    assert torch.equal(batch.t, full.t)


def test_empty_window_has_empty_split():
    """A zero-event window produces an empty split tensor."""
    g_empty, cfg = _make_window_full_emb(
        src=torch.tensor([], dtype=torch.long),
        dst=torch.tensor([], dtype=torch.long),
        t=torch.tensor([], dtype=torch.long),
    )
    # Inject produces the index fields
    _inject_event_indices(g_empty, cfg)
    # full_data should handle empty windows gracefully
    train = [g_empty]
    full = _inject_all(train, [], [], FakeCfg())
    assert full.split.shape[0] == 0


# --------------------------------------------------------------------------- #
# Full-data full-field alignment tests
# --------------------------------------------------------------------------- #
def test_full_data_all_fields_align():
    """
    Every per-event field in full_data aligns with per-window fields via
    global_event_index:
      g.src[i]       == full_data.src[k]
      g.dst[i]       == full_data.dst[k]
      g.t[i]         == full_data.t[k]
      g.msg[i]       == full_data.msg[k]
      g.edge_type[i] == full_data.edge_type[k]
      g.src_type[i]  == full_data.src_type[k]
      g.dst_type[i]  == full_data.dst_type[k]
      g.edge_type_index[i] == full_data.edge_type_index[k]
      g.split[i]     == full_data.split[k]  (or g.split == full_data.split[k])
    where k = g.global_event_index[i].
    """
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), torch.tensor([10, 11, 12]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())

    for g in train + val + test:
        for i in range(len(g)):
            k = g.global_event_index[i].item()
            assert g.src[i].item() == full.src[k].item(), f"src mismatch at event {i}"
            assert g.dst[i].item() == full.dst[k].item(), f"dst mismatch at event {i}"
            assert g.t[i].item() == full.t[k].item(), f"t mismatch at event {i}"
            assert torch.equal(g.msg[i], full.msg[k]), f"msg mismatch at event {i}"
            assert torch.equal(g.edge_type[i], full.edge_type[k]), f"edge_type mismatch at event {i}"
            assert g.src_type[i].item() == full.src_type[k].item(), f"src_type mismatch at event {i}"
            assert g.dst_type[i].item() == full.dst_type[k].item(), f"dst_type mismatch at event {i}"
            assert g.edge_type_index[i].item() == full.edge_type_index[k].item(), f"edge_type_index mismatch at event {i}"
            # Per-window split is integer; full_data split[k] is also integer
            assert g.split == full.split[k].item(), f"split mismatch at event {i}"


def test_full_data_src_dst_preserved(tmp_path):
    """full_data.src and full_data.dst match the concatenated window src/dst."""
    train = [_make_window_full_emb(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2]))[0]]
    val = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([10, 11]))[0]]
    test = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([20]))[0]]
    full = _inject_all(train, val, test, FakeCfg())

    all_src = torch.cat([g.src for g in train + val + test])
    all_dst = torch.cat([g.dst for g in train + val + test])
    assert torch.equal(full.src, all_src)
    assert torch.equal(full.dst, all_dst)


def test_full_data_edge_type_src_dst_types_align():
    """src_type, dst_type, edge_type_index on full_data match window fields."""
    train = [_make_window_full_emb(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 1]))[0]]
    val = [_make_window_full_emb(torch.tensor([0]), torch.tensor([1]), torch.tensor([10]))[0]]
    full = _inject_all(train, val, [], FakeCfg())

    all_src_type = torch.cat([g.src_type for g in train + val])
    all_dst_type = torch.cat([g.dst_type for g in train + val])
    all_edge_type_index = torch.cat([g.edge_type_index for g in train + val])

    assert torch.equal(full.src_type, all_src_type)
    assert torch.equal(full.dst_type, all_dst_type)
    assert torch.equal(full.edge_type_index, all_edge_type_index)