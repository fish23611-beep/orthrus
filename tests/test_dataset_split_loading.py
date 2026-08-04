"""Regression tests for selective dataset split loading."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import TemporalData

import data_utils


def _cfg():
    return SimpleNamespace(
        dataset=SimpleNamespace(name="SYNTHETIC"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir="unused")
        ),
    )


def _graph(split_index):
    src = torch.tensor([split_index], dtype=torch.long)
    dst = torch.tensor([split_index + 1], dtype=torch.long)
    return TemporalData(
        src=src,
        dst=dst,
        t=torch.tensor([split_index + 10], dtype=torch.long),
        msg=torch.tensor([[float(split_index)]], dtype=torch.float),
        edge_type=torch.tensor([[float(split_index + 20)]], dtype=torch.float),
    )


def _load_spy(calls):
    graphs = {"train": [_graph(0)], "val": [_graph(2)], "test": [_graph(4)]}

    def load_data_set(cfg, path, split):
        calls.append(split)
        return graphs[split]

    return load_data_set


def test_train_only_does_not_load_validation_or_test():
    calls = []
    with patch.object(data_utils, "load_data_set", side_effect=_load_spy(calls)):
        train, val, test, full_data, max_node = data_utils.load_all_datasets(
            _cfg(), required_splits=("train",)
        )

    assert calls == ["train"]
    assert len(train) == 1
    assert val == []
    assert test == []
    assert full_data.msg.tolist() == [[0.0]]
    assert max_node == 2


def test_replay_request_loads_train_validation_and_test_in_order():
    calls = []
    with patch.object(data_utils, "load_data_set", side_effect=_load_spy(calls)):
        train, val, test, full_data, max_node = data_utils.load_all_datasets(
            _cfg(), required_splits=("test", "train", "val")
        )

    assert calls == ["train", "val", "test"]
    assert [len(train), len(val), len(test)] == [1, 1, 1]
    assert full_data.msg.flatten().tolist() == [0.0, 2.0, 4.0]
    assert full_data.t.tolist() == [10, 12, 14]
    assert max_node == 6


def test_default_call_remains_backwards_compatible():
    calls = []
    with patch.object(data_utils, "load_data_set", side_effect=_load_spy(calls)):
        result = data_utils.load_all_datasets(_cfg())

    assert calls == ["train", "val", "test"]
    assert len(result) == 5
    assert [len(split) for split in result[:3]] == [1, 1, 1]


def test_unknown_or_empty_split_request_is_rejected_before_io():
    calls = []
    with patch.object(data_utils, "load_data_set", side_effect=_load_spy(calls)):
        with pytest.raises(ValueError, match="Unknown dataset split"):
            data_utils.load_all_datasets(_cfg(), required_splits=("validation",))
        with pytest.raises(ValueError, match="At least one"):
            data_utils.load_all_datasets(_cfg(), required_splits=())

    assert calls == []
