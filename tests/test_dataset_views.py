from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


views = load_file("c6_b8_dataset_views", SRC_ROOT / "mstc" / "dataset_views.py")


class EventData:
    def clone(self):
        return copy.deepcopy(self)


def make_data(offset=0):
    """Five events: S→F, S→N, N→F, N→N, F→S."""
    data = EventData()
    data.src = torch.tensor([10, 10, 30, 30, 20]) + offset
    data.dst = torch.tensor([20, 30, 20, 30, 10]) + offset
    data.t = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
    data.src_type = torch.tensor([1, 1, 3, 3, 2], dtype=torch.long)
    data.dst_type = torch.tensor([2, 3, 2, 3, 1], dtype=torch.long)
    data.src_type_onehot = torch.nn.functional.one_hot(data.src_type - 1, num_classes=3).float()
    data.dst_type_onehot = torch.nn.functional.one_hot(data.dst_type - 1, num_classes=3).float()
    src_sem = torch.arange(10, 20, dtype=torch.float).reshape(5, 2)
    dst_sem = torch.arange(20, 30, dtype=torch.float).reshape(5, 2)
    data.x_src = torch.cat([src_sem, data.src_type_onehot], dim=1)
    data.x_dst = torch.cat([dst_sem, data.dst_type_onehot], dim=1)
    data.edge_type = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 1, 0]), num_classes=3).float()
    data.edge_type_index = data.edge_type.argmax(dim=1).long()
    data.msg = torch.cat([data.x_src, data.x_dst, data.edge_type], dim=1)
    data.edge_feats = torch.cat([data.edge_type, data.msg], dim=1)
    data.edge_index = torch.stack([data.src, data.dst])
    data.local_event_index = torch.arange(5, dtype=torch.long)
    data.global_event_index = torch.arange(100, 105, dtype=torch.long)
    data.event_index = data.global_event_index.clone()
    data.split = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    data.window_id = 7
    data.loss_type = torch.arange(5, dtype=torch.float)
    data.some_event_field = torch.tensor([50, 51, 52, 53, 54])
    data.tags = ["a", "b", "c", "d", "e"]
    return data


def tensor_state(data):
    return {name: value.clone() for name, value in vars(data).items() if isinstance(value, torch.Tensor)}


def assert_tensor_state(data, state):
    for name, expected in state.items():
        assert torch.equal(getattr(data, name), expected), name


def test_host_only_keeps_only_host_events_and_all_event_fields_aligned():
    original = make_data()
    viewed = views.apply_dataset_view(original, "host_only")
    assert viewed.src.tolist() == [10, 20]
    assert viewed.dst.tolist() == [20, 10]
    assert viewed.t.tolist() == [10, 50]
    for field in ("src", "dst", "t", "msg", "edge_type", "edge_type_index", "x_src", "x_dst", "src_type", "dst_type", "local_event_index", "global_event_index", "event_index", "split", "loss_type", "some_event_field"):
        assert getattr(viewed, field).shape[0] == 2, field
    assert viewed.edge_index.shape == (2, 2)
    assert viewed.tags == ["a", "e"]
    assert viewed.local_event_index.tolist() == [0, 1]
    assert viewed.window_id == 7


def test_host_network_structure_keeps_structure_and_zeros_only_netflow_semantics():
    original = make_data()
    viewed = views.apply_dataset_view(original, "host_network_structure")
    for field in ("src", "dst", "t", "edge_index", "edge_type", "edge_type_index", "src_type", "dst_type"):
        assert torch.equal(getattr(viewed, field), getattr(original, field)), field
    src_net = original.src_type == views.NETFLOW_TYPE_INDEX
    dst_net = original.dst_type == views.NETFLOW_TYPE_INDEX
    assert torch.equal(viewed.x_src[src_net, 2:], original.x_src[src_net, 2:])
    assert torch.equal(viewed.x_dst[dst_net, 2:], original.x_dst[dst_net, 2:])
    assert torch.equal(viewed.x_src[src_net, :2], torch.zeros_like(viewed.x_src[src_net, :2]))
    assert torch.equal(viewed.x_dst[dst_net, :2], torch.zeros_like(viewed.x_dst[dst_net, :2]))
    assert torch.equal(viewed.x_src[~src_net, :2], original.x_src[~src_net, :2])
    assert torch.equal(viewed.x_dst[~dst_net, :2], original.x_dst[~dst_net, :2])
    assert not torch.equal(viewed.msg, original.msg)


def test_full_is_identity_copy_and_structure_does_not_mutate_input():
    original = make_data()
    before = tensor_state(original)
    views.apply_dataset_view(original, "host_network_structure")
    full = views.apply_dataset_view(original, "host_network_full")
    assert_tensor_state(original, before)
    assert_tensor_state(full, before)
    assert full is not original


@pytest.mark.parametrize("mode", ["unknown", "network_only", "", None])
def test_invalid_modes_fail_clearly(mode):
    with pytest.raises(ValueError, match="Invalid dataset view"):
        views.apply_dataset_view(make_data(), mode)


def test_structure_without_type_onehot_zeros_entire_netflow_feature_vector():
    data = make_data()
    del data.src_type_onehot
    del data.dst_type_onehot
    viewed = views.apply_dataset_view(data, "host_network_structure")
    assert torch.equal(viewed.x_src[data.src_type == 3], torch.zeros_like(viewed.x_src[data.src_type == 3]))
    assert torch.equal(viewed.x_dst[data.dst_type == 3], torch.zeros_like(viewed.x_dst[data.dst_type == 3]))
    assert torch.equal(viewed.src_type, data.src_type)
    assert torch.equal(viewed.dst_type, data.dst_type)


def test_empty_and_all_netflow_windows_are_safe():
    data = make_data()
    data.src_type[:] = 3
    data.dst_type[:] = 3
    viewed = views.apply_dataset_view(data, "host_only")
    assert viewed.src.numel() == 0
    assert viewed.local_event_index.dtype == torch.long
    assert viewed.local_event_index.numel() == 0


def load_data_utils_with_stubs(monkeypatch):
    class Data:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    temporal = types.ModuleType("torch_geometric.data")
    temporal.Data = Data
    temporal.TemporalData = Data
    loader = types.ModuleType("torch_geometric.loader")
    loader.TemporalDataLoader = object
    monkeypatch.setitem(sys.modules, "torch_geometric.data", temporal)
    monkeypatch.setitem(sys.modules, "torch_geometric.loader", loader)
    encoders = types.ModuleType("encoders")
    encoders.OrthrusEncoder = object
    monkeypatch.setitem(sys.modules, "encoders", encoders)
    return load_file("c6_b8_data_utils", SRC_ROOT / "data_utils.py")


def test_view_then_full_data_rebuilds_global_index_split_window_and_history_ids(monkeypatch):
    data_utils = load_data_utils_with_stubs(monkeypatch)
    train = [views.apply_dataset_view(make_data(0), "host_only"), views.apply_dataset_view(make_data(100), "host_only")]
    val = [views.apply_dataset_view(make_data(200), "host_only")]
    test = [views.apply_dataset_view(make_data(300), "host_only")]
    full = SimpleNamespace(
        msg=torch.cat([g.msg for g in train + val + test]), t=torch.cat([g.t for g in train + val + test]),
        edge_type=torch.cat([g.edge_type for g in train + val + test]), src=torch.cat([g.src for g in train + val + test]), dst=torch.cat([g.dst for g in train + val + test]),
    )
    full = data_utils._inject_full_data_event_fields(train, val, test, full)
    assert torch.equal(full.global_event_index, torch.arange(8))
    assert torch.equal(full.event_index, full.global_event_index)
    assert full.split.tolist() == [0, 0, 0, 0, 1, 1, 2, 2]
    assert [g.window_id for g in train + val + test] == [0, 1, 2, 3]
    for event_id in (0, 3, 5, 7):
        assert full.src[event_id].item() == full.src[full.global_event_index[event_id]].item()
        assert full.dst[event_id].item() == full.dst[full.global_event_index[event_id]].item()
        assert full.t[event_id].item() == full.t[full.global_event_index[event_id]].item()


def test_host_only_history_input_excludes_prior_netflow_event():
    data = make_data()
    data.src = torch.tensor([1, 1]); data.dst = torch.tensor([3, 2]); data.t = torch.tensor([10, 20])
    data.src_type = torch.tensor([1, 1]); data.dst_type = torch.tensor([3, 2])
    data.src_type_onehot = torch.nn.functional.one_hot(data.src_type - 1, 3).float()
    data.dst_type_onehot = torch.nn.functional.one_hot(data.dst_type - 1, 3).float()
    data.x_src = torch.cat([torch.ones(2, 2), data.src_type_onehot], 1); data.x_dst = torch.cat([torch.ones(2, 2), data.dst_type_onehot], 1)
    data.edge_type = torch.ones(2, 1); data.msg = torch.cat([data.x_src, data.x_dst, data.edge_type], 1); data.edge_feats = data.msg.clone(); data.edge_index = torch.stack([data.src, data.dst]); data.local_event_index = torch.arange(2)
    host = views.apply_dataset_view(data, "host_only")
    assert host.t.tolist() == [20]
    assert host.src.tolist() == [1]


def test_config_default_and_experiment_overlays():
    config = load_file("c6_b8_config", SRC_ROOT / "config.py")
    args = SimpleNamespace(cpu=True, from_weights=False, seed=0, skip_tracing=False, dataset="THEIA_E5")
    cfg = config.get_default_cfg(args)
    assert cfg.dataset_view.mode == "host_network_full"
    for name, expected in (("host_only", "host_only"), ("host_network_structure", "host_network_structure"), ("host_network_full", "host_network_full")):
        payload = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "experiments" / f"{name}.yml").read_text())
        assert payload["dataset_view"]["mode"] == expected
    for invalid in ("full", "unknown", ""):
        with pytest.raises(ValueError):
            config._validate_dataset_view_mode(invalid)

@pytest.mark.parametrize(
    ("mode", "expected_events"),
    [("host_only", 2), ("host_network_structure", 5), ("host_network_full", 5)],
)
def test_views_preserve_dtype_device_and_window_identity(mode, expected_events):
    data = make_data()
    viewed = views.apply_dataset_view(data, mode)
    assert viewed.src.dtype == data.src.dtype
    assert viewed.t.dtype == data.t.dtype
    assert viewed.src.device == data.src.device
    assert viewed.x_src.dtype == data.x_src.dtype
    assert viewed.src.shape[0] == expected_events
    assert viewed.window_id == 7


@pytest.mark.parametrize("mode", ["host_only", "host_network_structure", "host_network_full"])
def test_views_preserve_surviving_temporal_order(mode):
    data = make_data()
    viewed = views.apply_dataset_view(data, mode)
    assert torch.equal(viewed.t, torch.sort(viewed.t).values)
    if mode == "host_only":
        assert viewed.t.tolist() == [10, 50]


def test_netflow_index_is_single_official_theia_mapping():
    config = load_file("c6_b8_mapping_config", SRC_ROOT / "config.py")
    assert config.ntype2id["netflow"] == views.NETFLOW_TYPE_INDEX == 3


def test_full_preserves_extra_event_aligned_fields():
    data = make_data()
    viewed = views.apply_dataset_view(data, "host_network_full")
    assert torch.equal(viewed.some_event_field, data.some_event_field)
    assert viewed.tags == data.tags


def test_load_all_datasets_applies_view_before_full_data(monkeypatch, tmp_path):
    data_utils = load_data_utils_with_stubs(monkeypatch)
    mstc = types.ModuleType("mstc")
    mstc.__path__ = []
    monkeypatch.setitem(sys.modules, "mstc", mstc)
    monkeypatch.setitem(sys.modules, "mstc.dataset_views", views)
    mstc.dataset_views = views
    source = {"train": [make_data(0)], "val": [make_data(100)], "test": [make_data(200)]}
    monkeypatch.setattr(data_utils, "load_data_set", lambda cfg, path, split: [g.clone() for g in source[split]])
    cfg = SimpleNamespace(
        dataset_view=SimpleNamespace(mode="host_only"),
        dataset=SimpleNamespace(name="synthetic"),
        edge_featurization=SimpleNamespace(embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path))),
    )
    train, val, test, full, max_node = data_utils.load_all_datasets(cfg)
    assert [g.src.numel() for g in (train[0], val[0], test[0])] == [2, 2, 2]
    assert torch.equal(full.global_event_index, torch.arange(6))
    assert full.split.tolist() == [0, 0, 1, 1, 2, 2]
    assert max_node == 221

@pytest.mark.parametrize(
    ("mode", "events_per_window"),
    [("host_only", 2), ("host_network_structure", 5), ("host_network_full", 5)],
)
def test_synthetic_train_val_test_smoke_for_each_view(monkeypatch, tmp_path, mode, events_per_window):
    data_utils = load_data_utils_with_stubs(monkeypatch)
    mstc = types.ModuleType("mstc")
    mstc.__path__ = []
    monkeypatch.setitem(sys.modules, "mstc", mstc)
    monkeypatch.setitem(sys.modules, "mstc.dataset_views", views)
    source = {"train": [make_data(0)], "val": [make_data(100)], "test": [make_data(200)]}
    monkeypatch.setattr(data_utils, "load_data_set", lambda cfg, path, split: [g.clone() for g in source[split]])
    cfg = SimpleNamespace(dataset_view=SimpleNamespace(mode=mode), dataset=SimpleNamespace(name="synthetic"), edge_featurization=SimpleNamespace(embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path))))
    train, val, test, full, _ = data_utils.load_all_datasets(cfg)
    assert [g.src.numel() for g in (train[0], val[0], test[0])] == [events_per_window] * 3
    assert torch.equal(full.global_event_index, torch.arange(events_per_window * 3))
