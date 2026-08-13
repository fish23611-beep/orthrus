from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_utils


def _cfg(root: Path, limit=None):
    encoder = SimpleNamespace(
        edge_features="edge_type,msg",
        use_node_type_in_node_feats=True,
    )
    training = SimpleNamespace(
        encoder=encoder,
        decoder=SimpleNamespace(used_methods="custom"),
    )
    return SimpleNamespace(
        _test_mode=False,
        _max_windows_per_split=limit,
        dataset=SimpleNamespace(
            name="synthetic",
            num_node_types=3,
            num_edge_types=4,
        ),
        dataset_view=SimpleNamespace(mode="host_network_full"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(root)),
            embed_nodes=SimpleNamespace(used_method="only_type"),
        ),
        detection=SimpleNamespace(gnn_training=training),
    )


def _raw_window(base: int, events: int = 3) -> TemporalData:
    src = torch.arange(base, base + events, dtype=torch.long)
    dst = src + 10
    t = torch.arange(base * 100, base * 100 + events, dtype=torch.long)
    src_type = torch.nn.functional.one_hot(src.remainder(3), num_classes=3)
    dst_type = torch.nn.functional.one_hot(dst.remainder(3), num_classes=3)
    edge_type = torch.nn.functional.one_hot(
        torch.arange(events).remainder(4), num_classes=4
    )
    return TemporalData(
        src=src,
        dst=dst,
        t=t,
        msg=torch.cat([src_type, edge_type, dst_type], dim=-1).float(),
    )


def _artifacts(root: Path, counts=(3, 2, 4)):
    for split_index, (split, count) in enumerate(zip(("train", "val", "test"), counts)):
        directory = root / split
        directory.mkdir(parents=True)
        for window_index in reversed(range(count)):
            torch.save(
                _raw_window(split_index * 100 + window_index * 10),
                directory / f"window_{window_index:02d}.pt",
            )


def _eager_reference(cfg):
    train = data_utils.load_data_set(cfg, cfg.edge_featurization.embed_edges._edge_embeds_dir, "train")
    val = data_utils.load_data_set(cfg, cfg.edge_featurization.embed_edges._edge_embeds_dir, "val")
    test = data_utils.load_data_set(cfg, cfg.edge_featurization.embed_edges._edge_embeds_dir, "test")
    return data_utils._eager_load_all_datasets(cfg, train, val, test)


def test_full_mode_is_path_backed_and_visits_every_window_in_sorted_order(tmp_path):
    _artifacts(tmp_path)
    cfg = _cfg(tmp_path, limit=None)

    train, val, test, full, _ = data_utils.load_all_datasets(cfg)

    assert isinstance(train, data_utils.LazyTemporalWindowCollection)
    assert [len(train), len(val), len(test)] == [3, 2, 4]
    assert [Path(path).name for path in train.window_paths] == [
        "window_00.pt", "window_01.pt", "window_02.pt"
    ]
    assert full.cached_window_count == 0
    assert [int(g.t[0]) for g in train] == [0, 1000, 2000]


def test_bounded_smoke_limits_each_split_before_any_payload_iteration(tmp_path):
    _artifacts(tmp_path, counts=(4, 4, 4))
    train, val, test, full, _ = data_utils.load_all_datasets(_cfg(tmp_path, limit=2))

    assert [len(train), len(val), len(test)] == [2, 2, 2]
    assert full.num_events == 18
    assert all(Path(path).name in {"window_00.pt", "window_01.pt"}
               for collection in (train, val, test)
               for path in collection.window_paths)


def test_lazy_full_data_matches_eager_reference_for_every_event_field(tmp_path):
    _artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    eager_train, eager_val, eager_test, eager_full, eager_max = _eager_reference(cfg)
    train, val, test, bounded, bounded_max = data_utils.load_all_datasets(cfg)
    event_ids = torch.arange(bounded.num_events)

    assert [len(train), len(val), len(test)] == [
        len(eager_train), len(eager_val), len(eager_test)
    ]
    assert bounded_max == eager_max
    for field in (
        "msg", "t", "edge_type", "src", "dst", "src_type", "dst_type",
        "edge_type_index", "global_event_index", "event_index", "split", "event_split",
    ):
        expected = getattr(eager_full, field)
        actual = bounded.get_event_values(field, event_ids)
        assert torch.equal(actual, expected), field

    expected_offsets = torch.cat(
        [g.global_event_index for g in eager_train + eager_val + eager_test]
    )
    actual_offsets = torch.cat(
        [g.global_event_index for collection in (train, val, test) for g in collection]
    )
    assert torch.equal(actual_offsets, expected_offsets)


def test_collection_releases_yielded_window_and_full_data_cache_is_fixed(tmp_path):
    _artifacts(tmp_path)
    train, _, _, full, _ = data_utils.load_all_datasets(_cfg(tmp_path))

    iterator = iter(train)
    window = next(iterator)
    reference = weakref.ref(window)
    del window
    del iterator
    gc.collect()
    assert reference() is None

    ids_across_windows = torch.tensor([0, 4, full.num_events - 1])
    values = full.get_event_values("t", ids_across_windows)
    assert values.numel() == 3
    assert full.cached_window_count <= 1
    full.release_cache()
    assert full.cached_window_count == 0


def test_loader_does_not_touch_database_and_reports_phase_rss(tmp_path):
    _artifacts(tmp_path, counts=(1, 1, 1))
    cfg = _cfg(tmp_path)

    with patch("provnet_utils.init_database_connection", side_effect=AssertionError("DB access")):
        _, _, _, full, _ = data_utils.load_all_datasets(cfg)

    telemetry = full.loader_telemetry
    assert telemetry["architecture"] == "path_backed_bounded_memory"
    assert telemetry["dataset_loader_peak_rss_mb"] is not None
    assert list(telemetry["rss_mb_by_phase"]) == [
        "before dataset loading",
        "after train index/path resolution",
        "after val index/path resolution",
        "after test index/path resolution",
        "window scan peak",
        "after global metadata construction",
        "before model construction",
    ]

