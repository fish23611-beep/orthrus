"""C8 sparse-node sidecar and true warm-load acceptance tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data_utils


SPARSE_NODE_IDS = (3, 100_000, 5_000_000, 90_000_000)
SRC_ACTIVE = SPARSE_NODE_IDS[:3]
DST_ACTIVE = SPARSE_NODE_IDS[1:]
EMB_DIM = 2
NODE_TYPE_DIM = 3
EDGE_TYPE_DIM = 4
FEATURE_DIM = EMB_DIM + NODE_TYPE_DIM


def _cfg(root: Path) -> SimpleNamespace:
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
        _max_windows_per_split=None,
        dataset=SimpleNamespace(
            name="sparse_synthetic",
            num_node_types=NODE_TYPE_DIM,
            num_edge_types=EDGE_TYPE_DIM,
        ),
        dataset_view=SimpleNamespace(mode="host_network_full"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(root)),
            embed_nodes=SimpleNamespace(
                used_method="feature_word2vec",
                emb_dim=EMB_DIM,
            ),
        ),
        detection=SimpleNamespace(gnn_training=training),
    )


def _role_features(node_ids: tuple[int, ...], *, role_offset: float):
    embeddings = {}
    node_types = {}
    for index, node_id in enumerate(node_ids):
        embeddings[node_id] = torch.tensor(
            [role_offset + index, role_offset + index + 0.25],
            dtype=torch.float32,
        )
        node_types[node_id] = (index + int(role_offset)) % NODE_TYPE_DIM
    return embeddings, node_types


SRC_EMBEDDINGS, SRC_TYPES = _role_features(SRC_ACTIVE, role_offset=1.0)
DST_EMBEDDINGS, DST_TYPES = _role_features(DST_ACTIVE, role_offset=11.0)


def _raw_window(window_index: int, events: int = 20) -> TemporalData:
    # Node identifiers deliberately arrive in non-sorted order with duplicates.
    src = torch.tensor(
        [SRC_ACTIVE[(index * 2 + window_index) % len(SRC_ACTIVE)] for index in range(events)],
        dtype=torch.int64,
    )
    dst = torch.tensor(
        [DST_ACTIVE[(index + 2 * window_index) % len(DST_ACTIVE)] for index in range(events)],
        dtype=torch.int64,
    )
    src_type = torch.nn.functional.one_hot(
        torch.tensor([SRC_TYPES[int(node_id)] for node_id in src]),
        num_classes=NODE_TYPE_DIM,
    ).to(torch.float32)
    dst_type = torch.nn.functional.one_hot(
        torch.tensor([DST_TYPES[int(node_id)] for node_id in dst]),
        num_classes=NODE_TYPE_DIM,
    ).to(torch.float32)
    src_emb = torch.stack([SRC_EMBEDDINGS[int(node_id)] for node_id in src])
    dst_emb = torch.stack([DST_EMBEDDINGS[int(node_id)] for node_id in dst])
    edge_index = torch.arange(events).remainder(EDGE_TYPE_DIM)
    edge_type = torch.nn.functional.one_hot(
        edge_index, num_classes=EDGE_TYPE_DIM
    ).to(torch.float32)
    msg = torch.cat([src_type, src_emb, edge_type, dst_type, dst_emb], dim=-1)
    return TemporalData(
        src=src,
        dst=dst,
        t=torch.arange(events, dtype=torch.int64) + window_index * 1_000,
        msg=msg,
    )


def _empty_raw_window(window_index: int) -> TemporalData:
    msg_dim = 2 * EMB_DIM + 2 * NODE_TYPE_DIM + EDGE_TYPE_DIM
    return TemporalData(
        src=torch.empty(0, dtype=torch.int64),
        dst=torch.empty(0, dtype=torch.int64),
        t=torch.empty(0, dtype=torch.int64),
        msg=torch.empty((0, msg_dim), dtype=torch.float32),
    )


def _write_artifacts(root: Path, *, empty: bool = False) -> None:
    # 3 train + 1 val + 1 test windows = exactly 100 non-empty events.
    global_window = 0
    for split, count in (("train", 3), ("val", 1), ("test", 1)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for split_window in range(count):
            data = (
                _empty_raw_window(global_window)
                if empty else _raw_window(global_window)
            )
            torch.save(data, directory / f"window_{split_window:02d}.pt")
            global_window += 1


def _manifest(root: Path) -> tuple[Path, dict]:
    files = list(
        (root / data_utils._CompactIndex.PERSISTENT_DIR_NAME).glob("manifest__*.json")
    )
    assert len(files) == 1
    return files[0], json.loads(files[0].read_text(encoding="utf-8"))


def _eager_fields(cfg: SimpleNamespace):
    collections = [
        data_utils.load_data_set(
            cfg, cfg.edge_featurization.embed_edges._edge_embeds_dir, split
        )
        for split in ("train", "val", "test")
    ]
    windows = [window for collection in collections for window in collection]
    return {
        field: torch.cat([getattr(window, field) for window in windows])
        for field in ("x_src", "x_dst", "msg", "edge_type")
    }


def test_sparse_node_storage_tracks_active_nodes_and_matches_eager(tmp_path: Path):
    _write_artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    eager = _eager_fields(cfg)

    train, val, test, full, max_node = data_utils.load_all_datasets(cfg)
    assert full.num_events == 100
    assert [len(train), len(val), len(test)] == [3, 1, 1]
    assert max_node == max(SPARSE_NODE_IDS) + 1

    index = full._compact_index
    assert index._src_node_ids.tolist() == sorted(SRC_ACTIVE)
    assert index._dst_node_ids.tolist() == sorted(DST_ACTIVE)
    assert index._src_node_ids.numel() == len(SRC_ACTIVE)
    assert index._dst_node_ids.numel() == len(DST_ACTIVE)
    assert index._src_node_features.shape == (len(SRC_ACTIVE), FEATURE_DIM)
    assert index._dst_node_features.shape == (len(DST_ACTIVE), FEATURE_DIM)

    event_ids = torch.arange(full.num_events)
    assert torch.equal(full.get_event_values("x_src", event_ids), eager["x_src"])
    assert torch.equal(full.get_event_values("x_dst", event_ids), eager["x_dst"])
    assert torch.equal(full.get_event_values("msg", event_ids), eager["msg"])
    assert torch.equal(full.get_event_values("edge_type", event_ids), eager["edge_type"])

    expected_feature_bytes = (
        (len(SRC_ACTIVE) + len(DST_ACTIVE)) * FEATURE_DIM * 4
    )
    expected_id_bytes = (len(SRC_ACTIVE) + len(DST_ACTIVE)) * 8
    telemetry = index.telemetry
    assert telemetry["node_feature_storage_bytes"] == expected_feature_bytes
    assert telemetry["node_id_index_bytes"] == expected_id_bytes
    assert telemetry["node_table_bytes"] == expected_feature_bytes + expected_id_bytes
    assert telemetry["num_src_active_nodes"] == len(SRC_ACTIVE)
    assert telemetry["num_dst_active_nodes"] == len(DST_ACTIVE)

    # A src-only and a dst-only node prove role tables are not silently unified.
    assert SPARSE_NODE_IDS[0] not in index._dst_node_ids.tolist()
    assert SPARSE_NODE_IDS[-1] not in index._src_node_ids.tolist()
    with pytest.raises(IndexError, match="absent from sparse x_src"):
        index._lookup_node_features("x_src", torch.tensor([SPARSE_NODE_IDS[-1]]))

    _, manifest = _manifest(tmp_path)
    assert manifest["schema_version"] == 4
    assert manifest["num_src_active_nodes"] == len(SRC_ACTIVE)
    assert manifest["num_dst_active_nodes"] == len(DST_ACTIVE)
    assert manifest["node_feature_storage_bytes"] == expected_feature_bytes


def test_empty_events_keep_feature_shape_without_node_storage(tmp_path: Path):
    _write_artifacts(tmp_path, empty=True)
    cfg = _cfg(tmp_path)
    _, _, _, full, max_node = data_utils.load_all_datasets(cfg)
    index = full._compact_index

    assert full.num_events == 0
    assert max_node == 0
    assert index._src_node_ids.numel() == 0
    assert index._dst_node_ids.numel() == 0
    assert index._src_node_features.shape == (0, FEATURE_DIM)
    assert index._dst_node_features.shape == (0, FEATURE_DIM)
    assert full.get_event_values("x_src", torch.empty(0, dtype=torch.long)).shape == (
        0, FEATURE_DIM
    )
    assert index.telemetry["node_feature_storage_bytes"] == 0


def test_valid_warm_sidecar_never_calls_source_artifact_loader(tmp_path: Path):
    _write_artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    _, _, _, cold, _ = data_utils.load_all_datasets(cfg)
    cold_specs = [
        (
            spec.path, spec.split_name, spec.split_window_index,
            spec.global_window_id, spec.global_offset, spec.num_events,
        )
        for spec in cold._specs
    ]

    with patch.object(
        data_utils,
        "load_trusted_torch_artifact",
        side_effect=AssertionError("warm path loaded source TemporalData"),
    ):
        train, val, test, warm, _ = data_utils.load_all_datasets(cfg)

    warm_specs = [
        (
            spec.path, spec.split_name, spec.split_window_index,
            spec.global_window_id, spec.global_offset, spec.num_events,
        )
        for spec in warm._specs
    ]
    assert warm_specs == cold_specs
    assert [spec.split_name for spec in train._specs] == ["train"] * 3
    assert [spec.split_name for spec in val._specs] == ["val"]
    assert [spec.split_name for spec in test._specs] == ["test"]
    telemetry = warm._compact_index.telemetry
    assert telemetry["sidecar_manifest_hit"] is True
    assert telemetry["persistent_cache_hit"] is True
    assert telemetry["sidecar_tensor_load_count"] == 2
    assert telemetry["metadata_source_load_count"] == 0
    assert telemetry["compact_build_source_load_count"] == 0
    assert telemetry["total_source_artifact_load_count"] == 0
    assert telemetry["warm_source_temporaldata_load_count"] == 0


def test_schema_v3_manifest_automatically_misses_and_rebuilds(tmp_path: Path):
    _write_artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    data_utils.load_all_datasets(cfg)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["schema_version"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    original_loader = data_utils.load_trusted_torch_artifact
    with patch.object(
        data_utils, "load_trusted_torch_artifact", wraps=original_loader
    ) as source_loader:
        _, _, _, rebuilt, _ = data_utils.load_all_datasets(cfg)
    telemetry = rebuilt._compact_index.telemetry
    assert telemetry["persistent_cache_hit"] is False
    assert telemetry["sidecar_manifest_hit"] is False
    assert source_loader.call_count == 10  # 5 metadata + 5 compact-build loads
    assert telemetry["cold_source_temporaldata_load_count"] == 10
    assert _manifest(tmp_path)[1]["schema_version"] == 4


@pytest.mark.parametrize("mutation", ["mtime", "size", "path"])
def test_source_identity_changes_invalidate_sidecar(tmp_path: Path, mutation: str):
    _write_artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    data_utils.load_all_datasets(cfg)
    source = tmp_path / "train" / "window_00.pt"
    if mutation == "mtime":
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    elif mutation == "size":
        raw = data_utils.load_trusted_torch_artifact(
            str(source), expected_type=TemporalData
        )
        raw.identity_padding = torch.arange(17)
        torch.save(raw, source)
    else:
        source.rename(source.with_name("renamed_window_00.pt"))

    _, _, _, rebuilt, _ = data_utils.load_all_datasets(cfg)
    telemetry = rebuilt._compact_index.telemetry
    assert telemetry["persistent_cache_hit"] is False
    assert telemetry["sidecar_manifest_hit"] is False
    assert telemetry["cold_source_temporaldata_load_count"] == 10


def test_incomplete_manifest_is_not_a_completed_sidecar(tmp_path: Path):
    _write_artifacts(tmp_path)
    cfg = _cfg(tmp_path)
    data_utils.load_all_datasets(cfg)
    manifest_path, manifest = _manifest(tmp_path)
    manifest["completed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, _, _, rebuilt, _ = data_utils.load_all_datasets(cfg)
    telemetry = rebuilt._compact_index.telemetry
    assert telemetry["persistent_cache_hit"] is False
    assert telemetry["sidecar_manifest_hit"] is False
    assert telemetry["cold_source_temporaldata_load_count"] == 10
    assert _manifest(tmp_path)[1]["completed"] is True
