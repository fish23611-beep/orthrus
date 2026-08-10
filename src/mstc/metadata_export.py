"""Production metadata export for restartable preprocessing."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from config import ntype2id, rel2id
from mstc.metadata_cache import MetadataCache, MetadataCacheError


REQUIRED_METADATA = (
    "node_metadata",
    "uuid_to_node_id",
    "node_id_to_uuid",
    "ground_truth_nodes",
    "attack_to_nodes",
    "time_to_malicious_nodes",
    "relation_mapping",
    "dataset_manifest",
    "nodeid2msg",
)


def stream_query(cur, sql, params=None, batch_size=1024):
    """Yield query rows with bounded memory."""
    if params is None:
        cur.execute(sql)
    else:
        cur.execute(sql, params)
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield from rows


def _node_metadata_from_db(cur):
    metadata = {}

    for index_id, node_uuid, path, cmd in stream_query(
        cur, "SELECT index_id, node_uuid, path, cmd FROM subject_node_table;"
    ):
        metadata[int(index_id)] = {
            "uuid": str(node_uuid),
            "type": "subject",
            "path": str(path) if path else None,
            "cmd": str(cmd) if cmd else None,
            "local_ip": None,
            "local_port": None,
            "remote_ip": None,
            "remote_port": None,
            "display": f"subject:{path} {cmd}" if cmd else f"subject:{path}",
        }

    for index_id, node_uuid, path in stream_query(
        cur, "SELECT index_id, node_uuid, path FROM file_node_table;"
    ):
        metadata[int(index_id)] = {
            "uuid": str(node_uuid),
            "type": "file",
            "path": str(path) if path else None,
            "cmd": None,
            "local_ip": None,
            "local_port": None,
            "remote_ip": None,
            "remote_port": None,
            "display": f"file:{path}",
        }

    netflow_sql = (
        "SELECT index_id, node_uuid, src_addr, src_port, dst_addr, dst_port "
        "FROM netflow_node_table;"
    )
    for index_id, node_uuid, src_addr, src_port, dst_addr, dst_port in stream_query(
        cur, netflow_sql
    ):
        metadata[int(index_id)] = {
            "uuid": str(node_uuid),
            "type": "netflow",
            "path": None,
            "cmd": None,
            "local_ip": str(src_addr) if src_addr is not None else None,
            "local_port": str(src_port) if src_port is not None else None,
            "remote_ip": str(dst_addr) if dst_addr is not None else None,
            "remote_port": str(dst_port) if dst_port is not None else None,
            "display": f"netflow:{src_addr}:{src_port}->{dst_addr}:{dst_port}",
        }
    return metadata


def _ground_truth_mappings(cfg, uuid_to_node_id):
    ground_truth_nodes = set()
    attack_to_nodes = {}
    for attack_id, relative_path in enumerate(cfg.dataset.ground_truth_relative_path):
        attack_nodes = set()
        gt_path = Path(cfg._ground_truth_dir) / relative_path
        if gt_path.is_file():
            with gt_path.open("r", encoding="utf-8") as handle:
                for row in csv.reader(handle):
                    if not row:
                        continue
                    node_id = uuid_to_node_id.get(row[0])
                    if node_id is not None:
                        attack_nodes.add(int(node_id))
        attack_to_nodes[attack_id] = attack_nodes
        ground_truth_nodes.update(attack_nodes)
    return ground_truth_nodes, attack_to_nodes


def _time_to_malicious_nodes(cfg, cur, attack_to_nodes, node_id_to_uuid):
    from provnet_utils import datetime_to_ns_time_US

    result = defaultdict(list)
    path_to_attack = {
        path: attack_id
        for attack_id, path in enumerate(cfg.dataset.ground_truth_relative_path)
    }
    sql = (
        "SELECT src_index_id, dst_index_id, timestamp_rec FROM event_table "
        "WHERE timestamp_rec BETWEEN %s AND %s ORDER BY timestamp_rec, event_uuid;"
    )
    for relative_path, start, end in cfg.dataset.attack_to_time_window:
        malicious = attack_to_nodes.get(path_to_attack.get(relative_path, -1), set())
        if not malicious:
            continue
        params = (datetime_to_ns_time_US(start), datetime_to_ns_time_US(end))
        for src_id, dst_id, timestamp in stream_query(cur, sql, params, batch_size=8192):
            for node_id in (int(src_id), int(dst_id)):
                if node_id in malicious and node_id in node_id_to_uuid:
                    result[int(timestamp)].append(node_id_to_uuid[node_id])
    return dict(result)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preprocess_config_hash(cfg):
    return Path(cfg.graph_construction.build_graphs._task_path).name


def update_dataset_manifest(cfg, cache=None):
    """Create/update the manifest; include the model digest once available."""
    cache = cache or MetadataCache(cfg._metadata_dir)
    model_path = (
        Path(cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir)
        / "feature_word2vec.model"
    )
    model_hash = _sha256_file(model_path) if model_path.is_file() else None
    cache.save_dataset_manifest(
        dataset=cfg.dataset.name,
        num_node_types=cfg.dataset.num_node_types,
        num_edge_types=cfg.dataset.num_edge_types,
        train_files=list(cfg.dataset.train_files),
        val_files=list(cfg.dataset.val_files),
        test_files=list(cfg.dataset.test_files),
        word2vec_dim=cfg.edge_featurization.embed_nodes.emb_dim,
        corpus_scope=cfg.semantic_features.corpus_scope,
        preprocess_config_hash=_preprocess_config_hash(cfg),
        word2vec_model_hash=model_hash,
    )


def _write_completion_marker(cache):
    marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
    fd, tmp_path = tempfile.mkstemp(dir=str(cache.cache_root), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            from datetime import datetime, timezone
            handle.write(datetime.now(timezone.utc).isoformat())
        os.replace(tmp_path, marker)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def metadata_complete(cache):
    marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
    return marker.is_file() and not cache.validate_required(list(REQUIRED_METADATA))


def dump_from_postgres(cfg, cache):
    """Export every required cache with streaming reads and atomic writes."""
    from provnet_utils import init_database_connection

    try:
        cur, connection = init_database_connection(cfg)
    except Exception as exc:
        raise MetadataCacheError(
            f"Failed to connect to database for metadata dump: {exc}"
        ) from exc

    try:
        node_metadata = _node_metadata_from_db(cur)
        uuid_to_node_id = {
            meta["uuid"]: int(node_id) for node_id, meta in node_metadata.items()
        }
        node_id_to_uuid = {
            int(node_id): meta["uuid"] for node_id, meta in node_metadata.items()
        }
        ground_truth_nodes, attack_to_nodes = _ground_truth_mappings(
            cfg, uuid_to_node_id
        )
        time_to_nodes = _time_to_malicious_nodes(
            cfg, cur, attack_to_nodes, node_id_to_uuid
        )

        cache.save_node_metadata(node_metadata)
        cache.save_uuid_to_node_id(uuid_to_node_id)
        cache.save_node_id_to_uuid(node_id_to_uuid)
        cache.save_ground_truth_nodes(ground_truth_nodes)
        cache.save_attack_to_nodes(attack_to_nodes)
        cache.save_time_to_malicious_nodes(time_to_nodes)
        cache.save_relation_mapping({
            "relation_to_id": {
                key: value for key, value in rel2id.items() if isinstance(key, str)
            },
            "node_type_to_id": {
                key: value for key, value in ntype2id.items() if isinstance(key, str)
            },
        })
        cache.save_nodeid2msg({
            int(node_id): meta["display"] for node_id, meta in node_metadata.items()
        })
        update_dataset_manifest(cfg, cache)
        _write_completion_marker(cache)
    finally:
        cur.close()
        connection.close()


def export_metadata(cfg, force=False):
    """Idempotently ensure production metadata exists for the configured dataset."""
    cache = MetadataCache(cfg._metadata_dir)
    if metadata_complete(cache) and not force:
        return cache
    dump_from_postgres(cfg, cache)
    if not metadata_complete(cache):
        missing = cache.validate_required(list(REQUIRED_METADATA))
        raise MetadataCacheError(f"Metadata export incomplete; missing: {missing}")
    return cache
