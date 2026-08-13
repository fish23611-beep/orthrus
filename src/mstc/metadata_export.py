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
    """Extract ground truth node mappings from configured CSV files.

    Raises
    ------
    MetadataCacheError
        - When a configured ground truth CSV does not exist.
        - When a CSV has no valid rows.
        - When zero UUIDs from the CSV match the uuid_to_node_id mapping.
    """
    ground_truth_nodes = set()
    attack_to_nodes = {}

    for attack_id, relative_path in enumerate(cfg.dataset.ground_truth_relative_path):
        gt_path = Path(cfg._ground_truth_dir) / relative_path

        # C8: Fail closed — missing GT file is a hard error.
        if not gt_path.is_file():
            raise MetadataCacheError(
                f"Ground truth CSV not found: {gt_path}. "
                f"Ensure ground truth root resolves correctly and submodule is initialized."
            )

        attack_nodes = set()
        total_rows = 0
        matched_rows = 0

        with gt_path.open("r", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                total_rows += 1
                if not row:
                    continue
                node_id = uuid_to_node_id.get(row[0])
                if node_id is not None:
                    attack_nodes.add(int(node_id))
                    matched_rows += 1

        # C8: Fail closed — empty CSV or zero matched UUIDs.
        if total_rows == 0:
            raise MetadataCacheError(
                f"Ground truth CSV has no data rows: {gt_path}"
            )

        if matched_rows == 0:
            raise MetadataCacheError(
                f"Ground truth CSV has {total_rows} rows but 0 UUIDs matched "
                f"the uuid_to_node_id cache. File: {gt_path}"
            )

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


def metadata_complete(cache, cfg=None):
    """Check if metadata export is complete and semantically valid.

    Returns ``bool``. Diagnostic detail (which file/cache is missing or empty)
    is intentionally NOT exposed here to keep this function safe against
    accidental ``if metadata_complete(cache):`` truthiness checks on the
    legacy ``(bool, str)`` tuple shape. Use :func:`metadata_validation_status`
    when a human-readable reason is required (logging / error messages).

    A cache is considered complete only when:

    - the completion marker exists;
    - every required file is present;
    - :class:`MetadataCache` payloads (``uuid_to_node_id``,
      ``ground_truth_nodes``, ``attack_to_nodes``) are non-empty;
    - when ``cfg.dataset.attack_to_time_window`` is non-empty (i.e. the
      dataset declares a ground-truth attack timeline), the
      ``time_to_malicious_nodes`` cache must also be non-empty. Empty
      time cache while attacks are declared is the THEIA_E3 / CADETS_E5
      family symptom that must NOT be reported as complete.

    Parameters
    ----------
    cache : MetadataCache
    cfg : optional
        When provided, the function uses ``cfg.dataset.attack_to_time_window``
        and ``cfg.dataset.ground_truth_relative_path`` to decide whether the
        time cache emptiness is a hard contract violation. When ``cfg`` is
        ``None``, the manifest's ``dataset`` name is consulted through the
        built-in ``DATASET_HAS_TIME_WINDOW`` registry (best-effort). If both
        lookups are unavailable the function falls back to the legacy
        "complete when files exist" semantics, which matches the v1 contract
        used by pre-C8 callers.
    """
    is_complete, _detail = metadata_validation_status(cache, cfg=cfg)
    return is_complete


def metadata_validation_status(cache, cfg=None):
    """Return ``(is_complete, detail_message)`` for metadata export.

    This is the diagnostic twin of :func:`metadata_complete`. Callers MUST
    unpack the tuple explicitly (``ok, detail = ...``) and MUST NOT use it
    directly in boolean contexts. See :func:`metadata_complete` for the
    safe-by-construction bool wrapper.

    ``detail_message`` values:

    - ``"complete"``                     – all required files exist and pass
      semantic validation;
    - ``"missing: completion marker"``   – marker file absent;
    - ``"missing: [...]"``               – one or more required cache files
      are missing;
    - ``"empty: uuid_to_node_id"``       – uuid map is empty while required
      cache files exist (export incomplete);
    - ``"empty: ground_truth"``          – ``ground_truth_nodes`` is empty
      while ground truth is configured;
    - ``"empty: attack_to_nodes"``       – ``attack_to_nodes`` is empty /
      every attack has zero nodes;
    - ``"empty: time_to_malicious_nodes"`` – the configured dataset has
      ``attack_to_time_window`` declarations but the timestamp -> UUID
      cache is empty (THEIA_E3 / CADETS_E5 symptom);
    - ``"corrupt: <file> (<reason>)"``   – a cache file is present but
      cannot be loaded (pickle / JSON parse failure).
    """
    marker = cache.cache_root / cache.COMPLETION_MARKER_FILE
    if not marker.is_file():
        return False, "missing: completion marker"

    missing = cache.validate_required(list(REQUIRED_METADATA))
    if missing:
        return False, f"missing: {missing}"

    gt_declared = _dataset_declares_ground_truth(cache, cfg)
    time_window_declared = _dataset_declares_time_window(cache, cfg)

    try:
        uuid_map = cache.load_uuid_to_node_id()
        gt_nodes = cache.load_ground_truth_nodes()
        attack_map = cache.load_attack_to_nodes()
        time_to_nodes = cache.load_time_to_malicious_nodes()
    except MetadataCacheError as exc:
        return False, f"corrupt: {exc}"

    if not uuid_map:
        return False, "empty: uuid_to_node_id"

    # GT semantic checks only apply when the dataset declares ground truth
    # (legacy caches without ``ground_truth_relative_path`` are exempt).
    if gt_declared:
        if not gt_nodes:
            return False, "empty: ground_truth"
        if not attack_map or all(not nodes for nodes in attack_map.values()):
            return False, "empty: attack_to_nodes"

    # THEIA_E3 / CADETS_E5 contract: when the dataset declares
    # attack_to_time_window, an empty time_to_malicious_nodes cache is the
    # exact symptom of the C8 ground-truth metadata export bug. We refuse to
    # certify the cache as complete in that state.
    if time_window_declared and not time_to_nodes:
        return False, "empty: time_to_malicious_nodes"

    return True, "complete"


_DATASET_HAS_TIME_WINDOW = frozenset({
    "THEIA_E3",
    "THEIA_E5",
    "CADETS_E5",
    "CADETS_E3",
    "E3-THEIA",
    "E5-THEIA",
    "E5-CADETS",
})


def _dataset_declares_ground_truth(cache, cfg):
    """Return ``True`` when the active dataset declares ground truth CSVs.

    Resolution order:

    1. ``cfg.dataset.ground_truth_relative_path`` non-empty (preferred).
    2. ``cache.load_dataset_manifest()["dataset"]`` matched against
       the well-known DARPA dataset names that always declare GT.
    """
    if cfg is not None:
        dataset = getattr(cfg, "dataset", None)
        if dataset is not None:
            gt_rel = getattr(dataset, "ground_truth_relative_path", None)
            if gt_rel:
                return True
            # No GT configured: legacy cache.
            return False

    try:
        manifest = cache.load_dataset_manifest()
    except MetadataCacheError:
        return False
    dataset_name = manifest.get("dataset") if isinstance(manifest, dict) else None
    return dataset_name in _DATASET_HAS_TIME_WINDOW


def _dataset_declares_time_window(cache, cfg):
    """Return ``True`` when the active dataset is known to declare an
    ``attack_to_time_window`` block.

    Resolution order:

    1. ``cfg.dataset.attack_to_time_window`` (preferred; works for both
       production runs and tests).
    2. ``cfg.dataset.ground_truth_relative_path`` – if non-empty AND no
       time window attribute is present we still treat the dataset as
       configured (THEIA_E3 has both, but defensive callers might strip
       one).
    3. ``cache.load_dataset_manifest()["dataset"]`` matched against
       :data:`_DATASET_HAS_TIME_WINDOW`.
    """
    if cfg is not None:
        dataset = getattr(cfg, "dataset", None)
        if dataset is not None:
            time_window = getattr(dataset, "attack_to_time_window", None)
            if time_window:
                return True
            gt_rel = getattr(dataset, "ground_truth_relative_path", None)
            if not gt_rel:
                # No GT declarations at all: legacy cache has no time
                # contract obligation.
                return False
            name = getattr(dataset, "name", None)
            if name in _DATASET_HAS_TIME_WINDOW:
                return True

    try:
        manifest = cache.load_dataset_manifest()
    except MetadataCacheError:
        return False
    dataset_name = manifest.get("dataset") if isinstance(manifest, dict) else None
    return dataset_name in _DATASET_HAS_TIME_WINDOW


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
    """Idempotently ensure production metadata exists for the configured dataset.

    This is the metadata-only refresh entry point. It deliberately touches
    only the metadata cache (PostgreSQL → disk) and never re-runs:

    - ``build_graphs`` (graph construction)
    - ``embed_nodes``  (Word2Vec)
    - ``embed_edges``  (edge embeddings)

    Pre-existing graph artifacts, the Word2Vec model, and the edge embeddings
    are read-only inputs to the metadata refresh. If PostgreSQL is required
    (which is currently the case), only a database restore + this function
    call is needed; the user does not need to rerun ``--stages preprocess``.

    When ``force=False`` (the default), the function short-circuits if the
    metadata cache is already complete. When ``force=True``, the cache is
    unconditionally re-exported.
    """
    cache = MetadataCache(cfg._metadata_dir)
    is_complete, detail = metadata_validation_status(cache, cfg=cfg)
    if is_complete and not force:
        return cache
    dump_from_postgres(cfg, cache)
    is_complete, detail = metadata_validation_status(cache, cfg=cfg)
    if not is_complete:
        raise MetadataCacheError(
            f"Metadata export incomplete; reason: {detail}. "
            f"Call export_metadata(cfg, force=True) to retry, "
            f"or restore the PostgreSQL dump first."
        )
    return cache
