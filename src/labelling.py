from collections import defaultdict

from config import *
from provnet_utils import *


def _get_metadata_cache(cfg):
    """
    Get or create MetadataCache instance from cfg.

    Returns None if cfg._metadata_dir is not set.
    """
    if not hasattr(cfg, "_metadata_dir") or not cfg._metadata_dir:
        return None
    from mstc.metadata_cache import MetadataCache
    return MetadataCache(cfg._metadata_dir)


def _is_detection_only_mode(cfg) -> bool:
    """Check if pipeline mode is detection_only."""
    return (
        hasattr(cfg, "pipeline")
        and hasattr(cfg.pipeline, "mode")
        and cfg.pipeline.mode == "detection_only"
    )


def get_ground_truth(cfg):
    """
    Get ground truth nodes, paths, and UUID mapping.

    Cache-first strategy:
    - If cache hit: return cached data (no DB connection)
    - If cache miss + full_pipeline: fall back to DB
    - If cache miss + detection_only: raise clear error
    """
    cache = _get_metadata_cache(cfg)

    # Try cache first
    if cache is not None and cache.has_ground_truth_nodes():
        try:
            ground_truth_nids = cache.load_ground_truth_nodes()
            # Load other related caches if available
            uuid_to_node_id = {}
            if cache.has_uuid_to_node_id():
                uuid_to_node_id = cache.load_uuid_to_node_id()
            ground_truth_paths = {}
            # Return with cache data
            return ground_truth_nids, ground_truth_paths, uuid_to_node_id
        except Exception:
            pass  # Fall through to DB path

    # Cache miss - check mode
    if _is_detection_only_mode(cfg):
        if cache is not None:
            missing = cache.validate_required([
                "ground_truth_nodes", "uuid_to_node_id"
            ])
            raise FileNotFoundError(
                f"detection_only mode: required caches missing: {missing}. "
                f"Run in full_pipeline mode or provide cached metadata."
            )
        else:
            raise FileNotFoundError(
                "detection_only mode: no metadata cache configured and no DB fallback allowed."
            )

    # full_pipeline: use DB
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    for file in cfg.dataset.ground_truth_relative_path:
        with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
                uuid_to_node_id[node_uuid] = str(node_id)
    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id


def get_GP_of_each_attack(cfg):
    """
    Get ground truth nodes for each attack.

    Cache-first strategy:
    - If cache hit: return cached data (no DB connection)
    - If cache miss + full_pipeline: fall back to DB
    - If cache miss + detection_only: raise clear error
    """
    cache = _get_metadata_cache(cfg)

    # Try cache first
    if cache is not None and cache.has_attack_to_nodes():
        try:
            attack_to_nids = cache.load_attack_to_nodes()
            return attack_to_nids
        except Exception:
            pass  # Fall through to DB path

    # Cache miss - check mode
    if _is_detection_only_mode(cfg):
        if cache is not None:
            missing = cache.validate_required(["attack_to_nodes"])
            raise FileNotFoundError(
                f"detection_only mode: required caches missing: {missing}. "
                f"Run in full_pipeline mode or provide cached metadata."
            )
        else:
            raise FileNotFoundError(
                "detection_only mode: no metadata cache configured and no DB fallback allowed."
            )

    # full_pipeline: use DB
    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_to_nids = {}

    for i, file in enumerate(cfg.dataset.ground_truth_relative_path):
        attack_to_nids[i] = set()
        with open(os.path.join(cfg._ground_truth_dir, file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                attack_to_nids[i].add(int(node_id))
    return attack_to_nids


def get_uuid2nids(cur):
    queries = {
        "file": "SELECT index_id, node_uuid FROM file_node_table;",
        "netflow": "SELECT index_id, node_uuid FROM netflow_node_table;",
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;"
    }
    uuid2nids = {}
    nid2uuid = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            uuid2nids[row[1]] = row[0]
            nid2uuid[row[0]] = row[1]

    return uuid2nids, nid2uuid


def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(timeArray))
    timestamp = tz.localize(dt)
    timestamp = timestamp.timestamp()
    timeStamp = timestamp * 1000000000
    return int(timeStamp)

def get_events(cur,
               start_time,
               end_time,):
    # malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    # sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str});"
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"

    cur.execute(sql)
    rows = cur.fetchall()
    return rows

def get_t2malicious_node(cfg) -> dict[list]:
    """
    Get timestamp to malicious node UUIDs mapping.

    Cache-first strategy:
    - If cache hit: return cached data (no DB connection)
    - If cache miss + full_pipeline: fall back to DB
    - If cache miss + detection_only: raise clear error
    """
    cache = _get_metadata_cache(cfg)

    # Try cache first
    if cache is not None and cache.has_time_to_malicious_nodes():
        try:
            t_to_node = cache.load_time_to_malicious_nodes()
            return t_to_node  # An empty mapping is a valid exported cache.
        except Exception:
            pass  # Fall through to DB path

    # Cache miss - check mode
    if _is_detection_only_mode(cfg):
        if cache is not None:
            missing = cache.validate_required(["time_to_malicious_nodes"])
            raise FileNotFoundError(
                f"detection_only mode: required caches missing: {missing}. "
                f"Run in full_pipeline mode or provide cached metadata."
            )
        else:
            raise FileNotFoundError(
                "detection_only mode: no metadata cache configured and no DB fallback allowed."
            )

    # full_pipeline: use DB
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, attack), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))

            rows = get_events(cur, start_time, end_time)
            for row in rows:
                src_id = row[1]
                dst_id = row[4]
                t = row[6]
                if src_id in ground_truth_nids:
                    t_to_node[int(t)].append(nid2uuid[int(src_id)])
                if dst_id in ground_truth_nids:
                    t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    return t_to_node
