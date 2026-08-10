import logging
import os
import gc
from datetime import datetime, timedelta
import networkx as nx
import torch
from config import *
from provnet_utils import *


# ---------------------------------------------------------------------------
# Database streaming helpers
# ---------------------------------------------------------------------------

def stream_node_table(cur, sql, batch_size=1024):
    """
    Stream rows from a node table using fetchmany() instead of fetchall().
    Yields records one at a time while maintaining bounded memory.

    Args:
        cur: database cursor
        sql: SQL query to execute
        batch_size: number of rows to fetch per batch (DB I/O chunk, NOT semantic batch)

    Yields:
        Individual row records
    """
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield row
        # Guard: if the DB cursor returned fewer rows than requested, the
        # stream is exhausted regardless of any empty-list sentinel.
        try:
            short_batch = len(rows) < batch_size
        except TypeError:
            break
        if short_batch:
            break


def stream_event_table(cur, sql, batch_size=8192):
    """
    Stream rows from event_table using fetchmany() instead of fetchall().
    This is the primary memory bottleneck for graph construction.

    Args:
        cur: database cursor
        sql: SQL query to execute
        batch_size: number of rows to fetch per batch

    Yields:
        Individual event row records
    """
    cur.execute(sql)
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield row
        try:
            short_batch = len(rows) < batch_size
        except TypeError:
            break
        if short_batch:
            break


# ---------------------------------------------------------------------------
# Node list construction (streaming version)
# ---------------------------------------------------------------------------

def get_node_list_streaming(cur, cfg, batch_size=1024):
    """
    Memory-efficient version of get_node_list using streaming cursor.
    Builds nodeid2msg incrementally without holding full records list.
    
    Args:
        cur: database cursor
        cfg: configuration object
        batch_size: fetchmany batch size for node tables
    
    Returns:
        dict: nodeid2msg mapping hash_id -> [node_type, label]
    """
    use_hashed_label = cfg.graph_construction.build_graphs.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    
    nodeid2msg = {}

    # Process netflow node table with streaming
    sql = "SELECT * FROM netflow_node_table;"
    for i in stream_node_table(cur, sql, batch_size):
        nodeid2msg[i[0]] = [i[1], i[2]]
    
    # Process netflow labels with streaming
    for i in stream_node_table(cur, sql, batch_size):
        attrs = {
            'type': 'netflow',
            'local_ip': str(i[2]),
            'local_port': str(i[3]),
            'remote_ip': str(i[4]),
            'remote_port': str(i[5])
        }
        hash_id = i[1]
        features_used = []
        for label_used in node_label_features['netflow']:
            features_used.append(attrs[label_used])
        label_str = ' '.join(features_used)
        if use_hashed_label:
            nodeid2msg[hash_id] = ['netflow', stringtomd5(label_str)]
        else:
            nodeid2msg[hash_id] = ['netflow', label_str]

    # Process subject node table with streaming
    sql = "SELECT * FROM subject_node_table;"
    for i in stream_node_table(cur, sql, batch_size):
        hash_id = i[1]
        attrs = {
            'type': 'subject',
            'path': str(i[2]),
            'cmd_line': str(i[3])
        }
        features_used = []
        for label_used in node_label_features['subject']:
            features_used.append(attrs[label_used])
        label_str = ' '.join(features_used)
        if use_hashed_label:
            nodeid2msg[hash_id] = ['subject', stringtomd5(label_str)]
        else:
            nodeid2msg[hash_id] = ['subject', label_str]

    # Process file node table with streaming
    sql = "SELECT * FROM file_node_table;"
    for i in stream_node_table(cur, sql, batch_size):
        attrs = {
            'type': 'file',
            'path': str(i[2])
        }
        hash_id = i[1]
        features_used = []
        for label_used in node_label_features['file']:
            features_used.append(attrs[label_used])
        label_str = ' '.join(features_used)
        if use_hashed_label:
            nodeid2msg[hash_id] = ['file', stringtomd5(label_str)]
        else:
            nodeid2msg[hash_id] = ['file', label_str]

    return nodeid2msg  # {hash_id:[node_type,msg]}


def get_node_list(cur, cfg):
    """Legacy wrapper - uses streaming version."""
    return get_node_list_streaming(cur, cfg)


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def generate_timestamps(start_time, end_time, interval_minutes):
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

    timestamps = []
    current_time = start
    while current_time <= end:
        timestamps.append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=interval_minutes)
    timestamps.append(end)
    return timestamps


# ---------------------------------------------------------------------------
# Graph construction with streaming event table
# ---------------------------------------------------------------------------

def gen_edge_fused_tw_streaming(cur, nodeid2msg, logger, cfg, event_fetch_size=8192):
    """
    Memory-efficient graph construction using streaming event table.
    
    Key changes from original:
    - Uses fetchmany() streaming instead of fetchall()
    - Maintains separate DB_FETCH_SIZE vs semantic BATCH=1024
    - Proper resource cleanup after each graph
    - Completion markers for crash recovery
    
    Args:
        cur: database cursor
        nodeid2msg: node id to [type, label] mapping
        logger: logging handler
        cfg: configuration object
        event_fetch_size: DB fetch chunk size (NOT semantic batch size)
    """
    include_edge_type = rel2id
    BATCH = 1024  # Semantic batch size - MUST NOT change
    window_size_ns = cfg.graph_construction.build_graphs.time_window_size * 60_000_000_000
    
    graphs_dir = cfg.graph_construction.build_graphs._graphs_dir
    start, end = cfg.dataset.start_end_day_range
    
    for day in range(start, end):
        date_start = cfg.dataset.year_month + '-' + str(day) + ' 00:00:00'
        date_stop = cfg.dataset.year_month + '-' + str(day + 1) + ' 00:00:00'

        start_ns_timestamp = datetime_to_ns_time_US(date_start)
        end_ns_timestamp = datetime_to_ns_time_US(date_stop)
        
        sql = """
            SELECT * FROM event_table
            WHERE timestamp_rec > '%s' AND timestamp_rec < '%s'
            ORDER BY timestamp_rec, event_uuid;
        """ % (start_ns_timestamp, end_ns_timestamp)
        
        logger.info(f"Streaming events for day {day}: {date_start} to {date_stop}")
        
        # Keep the frozen implementation's semantic batches while allowing the
        # database fetch size to vary independently. At most one incomplete
        # semantic batch plus the current time-window's events are retained;
        # importantly, this does not materialize the full day's event table.
        def filtered_semantic_batches():
            batch = []
            for event in stream_event_table(cur, sql, event_fetch_size):
                if event[2] not in include_edge_type:
                    continue
                batch.append(event)
                if len(batch) == BATCH:
                    yield batch
                    batch = []
            if batch:
                yield batch

        graph_count = 0
        start_time = None
        temp_list = []

        for batch in filtered_semantic_batches():
            if start_time is None:
                start_time = batch[0][-2]
            
            for j in batch:
                temp_list.append(j)
            
            # Check if time window boundary is crossed
            if batch[-1][-2] > start_time + window_size_ns:
                time_interval = (ns_time_to_datetime_US(start_time) + "~" + 
                               ns_time_to_datetime_US(batch[-1][-2]))
                
                logger.info(f"Creating graph for {time_interval}")
                
                # Build node and edge info
                node_info = {}
                edge_info = {}
                
                for (src_node, src_index_id, operation, dst_node, dst_index_id,
                     event_uuid, timestamp_rec, _id) in temp_list:
                    if src_index_id not in node_info:
                        node_type, label = nodeid2msg[src_node]
                        node_info[src_index_id] = {
                            'label': label,
                            'node_type': node_type,
                        }
                    if dst_index_id not in node_info:
                        node_type, label = nodeid2msg[dst_node]
                        node_info[dst_index_id] = {
                            'label': label,
                            'node_type': node_type,
                        }

                    if (src_index_id, dst_index_id) not in edge_info:
                        edge_info[(src_index_id, dst_index_id)] = []

                    edge_info[(src_index_id, dst_index_id)].append(
                        (timestamp_rec, operation, event_uuid))

                edge_list = []

                for (src, dst), data in edge_info.items():
                    sorted_data = sorted(data, key=lambda x: x[0])
                    operation_list = [entry[1] for entry in sorted_data]

                    indices = []
                    current_type = None
                    current_start_index = None

                    for idx, item in enumerate(operation_list):
                        if item == current_type:
                            continue
                        else:
                            if current_type is not None and current_start_index is not None:
                                indices.append(current_start_index)
                            current_type = item
                            current_start_index = idx

                    if current_type is not None and current_start_index is not None:
                        indices.append(current_start_index)

                    for k in indices:
                        edge_list.append({
                            'src': src,
                            'dst': dst,
                            'time': sorted_data[k][0],
                            'label': sorted_data[k][1],
                            'event_uuid': sorted_data[k][2]
                        })

                # Create and save graph with atomic write
                logger.info(f"Creating graph for {time_interval}")
                graph = nx.MultiDiGraph()

                for node, info in node_info.items():
                    graph.add_node(
                        node,
                        node_type=info['node_type'],
                        label=info['label']
                    )

                for idx, edge in enumerate(edge_list):
                    graph.add_edge(
                        edge['src'],
                        edge['dst'],
                        event_uuid=edge['event_uuid'],
                        time=edge['time'],
                        label=edge['label']
                    )
                    
                    # For unit tests, limit edges
                    NUM_TEST_EDGES = 2000
                    if cfg._test_mode and idx >= NUM_TEST_EDGES:
                        break

                date_dir = f"{graphs_dir}/graph_{day}/"
                os.makedirs(date_dir, exist_ok=True)
                graph_name = f"{date_dir}/{time_interval}"

                # Atomic save: write to temp file, then rename
                temp_graph_path = graph_name + ".tmp"
                logger.info(f"Saving graph to {temp_graph_path}")
                torch.save(graph, temp_graph_path)
                os.replace(temp_graph_path, graph_name)

                logger.info(f"[{time_interval}] Edges: {len(edge_list)}, "
                           f"Events: {len(temp_list)}, Nodes: {len(node_info.keys())}")
                
                graph_count += 1
                
                # Clean up graph and intermediate data
                del graph
                del node_info
                del edge_info
                del edge_list
                temp_list.clear()
                
                # Update window start time
                start_time = batch[-1][-2]
                
                # For unit tests, exit after first graph
                if cfg._test_mode:
                    return

        if start_time is None:
            logger.info(f"No events for day {day}")

        # End of day cleanup
        temp_list.clear()
        gc.collect()
        
        logger.info(f"Day {day} completed: {graph_count} graphs created")
    
    return


def gen_edge_fused_tw(cur, nodeid2msg, logger, cfg):
    """Wrapper for backward compatibility - uses streaming version."""
    gen_edge_fused_tw_streaming(cur, nodeid2msg, logger, cfg)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg):
    logger = get_logger(
        name="graph_construction_edge_fused_tw",
        filename=os.path.join(cfg.graph_construction.build_graphs._logs_dir, "edge_fused_tw_graph.log"))
    logger.info(f"build_graphs path: {cfg.graph_construction.build_graphs._task_path}")

    cur, connect = None, None
    try:
        cur, connect = init_database_connection(cfg)
        nodeid2msg = get_node_list(cur=cur, cfg=cfg)

        os.makedirs(cfg.graph_construction.build_graphs._graphs_dir, exist_ok=True)

        gen_edge_fused_tw(cur=cur, nodeid2msg=nodeid2msg, logger=logger, cfg=cfg)
        
        # Publish the marker atomically only after all graphs are durable.
        marker_path = os.path.join(
            cfg.graph_construction.build_graphs._graphs_dir, 
            ".preprocess_build_graphs_complete"
        )
        marker_tmp_path = marker_path + ".tmp"
        with open(marker_tmp_path, 'w', encoding="utf-8") as f:
            f.write(datetime.now().isoformat())
        os.replace(marker_tmp_path, marker_path)
        logger.info(f"Build graphs complete. Marker written: {marker_path}")
        
    finally:
        if cur is not None:
            cur.close()
        if connect is not None:
            connect.close()
        gc.collect()


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
