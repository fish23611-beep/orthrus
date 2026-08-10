#!/usr/bin/env python3
"""Synthetic RSS benchmark for C8.1 graph construction.

The coordinator runs the frozen materialized strategy and the production
streaming strategy in separate subprocesses so ru_maxrss values are directly
comparable. Graph serialization is replaced by an in-process sink; graph
construction and edge-fusion semantics remain active.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


class SyntheticEventCursor:
    def __init__(self, event_count, operation, first_timestamp):
        self.event_count = event_count
        self.operation = operation
        self.first_timestamp = first_timestamp
        self.offset = 0

    def execute(self, _sql):
        self.offset = 0

    def _row(self, index):
        return (
            "src-hash",
            1,
            self.operation,
            "dst-hash",
            2,
            f"synthetic-event-{index:09d}",
            self.first_timestamp + index * 1_000_000_000,
            index,
        )

    def fetchmany(self, size):
        stop = min(self.offset + size, self.event_count)
        rows = [self._row(index) for index in range(self.offset, stop)]
        self.offset = stop
        return rows

    def fetchall(self):
        rows = [self._row(index) for index in range(self.event_count)]
        self.offset = self.event_count
        return rows


class QuietLogger:
    def info(self, _message):
        pass


class GraphSink:
    def __init__(self):
        self.graph_count = 0
        self.edge_count = 0
        self.timestamp_checksum = 0

    def consume(self, graph, _path=None):
        self.graph_count += 1
        edges = list(graph.edges(data=True, keys=True))
        self.edge_count += len(edges)
        self.timestamp_checksum += sum(int(attrs["time"]) for _, _, _, attrs in edges)

    def result(self):
        return {
            "graph_count": self.graph_count,
            "edge_count": self.edge_count,
            "timestamp_checksum": self.timestamp_checksum,
        }


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def build_graph(module, events, nodeid2msg):
    node_info = {}
    edge_info = {}
    for (src_node, src_index_id, operation, dst_node, dst_index_id,
         event_uuid, timestamp_rec, _id) in events:
        if src_index_id not in node_info:
            node_type, label = nodeid2msg[src_node]
            node_info[src_index_id] = {"label": label, "node_type": node_type}
        if dst_index_id not in node_info:
            node_type, label = nodeid2msg[dst_node]
            node_info[dst_index_id] = {"label": label, "node_type": node_type}
        edge_info.setdefault((src_index_id, dst_index_id), []).append(
            (timestamp_rec, operation, event_uuid)
        )

    graph = module.nx.MultiDiGraph()
    for node, info in node_info.items():
        graph.add_node(node, **info)
    for (src, dst), data in edge_info.items():
        sorted_data = sorted(data, key=lambda item: item[0])
        current_type = None
        for timestamp, operation, event_uuid in sorted_data:
            if operation == current_type:
                continue
            current_type = operation
            graph.add_edge(
                src,
                dst,
                event_uuid=event_uuid,
                time=timestamp,
                label=operation,
            )
    return graph


def run_materialized(module, cursor, nodeid2msg, cfg, sink):
    include_edge_type = module.rel2id
    window_size_ns = cfg.graph_construction.build_graphs.time_window_size * 60_000_000_000
    cursor.execute("SELECT * FROM event_table ORDER BY timestamp_rec, event_uuid")
    events = cursor.fetchall()
    events_list = [event for event in events if event[2] in include_edge_type]
    if not events_list:
        return

    start_time = events_list[0][-2]
    current_window = []
    for offset in range(0, len(events_list), 1024):
        semantic_batch = events_list[offset:offset + 1024]
        current_window.extend(semantic_batch)
        if semantic_batch[-1][-2] <= start_time + window_size_ns:
            continue
        sink.consume(build_graph(module, current_window, nodeid2msg))
        start_time = semantic_batch[-1][-2]
        current_window.clear()


def run_worker(strategy, event_count, fetch_size):
    sys.path.insert(0, str(SRC_DIR))
    from graph_construction import build_orthrus_graphs as module

    operation = next(key for key in module.rel2id if isinstance(key, str))
    first_timestamp = module.datetime_to_ns_time_US("2019-05-08 00:00:01")
    cursor = SyntheticEventCursor(event_count, operation, first_timestamp)
    nodeid2msg = {
        "src-hash": ["subject", "/usr/bin/python"],
        "dst-hash": ["file", "/tmp/output"],
    }
    sink = GraphSink()

    with tempfile.TemporaryDirectory(prefix="c8-memory-benchmark-") as graphs_dir:
        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(
                    time_window_size=15.0,
                    _graphs_dir=graphs_dir,
                )
            ),
            dataset=SimpleNamespace(
                start_end_day_range=(8, 9),
                year_month="2019-05",
            ),
            _test_mode=False,
        )
        started = time.perf_counter()
        if strategy == "old":
            run_materialized(module, cursor, nodeid2msg, cfg, sink)
        else:
            original_save = module.torch.save
            original_replace = module.os.replace
            module.torch.save = sink.consume
            module.os.replace = lambda _source, _destination: None
            try:
                module.gen_edge_fused_tw_streaming(
                    cursor,
                    nodeid2msg,
                    QuietLogger(),
                    cfg,
                    event_fetch_size=fetch_size,
                )
            finally:
                module.torch.save = original_save
                module.os.replace = original_replace
        duration = time.perf_counter() - started

    result = {
        "strategy": strategy,
        "event_count": event_count,
        "fetch_size": fetch_size,
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "duration_seconds": round(duration, 3),
    }
    result.update(sink.result())
    return result


def run_subprocess(strategy, event_count, fetch_size):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        strategy,
        "--events",
        str(event_count),
        "--fetch-size",
        str(fetch_size),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=200_000)
    parser.add_argument("--fetch-size", type=int, default=8192)
    parser.add_argument("--worker", choices=("old", "new"))
    args = parser.parse_args()

    if args.events <= 0 or args.fetch_size <= 0:
        parser.error("--events and --fetch-size must be positive")
    if args.worker:
        print(json.dumps(run_worker(args.worker, args.events, args.fetch_size)))
        return

    old = run_subprocess("old", args.events, args.fetch_size)
    new = run_subprocess("new", args.events, args.fetch_size)
    semantic_keys = ("graph_count", "edge_count", "timestamp_checksum")
    if any(old[key] != new[key] for key in semantic_keys):
        raise RuntimeError(f"old/new semantic mismatch: old={old}, new={new}")

    reduction_mb = old["peak_rss_mb"] - new["peak_rss_mb"]
    reduction_percent = reduction_mb / old["peak_rss_mb"] * 100
    report = {
        "scope": "Graph Construction synthetic RSS benchmark",
        "event_count": args.events,
        "fetch_size": args.fetch_size,
        "old_peak_rss_mb": old["peak_rss_mb"],
        "new_peak_rss_mb": new["peak_rss_mb"],
        "reduction_mb": round(reduction_mb, 2),
        "reduction_percent": round(reduction_percent, 2),
        "old_duration_seconds": old["duration_seconds"],
        "new_duration_seconds": new["duration_seconds"],
        "semantic_result": {key: old[key] for key in semantic_keys},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
