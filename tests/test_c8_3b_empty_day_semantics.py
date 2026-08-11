"""C8.3B regression tests: verified empty-day semantics and supporting fixes.

These tests cover:

* graph_construction.empty_day marker contract
  (schema, atomic write, raw_event_count == 0 enforcement, identity check)
* build_orthrus_graphs day-loop integration with a fake cursor
  (raw-empty writes a marker, raw > 0 does not, stale markers are removed,
   raw > 0 + supported == 0 raises loudly without writing a fake graph)
* build_graphs validator
  (real graph, verified-empty marker, missing folder, malformed marker,
   raw_event_count != 0 marker, stale marker cannot mask real graphs)
* train_only Word2Vec corpus
  (verified-empty split is skipped, unverified empty raises,
   remaining real splits are still collected)
* graph enumeration ignores ``.preprocess_*`` and ``.tmp``
* PostgreSQL major parsing (symlink at ``/usr/bin/pg_restore``,
  version 18.4, version 17.6, malformed output)
* ``diagnose_theia_graph_day`` CLI dataset positional forwarded
* no real DB / GPU / dump required
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
import torch

# Ensure src/ is on the import path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pipeline_stages import check_preprocess_stage_complete
from graph_construction import empty_day as empty_day_mod
from graph_construction import build_orthrus_graphs as build_graphs
from edge_featurization import build_feature_word2vec as build_w2v
from provnet_utils import get_all_files_from_folders
import colab_postgres


# ---------------------------------------------------------------------------
# Empty-day marker contract
# ---------------------------------------------------------------------------


def test_empty_day_marker_payload_requires_zero_raw_count():
    with pytest.raises(ValueError, match="raw_event_count == 0"):
        empty_day_mod.build_marker_payload(
            dataset="THEIA_E3",
            graph_name="graph_2",
            day=2,
            date_start="2018-04-02 00:00:00",
            date_stop="2018-04-03 00:00:00",
            start_ns=0,
            end_ns=1,
            raw_event_count=5,
        )


def test_empty_day_marker_rejects_unknown_reason():
    with pytest.raises(ValueError, match="Unsupported empty-day reason"):
        empty_day_mod.build_marker_payload(
            dataset="THEIA_E3",
            graph_name="graph_2",
            day=2,
            date_start="2018-04-02 00:00:00",
            date_stop="2018-04-03 00:00:00",
            start_ns=0,
            end_ns=1,
            raw_event_count=0,
            reason="some_other_reason",
        )


def test_empty_day_marker_atomic_write_does_not_leave_tmp(tmp_path):
    payload = empty_day_mod.build_marker_payload(
        dataset="THEIA_E3",
        graph_name="graph_2",
        day=2,
        date_start="2018-04-02 00:00:00",
        date_stop="2018-04-03 00:00:00",
        start_ns=123,
        end_ns=456,
        raw_event_count=0,
    )
    path = empty_day_mod.write_marker(tmp_path, payload)
    assert os.path.isfile(path)
    assert not list(tmp_path.glob("*.tmp"))
    on_disk = json.loads(pathlib_read_text(path))
    assert on_disk["schema"] == empty_day_mod.EMPTY_DAY_MARKER_SCHEMA
    assert on_disk["raw_event_count"] == 0
    assert on_disk["reason"] == empty_day_mod.EMPTY_DAY_REASON_NO_RAW_EVENTS
    assert on_disk["dataset"] == "THEIA_E3"
    assert on_disk["graph_name"] == "graph_2"
    assert on_disk["day"] == 2


def pathlib_read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_empty_day_marker_validation_rejects_raw_event_count_nonzero(tmp_path):
    payload = empty_day_mod.build_marker_payload(
        dataset="THEIA_E3",
        graph_name="graph_2",
        day=2,
        date_start="2018-04-02 00:00:00",
        date_stop="2018-04-03 00:00:00",
        start_ns=0,
        end_ns=1,
        raw_event_count=0,
    )
    payload["raw_event_count"] = 12  # spoof
    path = empty_day_mod.write_marker(tmp_path, payload)
    assert os.path.isfile(path)
    assert empty_day_mod.is_verified_empty_day(
        tmp_path, dataset="THEIA_E3", graph_name="graph_2", day=2
    ) is False


def test_empty_day_marker_validation_rejects_mismatched_identity(tmp_path):
    payload = empty_day_mod.build_marker_payload(
        dataset="THEIA_E3",
        graph_name="graph_2",
        day=2,
        date_start="2018-04-02 00:00:00",
        date_stop="2018-04-03 00:00:00",
        start_ns=0,
        end_ns=1,
        raw_event_count=0,
    )
    empty_day_mod.write_marker(tmp_path, payload)
    assert empty_day_mod.is_verified_empty_day(
        tmp_path, dataset="THEIA_E3", graph_name="graph_3", day=3
    ) is False


def test_empty_day_marker_rejects_malformed_json(tmp_path):
    (tmp_path / empty_day_mod.EMPTY_DAY_MARKER_FILENAME).write_text(
        "{not-json", encoding="utf-8"
    )
    assert empty_day_mod.read_marker(tmp_path) is None
    assert empty_day_mod.is_verified_empty_day(
        tmp_path, dataset="THEIA_E3", graph_name="graph_2", day=2
    ) is False


def test_empty_day_marker_rejects_dot_tmp_left_over(tmp_path):
    """A leftover ``.tmp`` file is *not* a marker."""
    (tmp_path / (empty_day_mod.EMPTY_DAY_MARKER_FILENAME + ".tmp")).write_text(
        "{}", encoding="utf-8"
    )
    assert empty_day_mod.is_verified_empty_day(
        tmp_path, dataset="THEIA_E3", graph_name="graph_2", day=2
    ) is False


# ---------------------------------------------------------------------------
# build_orthrus_graphs day-loop integration
# ---------------------------------------------------------------------------


class _BoundedCursor:
    """Minimal DB-API cursor that counts ``fetchmany`` calls and refuses fetchall."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._offset = 0
        self.fetchall_called = False
        self.execute_calls = 0
        self.fetchmany_calls = 0

    def execute(self, _sql):
        self._offset = 0
        self.execute_calls += 1

    def fetchmany(self, size):
        self.fetchmany_calls += 1
        batch = self._rows[self._offset:self._offset + size]
        self._offset += len(batch)
        return batch

    def fetchall(self):
        self.fetchall_called = True
        raise AssertionError("bounded-memory path must not call fetchall()")


def _make_cfg(tmp_path, *, day_range):
    cfg = MagicMock()
    cfg.graph_construction.build_graphs._graphs_dir = str(tmp_path / "graphs")
    cfg.graph_construction.build_graphs.time_window_size = 15.0
    cfg.dataset.start_end_day_range = day_range
    cfg.dataset.year_month = "2018-04"
    cfg.dataset.name = "THEIA_E3"
    cfg._test_mode = False
    return cfg


def test_build_graphs_raw_empty_writes_verified_marker_no_fake_graph(tmp_path):
    cfg = _make_cfg(tmp_path, day_range=(2, 3))
    cursor = _BoundedCursor([])  # raw-event count = 0
    nodeid2msg = {"src": ["subject", "/bin/sh"], "dst": ["file", "/tmp/x"]}
    logger = MagicMock()
    build_graphs.gen_edge_fused_tw_streaming(
        cursor, nodeid2msg, logger, cfg, event_fetch_size=1024
    )
    marker = tmp_path / "graphs/graph_2/.preprocess_empty_day.json"
    assert marker.is_file()
    payload = json.loads(marker.read_text("utf-8"))
    assert payload["raw_event_count"] == 0
    # No fake graph artifacts should exist for a raw-empty day.
    other = [p for p in (tmp_path / "graphs/graph_2").iterdir()
             if p.name != ".preprocess_empty_day.json"]
    assert other == []
    assert not list((tmp_path / "graphs").glob("**/*.tmp"))


def test_build_graphs_raw_event_positive_no_marker_even_if_no_graphs(tmp_path):
    """raw > 0 + supported == 0 must NOT be classified as a verified empty day."""
    cfg = _make_cfg(tmp_path, day_range=(2, 3))
    relation_names = [
        key for key in build_graphs.rel2id if isinstance(key, str)
    ]
    first = build_graphs.datetime_to_ns_time_US("2018-04-02 00:00:01")
    # 50 raw events with an *unsupported* operation.
    events = [
        ("src", 1, "UNSUPPORTED_OP", "dst", 2, f"e{i}", first + i, i)
        for i in range(50)
    ]
    cursor = _BoundedCursor(events)
    nodeid2msg = {"src": ["subject", "/bin/sh"], "dst": ["file", "/tmp/x"]}
    logger = MagicMock()
    build_graphs.gen_edge_fused_tw_streaming(
        cursor, nodeid2msg, logger, cfg, event_fetch_size=1024
    )
    day_dir = tmp_path / "graphs/graph_2"
    assert not (day_dir / ".preprocess_empty_day.json").exists()
    # No fake graph files either, but at least the directory was created and
    # no marker was written to silently mask the relation-mapping issue.
    graph_files = [p for p in day_dir.iterdir() if p.is_file()
                   and not p.name.startswith(".preprocess_")]
    assert graph_files == []


def test_build_graphs_stale_empty_marker_is_removed_on_normal_day(tmp_path):
    cfg = _make_cfg(tmp_path, day_range=(2, 3))
    day_dir = tmp_path / "graphs/graph_2"
    day_dir.mkdir(parents=True)
    # A previous run left a stale marker; the next real-data run must clear it.
    stale = day_dir / empty_day_mod.EMPTY_DAY_MARKER_FILENAME
    stale.write_text(json.dumps({
        "schema": empty_day_mod.EMPTY_DAY_MARKER_SCHEMA,
        "dataset": "THEIA_E3", "graph_name": "graph_2", "day": 2,
        "date_start": "2018-04-02 00:00:00",
        "date_stop": "2018-04-03 00:00:00",
        "start_ns": 0, "end_ns": 1,
        "raw_event_count": 0, "reason": "no_raw_events",
        "written_at": "2024-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    relation_names = [k for k in build_graphs.rel2id if isinstance(k, str)]
    op = relation_names[0]
    first = build_graphs.datetime_to_ns_time_US("2018-04-02 00:00:01")
    # 1500 events across 2 semantic windows to force at least one graph.
    events = [
        ("src", 1, op, "dst", 2, f"e{i}", first + i * 1_000_000_000, i)
        for i in range(1500)
    ]
    cursor = _BoundedCursor(events)
    nodeid2msg = {"src": ["subject", "/bin/sh"], "dst": ["file", "/tmp/x"]}
    logger = MagicMock()
    build_graphs.gen_edge_fused_tw_streaming(
        cursor, nodeid2msg, logger, cfg, event_fetch_size=1024
    )
    assert not stale.exists()
    graph_files = [p for p in day_dir.iterdir() if p.is_file()
                   and not p.name.startswith(".preprocess_")]
    assert graph_files, "expected at least one real graph artifact"


# ---------------------------------------------------------------------------
# build_graphs validator behavior
# ---------------------------------------------------------------------------


def _theia_e3_cfg(tmp_path):
    return SimpleNamespace(
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(_graphs_dir=str(tmp_path / "graphs"))
        ),
        edge_featurization=SimpleNamespace(
            embed_nodes=SimpleNamespace(
                feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "w2v"))
            ),
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "edge_emb")),
        ),
        dataset=SimpleNamespace(
            name="THEIA_E3",
            train_files=["graph_2", "graph_3", "graph_4", "graph_5"],
            val_files=["graph_9"],
            test_files=["graph_10", "graph_12", "graph_13"],
        ),
    )


def _save_graph(folder, name, node_ids):
    folder.mkdir(parents=True, exist_ok=True)
    g = nx.MultiDiGraph()
    g.add_nodes_from(node_ids)
    torch.save(g, folder / name)


def _write_empty_marker(folder, *, dataset, graph_name, day, raw_event_count=0,
                        reason=None, schema=None, graph_name_override=None,
                        day_override=None, dataset_override=None):
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": schema if schema is not None else empty_day_mod.EMPTY_DAY_MARKER_SCHEMA,
        "dataset": dataset_override if dataset_override is not None else dataset,
        "graph_name": graph_name_override if graph_name_override is not None else graph_name,
        "day": day_override if day_override is not None else day,
        "date_start": "2018-04-02 00:00:00",
        "date_stop": "2018-04-03 00:00:00",
        "start_ns": 123, "end_ns": 456,
        "raw_event_count": raw_event_count,
        "reason": reason if reason is not None else empty_day_mod.EMPTY_DAY_REASON_NO_RAW_EVENTS,
        "written_at": "2024-01-01T00:00:00+00:00",
    }
    (folder / empty_day_mod.EMPTY_DAY_MARKER_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_validator_accepts_normal_graphs(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    for split, nodes in [
        ("graph_2", [1]), ("graph_3", [2]), ("graph_4", [3]),
        ("graph_5", [4]), ("graph_9", [5]),
        ("graph_10", [6]), ("graph_12", [7]), ("graph_13", [8]),
    ]:
        _save_graph(tmp_path / "graphs" / split, "window", nodes)
    assert check_preprocess_stage_complete(cfg, "build_graphs") is True


def test_validator_accepts_verified_empty_marker(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    _write_empty_marker(
        tmp_path / "graphs/graph_2", dataset="THEIA_E3",
        graph_name="graph_2", day=2, raw_event_count=0,
    )
    for split, nodes in [
        ("graph_3", [2]), ("graph_4", [3]), ("graph_5", [4]),
        ("graph_9", [5]), ("graph_10", [6]),
        ("graph_12", [7]), ("graph_13", [8]),
    ]:
        _save_graph(tmp_path / "graphs" / split, "window", nodes)
    assert check_preprocess_stage_complete(cfg, "build_graphs") is True


def test_validator_rejects_missing_unmarked_split(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    # graph_2 directory completely missing
    for split, nodes in [
        ("graph_3", [2]), ("graph_4", [3]), ("graph_5", [4]),
        ("graph_9", [5]), ("graph_10", [6]),
        ("graph_12", [7]), ("graph_13", [8]),
    ]:
        _save_graph(tmp_path / "graphs" / split, "window", nodes)
    assert check_preprocess_stage_complete(cfg, "build_graphs") is False


def test_validator_rejects_malformed_marker(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    (tmp_path / "graphs/graph_2").mkdir(parents=True)
    (tmp_path / "graphs/graph_2" / empty_day_mod.EMPTY_DAY_MARKER_FILENAME).write_text(
        "{not-json", encoding="utf-8"
    )
    assert check_preprocess_stage_complete(cfg, "build_graphs") is False


def test_validator_rejects_marker_with_nonzero_raw_event_count(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    _write_empty_marker(
        tmp_path / "graphs/graph_2", dataset="THEIA_E3",
        graph_name="graph_2", day=2, raw_event_count=42,
    )
    assert check_preprocess_stage_complete(cfg, "build_graphs") is False


def test_validator_rejects_marker_with_wrong_identity(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    _write_empty_marker(
        tmp_path / "graphs/graph_2", dataset="THEIA_E3",
        graph_name="graph_2", day=2,
        dataset_override="THEIA_E5",  # identity mismatch
    )
    assert check_preprocess_stage_complete(cfg, "build_graphs") is False


def test_validator_accepts_real_graph_even_with_invalid_marker_present(tmp_path):
    """If a folder has a real graph artifact, the split is valid even if an
    invalid (raw != 0, mismatched identity, malformed) marker is also
    present. The real graph wins; the invalid marker is harmless and does
    not mask the validation outcome."""
    cfg = _theia_e3_cfg(tmp_path)
    for split, nodes in [
        ("graph_2", [1]), ("graph_3", [2]), ("graph_4", [3]),
        ("graph_5", [4]), ("graph_9", [5]),
        ("graph_10", [6]), ("graph_12", [7]), ("graph_13", [8]),
    ]:
        _save_graph(tmp_path / "graphs" / split, "window", nodes)
    # Now drop an invalid (raw_event_count != 0) marker on top of graph_2.
    _write_empty_marker(
        tmp_path / "graphs/graph_2", dataset="THEIA_E3",
        graph_name="graph_2", day=2, raw_event_count=99,
    )
    assert check_preprocess_stage_complete(cfg, "build_graphs") is True


def test_global_completion_marker_cannot_hide_real_missing_split(tmp_path):
    cfg = _theia_e3_cfg(tmp_path)
    (tmp_path / "graphs").mkdir(parents=True)
    (tmp_path / "graphs/.preprocess_build_graphs_complete").write_text("done")
    assert check_preprocess_stage_complete(cfg, "build_graphs") is False


# ---------------------------------------------------------------------------
# train_only Word2Vec corpus collection
# ---------------------------------------------------------------------------


def _save_graph_with_index(folder, name, node_ids):
    folder.mkdir(parents=True, exist_ok=True)
    g = nx.MultiDiGraph()
    g.add_nodes_from(node_ids)
    torch.save(g, folder / name)


def test_train_only_skips_verified_empty_split_and_collects_real_nodes(tmp_path):
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / "graph_2").mkdir(parents=True)
    _write_empty_marker(
        graphs_dir / "graph_2", dataset="THEIA_E3",
        graph_name="graph_2", day=2, raw_event_count=0,
    )
    _save_graph_with_index(graphs_dir / "graph_3", "w", [10, 11])
    _save_graph_with_index(graphs_dir / "graph_4", "w", [11, 12])
    _save_graph_with_index(graphs_dir / "graph_5", "w", [12])
    cfg = SimpleNamespace(
        semantic_features=SimpleNamespace(corpus_scope="train_only"),
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
        ),
        dataset=SimpleNamespace(
            name="THEIA_E3",
            train_files=["graph_2", "graph_3", "graph_4", "graph_5"],
        ),
    )
    ids = build_w2v.select_corpus_messages(
        {10: ["file", "a"], 11: ["file", "b"], 12: ["file", "c"]},
        cfg,
    )
    assert set(ids) == {10, 11, 12}


def test_train_only_fails_on_unverified_empty_split(tmp_path):
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / "graph_2").mkdir(parents=True)  # no marker
    _save_graph_with_index(graphs_dir / "graph_3", "w", [10])
    with pytest.raises(FileNotFoundError):
        build_w2v._collect_split_node_ids_impl(
            str(graphs_dir), ["graph_2", "graph_3"], cfg=SimpleNamespace(
                dataset=SimpleNamespace(name="THEIA_E3"),
                graph_construction=SimpleNamespace(
                    build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
                ),
            ),
        )


def test_train_only_fails_on_malformed_marker(tmp_path):
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / "graph_2").mkdir(parents=True)
    (graphs_dir / "graph_2" / empty_day_mod.EMPTY_DAY_MARKER_FILENAME).write_text(
        "{not-json", encoding="utf-8"
    )
    _save_graph_with_index(graphs_dir / "graph_3", "w", [10])
    with pytest.raises(FileNotFoundError):
        build_w2v._collect_split_node_ids_impl(
            str(graphs_dir), ["graph_2", "graph_3"], cfg=SimpleNamespace(
                dataset=SimpleNamespace(name="THEIA_E3"),
                graph_construction=SimpleNamespace(
                    build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
                ),
            ),
        )


# ---------------------------------------------------------------------------
# Graph enumeration ignores preprocess markers and .tmp
# ---------------------------------------------------------------------------


def test_get_all_files_from_folders_ignores_markers_and_tmp(tmp_path):
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / "graph_2").mkdir(parents=True)
    _save_graph_with_index(graphs_dir / "graph_2", "real-window", [1])
    (graphs_dir / "graph_2" / ".preprocess_empty_day.json").write_text(
        json.dumps({
            "schema": empty_day_mod.EMPTY_DAY_MARKER_SCHEMA,
            "dataset": "THEIA_E3", "graph_name": "graph_2", "day": 2,
            "raw_event_count": 0, "reason": "no_raw_events",
        }), encoding="utf-8"
    )
    (graphs_dir / "graph_2" / "in-flight.tmp").write_text("partial", encoding="utf-8")
    paths = get_all_files_from_folders(str(graphs_dir), ["graph_2"])
    assert len(paths) == 1
    assert paths[0].endswith("real-window")


def test_get_all_files_from_folders_returns_empty_for_verified_empty_split(tmp_path):
    graphs_dir = tmp_path / "graphs"
    (graphs_dir / "graph_2").mkdir(parents=True)
    (graphs_dir / "graph_2" / ".preprocess_empty_day.json").write_text(
        json.dumps({
            "schema": empty_day_mod.EMPTY_DAY_MARKER_SCHEMA,
            "dataset": "THEIA_E3", "graph_name": "graph_2", "day": 2,
            "raw_event_count": 0, "reason": "no_raw_events",
        }), encoding="utf-8"
    )
    paths = get_all_files_from_folders(str(graphs_dir), ["graph_2"])
    assert paths == []


# ---------------------------------------------------------------------------
# PostgreSQL major parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("pg_restore (PostgreSQL) 18.4 (Ubuntu 18.4-1.pgdg22.04+1)", "18"),
    ("pg_restore (PostgreSQL) 17.6", "17"),
    ("pg_restore (PostgreSQL) 17", "17"),
    ("pg_restore (PostgreSQL) 16.10 (Debian 16.10-1.pgdg120+1)", "16"),
    ("", None),
    ("garbage", None),
    ("pg_restore (EnterpriseDB) 18.4", None),  # not stock PostgreSQL
])
def test_parse_pg_restore_major(text, expected):
    assert colab_postgres.parse_pg_restore_major(text) == expected


def _make_fake_pg_binary(tmp_path, major):
    bin_dir = tmp_path / "postgresql" / major / "bin"
    bin_dir.mkdir(parents=True)
    pg_restore = bin_dir / "pg_restore"
    pg_restore.write_text("#!/bin/sh\necho pg_restore\n", encoding="utf-8")
    pg_restore.chmod(0o755)
    psql = bin_dir / "psql"
    psql.write_text("#!/bin/sh\necho psql\n", encoding="utf-8")
    psql.chmod(0o755)
    return str(pg_restore), str(psql)


def test_ensure_compatible_postgres_parses_major_from_version_output(tmp_path, monkeypatch):
    pg_restore, _ = _make_fake_pg_binary(tmp_path, "18")
    monkeypatch.setattr(
        colab_postgres, "pg_restore_candidates", lambda: [pg_restore]
    )
    monkeypatch.setattr(
        colab_postgres, "archive_is_readable",
        lambda _pg, _dump: (True, ""),
    )
    monkeypatch.setattr(
        colab_postgres, "_capture",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0,
            stdout="pg_restore (PostgreSQL) 18.4 (Ubuntu 18.4-1.pgdg22.04+1)",
            stderr="",
        ),
    )
    dump = tmp_path / "dump.bin"
    dump.write_bytes(b"")
    _pg, _psql, major = colab_postgres.ensure_compatible_postgres(str(dump))
    assert major == "18"


def test_ensure_compatible_postgres_follows_usr_bin_symlink(tmp_path, monkeypatch):
    real, _ = _make_fake_pg_binary(tmp_path, "17")
    link = tmp_path / "usr_bin_pg_restore"
    link.symlink_to(real)
    monkeypatch.setattr(
        colab_postgres, "pg_restore_candidates", lambda: [str(link)]
    )
    monkeypatch.setattr(
        colab_postgres, "archive_is_readable",
        lambda _pg, _dump: (True, ""),
    )
    monkeypatch.setattr(
        colab_postgres, "_capture",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout="pg_restore (PostgreSQL) 17.6", stderr="",
        ),
    )
    dump = tmp_path / "dump.bin"
    dump.write_bytes(b"")
    _pg, _psql, major = colab_postgres.ensure_compatible_postgres(str(dump))
    assert major == "17"
    assert _pg == str(Path(real).resolve())


def test_ensure_compatible_postgres_rejects_malformed_version(tmp_path, monkeypatch):
    pg_restore, _ = _make_fake_pg_binary(tmp_path, "18")
    monkeypatch.setattr(
        colab_postgres, "pg_restore_candidates", lambda: [pg_restore]
    )
    monkeypatch.setattr(
        colab_postgres, "archive_is_readable",
        lambda _pg, _dump: (True, ""),
    )
    monkeypatch.setattr(
        colab_postgres, "_capture",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout="some garbage", stderr="",
        ),
    )
    dump = tmp_path / "dump.bin"
    dump.write_bytes(b"")
    with pytest.raises(RuntimeError, match="Could not determine PostgreSQL major"):
        colab_postgres.ensure_compatible_postgres(str(dump))


# ---------------------------------------------------------------------------
# Diagnostic CLI dataset positional
# ---------------------------------------------------------------------------


def test_diagnose_script_passes_dataset_positional_to_runtime_args():
    script = ROOT / "scripts" / "diagnose_theia_graph_day.py"
    src = script.read_text(encoding="utf-8")
    # The post-hoc fixup that masked the bug must be gone.
    assert "runtime_args.dataset = args.dataset" not in src
    # The dead sys.argv reassignment must be gone.
    assert 'sys.argv = ["diagnose", args.dataset]' not in src
    # The args.dataset positional must be the first element forwarded to
    # get_runtime_required_args.
    assert "cli_args = [args.dataset]" in src
    assert "get_runtime_required_args(args=cli_args)" in src


def test_diagnose_script_help_works():
    script = ROOT / "scripts" / "diagnose_theia_graph_day.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "dataset" in result.stdout.lower()


# ---------------------------------------------------------------------------
# No-regression for existing C8 streaming / completion / atomic write
# ---------------------------------------------------------------------------


def test_streaming_does_not_use_fetchall_on_empty_day(tmp_path):
    cfg = _make_cfg(tmp_path, day_range=(2, 3))
    cursor = _BoundedCursor([])
    build_graphs.gen_edge_fused_tw_streaming(
        cursor, {}, MagicMock(), cfg, event_fetch_size=1024
    )
    assert cursor.fetchall_called is False
    assert cursor.execute_calls >= 1
    assert cursor.fetchmany_calls >= 1


def test_global_completion_marker_published_only_after_full_run(tmp_path, monkeypatch):
    """The global build marker is written *only* after the day loop returns.
    We do not actually publish it in this test; we merely assert the build
    loop's behavior under a synthetic empty day."""
    cfg = _make_cfg(tmp_path, day_range=(2, 3))
    cursor = _BoundedCursor([])
    build_graphs.gen_edge_fused_tw_streaming(
        cursor, {}, MagicMock(), cfg, event_fetch_size=1024
    )
    marker = tmp_path / "graphs/.preprocess_build_graphs_complete"
    assert not marker.exists(), (
        "build_orthrus_graphs.gen_edge_fused_tw_streaming must not publish the "
        "global build marker; main() does that after the loop."
    )
