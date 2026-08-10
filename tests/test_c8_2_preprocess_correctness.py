"""C8.2 preprocessing correctness and Colab production-hardening tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest
import torch

from config import get_runtime_required_args, get_yml_cfg
from edge_featurization import build_feature_word2vec as build_w2v
from edge_featurization import embed_edges_feature_word2vec as embed_w2v
from mstc.metadata_cache import MetadataCache, metadata_complete
from mstc.metadata_export import REQUIRED_METADATA, export_metadata, stream_query
from pipeline_stages import check_preprocess_stage_complete
import colab_postgres


def _corpus_cfg(tmp_path, scope):
    return SimpleNamespace(
        semantic_features=SimpleNamespace(corpus_scope=scope),
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(_graphs_dir=str(tmp_path / "graphs"))
        ),
        dataset=SimpleNamespace(train_files=["graph_2"]),
    )


def _save_graph(path, node_ids):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(node_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def test_train_only_corpus_excludes_val_and_test_only_tokens(tmp_path, monkeypatch):
    cfg = _corpus_cfg(tmp_path, "train_only")
    _save_graph(tmp_path / "graphs/graph_2/train-window", [1, 2])
    _save_graph(tmp_path / "graphs/graph_9/val-window", [3])
    _save_graph(tmp_path / "graphs/graph_10/test-window", [4])
    messages = {
        1: ["file", "train-only alpha"],
        2: ["file", "train shared"],
        3: ["file", "val-only sentinel"],
        4: ["file", "test-only sentinel"],
    }

    selected = build_w2v.select_corpus_messages(messages, cfg)
    assert set(selected) == {1, 2}
    monkeypatch.setattr(build_w2v, "tokenize_file", str.split)
    corpus_tokens = {
        token
        for sentence in build_w2v.load_corpus_from_database(selected, False)
        for token in sentence
    }
    assert "train-only" in corpus_tokens
    assert "val-only" not in corpus_tokens
    assert "test-only" not in corpus_tokens


def test_official_full_dataset_preserves_original_corpus_behavior(tmp_path):
    cfg = _corpus_cfg(tmp_path, "official_full_dataset")
    messages = {
        1: ["file", "train-token"],
        2: ["file", "val-token"],
        3: ["file", "test-token"],
    }
    assert build_w2v.select_corpus_messages(messages, cfg) is messages


class _FakeWV:
    def __init__(self):
        self.values = {
            "known": np.array([3.0, 4.0], dtype=np.float32),
            "also": np.array([1.0, 0.0], dtype=np.float32),
        }

    def __contains__(self, token):
        return token in self.values

    def __getitem__(self, token):
        return self.values[token]


class _FakeModel:
    vector_size = 2

    def __init__(self):
        self.wv = _FakeWV()
        self.train = MagicMock()


def test_partial_and_all_oov_are_deterministic_finite_and_do_not_retrain(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(embed_w2v.Word2Vec, "load", lambda _path: model)
    monkeypatch.setattr(embed_w2v, "tokenize_file", str.split)

    first = embed_w2v.get_indexid2vec(
        {1: ["file", "known missing"], 2: ["file", "missing entirely"]},
        "unused.model",
        use_node_types=False,
        decline_percentage=30,
    )
    second = embed_w2v.get_indexid2vec(
        {1: ["file", "known missing"], 2: ["file", "missing entirely"]},
        "unused.model",
        use_node_types=False,
        decline_percentage=30,
    )

    assert np.all(np.isfinite(first[1]))
    assert np.all(np.isfinite(first[2]))
    assert not np.allclose(first[1], 0)
    assert np.array_equal(first[2], np.zeros(2, dtype=np.float32))
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    model.train.assert_not_called()


def _task_paths(scope):
    args = get_runtime_required_args(args=[
        "THEIA_E3", "--config", "config/orthrus.yml",
        f"--semantic_features.corpus_scope={scope}",
    ])
    cfg = get_yml_cfg(args)
    return (
        Path(cfg.graph_construction.build_graphs._task_path).name,
        Path(cfg.edge_featurization.embed_nodes._task_path).name,
        Path(cfg.edge_featurization.embed_edges._task_path).name,
    )


def test_corpus_scope_changes_only_semantic_task_hash_chain():
    train_paths = _task_paths("train_only")
    official_paths = _task_paths("official_full_dataset")
    assert train_paths[0] == official_paths[0]
    assert train_paths[0] == "3eb70eba13d9538396c158583c8370762c55ba1c3a9228ca8f30b5d9f5dbeddc"
    assert train_paths[1] != official_paths[1]
    assert train_paths[2] != official_paths[2]


def _metadata_cfg(tmp_path):
    return SimpleNamespace(
        _metadata_dir=str(tmp_path / "metadata"),
        _ground_truth_dir=str(tmp_path / "gt"),
        semantic_features=SimpleNamespace(corpus_scope="train_only"),
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(_task_path=str(tmp_path / "graph-hash"))
        ),
        edge_featurization=SimpleNamespace(
            embed_nodes=SimpleNamespace(
                emb_dim=128,
                feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "w2v")),
            )
        ),
        dataset=SimpleNamespace(
            name="THEIA_E3",
            num_node_types=3,
            num_edge_types=10,
            train_files=["graph_2"],
            val_files=["graph_9"],
            test_files=["graph_10"],
            ground_truth_relative_path=[],
            attack_to_time_window=[],
        ),
    )


def test_metadata_export_generates_every_required_file_atomically(tmp_path, monkeypatch):
    from mstc import metadata_export

    cfg = _metadata_cfg(tmp_path)
    cursor = MagicMock()
    connection = MagicMock()
    monkeypatch.setattr(
        "provnet_utils.init_database_connection",
        lambda _cfg: (cursor, connection),
    )
    monkeypatch.setattr(metadata_export, "_node_metadata_from_db", lambda _cur: {
        7: {
            "uuid": "uuid-7", "type": "file", "path": "/tmp/x", "cmd": None,
            "local_ip": None, "local_port": None, "remote_ip": None,
            "remote_port": None, "display": "file:/tmp/x",
        }
    })
    monkeypatch.setattr(
        metadata_export, "_ground_truth_mappings", lambda _cfg, _mapping: (set(), {})
    )
    monkeypatch.setattr(
        metadata_export, "_time_to_malicious_nodes",
        lambda _cfg, _cur, _attacks, _ids: {},
    )

    cache = export_metadata(cfg)
    assert metadata_complete(cache)
    assert cache.validate_required(list(REQUIRED_METADATA)) == []
    assert not list(Path(cfg._metadata_dir).glob("*.tmp"))
    manifest = cache.load_dataset_manifest()
    assert manifest["corpus_scope"] == "train_only"
    assert manifest["word2vec_model_hash"] is None
    cursor.close.assert_called_once()
    connection.close.assert_called_once()


class _StreamingCursor:
    def __init__(self):
        self.batches = [[(1,), (2,)], [(3,)], []]
        self.fetchall = MagicMock(side_effect=AssertionError("fetchall forbidden"))

    def execute(self, *_args):
        return None

    def fetchmany(self, _size):
        return self.batches.pop(0)


def test_metadata_stream_query_never_uses_fetchall():
    cursor = _StreamingCursor()
    assert list(stream_query(cursor, "SELECT 1", batch_size=2)) == [(1,), (2,), (3,)]
    cursor.fetchall.assert_not_called()


def test_detection_metadata_path_lookup_never_connects_to_postgres(tmp_path, monkeypatch):
    import provnet_utils

    cfg = SimpleNamespace(
        _metadata_dir=str(tmp_path / "metadata"),
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(
                _node_id_to_path=str(tmp_path / "legacy-node-paths")
            )
        ),
        pipeline=SimpleNamespace(mode="detection_only"),
    )
    cache = MetadataCache(cfg._metadata_dir)
    cache.save_node_metadata({
        3: {
            "type": "subject", "path": "/bin/sh", "cmd": "-c true",
            "display": "subject:/bin/sh -c true",
        }
    })
    db = MagicMock(side_effect=AssertionError("database must not be used"))
    monkeypatch.setattr(provnet_utils, "init_database_connection", db)
    result = provnet_utils.get_node_to_path_and_type(cfg)
    assert result[3]["path"] == "/bin/sh"
    db.assert_not_called()


def test_existing_build_marker_skips_graph_rebuild_when_running_embed_nodes(tmp_path):
    import orthrus

    graph_dir = tmp_path / "graphs/graph_2"
    graph_dir.mkdir(parents=True)
    (graph_dir / "window").write_text("graph")
    (tmp_path / "graphs/.preprocess_build_graphs_complete").write_text("done")
    cfg = SimpleNamespace(
        graph_construction=SimpleNamespace(
            build_graphs=SimpleNamespace(_graphs_dir=str(tmp_path / "graphs"))
        ),
        edge_featurization=SimpleNamespace(
            embed_nodes=SimpleNamespace(
                feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "w2v"))
            ),
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "edges")),
        ),
        dataset=SimpleNamespace(train_files=["graph_2"], val_files=[], test_files=[]),
    )
    state = {"metadata"}
    real_complete = check_preprocess_stage_complete

    def complete(current_cfg, stage):
        if stage == "build_graphs":
            return real_complete(current_cfg, stage)
        return stage in state

    build = MagicMock()
    embed = MagicMock(side_effect=lambda _cfg: state.add("embed_nodes"))
    with patch.object(orthrus, "check_preprocess_stage_complete", side_effect=complete), \
         patch.object(orthrus, "export_metadata"), \
         patch.object(orthrus.build_orthrus_graphs, "main", build), \
         patch.object(orthrus.build_feature_word2vec, "main", embed), \
         patch.object(orthrus, "_get_memory_usage_mb", return_value=0.0), \
         patch.object(orthrus.torch.cuda, "is_available", return_value=False):
        timings = orthrus._run_preprocess_substages(
            cfg, ["build_graphs", "embed_nodes"]
        )

    build.assert_not_called()
    embed.assert_called_once_with(cfg)
    assert timings["skipped_build_graphs"] is True


def test_pg_restore_capability_selection_uses_first_readable_client(monkeypatch):
    candidates = ["/pg14/pg_restore", "/pg18/pg_restore"]
    monkeypatch.setattr(
        colab_postgres,
        "archive_is_readable",
        lambda candidate, _dump: (candidate.endswith("pg18/pg_restore"), "unsupported"),
    )
    monkeypatch.setattr(
        colab_postgres,
        "_capture",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="pg_restore mock", stderr=""),
    )
    assert colab_postgres.choose_compatible_pg_restore(
        "dump", candidates=candidates
    ) == candidates[1]


def test_password_setup_uses_redacted_display_and_stdin(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        colab_postgres,
        "run_cmd",
        lambda command, **kwargs: captured.update(command=command, **kwargs),
    )
    secret = "never-print-this"
    colab_postgres.set_postgres_password("/pg18/psql", secret)
    assert secret in captured["input_text"]
    assert secret not in captured["display_command"]
    assert all(secret not in str(part) for part in captured["command"])




def test_cluster_setup_stops_port_conflict_without_deleting_data(monkeypatch):
    clusters = [
        {"major": "14", "name": "main", "port": "5432", "status": "online"},
        {"major": "18", "name": "main", "port": "5433", "status": "down"},
    ]
    commands = []
    monkeypatch.setattr(colab_postgres, "_clusters", lambda: clusters)
    monkeypatch.setattr(
        colab_postgres, "run_cmd",
        lambda command, **kwargs: commands.append(list(command)),
    )
    monkeypatch.setattr(
        colab_postgres, "_capture",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    colab_postgres.ensure_cluster("18")
    assert ["pg_ctlcluster", "14", "main", "stop"] in commands
    assert ["pg_conftool", "18", "main", "set", "port", "5432"] in commands
    assert not any(command[0] in {"pg_dropcluster", "rm"} for command in commands)
def test_master_notebook_has_clean_outputs_safe_controls_and_status_refresh():
    path = Path("notebooks/ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else cell.get("source", "")
        for cell in notebook["cells"]
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
    assert "RUN_PREPROCESS_BENCHMARK = False" in source
    assert 'CORPUS_SCOPE = "train_only"' in source
    assert 'print_status("Before:"' in source
    assert 'print_status("After:"' in source
    assert "check_all_preprocess_stages_complete" in source
    assert "ALTER USER postgres WITH PASSWORD" not in source
    assert "secrets.token_urlsafe" not in source
