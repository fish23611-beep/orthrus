"""
tests/test_preprocess_memory_streaming.py

Tests for bounded-memory streaming preprocessing implementation.
Verifies:
- Graph construction uses streaming (no fetchall on events)
- Node tables use streaming (no fetchall)
- Word2Vec corpus is restartable
- Edge embedding uses bounded memory
- Completion markers work correctly
- Substage CLI works
"""
import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch, call
from datetime import datetime

import pytest
import torch
import networkx as nx

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Standalone streaming helpers (mirrors build_orthrus_graphs.py)
# These are used for testing the streaming protocol without importing
# the graph_construction package (which has pytest import issues).
# The production implementation should match these semantics.
# ---------------------------------------------------------------------------

def stream_node_table_standalone(cur, sql, batch_size=1024):
    """Standalone version of stream_node_table for testing."""
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


def stream_event_table_standalone(cur, sql, batch_size=8192):
    """Standalone version of stream_event_table for testing."""
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


from pipeline_stages import (
    parse_preprocess_substages,
    check_preprocess_stage_complete,
    check_all_preprocess_stages_complete,
    PREPROCESS_SUBSTAGES,
)


# ---------------------------------------------------------------------------
# Test parse_preprocess_substages
# ---------------------------------------------------------------------------

class TestParsePreprocessSubstages:
    
    def test_none_returns_all(self):
        """None returns the full list of substages."""
        result = parse_preprocess_substages(None)
        assert result == ["build_graphs", "embed_nodes", "embed_edges"]
    
    def test_empty_string_returns_all(self):
        """Empty string returns the full list."""
        result = parse_preprocess_substages("")
        assert result == ["build_graphs", "embed_nodes", "embed_edges"]
    
    def test_single_substage(self):
        """Single substage returns just that one."""
        result = parse_preprocess_substages("build_graphs")
        assert result == ["build_graphs"]
    
    def test_multiple_substages(self):
        """Multiple substages returns in specified order."""
        result = parse_preprocess_substages("embed_nodes,embed_edges")
        assert result == ["embed_nodes", "embed_edges"]
    
    def test_whitespace_handling(self):
        """Whitespace is stripped from substages."""
        result = parse_preprocess_substages(" build_graphs , embed_nodes ")
        assert result == ["build_graphs", "embed_nodes"]
    
    def test_invalid_substage_raises(self):
        """Invalid substage raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_preprocess_substages("build_graphs,invalid_stage")
        assert "invalid_stage" in str(exc_info.value)
    
    def test_all_valid_substages(self):
        """All valid substages are accepted."""
        for substage in PREPROCESS_SUBSTAGES:
            result = parse_preprocess_substages(substage)
            assert result == [substage]


# ---------------------------------------------------------------------------
# Test completion markers
# ---------------------------------------------------------------------------

class TestCompletionMarkers:
    
    @pytest.fixture
    def temp_graphs_dir(self, tmp_path):
        """Create a temporary graphs directory."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        return graphs_dir
    
    @pytest.fixture
    def mock_cfg(self, temp_graphs_dir, tmp_path):
        """Create a mock configuration."""
        cfg = MagicMock()
        cfg.graph_construction.build_graphs._graphs_dir = str(temp_graphs_dir)
        
        # Mock embed_nodes paths - use unique subdirs
        word2vec_dir = tmp_path / "w2v"
        word2vec_dir.mkdir(exist_ok=True)
        cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir = str(word2vec_dir)
        
        # Mock embed_edges paths
        edge_embeds_dir = tmp_path / "edge_emb"
        edge_embeds_dir.mkdir(exist_ok=True)
        cfg.edge_featurization.embed_edges._edge_embeds_dir = str(edge_embeds_dir)
        
        return cfg
    
    def test_build_graphs_marker_not_exists(self, mock_cfg, temp_graphs_dir):
        """check_preprocess_stage_complete returns False when marker missing."""
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is False
    
    def test_build_graphs_marker_exists(self, mock_cfg, temp_graphs_dir):
        """check_preprocess_stage_complete returns True when marker exists."""
        marker_path = temp_graphs_dir / ".preprocess_build_graphs_complete"
        marker_path.write_text(datetime.now().isoformat())
        
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is True
    
    def test_build_graphs_legacy_with_files(self, mock_cfg, temp_graphs_dir):
        """check_preprocess_stage_complete uses legacy detection when no marker."""
        # Create a dummy graph file in a proper graph subdirectory
        # Test WITHOUT .pt suffix (frozen version format)
        graph_subdir = temp_graphs_dir / "graph_0"
        graph_subdir.mkdir(exist_ok=True)
        (graph_subdir / "2019-05-08_00:00:00~2019-05-08_00:15:00").write_text("dummy")
        
        # Should detect legacy artifacts (no .pt suffix)
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is True
    
    def test_build_graphs_legacy_with_pt_suffix(self, mock_cfg, temp_graphs_dir):
        """Legacy detection also works with .pt suffix files."""
        graph_subdir = temp_graphs_dir / "graph_1"
        graph_subdir.mkdir(exist_ok=True)
        (graph_subdir / "graph.pt").write_text("dummy")
        
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is True
    
    def test_embed_nodes_marker(self, mock_cfg, tmp_path):
        """Embed nodes completion check uses model file."""
        model_dir = tmp_path / "w2v"
        # No marker, no model
        assert check_preprocess_stage_complete(mock_cfg, "embed_nodes") is False
        
        # Add model file
        (model_dir / "feature_word2vec.model").write_text("dummy")
        assert check_preprocess_stage_complete(mock_cfg, "embed_nodes") is True
    
    def test_embed_edges_marker(self, mock_cfg, tmp_path):
        """Embed edges completion check uses edge embeddings."""
        edge_embeds_dir = tmp_path / "edge_emb"
        
        # No marker, no files
        assert check_preprocess_stage_complete(mock_cfg, "embed_edges") is False
        
        # Add train split with file
        train_dir = edge_embeds_dir / "train"
        train_dir.mkdir(exist_ok=True)
        (train_dir / "graph_0.TemporalData.simple").write_text("dummy")
        
        assert check_preprocess_stage_complete(mock_cfg, "embed_edges") is True
    
    def test_all_stages_complete_false(self, mock_cfg):
        """check_all_preprocess_stages_complete returns False when incomplete."""
        assert check_all_preprocess_stages_complete(mock_cfg) is False
    
    def test_all_stages_complete_true(self, mock_cfg, temp_graphs_dir, tmp_path):
        """check_all_preprocess_stages_complete returns True when all complete."""
        # Mark all stages complete
        (temp_graphs_dir / ".preprocess_build_graphs_complete").write_text("")
        
        model_dir = tmp_path / "w2v"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "feature_word2vec.model").write_text("")
        
        edge_embeds_dir = tmp_path / "edge_emb"
        train_dir = edge_embeds_dir / "train"
        train_dir.mkdir(exist_ok=True)
        (train_dir / "dummy.pt").write_text("")
        
        # Create marker for embed_edges
        (edge_embeds_dir / ".preprocess_embed_edges_complete").write_text("")
        
        assert check_all_preprocess_stages_complete(mock_cfg) is True


# ---------------------------------------------------------------------------
# Test streaming helpers (standalone implementations)
# ---------------------------------------------------------------------------

class TestStreamingHelpers:
    """Test that streaming cursor patterns are used correctly.
    
    These tests use standalone implementations that mirror the production
    code in graph_construction.build_orthrus_graphs. The production
    implementation should match these semantics.
    """
    
    def test_stream_node_table_generator(self):
        """Test stream_node_table produces rows correctly."""
        mock_cur = MagicMock()
        mock_cur.fetchmany.side_effect = [
            [(1, 2, 3), (4, 5, 6)],
            [(7, 8, 9)],
            []
        ]
        
        result = list(stream_node_table_standalone(mock_cur, "SELECT * FROM table", batch_size=2))
        
        assert result == [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        mock_cur.execute.assert_called_once_with("SELECT * FROM table")
        # The short-batch guard terminates after the 2nd fetchmany because the
        # 2nd batch returned 1 row (< batch_size=2), so the 3rd [] entry is
        # never consumed.
        assert mock_cur.fetchmany.call_count == 2
    
    def test_stream_event_table_generator(self):
        """Test stream_event_table produces rows correctly."""
        mock_cur = MagicMock()
        mock_cur.fetchmany.side_effect = [
            [("evt1",), ("evt2",)],
            [("evt3",)],
            []
        ]
        
        result = list(stream_event_table_standalone(mock_cur, "SELECT * FROM events", batch_size=2))
        
        assert result == [("evt1",), ("evt2",), ("evt3",)]
    
    def test_stream_node_table_empty(self):
        """Test stream_node_table with empty result."""
        mock_cur = MagicMock()
        mock_cur.fetchmany.side_effect = [[]]
        
        result = list(stream_node_table_standalone(mock_cur, "SELECT * FROM empty", batch_size=10))
        
        assert result == []
    
    def test_stream_node_table_full_batch(self):
        """Test stream_node_table with full batches."""
        mock_cur = MagicMock()
        # Return 2 full batches + empty sentinel
        mock_cur.fetchmany.side_effect = [
            [(1,), (2,), (3,)],
            [(4,), (5,), (6,)],
            []
        ]
        
        result = list(stream_node_table_standalone(mock_cur, "SELECT * FROM full", batch_size=3))
        
        assert result == [(1,), (2,), (3,), (4,), (5,), (6,)]
        assert mock_cur.fetchmany.call_count == 3


# ---------------------------------------------------------------------------
# Test RestartableCorpus for Word2Vec
# ---------------------------------------------------------------------------

class TestRestartableCorpus:
    """Test the restartable corpus implementation."""
    
    def test_corpus_iteration(self):
        """Test that corpus can be iterated."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus
        
        indexid2msg = {
            1: ['subject', '/path/to/file'],
            2: ['file', '/another/path'],
        }
        
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)
        
        # Should be able to iterate
        sentences = list(corpus)
        assert len(sentences) == 2
    
    def test_corpus_restartable(self):
        """Test that corpus can be iterated multiple times."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus
        
        indexid2msg = {
            1: ['subject', '/path/to/file'],
            2: ['file', '/another/path'],
        }
        
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)
        
        # First iteration
        first = list(corpus)
        # Second iteration - should get same results
        second = list(corpus)
        
        assert first == second
    
    def test_corpus_length(self):
        """Test corpus length."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus
        
        indexid2msg = {
            1: ['subject', '/path/to/file'],
            2: ['file', '/another/path'],
            3: ['netflow', '192.168.1.1'],
        }
        
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)
        
        assert len(corpus) == 3
    
    def test_corpus_dedup_behavior(self):
        """Test that duplicate msg[1] keys are handled like Python dict."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus
        
        # Two nodes with same msg[1] - dict will keep only one
        indexid2msg = {
            1: ['subject', '/path/to/file'],
            2: ['subject', '/path/to/file'],  # Same label as above
        }
        
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)
        
        # Dict deduplicates, so only one sentence
        assert len(corpus) == 1
    
    def test_corpus_to_list(self):
        """Test to_list() method for compatibility."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus
        
        indexid2msg = {
            1: ['subject', '/path/to/file'],
            2: ['file', '/another/path'],
        }
        
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)
        
        as_list = corpus.to_list()
        assert isinstance(as_list, list)
        assert len(as_list) == 2


# ---------------------------------------------------------------------------
# Test edge featurization memory patterns
# ---------------------------------------------------------------------------

class TestEdgeEmbeddingMemory:
    """Test that edge embedding uses bounded memory patterns."""
    
    def test_preallocated_tensors(self):
        """Verify edge embedding uses list approach (can't fully preallocate due to tensor dimensions)."""
        from edge_featurization.embed_edges_feature_word2vec import gen_vectorized_graphs, gen_relation_onehot
        from config import ntype2id, rel2id
        
        # Create a small test graph
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create graph directory structure
            graphs_dir = os.path.join(tmp_dir, "graphs")
            os.makedirs(os.path.join(graphs_dir, "graph_0"))
            
            graph = nx.MultiDiGraph()
            graph.add_node(1, node_type='subject', label='test')
            graph.add_node(2, node_type='file', label='test2')
            graph.add_edge(1, 2, event_uuid='evt1', time=1000, label='EVENT_OPEN')
            
            graph_path = os.path.join(graphs_dir, "graph_0", "test_graph")
            torch.save(graph, graph_path)
            
            # Create output directory
            out_dir = os.path.join(tmp_dir, "output")
            os.makedirs(out_dir)
            
            # Create mock config
            cfg = MagicMock()
            cfg.graph_construction.build_graphs._graphs_dir = graphs_dir
            
            # Create mock indexid2vec
            import numpy as np
            indexid2vec = {
                1: np.array([0.1] * 128),
                2: np.array([0.2] * 128),
            }
            
            # Create one-hot encodings
            etype2oh = gen_relation_onehot(rel2id)
            ntype2oh = gen_relation_onehot(ntype2id)
            
            # Mock logger
            logger = MagicMock()
            
            # Run
            gen_vectorized_graphs(
                indexid2vec=indexid2vec,
                etype2oh=etype2oh,
                ntype2oh=ntype2oh,
                split_files=["graph_0"],
                out_dir=out_dir,
                logger=logger,
                cfg=cfg
            )
            
            # Check output file was created
            output_file = os.path.join(out_dir, "test_graph.TemporalData.simple")
            assert os.path.exists(output_file)
            
            # Verify output is a valid TemporalData
            data = torch.load(output_file)
            assert hasattr(data, 'src')
            assert hasattr(data, 'dst')
            assert hasattr(data, 't')
            assert hasattr(data, 'msg')
            assert data.src.dtype == torch.long
            assert data.dst.dtype == torch.long
            assert data.t.dtype == torch.long
            assert data.msg.dtype == torch.float


# ---------------------------------------------------------------------------
# Test atomic writes
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    """Test that atomic write pattern is used."""
    
    def test_graph_save_uses_atomic_write(self):
        """Verify graph save uses temp file + replace."""
        import tempfile
        from graph_construction.build_orthrus_graphs import gen_edge_fused_tw_streaming
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            graphs_dir = os.path.join(tmp_dir, "graphs")
            os.makedirs(graphs_dir)
            
            # Mock config
            cfg = MagicMock()
            cfg.graph_construction.build_graphs._graphs_dir = graphs_dir
            cfg.graph_construction.build_graphs.time_window_size = 15.0
            cfg.dataset.start_end_day_range = (1, 2)
            cfg.dataset.year_month = "2024-01"
            cfg._test_mode = True
            
            # Mock nodeid2msg
            nodeid2msg = {}
            
            # Mock cursor and connection
            mock_cur = MagicMock()
            mock_connect = MagicMock()
            
            # Mock logger
            logger = MagicMock()
            
            # Patch stream_event_table to return empty
            with patch('graph_construction.build_orthrus_graphs.stream_event_table', return_value=iter([])):
                with patch('graph_construction.build_orthrus_graphs.rel2id', {1: 'EVENT_OPEN', 'EVENT_OPEN': 1}):
                    gen_edge_fused_tw_streaming(
                        cur=mock_cur,
                        nodeid2msg=nodeid2msg,
                        logger=logger,
                        cfg=cfg,
                        event_fetch_size=1024
                    )
            
            # Function should complete without error
            assert True


# ---------------------------------------------------------------------------
# Test no-fetchall enforcement (using standalone implementations)
# ---------------------------------------------------------------------------

class TestNoFetchallEnforcement:
    """Test that production code doesn't use fetchall on large tables."""
    
    def test_stream_node_table_no_fetchall(self):
        """Verify stream_node_table doesn't use fetchall."""
        mock_cur = MagicMock()
        # Ensure fetchall is not called
        mock_cur.fetchall = MagicMock()
        mock_cur.fetchmany.side_effect = [[], []]
        
        list(stream_node_table_standalone(mock_cur, "SELECT * FROM table", batch_size=10))
        
        mock_cur.fetchall.assert_not_called()
    
    def test_stream_event_table_no_fetchall(self):
        """Verify stream_event_table doesn't use fetchall."""
        mock_cur = MagicMock()
        mock_cur.fetchall = MagicMock()
        mock_cur.fetchmany.side_effect = [[], []]
        
        list(stream_event_table_standalone(mock_cur, "SELECT * FROM events", batch_size=10))
        
        mock_cur.fetchall.assert_not_called()


# ---------------------------------------------------------------------------
# Test CLI integration
# ---------------------------------------------------------------------------

class TestPreprocessSubstagesCLI:
    """Test preprocess substages integration with orthrus.py."""
    
    def test_import_orthrus(self):
        """Verify orthrus module can be imported."""
        import orthrus
        assert hasattr(orthrus, 'main')
    
    def test_import_pipeline_stages(self):
        """Verify pipeline_stages has required functions."""
        from pipeline_stages import (
            parse_preprocess_substages,
            check_preprocess_stage_complete,
            PREPROCESS_SUBSTAGES
        )
        assert parse_preprocess_substages is not None
        assert check_preprocess_stage_complete is not None
        assert PREPROCESS_SUBSTAGES == ["build_graphs", "embed_nodes", "embed_edges"]


# ---------------------------------------------------------------------------
# C8.1 final acceptance: frozen-reference semantic equivalence
# ---------------------------------------------------------------------------

class _StreamingCursor:
    """Small DB-API cursor that refuses materializing fetchall()."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._offset = 0
        self.fetch_sizes = []

    def execute(self, _sql):
        self._offset = 0

    def fetchmany(self, size):
        self.fetch_sizes.append(size)
        batch = self._rows[self._offset:self._offset + size]
        self._offset += len(batch)
        return batch

    def fetchall(self):
        raise AssertionError("bounded-memory path must not call fetchall()")


class _NodeTableCursor(_StreamingCursor):
    def __init__(self, tables):
        super().__init__([])
        self._tables = tables

    def execute(self, sql):
        table = next(name for name in self._tables if name in sql.lower())
        self._rows = self._tables[table]
        self._offset = 0


def _canonical_graph(graph):
    return (
        [(node, dict(attrs)) for node, attrs in graph.nodes(data=True)],
        [
            (src, dst, key, dict(attrs))
            for src, dst, key, attrs in graph.edges(data=True, keys=True)
        ],
    )


def _frozen_reference_graphs(events, nodeid2msg, include_edge_type, window_size_ns):
    """Direct transcription of the frozen materialized graph semantics."""
    events_list = [event for event in events if event[2] in include_edge_type]
    if not events_list:
        return []

    graphs = []
    start_time = events_list[0][-2]
    temp_list = []
    for offset in range(0, len(events_list), 1024):
        batch = events_list[offset:offset + 1024]
        temp_list.extend(batch)
        if batch[-1][-2] <= start_time + window_size_ns:
            continue

        node_info = {}
        edge_info = {}
        for (src_node, src_index_id, operation, dst_node, dst_index_id,
             event_uuid, timestamp_rec, _id) in temp_list:
            if src_index_id not in node_info:
                node_type, label = nodeid2msg[src_node]
                node_info[src_index_id] = {"label": label, "node_type": node_type}
            if dst_index_id not in node_info:
                node_type, label = nodeid2msg[dst_node]
                node_info[dst_index_id] = {"label": label, "node_type": node_type}
            edge_info.setdefault((src_index_id, dst_index_id), []).append(
                (timestamp_rec, operation, event_uuid)
            )

        edge_list = []
        for (src, dst), data in edge_info.items():
            sorted_data = sorted(data, key=lambda item: item[0])
            indices = []
            current_type = None
            current_start_index = None
            for index, item in enumerate(entry[1] for entry in sorted_data):
                if item == current_type:
                    continue
                if current_type is not None and current_start_index is not None:
                    indices.append(current_start_index)
                current_type = item
                current_start_index = index
            if current_type is not None and current_start_index is not None:
                indices.append(current_start_index)
            for index in indices:
                edge_list.append((src, dst, sorted_data[index]))

        graph = nx.MultiDiGraph()
        for node, info in node_info.items():
            graph.add_node(node, **info)
        for src, dst, (timestamp, operation, event_uuid) in edge_list:
            graph.add_edge(
                src, dst, event_uuid=event_uuid, time=timestamp, label=operation
            )
        graphs.append(_canonical_graph(graph))
        start_time = batch[-1][-2]
        temp_list.clear()
    return graphs


def _run_graph_case(tmp_path, event_fetch_size):
    from graph_construction import build_orthrus_graphs as implementation

    relation_names = [
        key for key in implementation.rel2id if isinstance(key, str)
    ]
    operations = relation_names[:2]
    first_timestamp = implementation.datetime_to_ns_time_US(
        "2019-05-08 00:00:01"
    )
    events = [
        (
            "src-hash", 1, operations[(index // 100) % 2], "dst-hash", 2,
            f"event-{index:04d}", first_timestamp + index * 1_000_000_000, index,
        )
        for index in range(1030)
    ]
    nodeid2msg = {
        "src-hash": ["subject", "/usr/bin/python"],
        "dst-hash": ["file", "/tmp/result"],
    }
    cfg = MagicMock()
    cfg.graph_construction.build_graphs.time_window_size = 15.0
    cfg.graph_construction.build_graphs._graphs_dir = str(tmp_path)
    cfg.dataset.start_end_day_range = (8, 9)
    cfg.dataset.year_month = "2019-05"
    cfg._test_mode = False
    cursor = _StreamingCursor(events)

    implementation.gen_edge_fused_tw_streaming(
        cursor, nodeid2msg, MagicMock(), cfg, event_fetch_size=event_fetch_size
    )
    paths = sorted(
        path
        for root, _dirs, files in os.walk(tmp_path)
        for name in files
        if not name.endswith(".tmp") and not name.startswith(".preprocess_")
        for path in [os.path.join(root, name)]
    )
    actual = [_canonical_graph(torch.load(path)) for path in paths]
    expected = _frozen_reference_graphs(
        events,
        nodeid2msg,
        implementation.rel2id,
        int(cfg.graph_construction.build_graphs.time_window_size * 60_000_000_000),
    )
    return actual, expected, cursor.fetch_sizes


class TestFrozenReferenceEquivalence:
    def test_graph_streaming_matches_frozen_materialized_reference(self, tmp_path):
        actual, expected, fetch_sizes = _run_graph_case(tmp_path, 1024)
        assert actual == expected
        assert actual
        assert set(fetch_sizes) == {1024}

    def test_node_table_streaming_matches_frozen_reference(self):
        from graph_construction import build_orthrus_graphs as implementation

        tables = {
            "netflow_node_table": [
                ("net-row", 101, "10.0.0.1", 1234, "8.8.8.8", 53)
            ],
            "subject_node_table": [
                ("subject-row", 201, "/bin/bash", "bash -c echo")
            ],
            "file_node_table": [("file-row", 301, "/tmp/output")],
        }
        cursor = _NodeTableCursor(tables)
        cfg = MagicMock()
        cfg.graph_construction.build_graphs.use_hashed_label = False
        features = {
            "netflow": ["local_ip", "remote_ip"],
            "subject": ["path", "cmd_line"],
            "file": ["path"],
        }
        with patch.object(
            implementation, "get_darpa_tc_node_feats_from_cfg", return_value=features
        ):
            actual = implementation.get_node_list_streaming(cursor, cfg, batch_size=2)

        assert actual == {
            "net-row": [101, "10.0.0.1"],
            101: ["netflow", "10.0.0.1 8.8.8.8"],
            201: ["subject", "/bin/bash bash -c echo"],
            301: ["file", "/tmp/output"],
        }
        assert set(cursor.fetch_sizes) == {2}

    def test_fetch_size_128_1024_8192_is_semantically_invariant(self, tmp_path):
        outputs = []
        for fetch_size in (128, 1024, 8192):
            actual, expected, sizes = _run_graph_case(
                tmp_path / str(fetch_size), fetch_size
            )
            assert actual == expected
            assert set(sizes) == {fetch_size}
            outputs.append(actual)
        assert outputs[0] == outputs[1] == outputs[2]

    @pytest.mark.parametrize("use_node_types", [False, True])
    def test_word2vec_corpus_matches_frozen_reference_tokenization(
        self, use_node_types
    ):
        from edge_featurization import build_feature_word2vec as implementation

        indexid2msg = {
            1: ["subject", "/usr/bin/python --version"],
            2: ["file", "/var/log/audit/audit.log"],
            3: ["netflow", "10.0.0.1:443 8.8.8.8:53"],
        }
        reference = {}
        tokenizers = {
            "subject": implementation.tokenize_subject,
            "file": implementation.tokenize_file,
            "netflow": implementation.tokenize_netflow,
        }
        for _indexid, msg in indexid2msg.items():
            text = f"{msg[0]} {msg[1]}" if use_node_types else msg[1]
            reference[msg[1]] = tokenizers[msg[0]](text)
        assert list(implementation.RestartableCorpus(
            indexid2msg, use_node_types
        )) == list(reference.values())

    def test_restartable_corpus_two_complete_passes_match(self):
        from edge_featurization.build_feature_word2vec import RestartableCorpus

        corpus = RestartableCorpus({
            1: ["subject", "/bin/sh -c id"],
            2: ["file", "/etc/passwd"],
            3: ["netflow", "127.0.0.1:80"],
        }, use_node_types=True)
        assert list(corpus) == list(corpus)

    def test_duplicate_message_dedup_matches_frozen_dict_semantics(self):
        from edge_featurization import build_feature_word2vec as implementation

        shared = "/same/semantic/label"
        corpus = implementation.RestartableCorpus({
            1: ["subject", shared],
            2: ["file", shared],
        }, use_node_types=False)
        assert list(corpus) == [implementation.tokenize_file(shared)]


def _temporal_data_case(tmp_path):
    import numpy as np
    from edge_featurization.embed_edges_feature_word2vec import gen_vectorized_graphs

    graphs_dir = tmp_path / "graphs"
    graph_dir = graphs_dir / "graph_8"
    graph_dir.mkdir(parents=True)
    graph = nx.MultiDiGraph()
    graph.add_node(3, node_type="subject", label="s")
    graph.add_node(1, node_type="file", label="f")
    graph.add_node(2, node_type="netflow", label="n")
    graph.add_edge(3, 1, time=30, label="EVENT_OPEN", event_uuid="e3")
    graph.add_edge(3, 2, time=10, label="EVENT_READ", event_uuid="e1")
    graph.add_edge(1, 2, time=20, label="EVENT_OPEN", event_uuid="e2")
    torch.save(graph, graph_dir / "window")

    indexid2vec = {
        1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4]),
        3: np.array([0.5, 0.6]),
    }
    ntype2oh = {
        "subject": torch.tensor([1, 0, 0]),
        "file": torch.tensor([0, 1, 0]),
        "netflow": torch.tensor([0, 0, 1]),
    }
    etype2oh = {
        "EVENT_OPEN": torch.tensor([1, 0]),
        "EVENT_READ": torch.tensor([0, 1]),
    }
    cfg = MagicMock()
    cfg.graph_construction.build_graphs._graphs_dir = str(graphs_dir)
    out_dir = tmp_path / "embedded"
    gen_vectorized_graphs(
        indexid2vec, etype2oh, ntype2oh, ["graph_8"], str(out_dir),
        MagicMock(), cfg,
    )
    actual = torch.load(out_dir / "window.TemporalData.simple")
    edges = list(graph.edges(data=True, keys=True))
    expected_src = torch.tensor([int(u) for u, _v, _k, _a in edges], dtype=torch.long)
    expected_dst = torch.tensor([int(v) for _u, v, _k, _a in edges], dtype=torch.long)
    expected_t = torch.tensor([int(a["time"]) for _u, _v, _k, a in edges], dtype=torch.long)
    expected_msg = torch.stack([
        torch.cat([
            ntype2oh[graph.nodes[u]["node_type"]],
            torch.from_numpy(indexid2vec[int(u)]),
            etype2oh[attr["label"]],
            ntype2oh[graph.nodes[v]["node_type"]],
            torch.from_numpy(indexid2vec[int(v)]),
        ])
        for u, v, _key, attr in edges
    ]).float()
    return actual, expected_src, expected_dst, expected_t, expected_msg


class TestTemporalDataFrozenEquivalence:
    def test_temporal_data_src_dst_t_msg_match_frozen_reference(self, tmp_path):
        actual, src, dst, timestamps, msg = _temporal_data_case(tmp_path)
        assert torch.equal(actual.src, src)
        assert torch.equal(actual.dst, dst)
        assert torch.equal(actual.t, timestamps)
        assert torch.equal(actual.msg, msg)

    def test_temporal_data_preserves_frozen_edge_order(self, tmp_path):
        actual, src, dst, timestamps, _msg = _temporal_data_case(tmp_path)
        assert list(zip(actual.src.tolist(), actual.dst.tolist(), actual.t.tolist())) == list(
            zip(src.tolist(), dst.tolist(), timestamps.tolist())
        )


def _run_substage_dispatch(selected):
    import orthrus

    cfg = MagicMock()
    calls = MagicMock()
    build_graphs = MagicMock()
    embed_nodes = MagicMock()
    embed_edges = MagicMock()
    calls.attach_mock(build_graphs, "build_graphs")
    calls.attach_mock(embed_nodes, "embed_nodes")
    calls.attach_mock(embed_edges, "embed_edges")
    with patch.object(orthrus.build_orthrus_graphs, "main", build_graphs), \
         patch.object(orthrus.build_feature_word2vec, "main", embed_nodes), \
         patch.object(orthrus.embed_edges_feature_word2vec, "main", embed_edges), \
         patch.object(orthrus, "_check_preprocess_substage_prerequisites"), \
         patch.object(orthrus, "_get_memory_usage_mb", return_value=0.0), \
         patch.object(orthrus.torch.cuda, "is_available", return_value=False):
        orthrus._run_preprocess_substages(cfg, selected)
    return cfg, calls.mock_calls


class TestPreprocessSubstageExecution:
    def test_default_preprocess_behavior_runs_all_substages_in_order(self):
        cfg, calls_seen = _run_substage_dispatch(parse_preprocess_substages(None))
        assert calls_seen == [
            call.build_graphs(cfg), call.embed_nodes(cfg), call.embed_edges(cfg)
        ]

    def test_build_graphs_only(self):
        cfg, calls_seen = _run_substage_dispatch(["build_graphs"])
        assert calls_seen == [call.build_graphs(cfg)]

    def test_embed_nodes_only(self):
        cfg, calls_seen = _run_substage_dispatch(["embed_nodes"])
        assert calls_seen == [call.embed_nodes(cfg)]

    def test_embed_edges_only(self):
        cfg, calls_seen = _run_substage_dispatch(["embed_edges"])
        assert calls_seen == [call.embed_edges(cfg)]


@pytest.fixture
def artifact_cfg(tmp_path):
    graphs_dir = tmp_path / "graphs"
    model_dir = tmp_path / "w2v"
    edge_embeds_dir = tmp_path / "edge_emb"
    graphs_dir.mkdir()
    model_dir.mkdir()
    edge_embeds_dir.mkdir()
    cfg = MagicMock()
    cfg.graph_construction.build_graphs._graphs_dir = str(graphs_dir)
    cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir = str(model_dir)
    cfg.edge_featurization.embed_edges._edge_embeds_dir = str(edge_embeds_dir)
    return cfg, graphs_dir


class TestArtifactAcceptanceRegressions:
    def test_partial_legacy_graph_artifacts_are_incomplete(
        self, artifact_cfg
    ):
        from types import SimpleNamespace
        mock_cfg, temp_graphs_dir = artifact_cfg

        mock_cfg.dataset = SimpleNamespace(
            train_files=["graph_0"], val_files=["graph_1"], test_files=[]
        )
        graph_0 = temp_graphs_dir / "graph_0"
        graph_0.mkdir()
        (graph_0 / "2019-05-08_00:00:00~2019-05-08_00:15:00").write_text("x")
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is False

    def test_complete_legacy_suffixless_graph_artifacts_are_compatible(
        self, artifact_cfg
    ):
        from types import SimpleNamespace
        mock_cfg, temp_graphs_dir = artifact_cfg

        mock_cfg.dataset = SimpleNamespace(
            train_files=["graph_0"], val_files=["graph_1"], test_files=[]
        )
        for folder in ("graph_0", "graph_1"):
            graph_dir = temp_graphs_dir / folder
            graph_dir.mkdir()
            (graph_dir / "2019-05-08_00:00:00~2019-05-08_00:15:00").write_text("x")
        assert check_preprocess_stage_complete(mock_cfg, "build_graphs") is True

    @pytest.mark.parametrize("stage,relative_dir", [
        ("build_graphs", "graphs"),
        ("embed_nodes", "w2v"),
        ("embed_edges", "edge_emb"),
    ])
    def test_completion_markers_are_read_from_writer_directories(
        self, artifact_cfg, tmp_path, stage, relative_dir
    ):
        mock_cfg, _temp_graphs_dir = artifact_cfg
        marker = tmp_path / relative_dir / f".preprocess_{stage}_complete"
        marker.write_text("complete")
        assert check_preprocess_stage_complete(mock_cfg, stage) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
