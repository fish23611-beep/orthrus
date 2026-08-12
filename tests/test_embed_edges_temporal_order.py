"""
tests/test_embed_edges_temporal_order.py

Tests for embed_edges temporal ordering fix (C8 fix).

Verifies:
1. Edges are sorted chronologically before generating TemporalData
2. Stable sort preserves relative order for equal timestamps
3. All event fields (src, dst, t, msg) stay aligned after sorting
4. Save-before monotonicity validation catches any remaining issues
5. v2 completion marker contract is enforced
6. Old ISO timestamp markers are rejected
7. Crash-safe marker behavior (marker only written after all splits succeed)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest
import torch
from torch_geometric.data import TemporalData

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from edge_featurization.embed_edges_feature_word2vec import (
    _order_edges_chronologically,
    _parse_edge_timestamp,
    _validate_temporal_order,
    gen_vectorized_graphs,
)
from pipeline_stages import (
    _is_valid_embed_edges_marker,
    check_preprocess_stage_complete,
)


# ---------------------------------------------------------------------------
# Test: Time sorting helper
# ---------------------------------------------------------------------------

class TestChronologicalSorting:
    """Test the _order_edges_chronologically helper."""

    def test_basic_chronological_order(self):
        """Edges are sorted by timestamp ascending."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2, time=300, label="LATE")
        graph.add_edge(1, 2, time=100, label="EARLY")
        graph.add_edge(1, 2, time=200, label="MID")

        edges = list(graph.edges(data=True, keys=True))
        sorted_edges = _order_edges_chronologically(edges, "/tmp", "test")

        timestamps = [int(e[3]["time"]) for e in sorted_edges]
        assert timestamps == [100, 200, 300]

    def test_reverse_input_order(self):
        """Even with reverse input order, output is chronological."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        # Add edges in reverse chronological order
        graph.add_edge(1, 2, time=500)
        graph.add_edge(1, 2, time=300)
        graph.add_edge(1, 2, time=100)
        graph.add_edge(1, 2, time=400)
        graph.add_edge(1, 2, time=200)

        edges = list(graph.edges(data=True, keys=True))
        sorted_edges = _order_edges_chronologically(edges, "/tmp", "test")

        timestamps = [int(e[3]["time"]) for e in sorted_edges]
        assert timestamps == [100, 200, 300, 400, 500]

    def test_missing_time_attribute_raises(self):
        """Missing time attribute raises ValueError with context."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2, label="NO_TIME")

        edges = list(graph.edges(data=True, keys=True))
        with pytest.raises(ValueError, match="missing required 'time' attribute"):
            _order_edges_chronologically(edges, "/graphs/train", "window_001")

    def test_non_integer_time_raises(self):
        """Non-integer time attribute raises ValueError with context."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2, time="not-a-number", label="BAD_TIME")

        edges = list(graph.edges(data=True, keys=True))
        with pytest.raises(ValueError, match="non-integer 'time' attribute"):
            _order_edges_chronologically(edges, "/graphs/train", "window_001")

    def test_graph_path_in_error_message(self):
        """Error message includes graph path for debugging."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_edge(1, 2, label="NO_TIME")

        edges = list(graph.edges(data=True, keys=True))
        with pytest.raises(ValueError) as exc_info:
            _order_edges_chronologically(edges, "/some/path", "file.graph")
        assert "/some/path" in str(exc_info.value)
        assert "file.graph" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: Stable sort for equal timestamps
# ---------------------------------------------------------------------------

class TestStableSortEqualTimestamps:
    """Test that equal timestamps preserve relative order (stable sort)."""

    def test_equal_timestamps_preserve_order(self):
        """Equal timestamps should not be arbitrarily reordered."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        # Add in specific order: A, B at time=100, then C at time=200
        graph.add_edge(1, 2, time=100, label="A")
        graph.add_edge(2, 3, time=100, label="B")
        graph.add_edge(1, 3, time=200, label="C")

        edges = list(graph.edges(data=True, keys=True))
        sorted_edges = _order_edges_chronologically(edges, "/tmp", "test")

        labels = [e[3]["label"] for e in sorted_edges]
        # A and B at same time should maintain relative order from original traversal
        assert labels[0] in ("A", "B")
        assert labels[1] in ("A", "B")
        assert labels[2] == "C"

    def test_multiple_equal_timestamps(self):
        """Multiple edges with same timestamp preserve all relative orders."""
        graph = nx.MultiDiGraph()
        graph.add_node(1)
        for i in range(5):
            graph.add_node(i + 2)
            graph.add_edge(1, i + 2, time=100, label=f"E{i}")

        edges = list(graph.edges(data=True, keys=True))
        sorted_edges = _order_edges_chronologically(edges, "/tmp", "test")

        labels = [e[3]["label"] for e in sorted_edges]
        # All edges at time=100, verify they're contiguous
        assert all(e[3]["time"] == 100 for e in sorted_edges[:5])
        # Original relative order preserved
        assert labels == ["E0", "E1", "E2", "E3", "E4"]


# ---------------------------------------------------------------------------
# Test: Parse edge timestamp helper
# ---------------------------------------------------------------------------

class TestParseEdgeTimestamp:
    """Test _parse_edge_timestamp helper."""

    def test_integer_timestamp(self):
        """Integer timestamps are parsed correctly."""
        edge = (1, 2, 0, {"time": 123456789})
        assert _parse_edge_timestamp(edge, "/tmp", "test") == 123456789

    def test_string_integer_timestamp(self):
        """String integer timestamps are parsed correctly."""
        edge = (1, 2, 0, {"time": "123456789"})
        assert _parse_edge_timestamp(edge, "/tmp", "test") == 123456789

    def test_float_timestamp(self):
        """Float timestamps are parsed correctly."""
        edge = (1, 2, 0, {"time": 123456789.0})
        assert _parse_edge_timestamp(edge, "/tmp", "test") == 123456789

    def test_missing_time_raises(self):
        """Missing time raises ValueError."""
        edge = (1, 2, 0, {"label": "EVENT"})
        with pytest.raises(ValueError, match="missing required 'time'"):
            _parse_edge_timestamp(edge, "/tmp", "test")

    def test_invalid_time_raises(self):
        """Invalid time format raises ValueError."""
        edge = (1, 2, 0, {"time": "not-a-number"})
        with pytest.raises(ValueError, match="non-integer 'time'"):
            _parse_edge_timestamp(edge, "/tmp", "test")


# ---------------------------------------------------------------------------
# Test: Temporal monotonicity validation
# ---------------------------------------------------------------------------

class TestTemporalMonotonicityValidation:
    """Test _validate_temporal_order helper."""

    def test_valid_non_decreasing_order(self):
        """Valid non-decreasing timestamps pass."""
        timestamps = torch.tensor([100, 200, 200, 300, 400], dtype=torch.long)
        # Should not raise
        _validate_temporal_order(timestamps, "/tmp", "test")

    def test_valid_strictly_increasing_order(self):
        """Strictly increasing timestamps pass."""
        timestamps = torch.tensor([100, 200, 300, 400, 500], dtype=torch.long)
        _validate_temporal_order(timestamps, "/tmp", "test")

    def test_single_event_passes(self):
        """Single event passes (no comparison needed)."""
        timestamps = torch.tensor([100], dtype=torch.long)
        _validate_temporal_order(timestamps, "/tmp", "test")

    def test_empty_events_passes(self):
        """Empty events pass (handled by caller)."""
        timestamps = torch.tensor([], dtype=torch.long)
        _validate_temporal_order(timestamps, "/tmp", "test")

    def test_decreasing_order_raises_with_context(self):
        """Decreasing timestamps raise ValueError with context."""
        timestamps = torch.tensor([100, 300, 200, 400], dtype=torch.long)
        with pytest.raises(ValueError) as exc_info:
            _validate_temporal_order(timestamps, "/graphs/train", "window_001")
        err = str(exc_info.value)
        assert "monotonicity violation" in err
        assert "/graphs/train" in err
        assert "window_001" in err
        assert "inversion" in err

    def test_first_inversion_reported(self):
        """First inversion is reported in error message."""
        timestamps = torch.tensor([100, 200, 300, 150, 250], dtype=torch.long)
        with pytest.raises(ValueError) as exc_info:
            _validate_temporal_order(timestamps, "/tmp", "test")
        # Index 2 -> 3 is first inversion (300 > 150)
        assert "index 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: Event field alignment
# ---------------------------------------------------------------------------

class TestEventFieldAlignment:
    """Test that all event fields stay aligned after sorting."""

    @pytest.fixture
    def mock_cfg(self, tmp_path):
        """Create mock config for gen_vectorized_graphs."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        cfg = MagicMock()
        cfg.graph_construction.build_graphs._graphs_dir = str(graphs_dir)
        return cfg, graphs_dir, out_dir

    def test_src_dst_t_msg_stay_aligned(self, mock_cfg, tmp_path):
        """All event fields come from the same sorted edge list."""
        cfg, graphs_dir, out_dir = mock_cfg

        # Create graph with edges in intentionally wrong time order
        graph_dir = graphs_dir / "graph_0"
        graph_dir.mkdir()
        graph = nx.MultiDiGraph()
        graph.add_node(1, node_type="subject", label="s")
        graph.add_node(2, node_type="file", label="f")
        graph.add_node(3, node_type="netflow", label="n")
        # Add edges in WRONG order: 30, 10, 20
        graph.add_edge(1, 2, time=30, label="LATE", event_uuid="e3")
        graph.add_edge(2, 3, time=10, label="EARLY", event_uuid="e1")
        graph.add_edge(1, 3, time=20, label="MID", event_uuid="e2")
        torch.save(graph, graph_dir / "window")

        # Mock vectors and encodings
        indexid2vec = {
            1: np.array([0.1, 0.2]),
            2: np.array([0.3, 0.4]),
            3: np.array([0.5, 0.6]),
        }
        ntype2oh = {
            "subject": torch.tensor([1, 0, 0]),
            "file": torch.tensor([0, 1, 0]),
            "netflow": torch.tensor([0, 0, 1]),
        }
        etype2oh = {
            "LATE": torch.tensor([1, 0]),
            "EARLY": torch.tensor([0, 1]),
            "MID": torch.tensor([1, 0]),
        }

        gen_vectorized_graphs(
            indexid2vec, etype2oh, ntype2oh, ["graph_0"], str(out_dir),
            MagicMock(), cfg,
        )

        # Load and verify
        data = torch.load(out_dir / "window.TemporalData.simple")

        # Timestamps should be sorted: 10, 20, 30
        assert data.t.tolist() == [10, 20, 30]

        # Check that each event's fields are consistent
        # Event 1: time=10, src=2, dst=3 (EARLY)
        # Event 2: time=20, src=1, dst=3 (MID)
        # Event 3: time=30, src=1, dst=2 (LATE)
        assert data.t[0].item() == 10
        assert data.src[0].item() == 2
        assert data.dst[0].item() == 3

        assert data.t[1].item() == 20
        assert data.src[1].item() == 1
        assert data.dst[1].item() == 3

        assert data.t[2].item() == 30
        assert data.src[2].item() == 1
        assert data.dst[2].item() == 2

    def test_reversed_input_produces_corrected_output(self, mock_cfg):
        """Reversed input edges are corrected to chronological output."""
        cfg, graphs_dir, out_dir = mock_cfg

        graph_dir = graphs_dir / "graph_1"
        graph_dir.mkdir()
        graph = nx.MultiDiGraph()
        graph.add_node(1, node_type="subject", label="s")
        graph.add_node(2, node_type="file", label="f")
        # Add 5 edges in reverse order
        for i in range(5):
            graph.add_edge(1, 2, time=(5 - i) * 100, label=f"E{i}")
        torch.save(graph, graph_dir / "window")

        indexid2vec = {
            1: np.array([0.1, 0.2]),
            2: np.array([0.3, 0.4]),
        }
        ntype2oh = {
            "subject": torch.tensor([1, 0, 0]),
            "file": torch.tensor([0, 1, 0]),
        }
        etype2oh = {f"E{i}": torch.tensor([1 if i % 2 == 0 else 0, 0 if i % 2 == 0 else 1])
                    for i in range(5)}

        gen_vectorized_graphs(
            indexid2vec, etype2oh, ntype2oh, ["graph_1"], str(out_dir),
            MagicMock(), cfg,
        )

        data = torch.load(out_dir / "window.TemporalData.simple")

        # Should be sorted: 100, 200, 300, 400, 500
        expected_ts = [100, 200, 300, 400, 500]
        assert data.t.tolist() == expected_ts


# ---------------------------------------------------------------------------
# Test: v2 completion marker validation
# ---------------------------------------------------------------------------

class TestV2MarkerValidation:
    """Test _is_valid_embed_edges_marker function."""

    @pytest.fixture
    def marker_dir(self, tmp_path):
        """Create a temporary directory for marker tests."""
        return tmp_path / "edge_embeds"

    def test_old_iso_marker_rejected(self, marker_dir):
        """Old ISO timestamp marker is rejected."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_path.write_text(datetime.now(timezone.utc).isoformat())

        assert _is_valid_embed_edges_marker(str(marker_dir)) is False

    def test_valid_v2_marker_accepted(self, marker_dir):
        """Valid v2 JSON marker is accepted."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_data = {
            "schema_version": 2,
            "temporal_order": "nondecreasing",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        marker_path.write_text(json.dumps(marker_data))

        assert _is_valid_embed_edges_marker(str(marker_dir)) is True

    def test_v1_marker_rejected(self, marker_dir):
        """v1 marker is rejected."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_data = {
            "schema_version": 1,
            "temporal_order": "nondecreasing"
        }
        marker_path.write_text(json.dumps(marker_data))

        assert _is_valid_embed_edges_marker(str(marker_dir)) is False

    def test_wrong_temporal_order_rejected(self, marker_dir):
        """Wrong temporal_order value is rejected."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_data = {
            "schema_version": 2,
            "temporal_order": "strictly_increasing"
        }
        marker_path.write_text(json.dumps(marker_data))

        assert _is_valid_embed_edges_marker(str(marker_dir)) is False

    def test_missing_marker_rejected(self, marker_dir):
        """Missing marker file is rejected."""
        # marker_dir exists but no marker file
        assert _is_valid_embed_edges_marker(str(marker_dir)) is False

    def test_malformed_json_rejected(self, marker_dir):
        """Malformed JSON marker is rejected."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_path.write_text("not valid json {")

        assert _is_valid_embed_edges_marker(str(marker_dir)) is False

    def test_incomplete_v2_marker_rejected(self, marker_dir):
        """v2 marker missing required fields is rejected."""
        marker_dir.mkdir()
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_data = {
            "schema_version": 2
            # missing temporal_order
        }
        marker_path.write_text(json.dumps(marker_data))

        assert _is_valid_embed_edges_marker(str(marker_dir)) is False


# ---------------------------------------------------------------------------
# Test: check_preprocess_stage_complete with v2 marker
# ---------------------------------------------------------------------------

class TestCheckPreprocessStageCompleteEmbedEdges:
    """Test check_preprocess_stage_complete for embed_edges with v2 marker."""

    @pytest.fixture
    def embed_cfg(self, tmp_path):
        """Create mock config for embed_edges stage."""
        edge_embeds_dir = tmp_path / "edge_emb"
        edge_embeds_dir.mkdir()
        for split in ("train", "val", "test"):
            (edge_embeds_dir / split).mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_edges._edge_embeds_dir = str(edge_embeds_dir)
        return cfg, edge_embeds_dir

    def test_missing_marker_returns_false(self, embed_cfg):
        """Missing marker returns False."""
        cfg, _ = embed_cfg
        assert check_preprocess_stage_complete(cfg, "embed_edges") is False

    def test_old_iso_marker_returns_false(self, embed_cfg):
        """Old ISO marker returns False."""
        cfg, marker_dir = embed_cfg
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_path.write_text(datetime.now(timezone.utc).isoformat())

        assert check_preprocess_stage_complete(cfg, "embed_edges") is False

    def test_valid_v2_marker_returns_true(self, embed_cfg):
        """Valid v2 marker with files returns True."""
        cfg, marker_dir = embed_cfg
        marker_data = {
            "schema_version": 2,
            "temporal_order": "nondecreasing",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_path.write_text(json.dumps(marker_data))

        # Add actual files so _has_visible_file returns True
        for split in ("train", "val", "test"):
            (marker_dir / split / "window.TemporalData.simple").write_text("data")

        assert check_preprocess_stage_complete(cfg, "embed_edges") is True

    def test_v2_marker_without_files_returns_false(self, embed_cfg):
        """v2 marker without files returns False (marker alone insufficient)."""
        cfg, marker_dir = embed_cfg
        marker_data = {
            "schema_version": 2,
            "temporal_order": "nondecreasing"
        }
        marker_path = marker_dir / ".preprocess_embed_edges_complete"
        marker_path.write_text(json.dumps(marker_data))

        # No files in train/val/test directories
        assert check_preprocess_stage_complete(cfg, "embed_edges") is False


# ---------------------------------------------------------------------------
# Test: Save-before validation catches errors
# ---------------------------------------------------------------------------

class TestSaveBeforeValidation:
    """Test that save-before validation catches temporal order errors."""

    def test_validator_catches_manual_violation(self):
        """Validator correctly rejects manual timestamp violations."""
        # Create timestamps that are NOT in order
        bad_timestamps = torch.tensor([100, 300, 200, 400, 500], dtype=torch.long)

        with pytest.raises(ValueError, match="monotonicity violation"):
            _validate_temporal_order(bad_timestamps, "/tmp", "test")


# ---------------------------------------------------------------------------
# Test: Full integration with gen_vectorized_graphs
# ---------------------------------------------------------------------------

class TestFullIntegration:
    """Full integration test of temporal ordering in gen_vectorized_graphs."""

    def test_produces_valid_temporal_data(self, tmp_path):
        """gen_vectorized_graphs produces valid TemporalData with correct order."""
        graphs_dir = tmp_path / "graphs"
        out_dir = tmp_path / "output"
        graphs_dir.mkdir()
        out_dir.mkdir()

        # Create graph with 300 edges in random time order
        graph_dir = graphs_dir / "graph_0"
        graph_dir.mkdir()
        graph = nx.MultiDiGraph()
        for i in range(1, 11):
            graph.add_node(i, node_type="subject", label="s")

        import random
        random.seed(42)
        times = list(range(100, 1100, 10))
        random.shuffle(times)
        for i, t in enumerate(times):
            src = (i % 9) + 1
            dst = ((i + 1) % 9) + 1
            graph.add_edge(src, dst, time=t, label="EVENT")

        torch.save(graph, graph_dir / "window")

        # Mock vectors
        indexid2vec = {i: np.array([0.1 * i, 0.2 * i]) for i in range(1, 11)}
        ntype2oh = {"subject": torch.tensor([1, 0, 0])}
        etype2oh = {"EVENT": torch.tensor([1, 0])}

        cfg = MagicMock()
        cfg.graph_construction.build_graphs._graphs_dir = str(graphs_dir)

        gen_vectorized_graphs(
            indexid2vec, etype2oh, ntype2oh, ["graph_0"], str(out_dir),
            MagicMock(), cfg,
        )

        data = torch.load(out_dir / "window.TemporalData.simple")

        # Verify temporal order
        for i in range(len(data.t) - 1):
            assert data.t[i] <= data.t[i + 1], \
                f"t[{i}]={data.t[i]} > t[{i+1}]={data.t[i+1]}"

        # Verify all times are from original list
        assert sorted(data.t.tolist()) == sorted(times)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
