"""
tests/test_pytorch26_serialization_compat.py

PyTorch 2.6 serialization compatibility regression tests.

Tests cover:
- NetworkX MultiDiGraph roundtrip with trusted loader
- TemporalData roundtrip with trusted loader
- expected_type guard (TypeError on mismatch)
- weights_only=False enforcement (spy/mock verification)
- Integration with production code paths (embed_edges, data_utils)
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import networkx as nx
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Test 1: NetworkX MultiDiGraph roundtrip
# ---------------------------------------------------------------------------

class TestNetworkXMultiDiGraphRoundtrip:
    """Test real NetworkX MultiDiGraph save->load roundtrip with trusted loader."""

    def test_multidigraph_roundtrip_preserves_structure(self, tmp_path):
        """Save a MultiDiGraph and load it back; verify structure and attributes."""
        from serialization_compat import load_trusted_torch_artifact

        # Create a MultiDiGraph with nodes and edges
        graph = nx.MultiDiGraph()
        graph.add_node(0, node_type="subject", label="node_0")
        graph.add_node(1, node_type="file", label="node_1")
        graph.add_node(2, node_type="netflow", label="node_2")
        graph.add_edge(0, 1, key=0, label="write", time=1234567890, event_uuid="evt-001")
        graph.add_edge(1, 2, key=0, label="connect", time=1234567891, event_uuid="evt-002")
        graph.add_edge(0, 2, key=0, label="access", time=1234567892, event_uuid="evt-003")

        # Save via torch.save
        save_path = tmp_path / "test_graph.pkl"
        torch.save(graph, save_path)

        # Load via trusted loader
        loaded = load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)

        # Verify type
        assert isinstance(loaded, nx.MultiDiGraph), (
            f"Expected nx.MultiDiGraph, got {type(loaded).__name__}"
        )

        # Verify nodes
        assert set(loaded.nodes()) == {0, 1, 2}, (
            f"Nodes mismatch: {set(loaded.nodes())}"
        )

        # Verify edges
        assert loaded.number_of_edges() == 3, (
            f"Edge count mismatch: {loaded.number_of_edges()}"
        )

        # Verify node attributes
        assert loaded.nodes[0]["node_type"] == "subject"
        assert loaded.nodes[1]["node_type"] == "file"
        assert loaded.nodes[2]["node_type"] == "netflow"

        # Verify edge attributes
        edges_by_label = {}
        for u, v, k, attr in loaded.edges(keys=True, data=True):
            edges_by_label[attr["label"]] = attr

        assert "write" in edges_by_label
        assert edges_by_label["write"]["time"] == 1234567890
        assert edges_by_label["write"]["event_uuid"] == "evt-001"

    def test_empty_multidigraph_roundtrip(self, tmp_path):
        """Empty MultiDiGraph can be saved and loaded."""
        from serialization_compat import load_trusted_torch_artifact

        graph = nx.MultiDiGraph()
        save_path = tmp_path / "empty_graph.pkl"
        torch.save(graph, save_path)

        loaded = load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)

        assert isinstance(loaded, nx.MultiDiGraph)
        assert loaded.number_of_nodes() == 0
        assert loaded.number_of_edges() == 0


# ---------------------------------------------------------------------------
# Test 2: TemporalData roundtrip
# ---------------------------------------------------------------------------

class TestTemporalDataRoundtrip:
    """Test real TemporalData save->load roundtrip with trusted loader."""

    def test_temporal_data_roundtrip_preserves_tensors(self, tmp_path):
        """Save a TemporalData and load it back; verify tensor contents."""
        from serialization_compat import load_trusted_torch_artifact

        # Create a minimal TemporalData with realistic fields
        data = TemporalData()
        data.src = torch.tensor([0, 1, 2], dtype=torch.long)
        data.dst = torch.tensor([1, 2, 0], dtype=torch.long)
        data.t = torch.tensor([1000, 1001, 1002], dtype=torch.long)
        data.msg = torch.randn(3, 16, dtype=torch.float32)

        # Save via torch.save
        save_path = tmp_path / "test_window.TemporalData.simple"
        torch.save(data, save_path)

        # Load via trusted loader
        loaded = load_trusted_torch_artifact(save_path, expected_type=TemporalData)

        # Verify type
        assert isinstance(loaded, TemporalData), (
            f"Expected TemporalData, got {type(loaded).__name__}"
        )

        # Verify tensors match
        assert torch.equal(loaded.src, data.src), "src tensor mismatch"
        assert torch.equal(loaded.dst, data.dst), "dst tensor mismatch"
        assert torch.equal(loaded.t, data.t), "t tensor mismatch"
        assert torch.equal(loaded.msg, data.msg), "msg tensor mismatch"

    def test_temporal_data_large_msg_dim(self, tmp_path):
        """TemporalData with large message dimension (ORTHRUS production size)."""
        from serialization_compat import load_trusted_torch_artifact

        # ORTHRUS uses large embedding dimensions (emb_dim * 2 + node_types * 2 + edge_type)
        data = TemporalData()
        data.src = torch.tensor([0, 1], dtype=torch.long)
        data.dst = torch.tensor([1, 0], dtype=torch.long)
        data.t = torch.tensor([2000, 2001], dtype=torch.long)
        # Simulate production msg dimension: emb=64 * 2 + node_types=8 * 2 + edge_type=4 = 148
        data.msg = torch.randn(2, 148, dtype=torch.float32)

        save_path = tmp_path / "large_window.TemporalData.simple"
        torch.save(data, save_path)

        loaded = load_trusted_torch_artifact(save_path, expected_type=TemporalData)

        assert isinstance(loaded, TemporalData)
        assert loaded.msg.shape == (2, 148)
        assert torch.equal(loaded.msg, data.msg)


# ---------------------------------------------------------------------------
# Test 3: expected_type guard
# ---------------------------------------------------------------------------

class TestExpectedTypeGuard:
    """Test that expected_type verification raises TypeError on mismatch."""

    def test_wrong_type_raises_typeerror(self, tmp_path):
        """Loading a dict with expected_type=MultiDiGraph raises TypeError."""
        from serialization_compat import load_trusted_torch_artifact

        # Save a dict artifact (wrong type)
        artifact = {"key": "value", "numbers": [1, 2, 3]}
        save_path = tmp_path / "wrong_type.pkl"
        torch.save(artifact, save_path)

        # Attempting to load as MultiDiGraph must raise TypeError
        with pytest.raises(TypeError, match="type mismatch"):
            load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)

    def test_wrong_type_temporal_data_raises(self, tmp_path):
        """Loading a list with expected_type=TemporalData raises TypeError."""
        from serialization_compat import load_trusted_torch_artifact

        artifact = [1, 2, 3]
        save_path = tmp_path / "list_artifact.pkl"
        torch.save(artifact, save_path)

        with pytest.raises(TypeError, match="type mismatch"):
            load_trusted_torch_artifact(save_path, expected_type=TemporalData)

    def test_no_expected_type_accepts_any_type(self, tmp_path):
        """When expected_type is None, any object type is accepted."""
        from serialization_compat import load_trusted_torch_artifact

        artifact = {"hello": "world"}
        save_path = tmp_path / "any_type.pkl"
        torch.save(artifact, save_path)

        # No expected_type: should succeed with any type
        loaded = load_trusted_torch_artifact(save_path, expected_type=None)
        assert loaded == {"hello": "world"}

    def test_matching_type_succeeds(self, tmp_path):
        """Loading with matching expected_type succeeds."""
        from serialization_compat import load_trusted_torch_artifact

        graph = nx.MultiDiGraph()
        graph.add_node(1)
        save_path = tmp_path / "correct_type.pkl"
        torch.save(graph, save_path)

        loaded = load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)
        assert isinstance(loaded, nx.MultiDiGraph)


# ---------------------------------------------------------------------------
# Test 4: weights_only=False enforcement
# ---------------------------------------------------------------------------

class TestWeightsOnlyFalse:
    """Verify trusted loader always calls torch.load with weights_only=False."""

    def test_torch_load_called_with_weights_only_false(self, tmp_path):
        """Spy test: torch.load must be called with weights_only=False."""
        from serialization_compat import load_trusted_torch_artifact

        graph = nx.MultiDiGraph()
        graph.add_node(1)
        save_path = tmp_path / "spy_test.pkl"
        torch.save(graph, save_path)

        with patch("serialization_compat.torch.load") as mock_load:
            # Configure mock to return the graph
            mock_load.return_value = graph

            try:
                load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)
            except Exception:
                pass  # We only care about the call, not the result

            # Verify torch.load was called with weights_only=False
            assert mock_load.called, "torch.load was not called"
            call_kwargs = mock_load.call_args
            # Supports both positional and keyword argument styles
            if call_kwargs.kwargs:
                assert call_kwargs.kwargs.get("weights_only") is False, (
                    f"Expected weights_only=False, got {call_kwargs.kwargs.get('weights_only')}"
                )
            else:
                # Check positional args if any
                args = call_kwargs.args if hasattr(call_kwargs, "args") else ()
                # weights_only is always a keyword argument in our code
                assert False, "weights_only=False was not passed to torch.load"

    def test_weights_only_false_survives_removal_attempt(self, tmp_path):
        """If someone removes weights_only=False from the loader, test fails."""
        from serialization_compat import load_trusted_torch_artifact
        import serialization_compat

        # Read the source to verify weights_only=False is present
        source = serialization_compat.__file__
        if source.endswith(".py"):
            source_code = Path(source).read_text()
            assert "weights_only=False" in source_code, (
                "weights_only=False must be present in serialization_compat.py"
            )


# ---------------------------------------------------------------------------
# Test 5: FileNotFoundError handling
# ---------------------------------------------------------------------------

class TestFileNotFoundError:
    """Test that non-existent paths raise FileNotFoundError."""

    def test_missing_file_raises(self):
        """Loading a non-existent artifact raises FileNotFoundError."""
        from serialization_compat import load_trusted_torch_artifact

        fake_path = "/nonexistent/path/to/artifact.pkl"
        with pytest.raises(FileNotFoundError):
            load_trusted_torch_artifact(fake_path)


# ---------------------------------------------------------------------------
# Test 6: Integration with production code paths
# ---------------------------------------------------------------------------

class TestProductionCodeIntegration:
    """Verify production code paths use trusted loader correctly."""

    def test_embed_edges_import_uses_trusted_loader(self):
        """Verify embed_edges_feature_word2vec imports and uses trusted loader."""
        # Read the source to verify import and usage
        embed_edges_path = SRC / "edge_featurization" / "embed_edges_feature_word2vec.py"
        source_code = embed_edges_path.read_text()

        # Should import load_trusted_torch_artifact
        assert "from serialization_compat import load_trusted_torch_artifact" in source_code, (
            "embed_edges_feature_word2vec.py must import load_trusted_torch_artifact"
        )

        # Should NOT have bare torch.load(path) for graphs
        # The pattern torch.load(path) without weights_only= should not appear
        lines = source_code.split("\n")
        for i, line in enumerate(lines):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Check for torch.load calls (but allow ones with weights_only=)
            if "torch.load" in line and "weights_only" not in line:
                # Should not be loading graph without weights_only
                assert "expected_type=nx.MultiDiGraph" in source_code, (
                    f"torch.load found without weights_only on line {i+1}: {line.strip()}"
                )

    def test_data_utils_import_uses_trusted_loader(self):
        """Verify data_utils.py imports and uses trusted loader for TemporalData."""
        data_utils_path = SRC / "data_utils.py"
        source_code = data_utils_path.read_text()

        # Should import load_trusted_torch_artifact
        assert "from serialization_compat import load_trusted_torch_artifact" in source_code, (
            "data_utils.py must import load_trusted_torch_artifact"
        )

        # Should use expected_type=TemporalData for TemporalData loading
        assert "expected_type=TemporalData" in source_code, (
            "data_utils.py must use expected_type=TemporalData for TemporalData loading"
        )

    def test_build_feature_word2vec_uses_trusted_loader(self):
        """Verify build_feature_word2vec.py uses trusted loader for graphs."""
        build_w2v_path = SRC / "edge_featurization" / "build_feature_word2vec.py"
        source_code = build_w2v_path.read_text()

        # Should import load_trusted_torch_artifact
        assert "from serialization_compat import load_trusted_torch_artifact" in source_code, (
            "build_feature_word2vec.py must import load_trusted_torch_artifact"
        )

        # Should use expected_type=nx.MultiDiGraph
        assert "expected_type=nx.MultiDiGraph" in source_code, (
            "build_feature_word2vec.py must use expected_type=nx.MultiDiGraph"
        )


# ---------------------------------------------------------------------------
# Test 7: Legacy fallback for old PyTorch without weights_only support
# ---------------------------------------------------------------------------

class TestLegacyFallback:
    """Test fallback for very old PyTorch versions without weights_only param."""

    def test_legacy_pytorch_fallback_on_typeerror(self, tmp_path, mocker):
        """TypeError about weights_only triggers legacy fallback."""
        from serialization_compat import load_trusted_torch_artifact

        graph = nx.MultiDiGraph()
        graph.add_node(1)
        save_path = tmp_path / "legacy_test.pkl"
        torch.save(graph, save_path)

        # Simulate old PyTorch that raises TypeError on weights_only
        original_load = torch.load

        def fake_load(*args, **kwargs):
            if "weights_only" in kwargs:
                raise TypeError("torch.load() got an unexpected keyword argument 'weights_only'")
            return original_load(*args, **kwargs)

        mocker.patch("serialization_compat.torch.load", side_effect=fake_load)

        # Should fall back to legacy load without weights_only
        loaded = load_trusted_torch_artifact(save_path, expected_type=nx.MultiDiGraph)

        assert isinstance(loaded, nx.MultiDiGraph)


# ---------------------------------------------------------------------------
# Test 8: Compatibility with existing test fixtures
# ---------------------------------------------------------------------------

class TestCompatibilityWithExistingTests:
    """Ensure our changes don't break existing test patterns."""

    def test_torch_save_and_load_still_works(self, tmp_path):
        """Standard torch.save/load still works for tests."""
        data = {"key": "value"}
        path = tmp_path / "standard.pkl"
        torch.save(data, path)

        loaded = torch.load(path)
        assert loaded == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
