"""Tests for cache-first / DB-fallback behavior in labelling.py."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

import labelling


class TestCacheFirstGetGroundTruth:
    """Tests for get_ground_truth with cache-first strategy."""

    def test_cache_hit_returns_cached_data(self, mocker):
        """Cache hit returns cached data without calling DB."""
        # Create mock cache
        mock_cache = MagicMock()
        mock_cache.has_ground_truth_nodes.return_value = True
        mock_cache.has_uuid_to_node_id.return_value = True
        mock_cache.load_ground_truth_nodes.return_value = {1, 2, 3}
        mock_cache.load_uuid_to_node_id.return_value = {"uuid1": 1, "uuid2": 2}
        
        # Create cfg with metadata_dir
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        # Patch the cache getter
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        # Call get_ground_truth
        gt, paths, uuid_map = labelling.get_ground_truth(cfg)
        
        # Should return cached data
        assert gt == {1, 2, 3}
        assert uuid_map == {"uuid1": 1, "uuid2": 2}

    def test_detection_only_cache_miss_raises_clear_error(self, mocker):
        """Detection-only mode with cache miss raises clear error."""
        mock_cache = MagicMock()
        mock_cache.has_ground_truth_nodes.return_value = False
        mock_cache.validate_required.return_value = ["ground_truth_nodes"]
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        with pytest.raises(FileNotFoundError, match="detection_only mode"):
            labelling.get_ground_truth(cfg)


class TestCacheFirstGetGPOfEachAttack:
    """Tests for get_GP_of_each_attack with cache-first strategy."""

    def test_cache_hit_returns_cached_data(self, mocker):
        """Cache hit returns cached data without calling DB."""
        mock_cache = MagicMock()
        mock_cache.has_attack_to_nodes.return_value = True
        mock_cache.load_attack_to_nodes.return_value = {0: {1, 2}, 1: {3, 4}}
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        result = labelling.get_GP_of_each_attack(cfg)
        
        assert result == {0: {1, 2}, 1: {3, 4}}


class TestCacheFirstGetT2MaliciousNode:
    """Tests for get_t2malicious_node with cache-first strategy."""

    def test_cache_hit_returns_cached_data(self, mocker):
        """Cache hit returns cached data without calling DB."""
        mock_cache = MagicMock()
        mock_cache.has_time_to_malicious_nodes.return_value = True
        mock_cache.load_time_to_malicious_nodes.return_value = {1000: ["uuid1", "uuid2"]}
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        result = labelling.get_t2malicious_node(cfg)
        
        assert result == {1000: ["uuid1", "uuid2"]}

    def test_empty_cache_falls_through(self, mocker):
        """Empty cache (falsy) falls through to DB path."""
        mock_cache = MagicMock()
        mock_cache.has_time_to_malicious_nodes.return_value = True
        mock_cache.load_time_to_malicious_nodes.return_value = {}  # Empty = falsy
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="full_pipeline"),
            dataset=SimpleNamespace(
                ground_truth_relative_path=[],
                attack_to_time_window=[],
            ),
            _ground_truth_dir="/fake/gt",
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir="/fake/graphs")
            ),
        )
        
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        # Mock the DB connection to return a valid cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mocker.patch.object(labelling, "init_database_connection", return_value=(mock_cursor, MagicMock()))
        
        # Should fall through since cache is empty
        result = labelling.get_t2malicious_node(cfg)
        
        # Empty dict because no attacks defined
        assert result == {}


class TestDetectionOnlyModeHelpers:
    """Tests for detection_only mode helpers."""

    def test_is_detection_only_mode_true(self):
        """Returns True for detection_only mode."""
        cfg = SimpleNamespace(pipeline=SimpleNamespace(mode="detection_only"))
        assert labelling._is_detection_only_mode(cfg) is True

    def test_is_detection_only_mode_false(self):
        """Returns False for full_pipeline mode."""
        cfg = SimpleNamespace(pipeline=SimpleNamespace(mode="full_pipeline"))
        assert labelling._is_detection_only_mode(cfg) is False

    def test_is_detection_only_mode_missing(self):
        """Returns False when mode attribute is missing."""
        cfg = SimpleNamespace()
        assert labelling._is_detection_only_mode(cfg) is False


class TestNoRealDBConnection:
    """Tests ensuring no real DB connections in test environment."""

    def test_cache_hit_does_not_init_db(self, mocker):
        """Cache hit does not call init_database_connection."""
        mock_cache = MagicMock()
        mock_cache.has_ground_truth_nodes.return_value = True
        mock_cache.load_ground_truth_nodes.return_value = {1, 2}
        mock_cache.load_uuid_to_node_id.return_value = {}
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        db_mock = mocker.patch.object(labelling, "init_database_connection")
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        labelling.get_ground_truth(cfg)
        
        db_mock.assert_not_called()

    def test_detection_only_no_db_fallback(self, mocker):
        """Detection-only mode with cache miss does not call DB."""
        mock_cache = MagicMock()
        mock_cache.has_ground_truth_nodes.return_value = False
        mock_cache.validate_required.return_value = ["ground_truth_nodes"]
        
        cfg = SimpleNamespace(
            _metadata_dir="/fake/cache",
            pipeline=SimpleNamespace(mode="detection_only"),
        )
        
        db_mock = mocker.patch.object(labelling, "init_database_connection")
        mocker.patch.object(labelling, "_get_metadata_cache", return_value=mock_cache)
        
        with pytest.raises(FileNotFoundError):
            labelling.get_ground_truth(cfg)
        
        db_mock.assert_not_called()
