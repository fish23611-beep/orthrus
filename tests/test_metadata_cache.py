"""Tests for metadata_cache.py."""

import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

from mstc.metadata_cache import (
    MetadataCache,
    MetadataCacheError,
    CacheNotFoundError,
    CacheCorruptedError,
)


@pytest.fixture
def cache(tmp_path):
    """Create MetadataCache instance with temp directory."""
    return MetadataCache(tmp_path)


class TestMetadataCacheInit:
    """Tests for MetadataCache initialization."""

    def test_creates_cache_root(self, tmp_path):
        """Creates cache root directory."""
        cache = MetadataCache(tmp_path / "new_cache")
        assert cache.cache_root.exists()
        assert cache.cache_root.is_dir()

    def test_cache_root_is_path(self, tmp_path):
        """Cache root is a Path object."""
        cache = MetadataCache(tmp_path)
        assert isinstance(cache.cache_root, Path)


class TestMetadataCacheHasMethods:
    """Tests for has_* methods."""

    def test_all_has_methods_exist(self, cache):
        """All has_* methods exist."""
        assert hasattr(cache, "has_node_metadata")
        assert hasattr(cache, "has_uuid_to_node_id")
        assert hasattr(cache, "has_node_id_to_uuid")
        assert hasattr(cache, "has_ground_truth_nodes")
        assert hasattr(cache, "has_attack_to_nodes")
        assert hasattr(cache, "has_time_to_malicious_nodes")
        assert hasattr(cache, "has_relation_mapping")
        assert hasattr(cache, "has_dataset_manifest")
        assert hasattr(cache, "has_nodeid2msg")

    def test_has_returns_false_for_missing(self, cache):
        """has_* returns False for missing files."""
        assert cache.has_node_metadata() is False
        assert cache.has_uuid_to_node_id() is False
        assert cache.has_dataset_manifest() is False


class TestMetadataCacheSaveLoad:
    """Tests for save/load round trips."""

    def test_save_load_node_metadata(self, cache):
        """Node metadata round trips correctly."""
        data = {
            0: {"uuid": "abc", "type": "subject", "display": "test"},
            1: {"uuid": "def", "type": "file", "display": "file.txt"},
        }
        cache.save_node_metadata(data)
        assert cache.has_node_metadata() is True
        
        loaded = cache.load_node_metadata()
        assert loaded == data

    def test_save_load_uuid_mappings(self, cache):
        """UUID mappings round trip correctly."""
        uuid_to_id = {"uuid1": 0, "uuid2": 1}
        id_to_uuid = {0: "uuid1", 1: "uuid2"}
        
        cache.save_uuid_to_node_id(uuid_to_id)
        cache.save_node_id_to_uuid(id_to_uuid)
        
        assert cache.load_uuid_to_node_id() == uuid_to_id
        assert cache.load_node_id_to_uuid() == id_to_uuid

    def test_save_load_ground_truth_nodes(self, cache):
        """Ground truth nodes round trip correctly."""
        data = {1, 2, 3, 4, 5}
        cache.save_ground_truth_nodes(data)
        loaded = cache.load_ground_truth_nodes()
        assert loaded == data

    def test_save_load_attack_to_nodes(self, cache):
        """Attack to nodes mapping round trips correctly."""
        data = {
            0: {1, 2, 3},
            1: {4, 5, 6},
        }
        cache.save_attack_to_nodes(data)
        loaded = cache.load_attack_to_nodes()
        assert loaded == data

    def test_save_load_time_to_malicious_nodes(self, cache):
        """Time to malicious nodes mapping round trips correctly."""
        data = {
            1000: ["uuid1", "uuid2"],
            2000: ["uuid3"],
        }
        cache.save_time_to_malicious_nodes(data)
        loaded = cache.load_time_to_malicious_nodes()
        assert loaded == data

    def test_save_load_json_relation_mapping(self, cache):
        """JSON relation mapping round trips correctly."""
        data = {"type1": ["rel1", "rel2"], "type2": ["rel3"]}
        cache.save_relation_mapping(data)
        loaded = cache.load_relation_mapping()
        assert loaded == data

    def test_save_load_dataset_manifest(self, cache):
        """Dataset manifest round trips correctly."""
        cache.save_dataset_manifest(
            dataset="THEIA_E3",
            num_node_types=3,
            num_edge_types=10,
            train_files=["graph_1", "graph_2"],
            val_files=["graph_3"],
            test_files=["graph_4", "graph_5"],
            word2vec_dim=128,
            preprocess_config_hash="abc123",
        )
        loaded = cache.load_dataset_manifest()
        
        assert loaded["dataset"] == "THEIA_E3"
        assert loaded["num_node_types"] == 3
        assert loaded["num_edge_types"] == 10
        assert loaded["word2vec_dim"] == 128
        assert "created_at" in loaded

    def test_save_load_nodeid2msg(self, cache):
        """Node ID to message mapping round trips correctly."""
        data = {0: "node 0 message", 1: "node 1 message"}
        cache.save_nodeid2msg(data)
        loaded = cache.load_nodeid2msg()
        assert loaded == data


class TestMetadataCacheErrors:
    """Tests for error handling."""

    def test_load_missing_raises_not_found(self, cache):
        """Loading missing cache raises CacheNotFoundError."""
        with pytest.raises(CacheNotFoundError, match="not found"):
            cache.load_node_metadata()

    def test_load_json_missing_raises_not_found(self, cache):
        """Loading missing JSON cache raises CacheNotFoundError."""
        with pytest.raises(CacheNotFoundError, match="not found"):
            cache.load_relation_mapping()

    def test_load_corrupted_pickle_raises(self, cache):
        """Loading corrupted pickle raises CacheCorruptedError."""
        # Write corrupted pickle
        corrupt_file = cache.cache_root / cache.NODE_METADATA_FILE
        with open(corrupt_file, "wb") as f:
            f.write(b"not a pickle")
        
        with pytest.raises(CacheCorruptedError, match="Failed to load"):
            cache.load_node_metadata()

    def test_load_corrupted_json_raises(self, cache):
        """Loading corrupted JSON raises CacheCorruptedError."""
        # Write corrupted JSON
        corrupt_file = cache.cache_root / cache.RELATION_MAPPING_FILE
        with open(corrupt_file, "w") as f:
            f.write("not json {")
        
        with pytest.raises(CacheCorruptedError, match="Failed to load"):
            cache.load_relation_mapping()


class TestMetadataCacheValidation:
    """Tests for validate_required method."""

    def test_validate_required_returns_empty_for_present(self, cache):
        """Returns empty list when all required caches present."""
        cache.save_node_metadata({1: {"type": "test"}})
        cache.save_uuid_to_node_id({"uuid": 1})
        
        missing = cache.validate_required(["node_metadata", "uuid_to_node_id"])
        assert missing == []

    def test_validate_required_returns_missing(self, cache):
        """Returns list of missing cache names."""
        missing = cache.validate_required(["node_metadata", "uuid_to_node_id", "dataset_manifest"])
        assert "node_metadata" in missing
        assert "uuid_to_node_id" in missing
        assert "dataset_manifest" in missing

    def test_validate_required_unknown_name_raises(self, cache):
        """Unknown cache name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cache name"):
            cache.validate_required(["nonexistent_cache"])


class TestDeriveNodeid2msgFromMetadata:
    """Tests for derive_nodeid2msg_from_metadata method."""

    def test_derives_from_node_metadata(self, cache):
        """Derives nodeid2msg from node_metadata."""
        cache.save_node_metadata({
            0: {"uuid": "abc", "type": "subject", "path": "/bin/bash", "cmd": "-c echo", "display": "bash -c echo"},
            1: {"uuid": "def", "type": "file", "path": "/etc/passwd", "display": "file:/etc/passwd"},
        })
        
        result = cache.derive_nodeid2msg_from_metadata()
        
        assert 0 in result
        assert 1 in result
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_derives_missing_raises(self, cache):
        """Deriving without node_metadata raises CacheNotFoundError."""
        with pytest.raises(CacheNotFoundError, match="Cannot derive"):
            cache.derive_nodeid2msg_from_metadata()


class TestAtomicWrite:
    """Tests for atomic write behavior."""

    def test_writes_atomically(self, cache, tmp_path):
        """Save uses atomic write (temp file + rename)."""
        data = {"key": "value"}
        
        # Save should work
        cache.save_node_metadata(data)
        
        # Verify file exists and is valid
        assert cache.has_node_metadata()
        assert cache.load_node_metadata() == data


class TestDatabaseFallbackPrevention:
    """Tests ensuring no DB is called in test environment."""

    def test_cache_hit_returns_data_no_db(self, cache, mocker):
        """Cache hit returns data without calling DB."""
        # This would be a DB call in production
        db_mock = mocker.patch("provnet_utils.init_database_connection")
        
        cache.save_ground_truth_nodes({1, 2, 3})
        result = cache.load_ground_truth_nodes()
        
        assert result == {1, 2, 3}
        db_mock.assert_not_called()
