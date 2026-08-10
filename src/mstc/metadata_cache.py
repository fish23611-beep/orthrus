"""
src/mstc/metadata_cache.py

Unified metadata cache management for ORTHRUS pipeline.

Supports:
- node_metadata.pkl
- uuid_to_node_id.pkl
- node_id_to_uuid.pkl
- ground_truth_nodes.pkl
- attack_to_nodes.pkl
- time_to_malicious_nodes.pkl
- relation_mapping.json
- dataset_manifest.json
- nodeid2msg.pkl (for testing node messages)

Single-node metadata structure (node_metadata.pkl):
{
    "uuid": str,
    "type": "subject" | "file" | "netflow",
    "path": str | None,
    "cmd": str | None,
    "local_ip": str | None,
    "local_port": str | None,
    "remote_ip": str | None,
    "remote_port": str | None,
    "display": str,
}

dataset_manifest.json structure:
{
    "dataset": str,
    "num_node_types": int,
    "num_edge_types": int,
    "train_files": list[str],
    "val_files": list[str],
    "test_files": list[str],
    "word2vec_dim": int,
    "preprocess_config_hash": str,
    "created_at": str (ISO8601),
}
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class MetadataCacheError(Exception):
    """Base exception for metadata cache operations."""
    pass


class CacheNotFoundError(MetadataCacheError):
    """Raised when a cache file does not exist."""
    pass


class CacheCorruptedError(MetadataCacheError):
    """Raised when a cache file is corrupted or unreadable."""
    pass


class MetadataCache:
    """
    Unified metadata cache manager.
    
    Provides load/save for various cache files used by the ORTHRUS pipeline.
    All paths are derived from the cache root directory.
    
    Parameters
    ----------
    cache_root : str | Path
        Root directory for all metadata caches.
        Typically cfg._metadata_dir or a derived path.
    """

    # Cache file names
    NODE_METADATA_FILE = "node_metadata.pkl"
    UUID_TO_NODE_ID_FILE = "uuid_to_node_id.pkl"
    NODE_ID_TO_UUID_FILE = "node_id_to_uuid.pkl"
    GROUND_TRUTH_NODES_FILE = "ground_truth_nodes.pkl"
    ATTACK_TO_NODES_FILE = "attack_to_nodes.pkl"
    TIME_TO_MALICIOUS_NODES_FILE = "time_to_malicious_nodes.pkl"
    RELATION_MAPPING_FILE = "relation_mapping.json"
    DATASET_MANIFEST_FILE = "dataset_manifest.json"
    NODEID2MSG_FILE = "nodeid2msg.pkl"
    COMPLETION_MARKER_FILE = ".preprocess_metadata_complete"

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, filename: str) -> Path:
        """Get full path for a cache file."""
        return self.cache_root / filename

    def _check_exists(self, filename: str) -> bool:
        """Check if a cache file exists."""
        return self._get_cache_path(filename).exists()

    def _atomic_write_pickle(self, path: Path, data: Any) -> None:
        """Write pickle data atomically using temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        """Write JSON data atomically using temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            suffix=".tmp",
            text=True
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _load_pickle(self, filename: str) -> Any:
        """Load pickle data, raising clear errors on miss or corruption."""
        path = self._get_cache_path(filename)
        if not path.exists():
            raise CacheNotFoundError(
                f"Cache file not found: {path}"
            )
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise CacheCorruptedError(
                f"Failed to load cache {path}: {e}"
            ) from e

    def _load_json(self, filename: str) -> dict:
        """Load JSON data, raising clear errors on miss or corruption."""
        path = self._get_cache_path(filename)
        if not path.exists():
            raise CacheNotFoundError(
                f"Cache file not found: {path}"
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise CacheCorruptedError(
                f"Failed to load cache {path}: {e}"
            ) from e

    # ------------------------------------------------------------------- #
    # Has checks
    # ------------------------------------------------------------------- #

    def has_node_metadata(self) -> bool:
        return self._check_exists(self.NODE_METADATA_FILE)

    def has_uuid_to_node_id(self) -> bool:
        return self._check_exists(self.UUID_TO_NODE_ID_FILE)

    def has_node_id_to_uuid(self) -> bool:
        return self._check_exists(self.NODE_ID_TO_UUID_FILE)

    def has_ground_truth_nodes(self) -> bool:
        return self._check_exists(self.GROUND_TRUTH_NODES_FILE)

    def has_attack_to_nodes(self) -> bool:
        return self._check_exists(self.ATTACK_TO_NODES_FILE)

    def has_time_to_malicious_nodes(self) -> bool:
        return self._check_exists(self.TIME_TO_MALICIOUS_NODES_FILE)

    def has_relation_mapping(self) -> bool:
        return self._check_exists(self.RELATION_MAPPING_FILE)

    def has_dataset_manifest(self) -> bool:
        return self._check_exists(self.DATASET_MANIFEST_FILE)

    def has_nodeid2msg(self) -> bool:
        return self._check_exists(self.NODEID2MSG_FILE)

    # ------------------------------------------------------------------- #
    # Load methods (raise on cache miss)
    # ------------------------------------------------------------------- #

    def load_node_metadata(self) -> dict[int, dict]:
        """Load node metadata mapping node_id -> metadata dict."""
        return self._load_pickle(self.NODE_METADATA_FILE)

    def load_uuid_to_node_id(self) -> dict[str, int]:
        """Load UUID to node ID mapping."""
        return self._load_pickle(self.UUID_TO_NODE_ID_FILE)

    def load_node_id_to_uuid(self) -> dict[int, str]:
        """Load node ID to UUID mapping."""
        return self._load_pickle(self.NODE_ID_TO_UUID_FILE)

    def load_ground_truth_nodes(self) -> set[int]:
        """Load ground truth malicious node IDs."""
        return self._load_pickle(self.GROUND_TRUTH_NODES_FILE)

    def load_attack_to_nodes(self) -> dict[int, set[int]]:
        """Load attack ID to node IDs mapping."""
        return self._load_pickle(self.ATTACK_TO_NODES_FILE)

    def load_time_to_malicious_nodes(self) -> dict[int, list]:
        """Load timestamp to malicious nodes mapping."""
        return self._load_pickle(self.TIME_TO_MALICIOUS_NODES_FILE)

    def load_relation_mapping(self) -> dict:
        """Load relation mapping."""
        return self._load_json(self.RELATION_MAPPING_FILE)

    def load_dataset_manifest(self) -> dict:
        """Load dataset manifest."""
        return self._load_json(self.DATASET_MANIFEST_FILE)

    def load_nodeid2msg(self) -> dict:
        """Load node ID to message mapping for testing output."""
        return self._load_pickle(self.NODEID2MSG_FILE)

    # ------------------------------------------------------------------- #
    # Save methods
    # ------------------------------------------------------------------- #

    def save_node_metadata(self, data: dict[int, dict]) -> None:
        """Save node metadata."""
        path = self._get_cache_path(self.NODE_METADATA_FILE)
        self._atomic_write_pickle(path, data)

    def save_uuid_to_node_id(self, data: dict[str, int]) -> None:
        """Save UUID to node ID mapping."""
        path = self._get_cache_path(self.UUID_TO_NODE_ID_FILE)
        self._atomic_write_pickle(path, data)

    def save_node_id_to_uuid(self, data: dict[int, str]) -> None:
        """Save node ID to UUID mapping."""
        path = self._get_cache_path(self.NODE_ID_TO_UUID_FILE)
        self._atomic_write_pickle(path, data)

    def save_ground_truth_nodes(self, data: set[int]) -> None:
        """Save ground truth node IDs."""
        path = self._get_cache_path(self.GROUND_TRUTH_NODES_FILE)
        self._atomic_write_pickle(path, data)

    def save_attack_to_nodes(self, data: dict[int, set[int]]) -> None:
        """Save attack to nodes mapping."""
        path = self._get_cache_path(self.ATTACK_TO_NODES_FILE)
        self._atomic_write_pickle(path, data)

    def save_time_to_malicious_nodes(self, data: dict[int, list]) -> None:
        """Save time to malicious nodes mapping."""
        path = self._get_cache_path(self.TIME_TO_MALICIOUS_NODES_FILE)
        self._atomic_write_pickle(path, data)

    def save_relation_mapping(self, data: dict) -> None:
        """Save relation mapping."""
        path = self._get_cache_path(self.RELATION_MAPPING_FILE)
        self._atomic_write_json(path, data)

    def save_dataset_manifest(
        self,
        *,
        dataset: str,
        num_node_types: int,
        num_edge_types: int,
        train_files: list[str],
        val_files: list[str],
        test_files: list[str],
        word2vec_dim: int,
        preprocess_config_hash: str,
        corpus_scope: str = "official_full_dataset",
        word2vec_model_hash: Optional[str] = None,
    ) -> None:
        """Save dataset manifest with required fields."""
        manifest = {
            "dataset": dataset,
            "num_node_types": num_node_types,
            "num_edge_types": num_edge_types,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "word2vec_dim": word2vec_dim,
            "corpus_scope": corpus_scope,
            "preprocess_config_hash": preprocess_config_hash,
            "word2vec_model_hash": word2vec_model_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self._get_cache_path(self.DATASET_MANIFEST_FILE)
        self._atomic_write_json(path, manifest)

    def save_nodeid2msg(self, data: dict) -> None:
        """Save node ID to message mapping."""
        path = self._get_cache_path(self.NODEID2MSG_FILE)
        self._atomic_write_pickle(path, data)

    # ------------------------------------------------------------------- #
    # Validation
    # ------------------------------------------------------------------- #

    def validate_required(self, required_names: list[str]) -> list[str]:
        """
        Check which required caches are missing.

        Parameters
        ----------
        required_names : list[str]
            List of cache names to check. Valid names are:
            - "node_metadata"
            - "uuid_to_node_id"
            - "node_id_to_uuid"
            - "ground_truth_nodes"
            - "attack_to_nodes"
            - "time_to_malicious_nodes"
            - "relation_mapping"
            - "dataset_manifest"
            - "nodeid2msg"

        Returns
        -------
        list[str]
            List of missing cache names.
        """
        has_methods = {
            "node_metadata": self.has_node_metadata,
            "uuid_to_node_id": self.has_uuid_to_node_id,
            "node_id_to_uuid": self.has_node_id_to_uuid,
            "ground_truth_nodes": self.has_ground_truth_nodes,
            "attack_to_nodes": self.has_attack_to_nodes,
            "time_to_malicious_nodes": self.has_time_to_malicious_nodes,
            "relation_mapping": self.has_relation_mapping,
            "dataset_manifest": self.has_dataset_manifest,
            "nodeid2msg": self.has_nodeid2msg,
        }
        missing = []
        for name in required_names:
            checker = has_methods.get(name)
            if checker is None:
                raise ValueError(f"Unknown cache name: {name}")
            if not checker():
                missing.append(name)
        return missing

    # ------------------------------------------------------------------- #
    # Derive node messages from metadata
    # ------------------------------------------------------------------- #

    def derive_nodeid2msg_from_metadata(self) -> dict:
        """
        Derive node ID to message mapping from node metadata.
        
        Returns
        -------
        dict
            node_id -> message string
        """
        if not self.has_node_metadata():
            raise CacheNotFoundError(
                f"Cannot derive nodeid2msg: node_metadata not found at {self.cache_root}"
            )
        
        node_metadata = self.load_node_metadata()
        result = {}
        
        for node_id, meta in node_metadata.items():
            if isinstance(meta, dict):
                display = meta.get("display", "")
                if display:
                    result[node_id] = display
                else:
                    # Build display from available fields
                    parts = []
                    ntype = meta.get("type", "unknown")
                    if ntype == "subject":
                        path = meta.get("path", "")
                        cmd = meta.get("cmd", "")
                        parts.append(f"subject:{path}")
                        if cmd:
                            parts.append(cmd)
                    elif ntype == "file":
                        path = meta.get("path", "")
                        parts.append(f"file:{path}")
                    elif ntype == "netflow":
                        local_ip = meta.get("local_ip", "")
                        local_port = meta.get("local_port", "")
                        remote_ip = meta.get("remote_ip", "")
                        remote_port = meta.get("remote_port", "")
                        parts.append(f"netflow:{local_ip}:{local_port}->{remote_ip}:{remote_port}")
                    else:
                        parts.append(f"{ntype}:{meta.get('path', '')}")
                    
                    result[node_id] = " ".join(parts) if parts else f"node:{node_id}"
            else:
                result[node_id] = str(meta)
        
        return result

# Production implementations live separately to keep this cache container
# importable without database/config side effects.
def dump_from_postgres(cfg, cache: MetadataCache) -> None:
    from mstc.metadata_export import dump_from_postgres as _dump
    _dump(cfg, cache)


def export_metadata(cfg, force: bool = False) -> MetadataCache:
    from mstc.metadata_export import export_metadata as _export
    return _export(cfg, force=force)


def update_dataset_manifest(cfg, cache: Optional[MetadataCache] = None) -> None:
    from mstc.metadata_export import update_dataset_manifest as _update
    _update(cfg, cache=cache)


def metadata_complete(cache: MetadataCache) -> bool:
    from mstc.metadata_export import metadata_complete as _complete
    return _complete(cache)
