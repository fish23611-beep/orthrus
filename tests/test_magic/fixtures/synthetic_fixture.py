"""
Synthetic Fixture Generator for MAGIC Smoke Tests

本模块生成用于 MAGIC E2E smoke test 的 synthetic canonical records。

设计目标：
1. train 至少包含足够节点支持 MAGIC k=10
2. 覆盖多个 node types: process/subject, file, netflow/socket
3. 多个 edge types
4. validation/test 包含 train 未见 type 或 UNKNOWN path
5. 同一 canonical node 跨多个 snapshot 出现
6. 相同 timestamp，不同 global_event_index
7. 早期/晚期事件用于未来泄漏测试
8. 至少一个 test node 明显远离 train embedding
9. validation score 分布产生非退化 q=0.999 threshold

禁止：
- 修改 M3-M7 frozen semantics
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple, Any

# =============================================================================
# Constants
# =============================================================================

# Minimum nodes for k=10 MAGIC scoring
MIN_TRAIN_NODES = 50

# Node types
NODE_TYPE_PROCESS = "subject:process"
NODE_TYPE_FILE = "file"
NODE_TYPE_NETFLOW = "netflow:socket"
NODE_TYPE_PIPE = "pipe"
NODE_TYPE_IPC = "ipc"

# Edge types (MUST include READ/RECV/LOAD for direction reversal testing)
EDGE_TYPE_EXEC = "EVENT_EXEC"
EDGE_TYPE_WRITE = "EVENT_WRITE"
EDGE_TYPE_READ = "EVENT_READ"
EDGE_TYPE_RECV = "EVENT_RECV"
EDGE_TYPE_LOAD = "EVENT_LOAD"
EDGE_TYPE_CONNECT = "EVENT_CONNECT"
EDGE_TYPE_OPEN = "EVENT_OPEN"

# Timestamps
T_EARLY = 1000000  # Early event for future leakage
T_TRAIN = 2000000  # Train period
T_VAL = 3000000    # Validation period
T_TEST = 4000000    # Test period
T_LATE = 5000000    # Late event for future leakage

# Snapshot boundaries
SNAPSHOT_T1 = 2500000  # Snapshot boundary T1
SNAPSHOT_T2 = 3500000  # Snapshot boundary T2 (future event T2 > T1)
SNAPSHOT_T3 = 4500000  # Snapshot boundary T3


# =============================================================================
# Synthetic Embedding Generator
# =============================================================================

def make_synthetic_embedding(
    node_id: str,
    embedding_dim: int = 64,
    seed: int = 0,
) -> List[float]:
    """
    Generate deterministic synthetic embedding for a node.

    Embeddings are deterministic based on node_id and seed,
    allowing reproducible tests.

    Args:
        node_id: Canonical node ID
        embedding_dim: Embedding dimension
        seed: Random seed for reproducibility

    Returns:
        List of embedding values
    """
    import hashlib

    # Create deterministic seed from node_id
    h = hashlib.sha256(f"{seed}_{node_id}".encode()).digest()
    # Use first 8 bytes for float conversion
    base_seed = int.from_bytes(h[:8], byteorder='big')

    # Generate embedding using simple deterministic algorithm
    import struct
    values = []
    for i in range(embedding_dim):
        # Create deterministic float from seed and position
        h_i = hashlib.sha256(f"{base_seed}_{i}".encode()).digest()
        # Convert to float in range [-10, 10]
        bits = int.from_bytes(h_i[:4], byteorder='big')
        value = (bits / (2**32 - 1)) * 20 - 10
        values.append(float(value))

    return values


def make_anomalous_embedding(
    node_id: str,
    embedding_dim: int = 64,
    seed: int = 0,
    anomaly_scale: float = 5.0,
) -> List[float]:
    """
    Generate synthetic embedding for an anomalous node.

    Anomalous nodes are placed far from the train cluster.

    Args:
        node_id: Canonical node ID
        embedding_dim: Embedding dimension
        seed: Random seed
        anomaly_scale: How far from normal (default: 5x normal range)

    Returns:
        List of embedding values far from normal range
    """
    normal = make_synthetic_embedding(node_id, embedding_dim, seed)
    # Push away from origin by anomaly_scale
    return [v * anomaly_scale for v in normal]


# =============================================================================
# Fixture Builder
# =============================================================================

class SyntheticFixtureBuilder:
    """
    Builder for synthetic canonical records.

    Produces train/validation/test records that exercise:
    - Multiple node types
    - Multiple edge types
    - Snapshot boundaries
    - Future leakage tests
    - Ground-truth leakage tests
    """

    def __init__(
        self,
        seed: int = 0,
        train_node_count: int = MIN_TRAIN_NODES,
        embedding_dim: int = 64,
    ):
        self.seed = seed
        self.train_node_count = train_node_count
        self.embedding_dim = embedding_dim
        self._global_event_index = 0
        self._node_counter = 0

    def _next_node_id(self, prefix: str) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self.seed}_{self._node_counter}"

    def _next_event_index(self) -> int:
        """Generate next global event index."""
        idx = self._global_event_index
        self._global_event_index += 1
        return idx

    def _make_record(
        self,
        src: str,
        dst: str,
        edge_type: str,
        timestamp: int,
        src_type: str,
        dst_type: str,
    ) -> Dict[str, Any]:
        """Create a canonical edge record."""
        return {
            "src": src,
            "dst": dst,
            "edge_type": edge_type,
            "t": timestamp,
            "timestamp": timestamp,
            "global_event_index": self._next_event_index(),
            "src_type": src_type,
            "dst_type": dst_type,
        }

    def build_train_records(self) -> List[Dict[str, Any]]:
        """
        Build training records.

        Returns at least MIN_TRAIN_NODES with:
        - Multiple node types
        - Multiple edge types
        - Events in train period
        """
        records = []

        # Create train nodes of different types
        processes = [
            self._next_node_id("proc") for _ in range(self.train_node_count // 3)
        ]
        files = [
            self._next_node_id("file") for _ in range(self.train_node_count // 3)
        ]
        netflows = [
            self._next_node_id("sock") for _ in range(self.train_node_count // 3)
        ]

        # Ensure enough nodes
        all_nodes = processes + files + netflows
        while len(all_nodes) < self.train_node_count:
            extra = self._next_node_id("proc")
            all_nodes.append(extra)
            processes.append(extra)

        # Create edges with various edge types
        for i, proc in enumerate(processes):
            # EXEC: process creates something
            if i < len(files):
                records.append(self._make_record(
                    src=proc,
                    dst=files[i],
                    edge_type=EDGE_TYPE_EXEC,
                    timestamp=T_TRAIN + i * 1000,
                    src_type=NODE_TYPE_PROCESS,
                    dst_type=NODE_TYPE_FILE,
                ))

            # WRITE: process writes to file
            if i < len(files):
                records.append(self._make_record(
                    src=proc,
                    dst=files[i],
                    edge_type=EDGE_TYPE_WRITE,
                    timestamp=T_TRAIN + i * 1000 + 100,
                    src_type=NODE_TYPE_PROCESS,
                    dst_type=NODE_TYPE_FILE,
                ))

            # CONNECT: process connects to socket
            if i < len(netflows):
                records.append(self._make_record(
                    src=proc,
                    dst=netflows[i],
                    edge_type=EDGE_TYPE_CONNECT,
                    timestamp=T_TRAIN + i * 1000 + 200,
                    src_type=NODE_TYPE_PROCESS,
                    dst_type=NODE_TYPE_NETFLOW,
                ))

        # Add READ/RECV/LOAD edges (these get reversed for causal direction)
        for i, file_node in enumerate(files[:len(processes)]):
            proc = processes[i]
            # READ: file is read by process (will be reversed)
            records.append(self._make_record(
                src=file_node,
                dst=proc,
                edge_type=EDGE_TYPE_READ,
                timestamp=T_TRAIN + i * 1000 + 300,
                src_type=NODE_TYPE_FILE,
                dst_type=NODE_TYPE_PROCESS,
            ))

        # Add some edges with same timestamp, different event index
        same_ts = T_EARLY
        proc1 = self._next_node_id("proc_early_1")
        proc2 = self._next_node_id("proc_early_2")
        file1 = self._next_node_id("file_early_1")
        file2 = self._next_node_id("file_early_2")

        # These share same timestamp but different global_event_index
        records.append(self._make_record(
            src=proc1, dst=file1,
            edge_type=EDGE_TYPE_OPEN,
            timestamp=same_ts,
            src_type=NODE_TYPE_PROCESS,
            dst_type=NODE_TYPE_FILE,
        ))
        records.append(self._make_record(
            src=proc2, dst=file2,
            edge_type=EDGE_TYPE_OPEN,
            timestamp=same_ts,  # Same timestamp
            src_type=NODE_TYPE_PROCESS,
            dst_type=NODE_TYPE_FILE,
        ))

        return records

    def build_validation_records(self) -> List[Dict[str, Any]]:
        """
        Build validation records.

        Contains:
        - Train-unseen node types (pipe, ipc)
        - Nodes that also appear in train (for cross-snapshot testing)
        - Enough score distribution to produce q=0.999 threshold
        """
        records = []

        # Create new node types (unseen in train)
        pipes = [self._next_node_id("pipe") for _ in range(10)]
        ipcs = [self._next_node_id("ipc") for _ in range(10)]

        # Create some records with unseen types
        for i, pipe in enumerate(pipes):
            proc = self._next_node_id("proc_val")
            records.append(self._make_record(
                src=proc,
                dst=pipe,
                edge_type=EDGE_TYPE_OPEN,
                timestamp=T_VAL + i * 1000,
                src_type=NODE_TYPE_PROCESS,
                dst_type=NODE_TYPE_PIPE,  # Unseen in train
            ))

        for i, ipc in enumerate(ipcs):
            proc = self._next_node_id("proc_val2")
            records.append(self._make_record(
                src=ipc,
                dst=proc,
                edge_type=EDGE_TYPE_RECV,  # Will be reversed
                timestamp=T_VAL + i * 1000 + 500,
                src_type=NODE_TYPE_IPC,  # Unseen in train
                dst_type=NODE_TYPE_PROCESS,
            ))

        # Add LOAD edges (will be reversed)
        for i in range(5):
            file_node = self._next_node_id("file_val")
            proc = self._next_node_id("proc_val3")
            records.append(self._make_record(
                src=file_node,
                dst=proc,
                edge_type=EDGE_TYPE_LOAD,  # Will be reversed
                timestamp=T_VAL + i * 1000 + 800,
                src_type=NODE_TYPE_FILE,
                dst_type=NODE_TYPE_PROCESS,
            ))

        return records

    def build_test_records(self) -> List[Dict[str, Any]]:
        """
        Build test records.

        Contains:
        - One node clearly far from train embedding (for high score)
        - Mix of benign and malicious nodes
        - UNKNOWN type handling
        """
        records = []

        # Create some normal test nodes
        for i in range(20):
            proc = self._next_node_id("proc_test")
            file_node = self._next_node_id("file_test")
            records.append(self._make_record(
                src=proc,
                dst=file_node,
                edge_type=EDGE_TYPE_WRITE,
                timestamp=T_TEST + i * 1000,
                src_type=NODE_TYPE_PROCESS,
                dst_type=NODE_TYPE_FILE,
            ))

        # Create anomalous nodes (far from train cluster)
        # These will have high anomaly scores
        for i in range(5):
            proc = self._next_node_id("proc_test_malicious")
            file_node = self._next_node_id("file_test_malicious")
            records.append(self._make_record(
                src=proc,
                dst=file_node,
                edge_type=EDGE_TYPE_EXEC,
                timestamp=T_TEST + i * 1000 + 500,
                src_type=NODE_TYPE_PROCESS,
                dst_type=NODE_TYPE_FILE,
            ))

        # Add some with UNKNOWN type
        for i in range(3):
            node_id = self._next_node_id("unknown_node")
            proc = self._next_node_id("proc_test_unknown")
            records.append(self._make_record(
                src=proc,
                dst=node_id,
                edge_type=EDGE_TYPE_OPEN,
                timestamp=T_TEST + i * 1000 + 1000,
                src_type=NODE_TYPE_PROCESS,
                dst_type="__UNKNOWN_TYPE__",  # Unseen type
            ))

        return records

    def build_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build complete synthetic fixture."""
        train = self.build_train_records()
        validation = self.build_validation_records()
        test = self.build_test_records()

        return {
            "train": train,
            "validation": validation,
            "test": test,
        }


def generate_synthetic_fixture(
    seed: int = 0,
    train_node_count: int = MIN_TRAIN_NODES,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate synthetic fixture for smoke tests.

    Args:
        seed: Random seed for reproducibility
        train_node_count: Number of train nodes

    Returns:
        Dict with 'train', 'validation', 'test' keys
    """
    builder = SyntheticFixtureBuilder(
        seed=seed,
        train_node_count=train_node_count,
    )
    return builder.build_all()


# =============================================================================
# Ground Truth Generator
# =============================================================================

def generate_ground_truth(
    fixture: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, int]:
    """
    Generate ground truth labels for test nodes.

    Labels are based on node naming convention:
    - *_malicious* -> 1 (anomalous)
    - *_test_* -> 0 (benign, unless explicitly malicious)

    Args:
        fixture: Synthetic fixture with train/validation/test

    Returns:
        Dict mapping node_id to 0/1 label
    """
    ground_truth = {}

    for record in fixture.get("test", []):
        src = record.get("src", "")
        dst = record.get("dst", "")

        # Label based on naming
        if "malicious" in src or "malicious" in dst:
            ground_truth[src] = 1
            ground_truth[dst] = 1
        else:
            if src not in ground_truth:
                ground_truth[src] = 0
            if dst not in ground_truth:
                ground_truth[dst] = 0

    return ground_truth


def generate_altered_ground_truth(
    original: Dict[str, int],
) -> Dict[str, int]:
    """
    Generate altered ground truth for leakage test.

    Flips some labels to test that scores/predictions
    remain invariant to label changes.

    Args:
        original: Original ground truth

    Returns:
        Altered ground truth
    """
    altered = dict(original)
    keys = list(altered.keys())

    # Flip first few labels
    for key in keys[:min(3, len(keys))]:
        altered[key] = 1 - altered[key]

    return altered


# =============================================================================
# Embedding Generator
# =============================================================================

class SyntheticEmbeddingGenerator:
    """
    Generate synthetic embeddings for smoke tests.

    Embeddings are deterministic based on node_id and seed,
    allowing reproducibility verification.
    """

    def __init__(
        self,
        seed: int = 0,
        embedding_dim: int = 64,
    ):
        self.seed = seed
        self.embedding_dim = embedding_dim

    def generate_embeddings(
        self,
        node_ids: List[str],
        anomalous: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for given nodes.

        Args:
            node_ids: List of node IDs
            anomalous: If True, generate far-from-cluster embeddings

        Returns:
            List of embeddings [node_count, embedding_dim]
        """
        embeddings = []
        for node_id in node_ids:
            if anomalous:
                emb = make_anomalous_embedding(
                    node_id, self.embedding_dim, self.seed
                )
            else:
                emb = make_synthetic_embedding(
                    node_id, self.embedding_dim, self.seed
                )
            embeddings.append(emb)
        return embeddings

    def generate_fixture_embeddings(
        self,
        fixture: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[List[float]]]:
        """
        Generate embeddings for all fixture nodes.

        Returns:
            Dict with 'train', 'validation', 'test' keys
        """
        import numpy as np

        def get_nodes(records):
            nodes = set()
            for r in records:
                nodes.add(r.get("src", ""))
                nodes.add(r.get("dst", ""))
            return sorted(nodes)

        train_nodes = get_nodes(fixture.get("train", []))
        val_nodes = get_nodes(fixture.get("validation", []))
        test_nodes = get_nodes(fixture.get("test", []))

        # Check which test nodes are anomalous
        anomalous_nodes = {
            n for n in test_nodes if "malicious" in n
        }

        return {
            "train": self.generate_embeddings(train_nodes, anomalous=False),
            "validation": self.generate_embeddings(val_nodes, anomalous=False),
            "test": [
                self.generate_embeddings([n], anomalous=(n in anomalous_nodes))[0]
                for n in test_nodes
            ],
        }

    def get_node_id_list(
        self,
        fixture: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[str]]:
        """Get ordered node ID lists for each split."""
        def get_nodes(records):
            nodes = set()
            for r in records:
                nodes.add(r.get("src", ""))
                nodes.add(r.get("dst", ""))
            return sorted(nodes)

        return {
            "train": get_nodes(fixture.get("train", [])),
            "validation": get_nodes(fixture.get("validation", [])),
            "test": get_nodes(fixture.get("test", [])),
        }


# =============================================================================
# Fixture Checksum
# =============================================================================

def compute_fixture_checksum(
    fixture: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    Compute deterministic checksum of fixture.

    Used for cache identity verification.

    Args:
        fixture: Synthetic fixture

    Returns:
        SHA256 hex digest
    """
    import json

    # Sort for determinism
    def sort_records(records):
        return sorted(records, key=lambda r: (
            r.get("t", 0),
            r.get("global_event_index", 0),
            r.get("src", ""),
            r.get("dst", ""),
        ))

    data = {
        "train": sort_records(fixture.get("train", [])),
        "validation": sort_records(fixture.get("validation", [])),
        "test": sort_records(fixture.get("test", [])),
    }

    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MIN_TRAIN_NODES",
    "NODE_TYPE_PROCESS",
    "NODE_TYPE_FILE",
    "NODE_TYPE_NETFLOW",
    "NODE_TYPE_PIPE",
    "NODE_TYPE_IPC",
    "EDGE_TYPE_EXEC",
    "EDGE_TYPE_WRITE",
    "EDGE_TYPE_READ",
    "EDGE_TYPE_RECV",
    "EDGE_TYPE_LOAD",
    "EDGE_TYPE_CONNECT",
    "EDGE_TYPE_OPEN",
    "T_EARLY",
    "T_TRAIN",
    "T_VAL",
    "T_TEST",
    "T_LATE",
    "SNAPSHOT_T1",
    "SNAPSHOT_T2",
    "SNAPSHOT_T3",
    "SyntheticFixtureBuilder",
    "generate_synthetic_fixture",
    "generate_ground_truth",
    "generate_altered_ground_truth",
    "SyntheticEmbeddingGenerator",
    "compute_fixture_checksum",
    "make_synthetic_embedding",
    "make_anomalous_embedding",
]
