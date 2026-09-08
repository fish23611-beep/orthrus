"""
MAGIC Canonical Input Adapter - Neutral Graph Contract

本模块定义了 MAGIC Input Adapter 的中间数据合同，与 DGL 无关。
后续服务器 backend 负责将该 neutral contract 转换为 DGL graph。

设计依据：
- 冻结合同: docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md
- 审计报告: docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md
- 当前项目 canonical artifacts: src/data_utils.py

关键设计原则：
1. Neutral Graph Contract 与 DGL 解耦
2. Train-only type vocabulary (transductive 风险缓解)
3. Deterministic ordering: (timestamp, global_event_index)
4. Split isolation: train/val/test 显式分离

Python 3.8 兼容性：
- 不使用 slots=True (Python 3.10+)
- 使用 typing.FrozenSet 而非内置 frozenset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

# =============================================================================
# Type Definitions
# =============================================================================

class SplitType(Enum):
    """Canonical data split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class NodeRecord:
    """
    Neutral node record.

    对应 canonical artifact 字段:
    - node_id: 节点全局唯一标识 (UUID string)
    - node_type: 节点类型 (subject/file/netflow)
    - split: 该节点所属的 split
    """
    node_id: str
    node_type: str
    split: SplitType

    def __hash__(self) -> int:
        return hash((self.node_id, self.node_type, self.split))


@dataclass(frozen=True)
class EdgeRecord:
    """
    Neutral edge record.

    对应 canonical artifact 字段:
    - src: 源节点 ID
    - dst: 目标节点 ID
    - edge_type: 边类型 (EVENT_*)
    - timestamp: 事件时间戳 (纳秒)
    - global_event_index: 全局事件序号 (用于 timestamp tie-break)
    - split: 该边所属的 split
    """
    src: str
    dst: str
    edge_type: str
    timestamp: int
    global_event_index: int
    split: SplitType

    def __hash__(self) -> int:
        return hash((self.src, self.dst, self.edge_type, self.timestamp,
                     self.global_event_index, self.split))

    @property
    def src_dst_pair(self) -> Tuple[str, str]:
        """Return (src, dst) pair for simple graph deduplication."""
        return (self.src, self.dst)


@dataclass
class GraphSnapshot:
    """
    A causal graph snapshot ending at a specific time boundary.

    合同要求:
    - 只能包含 timestamp <= snapshot_end 的合法历史
    - 不包含未来事件
    - 使用确定性 ordering: (timestamp, global_event_index)
    """
    snapshot_id: str
    snapshot_end: int  # timestamp boundary
    nodes: FrozenSet = field(default_factory=frozenset)  # node IDs in this snapshot
    edges: FrozenSet = field(default_factory=frozenset)  # (src, dst) pairs

    # Type mappings for nodes and edges in this snapshot
    node_type_map: Mapping = field(default_factory=dict)
    edge_type_map: Mapping = field(default_factory=dict)

    def __post_init__(self):
        # Validate consistency
        if not isinstance(self.snapshot_end, int):
            raise TypeError("snapshot_end must be int")
        if self.snapshot_end < 0:
            raise ValueError("snapshot_end must be non-negative")


@dataclass
class TypeVocabulary:
    """
    Train-only type vocabulary.

    合同要求:
    - fit() 在 train 数据上建立
    - transform() 不扩展 vocabulary
    - 未知 type 映射到 UNKNOWN token
    """
    node_types: FrozenSet = field(default_factory=frozenset)
    edge_types: FrozenSet = field(default_factory=frozenset)
    _node_type_to_idx: Mapping = field(default_factory=dict)
    _edge_type_to_idx: Mapping = field(default_factory=dict)
    _fitted: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self._fitted and not self._node_type_to_idx:
            # Rebuild index mappings
            self._node_type_to_idx = {
                t: i for i, t in enumerate(sorted(self.node_types))
            }
            self._edge_type_to_idx = {
                t: i for i, t in enumerate(sorted(self.edge_types))
            }

    def fit(self, node_types, edge_types):
        """
        Fit vocabulary from training data.

        Args:
            node_types: Mapping from node_id to node_type
            edge_types: Mapping from (src, dst) to edge_type

        Returns:
            self with vocabulary fitted
        """
        unique_node_types = frozenset(node_types.values())
        unique_edge_types = frozenset(edge_types.values())

        return TypeVocabulary(
            node_types=unique_node_types,
            edge_types=unique_edge_types,
            _node_type_to_idx={
                t: i for i, t in enumerate(sorted(unique_node_types))
            },
            _edge_type_to_idx={
                t: i for i, t in enumerate(sorted(unique_edge_types))
            },
            _fitted=True,
        )

    def transform_node_type(self, node_type: str) -> Tuple[int, bool]:
        """
        Transform a node type to index.

        Args:
            node_type: The node type string

        Returns:
            Tuple of (type_index, is_seen)
            - type_index: 0-based index in vocabulary
            - is_seen: True if type was in training vocabulary
        """
        if node_type in self._node_type_to_idx:
            return self._node_type_to_idx[node_type], True
        else:
            # Unknown type - map to UNKNOWN
            # Use a fixed index for UNKNOWN
            return len(self._node_type_to_idx), False

    def transform_edge_type(self, edge_type: str) -> Tuple[int, bool]:
        """
        Transform an edge type to index.

        Args:
            edge_type: The edge type string

        Returns:
            Tuple of (type_index, is_seen)
        """
        if edge_type in self._edge_type_to_idx:
            return self._edge_type_to_idx[edge_type], True
        else:
            return len(self._edge_type_to_idx), False

    @property
    def node_feature_dim(self) -> int:
        """Node one-hot dimension (includes UNKNOWN placeholder)."""
        return len(self._node_type_to_idx) + 1

    @property
    def edge_feature_dim(self) -> int:
        """Edge one-hot dimension (includes UNKNOWN placeholder)."""
        return len(self._edge_type_to_idx) + 1

    def is_fitted(self) -> bool:
        """Check if vocabulary has been fitted."""
        return self._fitted


@dataclass
class NeutralGraphContract:
    """
    Complete neutral graph contract for MAGIC adapter.

    包含:
    - Node table
    - Edge table
    - Type vocabulary
    - Split information
    - Node-ID to local-ID mapping
    - Duplicate edge policy (set by adapter based on profile)
    """
    # Node and edge records
    nodes: Mapping = field(default_factory=dict)
    edges: List = field(default_factory=list)

    # Type vocabulary (fitted on train)
    type_vocabulary: Optional[TypeVocabulary] = field(default=None)

    # Duplicate edge policy - determines backend dedup behavior
    # Set by MAGICInputAdapter based on profile configuration
    duplicate_policy: Optional[DuplicateEdgePolicy] = field(default=None)

    # Split statistics
    train_node_count: int = field(default=0, repr=False)
    val_node_count: int = field(default=0, repr=False)
    test_node_count: int = field(default=0, repr=False)
    train_edge_count: int = field(default=0, repr=False)
    val_edge_count: int = field(default=0, repr=False)
    test_edge_count: int = field(default=0, repr=False)

    # Global event index counter
    global_event_index: int = field(default=0, repr=False)

    # Unseen type statistics
    unseen_node_types: Dict = field(default_factory=dict, repr=False)
    unseen_edge_types: Dict = field(default_factory=dict, repr=False)

    def should_dedup(self) -> bool:
        """
        Check if duplicate edges should be deduplicated based on policy.

        Returns:
            True if first-edge dedup should be applied

        Raises:
            ValueError: If duplicate_policy is not set
        """
        if self.duplicate_policy is None:
            raise ValueError(
                "NeutralGraphContract.duplicate_policy must be set. "
                "Call MAGICInputAdapter with a valid profile before transform."
            )
        return self.duplicate_policy == DuplicateEdgePolicy.DEDUP_FIRST

    def add_node(self, node_id: str, node_type: str, split: SplitType) -> None:
        """Add a node record."""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeRecord(
                node_id=node_id,
                node_type=node_type,
                split=split
            )
            if split == SplitType.TRAIN:
                self.train_node_count += 1
            elif split == SplitType.VALIDATION:
                self.val_node_count += 1
            else:
                self.test_node_count += 1

    def add_edge(self, edge: EdgeRecord) -> None:
        """Add an edge record with global event index."""
        edge_with_index = EdgeRecord(
            src=edge.src,
            dst=edge.dst,
            edge_type=edge.edge_type,
            timestamp=edge.timestamp,
            global_event_index=self.global_event_index,
            split=edge.split
        )
        self.edges.append(edge_with_index)
        self.global_event_index += 1

        if edge.split == SplitType.TRAIN:
            self.train_edge_count += 1
        elif edge.split == SplitType.VALIDATION:
            self.val_edge_count += 1
        else:
            self.test_edge_count += 1

    def add_edge_preserve_index(self, edge: EdgeRecord) -> None:
        """Add an edge record preserving its original global_event_index."""
        self.edges.append(edge)
        # Update global counter to max+1
        if edge.global_event_index >= self.global_event_index:
            self.global_event_index = edge.global_event_index + 1

        if edge.split == SplitType.TRAIN:
            self.train_edge_count += 1
        elif edge.split == SplitType.VALIDATION:
            self.val_edge_count += 1
        else:
            self.test_edge_count += 1

    def get_sorted_edges(self) -> List[EdgeRecord]:
        """
        Get edges sorted by deterministic ordering.

        Ordering: (timestamp, global_event_index)
        This ensures consistent ordering across runs.
        """
        return sorted(
            self.edges,
            key=lambda e: (e.timestamp, e.global_event_index)
        )

    def get_split_edges(self, split: SplitType) -> List[EdgeRecord]:
        """Get edges for a specific split."""
        return [e for e in self.edges if e.split == split]

    def get_split_nodes(self, split: SplitType) -> Mapping:
        """Get nodes for a specific split."""
        return {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.split == split
        }


@dataclass
class SnapshotScores:
    """
    Scores from a single causal snapshot.

    用于 M4 node merge。
    """
    snapshot_id: str
    snapshot_end: int
    node_scores: Mapping = field(default_factory=dict)  # node_id -> score

    def merge_with(self, other: "SnapshotScores") -> "SnapshotScores":
        """
        Merge scores with another snapshot using max policy.

        final_score(node) = max(self[node], other[node])
        """
        merged_scores = dict(self.node_scores)
        for node_id, score in other.node_scores.items():
            if node_id in merged_scores:
                merged_scores[node_id] = max(merged_scores[node_id], score)
            else:
                merged_scores[node_id] = score
        return SnapshotScores(
            snapshot_id="{}_merged".format(self.snapshot_id),
            snapshot_end=max(self.snapshot_end, other.snapshot_end),
            node_scores=merged_scores
        )


@dataclass
class ThresholdConfig:
    """
    Frozen threshold configuration.

    合同要求:
    - method: validation_quantile
    - q: 0.999
    - 仅使用 validation scores
    """
    method: str = "validation_quantile"
    q: float = 0.999
    threshold_value: Optional[float] = None
    validation_score_count: int = field(default=0, repr=False)
    provenance: str = "validation_only"

    def freeze(self, threshold_value: float, score_count: int) -> "ThresholdConfig":
        """Freeze the threshold with computed value."""
        return ThresholdConfig(
            method=self.method,
            q=self.q,
            threshold_value=threshold_value,
            validation_score_count=score_count,
            provenance="validation_only"
        )


# =============================================================================
# Constants
# =============================================================================

# Unknown type token for unseen types
UNKNOWN_TYPE = "__UNKNOWN__"

# =============================================================================
# Profile Definitions
# =============================================================================

class ProfileType(Enum):
    """MAGIC adapter profile types."""
    LEGACY_UPSTREAM = "legacy_upstream"
    ORTHRUS_UNIFIED = "orthrus_unified"


class DirectionPolicy(Enum):
    """Edge direction handling policy."""
    # Apply reversal for READ/RECV/LOAD (upstream-compatible)
    REVERSE_CAUSAL = "reverse_causal"
    # Keep source direction as-is (ORTHRUS nx already causal)
    KEEP_SOURCE = "keep_source"


class DuplicateEdgePolicy(Enum):
    """Duplicate edge handling policy."""
    # Only keep first edge for (src, dst) pair (upstream-compatible)
    DEDUP_FIRST = "dedup_first"
    # Preserve all parallel edges (ORTHRUS nx preserves all)
    PRESERVE_ALL = "preserve_all"


@dataclass(frozen=True)
class ProfileConfig:
    """
    Adapter profile configuration.

    Defines how the adapter handles direction, duplicates, and other semantics.
    """
    name: ProfileType
    direction_policy: DirectionPolicy
    duplicate_policy: DuplicateEdgePolicy
    description: str

    def should_reverse(self, edge_type: str) -> bool:
        """
        Check if an edge type should be reversed based on profile.

        Args:
            edge_type: The edge type string

        Returns:
            True if direction should be reversed
        """
        if self.direction_policy == DirectionPolicy.KEEP_SOURCE:
            return False
        elif self.direction_policy == DirectionPolicy.REVERSE_CAUSAL:
            return any(
                edge_type.startswith(prefix)
                for prefix in LEGACY_REVERSED_EDGE_PREFIXES
            )
        return False

    def should_dedup(self) -> bool:
        """
        Check if duplicate edges should be deduplicated.

        Returns:
            True if first-edge dedup should be applied
        """
        return self.duplicate_policy == DuplicateEdgePolicy.DEDUP_FIRST


# Profile configurations
LEGACY_UPSTREAM_CONFIG = ProfileConfig(
    name=ProfileType.LEGACY_UPSTREAM,
    direction_policy=DirectionPolicy.REVERSE_CAUSAL,
    duplicate_policy=DuplicateEdgePolicy.DEDUP_FIRST,
    description="Upstream-compatible behavior: READ/RECV/LOAD reversed, first-edge dedup"
)

ORTHRUS_UNIFIED_CONFIG = ProfileConfig(
    name=ProfileType.ORTHRUS_UNIFIED,
    direction_policy=DirectionPolicy.KEEP_SOURCE,
    duplicate_policy=DuplicateEdgePolicy.PRESERVE_ALL,
    description="ORTHRUS production graph protocol: causal direction preserved, all edges kept"
)

PROFILE_CONFIGS = {
    ProfileType.LEGACY_UPSTREAM: LEGACY_UPSTREAM_CONFIG,
    ProfileType.ORTHRUS_UNIFIED: ORTHRUS_UNIFIED_CONFIG,
}


# Legacy constants (for backward compatibility)
# Events with READ/RECV/LOAD are reversed for causal direction
LEGACY_REVERSED_EDGE_PREFIXES = ("EVENT_READ", "EVENT_RECV", "EVENT_LOAD")
MAGIC_REVERSED_EDGE_PREFIXES = LEGACY_REVERSED_EDGE_PREFIXES  # Alias for backward compat

# Legacy simple graph policy: only keep first edge for (src, dst) pair
MAGIC_SIMPLE_GRAPH_POLICY = "first"
