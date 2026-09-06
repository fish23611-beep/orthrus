"""
MAGIC Unified Causal Protocol

本模块实现 M4: MAGIC Unified Causal Protocol。

关键组件：
1. Threshold Protocol - validation_quantile(q=0.999)
2. Causal Snapshot Construction
3. Max Node Merge

设计依据：
- 冻结合同: docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md
- 审计报告: docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md

禁止行为：
- 不接受 test labels
- 不接受 test scores 用于 threshold selection
- 不使用 ground truth 筛选节点
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .contracts import (
    EdgeRecord,
    GraphSnapshot,
    SnapshotScores,
    ThresholdConfig,
    SplitType,
)


# =============================================================================
# Threshold Protocol
# =============================================================================

def compute_validation_quantile_threshold(
    validation_node_scores: Mapping,
    quantile: float = 0.999,
) -> Tuple[float, int]:
    """
    Compute threshold from validation node scores.

    合同要求：
    - method: validation_quantile
    - q: 0.999
    - 仅使用 validation scores
    - 不接受 test scores 或 labels

    Args:
        validation_node_scores: Mapping from node_id to score
        quantile: Quantile value (default: 0.999)

    Returns:
        Tuple of (threshold_value, score_count)
    """
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")

    # Extract finite scores
    finite_scores = []
    for node_id, score in validation_node_scores.items():
        try:
            value = float(score)
            # Check for NaN and inf
            if value != value or value == float('inf') or value == float('-inf'):
                # Skip NaN and inf
                continue
            finite_scores.append(value)
        except (TypeError, ValueError):
            continue

    if not finite_scores:
        raise ValueError("validation_node_scores contains no finite values")

    score_count = len(finite_scores)

    # Use numpy-style quantile (linear interpolation)
    # Sort scores
    sorted_scores = sorted(finite_scores)
    n = len(sorted_scores)

    # Compute quantile index
    # This matches numpy's quantile with linear interpolation (type 7)
    index = quantile * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        # At or beyond max
        threshold = sorted_scores[-1]
    else:
        weight = index - lower
        threshold = sorted_scores[lower] * (1 - weight) + sorted_scores[upper] * weight

    return float(threshold), score_count


def fit_threshold(
    validation_node_scores: Mapping,
    method: str = "validation_quantile",
    q: float = 0.999,
) -> ThresholdConfig:
    """
    Fit threshold configuration.

    Args:
        validation_node_scores: Validation node scores
        method: Threshold method (validation_quantile)
        q: Quantile value

    Returns:
        Frozen ThresholdConfig
    """
    if method != "validation_quantile":
        raise ValueError(
            "Only validation_quantile method is supported. "
            "Received: {}".format(method)
        )

    threshold_value, score_count = compute_validation_quantile_threshold(
        validation_node_scores, q
    )

    config = ThresholdConfig(
        method=method,
        q=q,
        threshold_value=threshold_value,
        validation_score_count=score_count,
        provenance="validation_only",
    )
    return config


def apply_threshold(
    node_scores: Mapping,
    threshold_config: ThresholdConfig,
) -> Dict[str, int]:
    """
    Apply frozen threshold to node scores.

    Args:
        node_scores: Mapping from node_id to score
        threshold_config: Frozen threshold configuration

    Returns:
        Dict from node_id to prediction (0 or 1)
    """
    if threshold_config.threshold_value is None:
        raise ValueError("Threshold not frozen. Call fit_threshold first.")

    threshold = threshold_config.threshold_value
    predictions = {}

    for node_id, score in node_scores.items():
        try:
            value = float(score)
            predictions[node_id] = 1 if value > threshold else 0
        except (TypeError, ValueError):
            # Non-numeric scores treated as non-anomalous
            predictions[node_id] = 0

    return predictions


# =============================================================================
# Causal Snapshot Construction
# =============================================================================

def build_causal_snapshots(
    edges: List[EdgeRecord],
    explicit_boundaries: List[int],
) -> List[GraphSnapshot]:
    """
    Build causal snapshots from edges with explicit boundaries.

    合同要求：
    - snapshot 只包含 timestamp <= snapshot_end 的事件
    - 不包含未来事件
    - 使用 explicit_boundaries 定义边界

    Args:
        edges: List of EdgeRecords sorted by (timestamp, global_event_index)
        explicit_boundaries: List of snapshot end timestamps

    Returns:
        List of GraphSnapshots
    """
    snapshots = []

    for boundary in sorted(explicit_boundaries):
        # Filter edges for this snapshot
        snapshot_edges = []
        snapshot_nodes: Set[str] = set()

        for edge in edges:
            if edge.timestamp <= boundary:
                snapshot_edges.append((edge.src, edge.dst))
                snapshot_nodes.add(edge.src)
                snapshot_nodes.add(edge.dst)
            else:
                # Edges are sorted, so no need to check further
                break

        snapshot = GraphSnapshot(
            snapshot_id="snapshot_{}".format(boundary),
            snapshot_end=boundary,
            nodes=frozenset(snapshot_nodes),
            edges=frozenset(snapshot_edges),
        )
        snapshots.append(snapshot)

    return snapshots


def build_snapshot_from_window(
    edges: List[EdgeRecord],
    window_start: int,
    window_end: int,
    snapshot_id: str,
) -> GraphSnapshot:
    """
    Build a single snapshot from a time window.

    Args:
        edges: All edges (assumed sorted by timestamp)
        window_start: Start of window (inclusive)
        window_end: End of window (inclusive)
        snapshot_id: Unique snapshot identifier

    Returns:
        GraphSnapshot
    """
    snapshot_edges = []
    snapshot_nodes: Set[str] = set()

    for edge in edges:
        if window_start <= edge.timestamp <= window_end:
            snapshot_edges.append((edge.src, edge.dst))
            snapshot_nodes.add(edge.src)
            snapshot_nodes.add(edge.dst)

    return GraphSnapshot(
        snapshot_id=snapshot_id,
        snapshot_end=window_end,
        nodes=frozenset(snapshot_nodes),
        edges=frozenset(snapshot_edges),
    )


# =============================================================================
# Node Score Merge
# =============================================================================

def merge_snapshot_scores(
    snapshot_scores_list: List[SnapshotScores],
) -> SnapshotScores:
    """
    Merge scores from multiple snapshots using max policy.

    合同要求：
    - final_score(node) = max over all valid snapshot scores
    - 与标签无关

    Args:
        snapshot_scores_list: List of SnapshotScores

    Returns:
        Merged SnapshotScores with max policy
    """
    if not snapshot_scores_list:
        raise ValueError("snapshot_scores_list cannot be empty")

    if len(snapshot_scores_list) == 1:
        return snapshot_scores_list[0]

    # Start with the first snapshot
    merged = snapshot_scores_list[0]

    for scores in snapshot_scores_list[1:]:
        merged = merged.merge_with(scores)

    return merged


def merge_node_scores(
    node_scores_per_snapshot: Mapping[str, float],
) -> Mapping[str, float]:
    """
    Simple max merge for node scores across snapshots.

    When a node appears in multiple snapshots, use the max score.

    Args:
        node_scores_per_snapshot: Dict mapping (snapshot_id, node_id) to score

    Returns:
        Dict mapping node_id to merged score
    """
    # Group by node_id
    node_to_scores: Dict[str, List[float]] = {}

    for (snapshot_id, node_id), score in node_scores_per_snapshot.items():
        if node_id not in node_to_scores:
            node_to_scores[node_id] = []
        node_to_scores[node_id].append(score)

    # Merge with max
    merged_scores = {}
    for node_id, scores in node_to_scores.items():
        merged_scores[node_id] = max(scores)

    return merged_scores


# =============================================================================
# Threshold Protocol Class
# =============================================================================

class ThresholdProtocol:
    """
    Unified threshold protocol for MAGIC.

    合同要求：
    - method: validation_quantile
    - q: 0.999
    - validation-only
    """

    def __init__(
        self,
        method: str = "validation_quantile",
        q: float = 0.999,
    ):
        """
        Initialize threshold protocol.

        Args:
            method: Threshold method
            q: Quantile value
        """
        self.method = method
        self.q = q
        self._config: Optional[ThresholdConfig] = None

    def fit(self, validation_node_scores: Mapping) -> ThresholdConfig:
        """
        Fit threshold on validation scores.

        Args:
            validation_node_scores: Validation node scores

        Returns:
            Frozen ThresholdConfig
        """
        self._config = fit_threshold(
            validation_node_scores,
            method=self.method,
            q=self.q,
        )
        return self._config

    def predict(self, node_scores: Mapping) -> Dict[str, int]:
        """
        Apply threshold to node scores.

        Args:
            node_scores: Node scores

        Returns:
            Predictions (0 or 1)
        """
        if self._config is None:
            raise RuntimeError("Threshold not fitted. Call fit() first.")
        return apply_threshold(node_scores, self._config)

    @property
    def threshold_value(self) -> Optional[float]:
        """Get frozen threshold value."""
        if self._config:
            return self._config.threshold_value
        return None

    @property
    def is_fitted(self) -> bool:
        """Check if threshold is fitted."""
        return self._config is not None


# =============================================================================
# Validation Helper
# =============================================================================

def validate_future_event_isolation(
    snapshot: GraphSnapshot,
    all_edges: List[EdgeRecord],
) -> bool:
    """
    Validate that snapshot contains no future events.

    Args:
        snapshot: The snapshot to validate
        all_edges: All edges in chronological order

    Returns:
        True if no future events, False otherwise
    """
    # Find the first edge that starts after snapshot_end
    for edge in all_edges:
        if edge.timestamp > snapshot.snapshot_end:
            # Check if this edge's nodes are in the snapshot
            if edge.src in snapshot.nodes or edge.dst in snapshot.nodes:
                return False
    return True


def validate_snapshot_determinism(
    snapshot: GraphSnapshot,
) -> bool:
    """
    Validate snapshot construction is deterministic.

    Args:
        snapshot: The snapshot to validate

    Returns:
        True if deterministic
    """
    # Check that nodes and edges are frozen sets
    return isinstance(snapshot.nodes, frozenset) and \
           isinstance(snapshot.edges, frozenset)
