"""
MAGIC Raw Score Export

本模块实现 M5: MAGIC Raw Score Export。

核心功能：
1. MAGIC KNN anomaly score 计算
2. Train-only detector fitting
3. Canonical node identity mapping
4. Causal snapshot integration

设计依据：
- 冻结合同: docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md
- 审计报告: docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md
- 官方 entity-level scoring: /opt/magic-upstream/model/eval.py

MAGIC raw score 公式:
    dist_target = mean(kNN distance from target to benign train)
    dist_reference = mean(train to its kNN distance)
    score_raw = dist_target / dist_reference

禁止行为：
- 不接受 labels 传入 scorer API
- 不读取 ground truth
- 不修改 /opt/magic-upstream
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from .contracts import (
    SnapshotScores,
    GraphSnapshot,
    SplitType,
)
from .protocol import merge_snapshot_scores


# =============================================================================
# Constants
# =============================================================================

# MAGIC official KNN k value for THEIA
MAGIC_K_NEIGHBORS = 10

# Upstream uses at most 50,000 train embeddings for reference distance
MAX_TRAIN_REFERENCE_SAMPLES = 50000


# =============================================================================
# Score Record
# =============================================================================

@dataclass
class NodeScoreRecord:
    """
    A single node's anomaly score record.

    至少保存:
    - canonical_node_id
    - snapshot_id
    - score_raw
    """
    canonical_node_id: str
    snapshot_id: str
    snapshot_boundary: int
    score_raw: float
    node_type: Optional[str] = None
    split: Optional[str] = None

    def __post_init__(self):
        # Validate score is finite
        if not np.isfinite(self.score_raw):
            raise ValueError(
                "score_raw must be finite. Got {} for node {}".format(
                    self.score_raw, self.canonical_node_id
                )
            )


# =============================================================================
# Node Identity Mapping
# =============================================================================

@dataclass
class NodeIdentityMap:
    """
    Canonical node ID to backend local index mapping.

    合同要求：
    - DGL backend 使用 local contiguous index
    - 统一 evaluator 必须使用 canonical_node_id
    - 建立可逆映射
    """
    # Canonical node ID -> backend local index
    _canonical_to_local: Dict[str, int] = field(default_factory=dict)
    # Backend local index -> canonical node ID
    _local_to_canonical: Dict[int, str] = field(default_factory=dict)

    def add(self, canonical_node_id: str, local_index: int) -> None:
        """Add a mapping."""
        self._canonical_to_local[canonical_node_id] = local_index
        self._local_to_canonical[local_index] = canonical_node_id

    def to_local(self, canonical_node_id: str) -> Optional[int]:
        """Convert canonical ID to local index."""
        return self._canonical_to_local.get(canonical_node_id)

    def to_canonical(self, local_index: int) -> Optional[str]:
        """Convert local index to canonical ID."""
        return self._local_to_canonical.get(local_index)

    @property
    def size(self) -> int:
        """Number of mapped nodes."""
        return len(self._canonical_to_local)

    def get_all_canonical_ids(self) -> List[str]:
        """Get all canonical node IDs."""
        return list(self._canonical_to_local.keys())


# =============================================================================
# MAGIC Entity Scorer
# =============================================================================

class MAGICEntityScorer:
    """
    MAGIC KNN-based entity-level anomaly scorer.

    合同要求：
    1. fit() 不接受 labels / malicious nodes / ground truth / test data
    2. score() 不接受 labels / malicious nodes / ground truth
    3. Train-only statistics: train mean, train std, KNN reference
    4. Deterministic with frozen seed (caller must set via MagicSeedController)

    注意：
    - 本类不再调用 np.random.seed() 污染全局 RNG
    - 当 use_local_rng=True 时，使用局部 np.random.RandomState(seed)
    - 该 RandomState 与 legacy M5 实现的 np.random.seed()/np.random.permutation()
      行为完全一致，保证 M5 sampling semantics 不被 M7 改变
    """

    def __init__(
        self,
        k: int = MAGIC_K_NEIGHBORS,
        seed: Optional[int] = None,  # Explicit None forces caller to pass seed
        max_train_reference_samples: int = MAX_TRAIN_REFERENCE_SAMPLES,
        use_local_rng: bool = False,  # New: use independent legacy-compatible RNG
    ):
        """
        Initialize the scorer.

        Args:
            k: Number of nearest neighbors (default: 10 for THEIA)
            seed: Random seed for deterministic behavior.
                  MUST be set explicitly by the caller.
                  Official experiment seeds: 0, 1, 2.
                  If None, fit() will raise ValueError to fail-fast.
            max_train_reference_samples: Max train samples for reference distance
            use_local_rng: If True, use an independent np.random.RandomState(seed)
                           instead of np.random.seed(). This prevents scorer.fit()
                           from polluting the caller's global RNG state while
                           preserving M5 legacy sampling semantics exactly.
        """
        self.k = k
        self.seed = seed
        self.max_train_reference_samples = max_train_reference_samples
        self.use_local_rng = use_local_rng
        self._rng: Optional[np.random.RandomState] = None

        # Fitted state
        self._fitted = False
        self._train_embeddings: Optional[np.ndarray] = None
        self._train_node_ids: Optional[List[str]] = None
        self._reference_distance: Optional[float] = None
        self._identity_map: Optional[NodeIdentityMap] = None

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_node_ids: List[str],
    ) -> "MAGICEntityScorer":
        """
        Fit the scorer on training data.

        Args:
            train_embeddings: [N, D] array of train node embeddings
            train_node_ids: List of N canonical node IDs

        Returns:
            self

        Raises:
            ValueError: If inputs are invalid or contain non-finite values
        """
        # Validate inputs
        if len(train_embeddings) == 0:
            raise ValueError("train_embeddings cannot be empty")
        if len(train_embeddings) != len(train_node_ids):
            raise ValueError(
                "train_embeddings and train_node_ids must have same length"
            )

        # Check for non-finite values
        if not np.all(np.isfinite(train_embeddings)):
            raise ValueError("train_embeddings contains non-finite values")

        # Seed handling: fail-fast if not explicitly set.
        # We do NOT allow a default 0 seed - official seeds (0, 1, 2)
        # must be passed explicitly by the caller.
        if self.seed is None:
            raise ValueError(
                "MAGICEntityScorer.seed must be explicitly set by the caller. "
                "Official experiment seeds: 0, 1, 2. "
                "Use MagicSeedController or pass seed explicitly."
            )

        # Set seed for determinism
        # M7 change: use np.random.RandomState(seed) which is legacy-compatible
        # with the original M5 np.random.seed() / np.random.permutation() flow.
        # We NEVER use np.random.default_rng() because it uses a different
        # RNG algorithm and would change M5 sampling semantics.
        if self.use_local_rng:
            # Local legacy-compatible RNG - does NOT pollute global NumPy state
            self._rng = np.random.RandomState(self.seed)
        else:
            # Local legacy-compatible RNG even without explicit isolation flag,
            # because we must not change M5 sampling semantics.
            # The local RandomState produces the same permutation() as
            # np.random.seed(s); np.random.permutation(n).
            self._rng = np.random.RandomState(self.seed)

        # Build identity map
        identity_map = NodeIdentityMap()
        for i, node_id in enumerate(train_node_ids):
            identity_map.add(node_id, i)

        # Normalize train embeddings
        train_mean = train_embeddings.mean(axis=0)
        train_std = train_embeddings.std(axis=0)

        # Check for zero variance
        if np.any(train_std == 0):
            raise ValueError(
                "train_embeddings has zero variance in some dimensions. "
                "Add epsilon or normalize before fitting."
            )

        train_normalized = (train_embeddings - train_mean) / (train_std + 1e-10)

        # Compute reference distance
        # Use sklearn NearestNeighbors
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError(
                "scikit-learn is required for MAGICEntityScorer. "
                "Install with: pip install scikit-learn"
            )

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        nbrs.fit(train_normalized)

        # Sample train embeddings for reference distance (upstream uses max 50k)
        # M7 change: always use local RandomState for sampling to:
        # 1. Avoid polluting global NumPy RNG
        # 2. Preserve M5 legacy semantics (same RandomState algorithm)
        n_train = len(train_normalized)
        if n_train > self.max_train_reference_samples:
            assert self._rng is not None
            indices = self._rng.permutation(n_train)[:self.max_train_reference_samples]
            ref_embeddings = train_normalized[indices]
        else:
            ref_embeddings = train_normalized

        # Compute reference distance
        distances, _ = nbrs.kneighbors(ref_embeddings, n_neighbors=self.k)
        # Upstream formula: mean_distance * k / (k - 1)
        self._reference_distance = distances.mean() * self.k / (self.k - 1)

        # Check for zero reference distance
        if self._reference_distance <= 0:
            raise ValueError(
                "Reference distance is non-positive ({}). "
                "Check train embeddings.".format(self._reference_distance)
            )

        # Store normalized train embeddings for scoring
        self._train_embeddings = train_normalized
        self._train_node_ids = train_node_ids
        self._identity_map = identity_map
        self._train_mean = train_mean
        self._train_std = train_std
        self._fitted = True

        return self

    def score(
        self,
        target_embeddings: np.ndarray,
        target_node_ids: List[str],
    ) -> List[NodeScoreRecord]:
        """
        Score target nodes using KNN anomaly detection.

        Args:
            target_embeddings: [M, D] array of target node embeddings
            target_node_ids: List of M canonical node IDs

        Returns:
            List of NodeScoreRecord

        Raises:
            RuntimeError: If scorer is not fitted
            ValueError: If inputs are invalid
        """
        if not self._fitted:
            raise RuntimeError(
                "Scorer not fitted. Call fit() before score()."
            )

        if len(target_embeddings) == 0:
            return []

        if len(target_embeddings) != len(target_node_ids):
            raise ValueError(
                "target_embeddings and target_node_ids must have same length"
            )

        if not np.all(np.isfinite(target_embeddings)):
            raise ValueError("target_embeddings contains non-finite values")

        # Normalize target embeddings using train statistics
        target_normalized = (
            (target_embeddings - self._train_mean) / (self._train_std + 1e-10)
        )

        # Build identity map for targets
        target_identity_map = NodeIdentityMap()
        for i, node_id in enumerate(target_node_ids):
            target_identity_map.add(node_id, i)

        # Compute KNN distances
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm="auto")
        nbrs.fit(self._train_embeddings)

        distances, _ = nbrs.kneighbors(target_normalized, n_neighbors=self.k)
        dist_target = distances.mean(axis=1)

        # Check for non-finite distances
        if not np.all(np.isfinite(dist_target)):
            raise ValueError(
                "Target distances contain non-finite values. "
                "Check embeddings."
            )

        # Compute raw anomaly score
        scores = dist_target / self._reference_distance

        # Build score records
        score_records = []
        for i, node_id in enumerate(target_node_ids):
            record = NodeScoreRecord(
                canonical_node_id=node_id,
                snapshot_id="full",
                snapshot_boundary=-1,  # No snapshot
                score_raw=float(scores[i]),
            )
            score_records.append(record)

        return score_records

    def score_snapshot(
        self,
        snapshot: GraphSnapshot,
        embeddings: np.ndarray,
        node_ids: List[str],
    ) -> List[NodeScoreRecord]:
        """
        Score nodes within a causal snapshot.

        Args:
            snapshot: The GraphSnapshot defining visible nodes
            embeddings: [M, D] array of embeddings for all nodes
            node_ids: List of M canonical node IDs

        Returns:
            List of NodeScoreRecord for nodes in snapshot
        """
        if not self._fitted:
            raise RuntimeError(
                "Scorer not fitted. Call fit() before score_snapshot()."
            )

        # Filter to nodes in snapshot
        visible_node_ids = set(snapshot.nodes)

        # Find indices of visible nodes
        visible_indices = [
            i for i, nid in enumerate(node_ids) if nid in visible_node_ids
        ]

        if not visible_indices:
            return []

        visible_embeddings = embeddings[visible_indices]
        visible_node_ids_filtered = [node_ids[i] for i in visible_indices]

        # Score visible nodes
        scores = self.score(visible_embeddings, visible_node_ids_filtered)

        # Update snapshot info
        for score in scores:
            score.snapshot_id = snapshot.snapshot_id
            score.snapshot_boundary = snapshot.snapshot_end

        return scores

    @property
    def is_fitted(self) -> bool:
        """Check if scorer is fitted."""
        return self._fitted

    @property
    def reference_distance(self) -> Optional[float]:
        """Get the computed reference distance."""
        return self._reference_distance

    @property
    def identity_map(self) -> Optional[NodeIdentityMap]:
        """Get the canonical node identity map."""
        return self._identity_map


# =============================================================================
# Snapshot Score Helper
# =============================================================================

def compute_snapshot_scores(
    scorer: MAGICEntityScorer,
    snapshot: GraphSnapshot,
    all_embeddings: np.ndarray,
    all_node_ids: List[str],
) -> SnapshotScores:
    """
    Compute snapshot scores using a fitted scorer.

    Args:
        scorer: Fitted MAGICEntityScorer
        snapshot: The snapshot to score
        all_embeddings: All node embeddings
        all_node_ids: All canonical node IDs

    Returns:
        SnapshotScores
    """
    score_records = scorer.score_snapshot(
        snapshot, all_embeddings, all_node_ids
    )

    node_scores = {
        record.canonical_node_id: record.score_raw
        for record in score_records
    }

    return SnapshotScores(
        snapshot_id=snapshot.snapshot_id,
        snapshot_end=snapshot.snapshot_end,
        node_scores=node_scores,
    )


def merge_and_score_snapshots(
    scorer: MAGICEntityScorer,
    snapshots: List[GraphSnapshot],
    all_embeddings: np.ndarray,
    all_node_ids: List[str],
) -> Dict[str, float]:
    """
    Score all snapshots and merge with max policy.

    final_score(node) = max(snapshot_scores)

    Args:
        scorer: Fitted MAGICEntityScorer
        snapshots: List of causal snapshots
        all_embeddings: All node embeddings
        all_node_ids: All canonical node IDs

    Returns:
        Dict mapping canonical_node_id to merged score
    """
    snapshot_scores_list = []

    for snapshot in snapshots:
        scores = compute_snapshot_scores(
            scorer, snapshot, all_embeddings, all_node_ids
        )
        snapshot_scores_list.append(scores)

    if not snapshot_scores_list:
        return {}

    # Merge with max policy
    merged = merge_snapshot_scores(snapshot_scores_list)
    return dict(merged.node_scores)
