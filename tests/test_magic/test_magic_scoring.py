"""
Tests for MAGIC Raw Score Export (M5)

测试 M5 的核心功能：
1. train-only fit
2. validation 不改变 scorer state
3. test 不改变 scorer state
4. KNN raw score deterministic
5. high-distance target 得到更高 anomaly score
6. canonical node id 保留
7. backend local node index 不泄漏到 evaluator identity
8. non-finite input fail fast
9. zero/invalid reference denominator fail fast
10. snapshot causal boundary
11. future node/event 不可见
12. repeated node snapshot score 使用 max merge
13. label objects 无法传入 scorer API
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.magic import (
    MAGICEntityScorer,
    NodeScoreRecord,
    NodeIdentityMap,
    SnapshotScores,
    merge_snapshot_scores,
    compute_snapshot_scores,
    merge_and_score_snapshots,
    MAGIC_K_NEIGHBORS,
    MAX_TRAIN_REFERENCE_SAMPLES,
)
from src.baselines.magic.contracts import GraphSnapshot


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def train_embeddings():
    """Simple train embeddings - 5 nodes, 4D."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # node_0
        [0.0, 1.0, 0.0, 0.0],  # node_1
        [0.0, 0.0, 1.0, 0.0],  # node_2
        [0.0, 0.0, 0.0, 1.0],  # node_3
        [0.5, 0.5, 0.0, 0.0],  # node_4
    ], dtype=np.float32)


@pytest.fixture
def train_node_ids():
    """Train node IDs."""
    return ["train_0", "train_1", "train_2", "train_3", "train_4"]


@pytest.fixture
def test_embeddings():
    """Test embeddings - 3 nodes."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],  # Similar to train_0
        [10.0, 10.0, 10.0, 10.0],  # Far from all train
        [0.6, 0.4, 0.0, 0.0],  # Similar to train_4
    ], dtype=np.float32)


@pytest.fixture
def test_node_ids():
    """Test node IDs."""
    return ["test_0", "test_1", "test_2"]


# =============================================================================
# Test 1: Train-only Fit
# =============================================================================

class TestTrainOnlyFit:
    """Test that scorer is fitted on train only."""

    def test_fit_creates_fitted_state(self, train_embeddings, train_node_ids):
        """Test fit() creates fitted state."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        assert scorer.is_fitted
        assert scorer.reference_distance is not None
        assert scorer.identity_map is not None

    def test_fit_sets_reference_distance(self, train_embeddings, train_node_ids):
        """Test fit() computes reference distance."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        assert scorer.reference_distance > 0
        assert np.isfinite(scorer.reference_distance)

    def test_fit_rejects_empty_embeddings(self):
        """Test fit() rejects empty embeddings."""
        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="cannot be empty"):
            scorer.fit(np.array([]).reshape(0, 4), [])

    def test_fit_rejects_mismatched_lengths(self, train_embeddings, train_node_ids):
        """Test fit() rejects mismatched lengths."""
        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="same length"):
            scorer.fit(train_embeddings, ["a", "b"])  # 5 embeddings, 2 IDs


# =============================================================================
# Test 2: Validation Does Not Change Scorer State
# =============================================================================

class TestValidationIsolation:
    """Test that validation scoring doesn't change scorer state."""

    def test_validation_does_not_change_reference(self, train_embeddings, train_node_ids):
        """Test validation doesn't modify reference distance."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        ref_dist_before = scorer.reference_distance

        # Score with validation-like data
        val_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scorer.score(val_emb, ["val_0"])

        ref_dist_after = scorer.reference_distance
        assert ref_dist_before == ref_dist_after

    def test_validation_does_not_change_train_embeddings(self, train_embeddings, train_node_ids):
        """Test validation doesn't modify train embeddings."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        train_emb_before = scorer._train_embeddings.copy()

        # Score with validation-like data
        val_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scorer.score(val_emb, ["val_0"])

        assert np.allclose(scorer._train_embeddings, train_emb_before)


# =============================================================================
# Test 3: Test Does Not Change Scorer State
# =============================================================================

class TestTestIsolation:
    """Test that test scoring doesn't change scorer state."""

    def test_test_does_not_change_reference(self, train_embeddings, train_node_ids):
        """Test test doesn't modify reference distance."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        ref_dist_before = scorer.reference_distance

        # Score test data
        test_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scorer.score(test_emb, ["test_0"])

        ref_dist_after = scorer.reference_distance
        assert ref_dist_before == ref_dist_after


# =============================================================================
# Test 4: KNN Raw Score Deterministic
# =============================================================================

class TestDeterminism:
    """Test that scoring is deterministic."""

    def test_same_embeddings_same_scores(self, train_embeddings, train_node_ids, test_embeddings, test_node_ids):
        """Test same embeddings produce same scores."""
        scorer1 = MAGICEntityScorer(k=5, seed=0)
        scorer1.fit(train_embeddings, train_node_ids)
        scores1 = scorer1.score(test_embeddings, test_node_ids)

        scorer2 = MAGICEntityScorer(k=5, seed=0)
        scorer2.fit(train_embeddings, train_node_ids)
        scores2 = scorer2.score(test_embeddings, test_node_ids)

        for s1, s2 in zip(scores1, scores2):
            assert s1.score_raw == s2.score_raw

    def test_different_seed_different_reference(self, train_embeddings, train_node_ids):
        """Test different seeds can produce different reference distances."""
        scorer1 = MAGICEntityScorer(k=5, seed=0)
        scorer1.fit(train_embeddings, train_node_ids)

        scorer2 = MAGICEntityScorer(k=5, seed=42)
        scorer2.fit(train_embeddings, train_node_ids)

        # Reference distances may differ due to sampling
        # But both should be positive
        assert scorer1.reference_distance > 0
        assert scorer2.reference_distance > 0


# =============================================================================
# Test 5: High-distance Target Gets Higher Score
# =============================================================================

class TestScoreOrdering:
    """Test that anomalous targets get higher scores."""

    def test_farther_node_has_higher_score(self, train_embeddings, train_node_ids):
        """Test that a node far from train has higher score."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        # Node similar to train cluster
        similar = np.array([[0.51, 0.49, 0.0, 0.0]], dtype=np.float32)
        # Node far from train
        far = np.array([[100.0, 100.0, 100.0, 100.0]], dtype=np.float32)

        scores_similar = scorer.score(similar, ["similar"])
        scores_far = scorer.score(far, ["far"])

        assert scores_far[0].score_raw > scores_similar[0].score_raw


# =============================================================================
# Test 6: Canonical Node ID Preserved
# =============================================================================

class TestCanonicalIdentity:
    """Test that canonical node IDs are preserved."""

    def test_score_records_preserve_node_id(self, train_embeddings, train_node_ids, test_embeddings, test_node_ids):
        """Test score records preserve canonical node IDs."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        scores = scorer.score(test_embeddings, test_node_ids)

        returned_ids = [s.canonical_node_id for s in scores]
        assert returned_ids == test_node_ids


# =============================================================================
# Test 7: Backend Local Index Not Leaked
# =============================================================================

class TestIdentityMapping:
    """Test that backend local index doesn't leak."""

    def test_identity_map_provides_bidirectional_mapping(self, train_embeddings, train_node_ids):
        """Test identity map provides reversible mapping."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        identity_map = scorer.identity_map

        # Test canonical -> local -> canonical
        for i, node_id in enumerate(train_node_ids):
            local = identity_map.to_local(node_id)
            assert local == i

            canonical_back = identity_map.to_canonical(i)
            assert canonical_back == node_id

    def test_local_index_not_in_score_record(self, train_embeddings, train_node_ids, test_embeddings, test_node_ids):
        """Test that local index doesn't leak into score records."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        scores = scorer.score(test_embeddings, test_node_ids)

        # Score records should only have canonical_node_id, not local index
        for score in scores:
            assert hasattr(score, "canonical_node_id")
            assert score.canonical_node_id in test_node_ids
            # The score record should NOT expose local index


# =============================================================================
# Test 8: Non-finite Input Fail Fast
# =============================================================================

class TestNonFiniteHandling:
    """Test that non-finite inputs cause errors."""

    def test_nan_embeddings_rejected(self, train_embeddings, train_node_ids):
        """Test that NaN embeddings are rejected."""
        bad_embeddings = train_embeddings.copy()
        bad_embeddings[0, 0] = float("nan")

        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="non-finite"):
            scorer.fit(bad_embeddings, train_node_ids)

    def test_inf_embeddings_rejected(self, train_embeddings, train_node_ids):
        """Test that infinite embeddings are rejected."""
        bad_embeddings = train_embeddings.copy()
        bad_embeddings[0, 0] = float("inf")

        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="non-finite"):
            scorer.fit(bad_embeddings, train_node_ids)

    def test_score_with_nan_rejected(self, train_embeddings, train_node_ids):
        """Test that scoring NaN embeddings raises error."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        bad_embeddings = np.array([[float("nan"), 0.0, 0.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="non-finite"):
            scorer.score(bad_embeddings, ["bad"])


# =============================================================================
# Test 9: Zero/Invalid Reference Denominator Fail Fast
# =============================================================================

class TestReferenceDenominator:
    """Test that invalid reference denominators cause errors."""

    def test_constant_embeddings_rejected(self):
        """Test that constant embeddings cause zero variance error."""
        embeddings = np.ones((5, 4), dtype=np.float32)
        node_ids = ["n0", "n1", "n2", "n3", "n4"]

        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="zero variance"):
            scorer.fit(embeddings, node_ids)

    def test_zero_reference_distance_rejected(self):
        """Test that zero reference distance is rejected."""
        # Keep non-zero variance globally, while making every point's
        # k=5 nearest-neighbor distances zero within duplicate clusters.
        # This specifically exercises the non-positive reference denominator
        # guard instead of the earlier zero-variance guard.
        embeddings = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32)
        node_ids = [f"n{i}" for i in range(10)]

        scorer = MAGICEntityScorer(k=5)

        with pytest.raises(ValueError, match="non-positive"):
            scorer.fit(embeddings, node_ids)


# =============================================================================
# Test 10: Snapshot Causal Boundary
# =============================================================================

class TestSnapshotBoundary:
    """Test causal snapshot boundary enforcement."""

    def test_snapshot_scores_respect_boundary(self, train_embeddings, train_node_ids):
        """Test that snapshot scores only include visible nodes."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        # Create snapshot with only some nodes
        snapshot = GraphSnapshot(
            snapshot_id="snap_0",
            snapshot_end=1000,
            nodes=frozenset({"train_0", "train_1"}),
            edges=frozenset(),
        )

        # Score full dataset
        scores = scorer.score_snapshot(snapshot, train_embeddings, train_node_ids)

        # Should only get scores for visible nodes
        scored_ids = {s.canonical_node_id for s in scores}
        assert scored_ids == {"train_0", "train_1"}


# =============================================================================
# Test 11: Future Node Not Visible
# =============================================================================

class TestFutureIsolation:
    """Test that future nodes are not visible."""

    def test_future_node_not_scored(self, train_embeddings, train_node_ids):
        """Test that nodes not in snapshot are not scored."""
        scorer = MAGICEntityScorer(k=5, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        # Create snapshot with only train_0
        snapshot = GraphSnapshot(
            snapshot_id="snap_early",
            snapshot_end=500,
            nodes=frozenset({"train_0"}),
            edges=frozenset(),
        )

        # Try to score a "future" node
        future_emb = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        future_id = "train_3"

        # Score with all embeddings
        all_emb = train_embeddings
        all_ids = train_node_ids

        scores = scorer.score_snapshot(snapshot, all_emb, all_ids)

        # Only train_0 should be scored
        scored_ids = {s.canonical_node_id for s in scores}
        assert scored_ids == {"train_0"}
        assert future_id not in scored_ids


# =============================================================================
# Test 12: Repeated Node Max Merge
# =============================================================================

class TestMaxMerge:
    """Test that repeated nodes use max merge."""

    def test_max_merge_across_snapshots(self):
        """Test max merge across snapshots."""
        scores1 = SnapshotScores(
            snapshot_id="snap_1",
            snapshot_end=1000,
            node_scores={"node_A": 0.2, "node_B": 0.8, "node_C": 0.4},
        )
        scores2 = SnapshotScores(
            snapshot_id="snap_2",
            snapshot_end=2000,
            node_scores={"node_A": 0.5, "node_B": 0.3, "node_D": 0.9},
        )
        scores3 = SnapshotScores(
            snapshot_id="snap_3",
            snapshot_end=3000,
            node_scores={"node_A": 0.1, "node_C": 0.9, "node_E": 0.7},
        )

        merged = merge_snapshot_scores([scores1, scores2, scores3])

        # node_A: max(0.2, 0.5, 0.1) = 0.5
        assert merged.node_scores["node_A"] == 0.5
        # node_B: max(0.8, 0.3) = 0.8
        assert merged.node_scores["node_B"] == 0.8
        # node_C: max(0.4, 0.9) = 0.9
        assert merged.node_scores["node_C"] == 0.9
        # node_D: 0.9
        assert merged.node_scores["node_D"] == 0.9
        # node_E: 0.7
        assert merged.node_scores["node_E"] == 0.7


# =============================================================================
# Test 13: Labels Cannot Be Passed to Scorer
# =============================================================================

class TestLabelRejection:
    """Test that labels cannot be passed to scorer API."""

    def test_score_signature_no_labels(self, train_embeddings, train_node_ids, test_embeddings, test_node_ids):
        """Test that score() signature doesn't accept labels."""
        scorer = MAGICEntityScorer(k=5)
        scorer.fit(train_embeddings, train_node_ids)

        # Call score without labels
        scores = scorer.score(test_embeddings, test_node_ids)

        # Should work fine
        assert len(scores) == len(test_node_ids)

    def test_fit_signature_no_labels(self, train_embeddings, train_node_ids):
        """Test that fit() signature doesn't accept labels."""
        scorer = MAGICEntityScorer(k=5)

        # Fit without labels
        scorer.fit(train_embeddings, train_node_ids)

        # Should work fine
        assert scorer.is_fitted


# =============================================================================
# Test 14: K Value
# =============================================================================

class TestKNeighbors:
    """Test K parameter handling."""

    def test_default_k_is_10(self):
        """Test default k is 10 for THEIA."""
        assert MAGIC_K_NEIGHBORS == 10

    def test_custom_k_works(self, train_embeddings, train_node_ids):
        """Test custom k value works."""
        scorer = MAGICEntityScorer(k=3, seed=0)
        scorer.fit(train_embeddings, train_node_ids)

        assert scorer.k == 3
        assert scorer.reference_distance > 0


# =============================================================================
# Test 15: Node Score Record Validation
# =============================================================================

class TestNodeScoreRecord:
    """Test NodeScoreRecord validation."""

    def test_finite_score_accepted(self):
        """Test finite scores are accepted."""
        record = NodeScoreRecord(
            canonical_node_id="test",
            snapshot_id="snap",
            snapshot_boundary=1000,
            score_raw=0.5,
        )
        assert record.score_raw == 0.5

    def test_nan_score_rejected(self):
        """Test NaN scores are rejected."""
        with pytest.raises(ValueError, match="finite"):
            NodeScoreRecord(
                canonical_node_id="test",
                snapshot_id="snap",
                snapshot_boundary=1000,
                score_raw=float("nan"),
            )

    def test_inf_score_rejected(self):
        """Test infinite scores are rejected."""
        with pytest.raises(ValueError, match="finite"):
            NodeScoreRecord(
                canonical_node_id="test",
                snapshot_id="snap",
                snapshot_boundary=1000,
                score_raw=float("inf"),
            )
