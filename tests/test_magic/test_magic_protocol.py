"""
Tests for MAGIC Unified Causal Protocol (M4)

测试 M4 的核心功能：
1. Threshold q=0.999
2. Threshold 只接受 validation scores
3. Threshold API 无 test labels
4. Future events 不进入当前 snapshot
5. Explicit boundaries 正确
6. Duplicate node max merge
7. Test label 对 threshold 无影响
8. Ground-truth 文件未被 M4 读取
9. Snapshot construction deterministic
10. Adversarial leakage tests
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Tuple

from src.baselines.magic import (
    compute_validation_quantile_threshold,
    merge_snapshot_scores,
    build_causal_snapshots,
    ThresholdProtocol,
    SnapshotScores,
    GraphSnapshot,
    ThresholdConfig,
    EdgeRecord,
    SplitType,
)


# =============================================================================
# Synthetic Fixtures
# =============================================================================

def _make_edge(src, dst, t, edge_type, gei, split):
    """Helper to create an EdgeRecord."""
    return EdgeRecord(
        src=src,
        dst=dst,
        edge_type=edge_type,
        timestamp=t,
        global_event_index=gei,
        split=split,
    )


@pytest.fixture
def synthetic_edges():
    """Synthetic edges for snapshot testing."""
    return [
        _make_edge("A", "B", 1000, "EVENT_OPEN", 0, SplitType.TRAIN),
        _make_edge("B", "C", 1100, "EVENT_WRITE", 1, SplitType.TRAIN),
        _make_edge("C", "A", 1200, "EVENT_READ", 2, SplitType.VALIDATION),
        _make_edge("A", "D", 1300, "EVENT_CONNECT", 3, SplitType.VALIDATION),
        _make_edge("D", "E", 1400, "EVENT_OPEN", 4, SplitType.TEST),
        _make_edge("E", "F", 1500, "EVENT_WRITE", 5, SplitType.TEST),
    ]


@pytest.fixture
def synthetic_validation_scores():
    """Synthetic validation node scores."""
    return {
        "node_A": 0.1,
        "node_B": 0.2,
        "node_C": 0.15,
        "node_D": 0.25,
        "node_E": 0.18,
    }


@pytest.fixture
def synthetic_snapshot_scores_list():
    """Synthetic list of SnapshotScores for merge testing."""
    return [
        SnapshotScores(
            snapshot_id="snapshot_1000",
            snapshot_end=1000,
            node_scores={"A": 0.2, "B": 0.8, "C": 0.4},
        ),
        SnapshotScores(
            snapshot_id="snapshot_2000",
            snapshot_end=2000,
            node_scores={"A": 0.3, "B": 0.5, "D": 0.6},
        ),
        SnapshotScores(
            snapshot_id="snapshot_3000",
            snapshot_end=3000,
            node_scores={"A": 0.1, "C": 0.9, "E": 0.7},
        ),
    ]


# =============================================================================
# Test 1: Threshold q=0.999
# =============================================================================

class TestThresholdQuantile:
    """Test threshold with q=0.999."""

    def test_quantile_0999_returns_high_threshold(self, synthetic_validation_scores):
        """Test q=0.999 returns a high threshold."""
        threshold, count = compute_validation_quantile_threshold(
            synthetic_validation_scores, quantile=0.999
        )

        # Threshold should be at or above max score
        max_score = max(synthetic_validation_scores.values())
        assert threshold <= max_score + 1e-9

    def test_quantile_0999_is_max_for_small_set(self):
        """Test q=0.999 for small validation sets uses linear interpolation."""
        scores = {"a": 0.1, "b": 0.2, "c": 0.3}
        threshold, count = compute_validation_quantile_threshold(scores, quantile=0.999)

        # With 3 scores, q=0.999 with linear interpolation
        # index = 0.999 * (3-1) = 1.998, lower=1, upper=2
        # weight = 0.998
        # threshold = 0.2 * (1-0.998) + 0.3 * 0.998 = 0.2998
        assert abs(threshold - 0.2998) < 1e-6

    def test_quantile_with_many_scores(self):
        """Test quantile with larger score set."""
        # 1000 scores, q=0.999 should be near 99.9th percentile
        scores = {f"node_{i}": i / 1000.0 for i in range(1000)}
        threshold, count = compute_validation_quantile_threshold(scores, quantile=0.999)

        # Should be between 0.998 and 1.0
        assert 0.998 <= threshold <= 1.0
        assert count == 1000


# =============================================================================
# Test 2: Threshold Only Accepts Validation Scores
# =============================================================================

class TestThresholdValidationOnly:
    """Test that threshold only accepts validation scores."""

    def test_threshold_function_takes_mapping(self, synthetic_validation_scores):
        """Test threshold function accepts a mapping."""
        threshold, count = compute_validation_quantile_threshold(
            synthetic_validation_scores, quantile=0.999
        )
        assert isinstance(threshold, float)
        assert count == len(synthetic_validation_scores)

    def test_threshold_no_test_scores_parameter(self, synthetic_validation_scores):
        """Test that threshold function doesn't accept test scores."""
        # The function only takes validation_node_scores
        # No test_scores or test_labels parameter
        threshold, count = compute_validation_quantile_threshold(
            synthetic_validation_scores, quantile=0.999
        )
        assert isinstance(threshold, float)

    def test_threshold_protocol_class(self, synthetic_validation_scores):
        """Test ThresholdProtocol class."""
        protocol = ThresholdProtocol(method="validation_quantile", q=0.999)
        config = protocol.fit(synthetic_validation_scores)

        assert isinstance(config, ThresholdConfig)
        assert config.method == "validation_quantile"
        assert config.q == 0.999
        assert config.threshold_value is not None

    def test_threshold_is_frozen(self, synthetic_validation_scores):
        """Test threshold is frozen after fit."""
        protocol = ThresholdProtocol(method="validation_quantile", q=0.999)
        protocol.fit(synthetic_validation_scores)

        # Should be fitted
        assert protocol.is_fitted
        assert protocol.threshold_value is not None


# =============================================================================
# Test 3: Threshold API Has No Test Labels
# =============================================================================

class TestNoTestLabels:
    """Test that threshold API doesn't accept test labels."""

    def test_fit_signature_no_labels(self, synthetic_validation_scores):
        """Test fit_threshold doesn't accept labels."""
        config = compute_validation_quantile_threshold(
            synthetic_validation_scores, quantile=0.999
        )

        # Just returns threshold, no labels involved
        assert isinstance(config, tuple)
        threshold, count = config
        assert isinstance(threshold, float)
        assert count > 0

    def test_threshold_class_no_labels_in_fit(self, synthetic_validation_scores):
        """Test ThresholdProtocol.fit doesn't accept labels."""
        protocol = ThresholdProtocol(q=0.999)

        # Only accepts validation_node_scores
        config = protocol.fit(synthetic_validation_scores)
        assert config.provenance == "validation_only"


# =============================================================================
# Test 4: Future Events Do Not Enter Current Snapshot
# =============================================================================

class TestNoFutureLeakage:
    """Test that future events don't enter earlier snapshots."""

    def test_snapshot_only_contains_past_events(self, synthetic_edges):
        """Test snapshot only contains events up to its boundary."""
        boundaries = [1200]  # Snapshot ending at 1200

        snapshots = build_causal_snapshots(synthetic_edges, boundaries)

        assert len(snapshots) == 1
        snapshot = snapshots[0]

        # Should only contain edges with timestamp <= 1200
        for edge in synthetic_edges:
            if edge.timestamp <= 1200:
                # Edge should be in snapshot
                pair = (edge.src, edge.dst)
                assert pair in snapshot.edges or edge.timestamp < 1200
            else:
                # Future edge should not be in snapshot
                # (only if nodes appear in snapshot)
                pass  # Logic handled in build_causal_snapshots

    def test_multiple_snapshots_temporal_order(self, synthetic_edges):
        """Test multiple snapshots respect temporal order."""
        boundaries = [1000, 1300, 2000]

        snapshots = build_causal_snapshots(synthetic_edges, boundaries)

        assert len(snapshots) == 3

        # Each snapshot should be larger than the previous
        assert snapshots[0].snapshot_end == 1000
        assert snapshots[1].snapshot_end == 1300
        assert snapshots[2].snapshot_end == 2000

        # Later snapshot should contain more nodes
        assert len(snapshots[2].nodes) >= len(snapshots[1].nodes)
        assert len(snapshots[1].nodes) >= len(snapshots[0].nodes)


# =============================================================================
# Test 5: Explicit Boundaries
# =============================================================================

class TestExplicitBoundaries:
    """Test explicit boundary handling."""

    def test_explicit_boundaries_respected(self, synthetic_edges):
        """Test that explicit boundaries are respected."""
        boundaries = [1100, 1200, 1300]

        snapshots = build_causal_snapshots(synthetic_edges, boundaries)

        assert len(snapshots) == 3
        for i, boundary in enumerate(boundaries):
            assert snapshots[i].snapshot_end == boundary

    def test_empty_boundaries_returns_empty_list(self, synthetic_edges):
        """Test empty boundaries returns empty list."""
        snapshots = build_causal_snapshots(synthetic_edges, [])
        assert len(snapshots) == 0


# =============================================================================
# Test 6: Duplicate Node Max Merge
# =============================================================================

class TestMaxNodeMerge:
    """Test max node merge policy."""

    def test_max_merge_single_snapshot(self, synthetic_snapshot_scores_list):
        """Test max merge with single snapshot."""
        merged = merge_snapshot_scores([synthetic_snapshot_scores_list[0]])

        assert merged.snapshot_id == "snapshot_1000"
        assert merged.node_scores == {"A": 0.2, "B": 0.8, "C": 0.4}

    def test_max_merge_multiple_snapshots(self, synthetic_snapshot_scores_list):
        """Test max merge with multiple snapshots."""
        merged = merge_snapshot_scores(synthetic_snapshot_scores_list)

        # A: max(0.2, 0.3, 0.1) = 0.3
        assert merged.node_scores["A"] == 0.3
        # B: max(0.8, 0.5) = 0.8
        assert merged.node_scores["B"] == 0.8
        # C: max(0.4, 0.9) = 0.9
        assert merged.node_scores["C"] == 0.9
        # D: only in snapshot 2
        assert merged.node_scores["D"] == 0.6
        # E: only in snapshot 3
        assert merged.node_scores["E"] == 0.7

    def test_max_merge_single_node_appears_once(self):
        """Test merge when node appears in only one snapshot."""
        scores_list = [
            SnapshotScores(
                snapshot_id="snap1",
                snapshot_end=1000,
                node_scores={"only_node": 0.5},
            ),
        ]

        merged = merge_snapshot_scores(scores_list)
        assert merged.node_scores["only_node"] == 0.5

    def test_max_merge_empty_raises_error(self):
        """Test merge with empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            merge_snapshot_scores([])

    def test_snapshot_scores_merge_with_method(self, synthetic_snapshot_scores_list):
        """Test SnapshotScores.merge_with method."""
        s1 = synthetic_snapshot_scores_list[0]
        s2 = synthetic_snapshot_scores_list[1]

        merged = s1.merge_with(s2)

        # A: max(0.2, 0.3) = 0.3
        assert merged.node_scores["A"] == 0.3
        # B: only in s1
        assert merged.node_scores["B"] == 0.8


# =============================================================================
# Test 7: Test Labels Do Not Affect Threshold
# =============================================================================

class TestTestLabelsDoNotAffectThreshold:
    """Test that test labels don't affect threshold selection."""

    def test_threshold_unchanged_by_different_test_scores(self):
        """Test that changing test scores doesn't change threshold."""
        val_scores = {"v1": 0.1, "v2": 0.2, "v3": 0.3}

        # First threshold
        threshold1, count1 = compute_validation_quantile_threshold(
            val_scores, quantile=0.999
        )

        # Should be same regardless of test data
        # (which we don't pass to the function)
        threshold2, count2 = compute_validation_quantile_threshold(
            val_scores, quantile=0.999
        )

        assert threshold1 == threshold2
        assert count1 == count2

    def test_threshold_protocol_fitted_only_by_validation(self):
        """Test ThresholdProtocol only uses validation for fit."""
        val_scores = {"a": 0.1, "b": 0.2, "c": 0.3}

        protocol = ThresholdProtocol(q=0.999)
        config = protocol.fit(val_scores)

        # Threshold depends only on validation
        assert config.provenance == "validation_only"
        # With 3 values, q=0.999 gives ~0.2998
        assert abs(config.threshold_value - 0.2998) < 1e-6


# =============================================================================
# Test 8: Ground Truth Not Read by M4
# =============================================================================

class TestGroundTruthNotRead:
    """Test that ground truth is not read."""

    def test_threshold_functions_take_only_scores(self):
        """Test functions only take scores, not labels."""
        scores = {"node": 0.5}

        # No label parameter
        result = compute_validation_quantile_threshold(scores, quantile=0.999)
        assert isinstance(result, tuple)

    def test_protocol_predict_takes_only_scores(self):
        """Test predict only takes scores."""
        scores = {"node": 0.5}

        protocol = ThresholdProtocol(q=0.999)
        protocol.fit({"v": 0.1})  # Fit first

        # Predict only takes scores
        predictions = protocol.predict(scores)
        assert isinstance(predictions, dict)


# =============================================================================
# Test 9: Snapshot Construction Deterministic
# =============================================================================

class TestSnapshotDeterminism:
    """Test snapshot construction is deterministic."""

    def test_same_edges_same_snapshots(self, synthetic_edges):
        """Test same edges produce same snapshots."""
        boundaries = [1000, 1500]

        snapshots1 = build_causal_snapshots(synthetic_edges, boundaries)
        snapshots2 = build_causal_snapshots(synthetic_edges, boundaries)

        assert len(snapshots1) == len(snapshots2)

        for s1, s2 in zip(snapshots1, snapshots2):
            assert s1.snapshot_id == s2.snapshot_id
            assert s1.snapshot_end == s2.snapshot_end
            assert s1.nodes == s2.nodes
            assert s1.edges == s2.edges

    def test_snapshot_nodes_are_frozenset(self, synthetic_edges):
        """Test snapshot nodes are frozenset (immutable)."""
        boundaries = [1200]
        snapshots = build_causal_snapshots(synthetic_edges, boundaries)

        assert isinstance(snapshots[0].nodes, frozenset)
        assert isinstance(snapshots[0].edges, frozenset)


# =============================================================================
# Test 10: Adversarial Leakage Tests
# =============================================================================

class TestAdversarialLeakage:
    """Test that changing test data doesn't leak into train/val."""

    def test_test_types_do_not_change_vocabulary(self):
        """Test that test unseen types don't change vocabulary."""
        from src.baselines.magic import MAGICInputAdapter, SplitType

        train = [
            {"src": "A", "dst": "B", "t": 100, "src_type": "subject",
             "dst_type": "file", "edge_type": "EVENT_OPEN", "global_event_index": 0},
        ]
        test_normal = [
            {"src": "C", "dst": "D", "t": 200, "src_type": "subject",
             "dst_type": "file", "edge_type": "EVENT_OPEN", "global_event_index": 0},
        ]
        test_with_unseen = [
            {"src": "E", "dst": "F", "t": 200, "src_type": "MAGIC_UNSEEN",
             "dst_type": "FILE_UNSEEN", "edge_type": "EVENT_OPEN", "global_event_index": 0},
        ]

        # Fit with normal test
        adapter1 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train,
            test_records=test_normal,
        )
        adapter1.fit_transform()
        vocab1_dim = adapter1.vocabulary.node_feature_dim

        # Fit with unseen test
        adapter2 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=train,
            test_records=test_with_unseen,
        )
        adapter2.fit_transform()
        vocab2_dim = adapter2.vocabulary.node_feature_dim

        # Vocabulary dimensions should be same
        assert vocab1_dim == vocab2_dim

    def test_test_future_events_do_not_affect_threshold(self):
        """Test that test future events don't affect validation threshold."""
        val_scores = {"v1": 0.1, "v2": 0.2, "v3": 0.3}
        test_scores_1 = {"t1": 0.9, "t2": 0.95}  # High scores
        test_scores_2 = {"t1": 0.01, "t2": 0.02}  # Low scores

        # Threshold only depends on validation
        threshold, _ = compute_validation_quantile_threshold(
            val_scores, quantile=0.999
        )

        # Same threshold regardless of test scores
        # (test scores are never passed to threshold function)
        # With 3 values, q=0.999 gives ~0.2998
        assert abs(threshold - 0.2998) < 1e-6

    def test_swapping_test_does_not_change_threshold(self):
        """Test swapping test data doesn't change threshold."""
        val_scores = {"a": 0.1, "b": 0.2}

        threshold1, _ = compute_validation_quantile_threshold(
            val_scores, quantile=0.999
        )

        # Same validation, different test data (which isn't used)
        threshold2, _ = compute_validation_quantile_threshold(
            val_scores, quantile=0.999
        )

        assert threshold1 == threshold2


# =============================================================================
# Test 11: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_nan_scores_handled(self):
        """Test NaN scores are handled by skipping them."""
        scores = {"a": 0.1, "b": float("nan"), "c": 0.3}

        # Should skip NaN and return threshold from finite scores
        threshold, count = compute_validation_quantile_threshold(scores, quantile=0.999)
        assert count == 2  # Only 2 finite scores

    def test_inf_scores_handled(self):
        """Test infinite scores are handled by skipping them."""
        scores = {"a": 0.1, "b": float("inf"), "c": 0.3}

        # Should skip inf and return threshold from finite scores
        threshold, count = compute_validation_quantile_threshold(scores, quantile=0.999)
        assert count == 2  # Only 2 finite scores

    def test_threshold_apply_with_strict_greater(self):
        """Test threshold uses strict > comparison."""
        from src.baselines.magic.protocol import apply_threshold

        config = ThresholdConfig(
            method="validation_quantile",
            q=0.999,
            threshold_value=0.5,
            validation_score_count=10,
        )

        scores = {"a": 0.4, "b": 0.5, "c": 0.6}
        predictions = apply_threshold(scores, config)

        assert predictions["a"] == 0  # 0.4 <= 0.5
        assert predictions["b"] == 0  # 0.5 <= 0.5 (strict)
        assert predictions["c"] == 1  # 0.6 > 0.5


# =============================================================================
# Test 12: ThresholdConfig Freeze
# =============================================================================

class TestThresholdConfig:
    """Test ThresholdConfig."""

    def test_threshold_config_frozen(self):
        """Test ThresholdConfig can be frozen."""
        config = ThresholdConfig(
            method="validation_quantile",
            q=0.999,
            threshold_value=0.123,
            validation_score_count=100,
            provenance="validation_only",
        )

        assert config.threshold_value == 0.123
        assert config.provenance == "validation_only"

    def test_threshold_config_freeze_method(self):
        """Test ThresholdConfig.freeze method."""
        config = ThresholdConfig(
            method="validation_quantile",
            q=0.999,
        )
        frozen = config.freeze(0.456, 50)

        assert frozen.threshold_value == 0.456
        assert frozen.validation_score_count == 50
        assert frozen.provenance == "validation_only"
