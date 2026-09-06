"""
Tests for MAGIC Input Adapter (M3)

测试 M3 的核心功能：
1. Train-only vocabulary fit
2. Validation 不扩 vocab
3. Test 不扩 vocab
4. Unseen type deterministic
5. E3/E5 共用 adapter
6. Timestamp + tie-break ordering
7. Split isolation
8. No ground-truth dependency
9. Deterministic repeated run
"""

from __future__ import annotations

import pytest
from typing import Dict, List

from src.baselines.magic import (
    MAGICInputAdapter,
    SplitType,
    EdgeRecord,
    NeutralGraphContract,
    TypeVocabulary,
    build_magic_input,
)


# =============================================================================
# Synthetic Fixtures
# =============================================================================

def _make_record(src, dst, t, src_type, dst_type, edge_type, global_event_index):
    """Helper to create a canonical record."""
    return {
        "src": src,
        "dst": dst,
        "t": t,
        "src_type": src_type,
        "dst_type": dst_type,
        "edge_type": edge_type,
        "global_event_index": global_event_index,
    }


@pytest.fixture
def synthetic_train_records():
    """Synthetic training records."""
    return [
        _make_record("node_A", "node_B", 1000, "subject", "file", "EVENT_OPEN", 0),
        _make_record("node_B", "node_C", 1100, "file", "netflow", "EVENT_WRITE", 1),
        _make_record("node_C", "node_A", 1200, "netflow", "subject", "EVENT_READ", 2),
        _make_record("node_A", "node_D", 1300, "subject", "file", "EVENT_CONNECT", 3),
    ]


@pytest.fixture
def synthetic_val_records():
    """Synthetic validation records."""
    return [
        _make_record("node_A", "node_B", 2000, "subject", "file", "EVENT_OPEN", 0),
        _make_record("node_B", "node_E", 2100, "file", "netflow", "EVENT_WRITE", 1),
    ]


@pytest.fixture
def synthetic_test_records():
    """Synthetic test records."""
    return [
        _make_record("node_X", "node_Y", 3000, "subject", "file", "EVENT_OPEN", 0),
        _make_record("node_Y", "node_Z", 3100, "file", "netflow", "EVENT_WRITE", 1),
    ]


@pytest.fixture
def synthetic_test_records_with_unseen_types():
    """Synthetic test records with unseen types."""
    return [
        _make_record("node_A", "node_B", 3000, "subject", "file", "EVENT_OPEN", 0),
        _make_record("node_B", "node_NEW", 3100, "file", "MAGIC_UNSEEN_TYPE", "EVENT_WRITE", 1),
        _make_record("node_NEW", "node_Z", 3200, "MAGIC_UNSEEN_NODE_TYPE", "netflow", "EVENT_WRITE", 2),
    ]


# =============================================================================
# Test 1: Train-only Vocabulary Fit
# =============================================================================

class TestTrainOnlyVocabulary:
    """Test that vocabulary is built from training data only."""

    def test_vocabulary_fit_creates_correct_types(self, synthetic_train_records):
        """Test vocabulary is correctly fitted from train data."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()

        vocab = adapter.vocabulary
        assert vocab is not None
        assert vocab.is_fitted()

    def test_vocabulary_includes_train_node_types(self, synthetic_train_records):
        """Test that train node types are in vocabulary."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()

        vocab = adapter.vocabulary
        # Check node types from train
        idx, seen = vocab.transform_node_type("subject")
        assert seen
        idx, seen = vocab.transform_node_type("file")
        assert seen
        idx, seen = vocab.transform_node_type("netflow")
        assert seen

    def test_vocabulary_includes_train_edge_types(self, synthetic_train_records):
        """Test that train edge types are in vocabulary."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()

        vocab = adapter.vocabulary
        # Check edge types from train
        idx, seen = vocab.transform_edge_type("EVENT_OPEN")
        assert seen
        idx, seen = vocab.transform_edge_type("EVENT_WRITE")
        assert seen
        idx, seen = vocab.transform_edge_type("EVENT_READ")
        assert seen

    def test_vocabulary_feature_dimensions(self, synthetic_train_records):
        """Test vocabulary feature dimensions are correct."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()

        vocab = adapter.vocabulary
        # 3 node types + 1 UNKNOWN = 4
        assert vocab.node_feature_dim == 4
        # 4 edge types + 1 UNKNOWN = 5
        assert vocab.edge_feature_dim == 5


# =============================================================================
# Test 2 & 3: Validation/Test Do Not Expand Vocabulary
# =============================================================================

class TestVocabularyIsolation:
    """Test that validation and test do not expand vocabulary."""

    def test_val_records_do_not_expand_vocabulary(
        self, synthetic_train_records, synthetic_val_records
    ):
        """Test validation records don't expand vocabulary."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            val_records=synthetic_val_records,
        )
        adapter.fit()

        train_vocab = adapter.vocabulary
        train_node_dim = train_vocab.node_feature_dim
        train_edge_dim = train_vocab.edge_feature_dim

        # Transform validation
        adapter.transform(synthetic_val_records, SplitType.VALIDATION)

        # Vocabulary should not change
        assert train_vocab.node_feature_dim == train_node_dim
        assert train_vocab.edge_feature_dim == train_edge_dim

    def test_test_records_do_not_expand_vocabulary(
        self, synthetic_train_records, synthetic_test_records
    ):
        """Test test records don't expand vocabulary."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            test_records=synthetic_test_records,
        )
        adapter.fit()

        train_vocab = adapter.vocabulary
        train_node_dim = train_vocab.node_feature_dim
        train_edge_dim = train_vocab.edge_feature_dim

        # Transform test
        adapter.transform(synthetic_test_records, SplitType.TEST)

        # Vocabulary should not change
        assert train_vocab.node_feature_dim == train_node_dim
        assert train_vocab.edge_feature_dim == train_edge_dim

    def test_fit_transform_maintains_vocabulary(
        self, synthetic_train_records, synthetic_val_records, synthetic_test_records
    ):
        """Test fit_transform maintains vocabulary across all splits."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            val_records=synthetic_val_records,
            test_records=synthetic_test_records,
        )

        train_contract, val_contract, test_contract = adapter.fit_transform()

        # All contracts should have same vocabulary dimensions
        train_vocab = train_contract.type_vocabulary
        val_vocab = val_contract.type_vocabulary
        test_vocab = test_contract.type_vocabulary

        assert train_vocab is val_vocab
        assert train_vocab is test_vocab
        assert train_vocab.node_feature_dim == val_vocab.node_feature_dim
        assert train_vocab.edge_feature_dim == val_vocab.edge_feature_dim


# =============================================================================
# Test 4: Unseen Type Deterministic
# =============================================================================

class TestUnseenTypeHandling:
    """Test deterministic handling of unseen types."""

    def test_unseen_node_type_returns_deterministic_index(
        self, synthetic_train_records, synthetic_test_records_with_unseen_types
    ):
        """Test unseen node types return consistent index."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            test_records=synthetic_test_records_with_unseen_types,
        )
        adapter.fit()

        vocab = adapter.vocabulary

        # Multiple calls should return same index
        idx1, seen1 = vocab.transform_node_type("MAGIC_UNSEEN_TYPE")
        idx2, seen2 = vocab.transform_node_type("MAGIC_UNSEEN_TYPE")
        assert not seen1
        assert not seen2
        assert idx1 == idx2

    def test_unseen_edge_type_returns_deterministic_index(
        self, synthetic_train_records
    ):
        """Test unseen edge types return consistent index."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()

        vocab = adapter.vocabulary

        # Multiple calls should return same index
        idx1, seen1 = vocab.transform_edge_type("EVENT_TOTALLY_NEW")
        idx2, seen2 = vocab.transform_edge_type("EVENT_TOTALLY_NEW")
        assert not seen1
        assert not seen2
        assert idx1 == idx2

    def test_unseen_types_tracked_in_contract(
        self, synthetic_train_records, synthetic_test_records_with_unseen_types
    ):
        """Test unseen types are tracked in contract."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            test_records=synthetic_test_records_with_unseen_types,
        )
        adapter.fit()
        test_contract = adapter.transform(
            synthetic_test_records_with_unseen_types, SplitType.TEST
        )

        # Unseen types should be tracked
        assert "MAGIC_UNSEEN_TYPE" in test_contract.unseen_node_types
        assert "MAGIC_UNSEEN_NODE_TYPE" in test_contract.unseen_node_types


# =============================================================================
# Test 5: E3/E5 Share Same Adapter
# =============================================================================

class TestE3E5SharedAdapter:
    """Test that E3 and E5 use the same adapter class."""

    def test_e3_adapter_creation(self, synthetic_train_records):
        """Test E3 adapter creation."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        assert adapter.dataset == "THEIA_E3"
        assert not adapter.is_fitted

    def test_e5_adapter_creation(self, synthetic_train_records):
        """Test E5 adapter creation."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E5",
            train_records=synthetic_train_records,
        )
        assert adapter.dataset == "THEIA_E5"
        assert not adapter.is_fitted

    def test_e3_e5_same_interface(self, synthetic_train_records):
        """Test E3 and E5 have same interface."""
        adapter_e3 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter_e5 = MAGICInputAdapter(
            dataset="THEIA_E5",
            train_records=synthetic_train_records,
        )

        # Same methods
        assert hasattr(adapter_e3, "fit")
        assert hasattr(adapter_e3, "transform")
        assert hasattr(adapter_e5, "fit")
        assert hasattr(adapter_e5, "transform")

        # Same fit/transform behavior
        adapter_e3.fit()
        adapter_e5.fit()

        assert adapter_e3.is_fitted == adapter_e5.is_fitted
        assert adapter_e3.vocabulary is not None
        assert adapter_e5.vocabulary is not None


# =============================================================================
# Test 6: Timestamp + Tie-break Ordering
# =============================================================================

class TestDeterministicOrdering:
    """Test deterministic ordering by timestamp + global_event_index."""

    def test_edges_sorted_by_timestamp(
        self, synthetic_train_records
    ):
        """Test edges are sorted by timestamp."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter.fit()
        contract = adapter.transform(synthetic_train_records, SplitType.TRAIN)

        sorted_edges = contract.get_sorted_edges()

        # Check timestamps are non-decreasing
        timestamps = [e.timestamp for e in sorted_edges]
        assert timestamps == sorted(timestamps)

    def test_tie_break_by_global_event_index(self):
        """Test tie-break by global_event_index."""
        # Create records with same timestamp
        records = [
            _make_record("A", "B", 1000, "s", "f", "EVENT_OPEN", 5),
            _make_record("C", "D", 1000, "s", "f", "EVENT_OPEN", 0),
            _make_record("E", "F", 1000, "s", "f", "EVENT_OPEN", 3),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        sorted_edges = contract.get_sorted_edges()

        # Should be sorted by global_event_index as tie-break
        indices = [e.global_event_index for e in sorted_edges]
        # Records come in order: gei=5, gei=0, gei=3
        # After sorting by (timestamp, gei): gei=0, gei=3, gei=5
        assert indices == [0, 3, 5]

    def test_deterministic_repeated_run(self, synthetic_train_records):
        """Test deterministic output across repeated runs."""
        adapter1 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter1.fit()
        contract1 = adapter1.transform(synthetic_train_records, SplitType.TRAIN)

        adapter2 = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )
        adapter2.fit()
        contract2 = adapter2.transform(synthetic_train_records, SplitType.TRAIN)

        # Same sorted edges
        edges1 = contract1.get_sorted_edges()
        edges2 = contract2.get_sorted_edges()

        assert len(edges1) == len(edges2)
        for e1, e2 in zip(edges1, edges2):
            assert e1.timestamp == e2.timestamp
            assert e1.global_event_index == e2.global_event_index
            assert e1.src == e2.src
            assert e1.dst == e2.dst


# =============================================================================
# Test 7: Split Isolation
# =============================================================================

class TestSplitIsolation:
    """Test split isolation."""

    def test_train_val_test_are_different_objects(
        self, synthetic_train_records, synthetic_val_records, synthetic_test_records
    ):
        """Test train/val/test are distinct data objects."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            val_records=synthetic_val_records,
            test_records=synthetic_test_records,
        )

        train_contract, val_contract, test_contract = adapter.fit_transform()

        # Should be different objects
        assert train_contract is not val_contract
        assert train_contract is not test_contract
        assert val_contract is not test_contract

    def test_split_statistics_correct(
        self, synthetic_train_records, synthetic_val_records, synthetic_test_records
    ):
        """Test split statistics are correct."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            val_records=synthetic_val_records,
            test_records=synthetic_test_records,
        )

        train_contract, val_contract, test_contract = adapter.fit_transform()

        # Check split edges
        train_edges = train_contract.get_split_edges(SplitType.TRAIN)
        val_edges = val_contract.get_split_edges(SplitType.VALIDATION)
        test_edges = test_contract.get_split_edges(SplitType.TEST)

        assert len(train_edges) > 0
        assert len(val_edges) > 0
        assert len(test_edges) > 0

    def test_no_concat_train_val_test(self, synthetic_train_records, synthetic_val_records):
        """Test that train+val+test are not concatenated."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            val_records=synthetic_val_records,
        )
        adapter.fit()

        train_contract = adapter.transform(synthetic_train_records, SplitType.TRAIN)
        val_contract = adapter.transform(synthetic_val_records, SplitType.VALIDATION)

        # Each contract should only contain its own split
        train_nodes = set(train_contract.nodes.keys())
        val_nodes = set(val_contract.nodes.keys())

        # Should have some overlap (node_A appears in both)
        # But not complete overlap
        assert train_nodes != val_nodes or len(train_nodes) == 0


# =============================================================================
# Test 8: No Ground Truth Dependency
# =============================================================================

class TestNoGroundTruthDependency:
    """Test that adapter does not depend on ground truth."""

    def test_adapter_works_without_ground_truth(self, synthetic_train_records):
        """Test adapter works without any ground truth data."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
            # No val or test
        )
        adapter.fit()
        contract = adapter.transform(synthetic_train_records, SplitType.TRAIN)

        assert contract is not None
        assert contract.type_vocabulary is not None

    def test_records_do_not_contain_labels(self, synthetic_train_records):
        """Test that records don't need label fields."""
        # Records should not require y_true, is_malicious, etc.
        for record in synthetic_train_records:
            assert "y_true" not in record
            assert "is_malicious" not in record
            assert "label" not in record


# =============================================================================
# Test 9: Direction Rule (READ/RECV/LOAD reversed)
# =============================================================================

class TestDirectionRule:
    """Test MAGIC direction rule for edge reversal."""

    def test_read_edge_reversed(self):
        """Test READ edges are reversed."""
        records = [
            _make_record("node_A", "node_B", 1000, "s", "f", "EVENT_READ", 0),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        # Should be reversed: dst -> src
        assert edges[0].src == "node_B"
        assert edges[0].dst == "node_A"

    def test_recv_edge_reversed(self):
        """Test RECV edges are reversed."""
        records = [
            _make_record("node_A", "node_B", 1000, "s", "f", "EVENT_RECVMSG", 0),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        assert edges[0].src == "node_B"
        assert edges[0].dst == "node_A"

    def test_load_edge_reversed(self):
        """Test LOAD edges are reversed."""
        records = [
            _make_record("node_A", "node_B", 1000, "s", "f", "EVENT_LOAD", 0),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        assert edges[0].src == "node_B"
        assert edges[0].dst == "node_A"

    def test_write_edge_not_reversed(self):
        """Test WRITE edges are not reversed."""
        records = [
            _make_record("node_A", "node_B", 1000, "s", "f", "EVENT_WRITE", 0),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        assert len(edges) == 1
        assert edges[0].src == "node_A"
        assert edges[0].dst == "node_B"


# =============================================================================
# Test 10: Simple Graph Policy (Deduplication)
# =============================================================================

class TestSimpleGraphPolicy:
    """Test simple graph policy (first edge only)."""

    def test_duplicate_edge_removed(self):
        """Test duplicate (src, dst) edges are removed."""
        records = [
            _make_record("node_A", "node_B", 1000, "s", "f", "EVENT_OPEN", 0),
            _make_record("node_A", "node_B", 1100, "s", "f", "EVENT_WRITE", 1),
        ]

        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=records,
        )
        adapter.fit()
        contract = adapter.transform(records, SplitType.TRAIN)

        edges = contract.get_sorted_edges()
        # Only first edge should be kept
        assert len(edges) == 1
        assert edges[0].global_event_index == 0


# =============================================================================
# Test 11: build_magic_input Helper
# =============================================================================

class TestBuildMagicInput:
    """Test the build_magic_input helper function."""

    def test_build_magic_input_returns_three_contracts(
        self, synthetic_train_records, synthetic_val_records, synthetic_test_records
    ):
        """Test build_magic_input returns train/val/test contracts."""
        artifacts = {
            "train": synthetic_train_records,
            "validation": synthetic_val_records,
            "test": synthetic_test_records,
        }

        train, val, test = build_magic_input("THEIA_E3", artifacts)

        assert isinstance(train, NeutralGraphContract)
        assert isinstance(val, NeutralGraphContract)
        assert isinstance(test, NeutralGraphContract)

    def test_build_magic_input_with_empty_val(
        self, synthetic_train_records, synthetic_test_records
    ):
        """Test build_magic_input with empty validation."""
        artifacts = {
            "train": synthetic_train_records,
            "validation": [],
            "test": synthetic_test_records,
        }

        train, val, test = build_magic_input("THEIA_E3", artifacts)

        assert isinstance(train, NeutralGraphContract)
        assert isinstance(val, NeutralGraphContract)
        assert isinstance(test, NeutralGraphContract)


# =============================================================================
# Test 12: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_transform_before_fit_raises_error(self, synthetic_train_records):
        """Test transform before fit raises RuntimeError."""
        adapter = MAGICInputAdapter(
            dataset="THEIA_E3",
            train_records=synthetic_train_records,
        )

        with pytest.raises(RuntimeError, match="fit"):
            adapter.transform(synthetic_train_records, SplitType.TRAIN)

    def test_invalid_quantile_raises_error(self):
        """Test invalid quantile raises ValueError."""
        from src.baselines.magic.protocol import compute_validation_quantile_threshold

        with pytest.raises(ValueError):
            compute_validation_quantile_threshold({"node": 0.5}, quantile=1.5)

    def test_empty_scores_raises_error(self):
        """Test empty scores raises ValueError."""
        from src.baselines.magic.protocol import compute_validation_quantile_threshold

        with pytest.raises(ValueError, match="no finite"):
            compute_validation_quantile_threshold({}, quantile=0.999)
