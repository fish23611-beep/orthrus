"""
MAGIC Synthetic Smoke Tests

本模块实现 M8: MAGIC Synthetic End-to-End Smoke Tests。

覆盖测试：
1. synthetic fixture validity
2. enough nodes for k=10
3. full train→val→test synthetic E2E
4. no ground-truth leakage
5. no future leakage
6. max node merge
7. q=0.999 threshold
8. all canonical test nodes retained
9. seed=0 repeated checksum identical
10. seed=1 stochastic trace differs
11. artifact schema
12. raw artifact has no y_true
13. final prediction may contain y_true
14. threshold provenance
15. is_smoke=true
16. dataset=SYNTHETIC_MAGIC
17. backend=synthetic_smoke
18. cache same identity reusable
19. seed change invalidates cache
20. fixture change invalidates cache
21. config change invalidates cache
22. upstream SHA change invalidates cache
23. collector default excludes smoke
24. explicit debug include smoke if implemented
25. smoke does not require DGL
26. smoke does not require CUDA
27. smoke does not require PostgreSQL
28. smoke does not access network
29. M3-M7 regressions remain green
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
import numpy as np

# Setup paths
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from tests.test_magic.fixtures.synthetic_fixture import (
    generate_synthetic_fixture,
    generate_ground_truth,
    generate_altered_ground_truth,
    SyntheticFixtureBuilder,
    compute_fixture_checksum,
    MIN_TRAIN_NODES,
    SNAPSHOT_T1,
    SNAPSHOT_T2,
    T_EARLY,
    T_LATE,
)
from src.baselines.magic.smoke import (
    SmokeConfig,
    run_synthetic_e2e,
    SmokeResult,
    compute_config_fingerprint,
    verify_cache_identity,
    MAGIC_K_NEIGHBORS,
    MAGIC_Q,
    MAGIC_BACKEND,
    MAGIC_DATASET,
    UPSTREAM_SHA_FROZEN,
)
from src.baselines.magic.input_adapter import MAGICInputAdapter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_fixture():
    """Generate synthetic fixture for testing."""
    return generate_synthetic_fixture(seed=0, train_node_count=MIN_TRAIN_NODES)


@pytest.fixture
def synthetic_fixture_seed1():
    """Generate synthetic fixture with seed=1."""
    return generate_synthetic_fixture(seed=1, train_node_count=MIN_TRAIN_NODES)


@pytest.fixture
def ground_truth(synthetic_fixture):
    """Generate ground truth labels."""
    return generate_ground_truth(synthetic_fixture)


@pytest.fixture
def altered_ground_truth(ground_truth):
    """Generate altered ground truth for leakage test."""
    return generate_altered_ground_truth(ground_truth)


@pytest.fixture
def smoke_config(synthetic_fixture):
    """Generate smoke config."""
    fixture_checksum = compute_fixture_checksum(synthetic_fixture)
    return SmokeConfig(
        seed=0,
        k=MAGIC_K_NEIGHBORS,
        q=MAGIC_Q,
        output_dir=Path(tempfile.mkdtemp()),
        backend=MAGIC_BACKEND,
        dataset=MAGIC_DATASET,
        is_smoke=True,
        upstream_sha=UPSTREAM_SHA_FROZEN,
        fixture_checksum=fixture_checksum,
    )


# =============================================================================
# Test 1-2: Synthetic Fixture Validity
# =============================================================================

class TestSyntheticFixtureValidity:
    """Test synthetic fixture generation and validity."""

    def test_fixture_has_required_splits(self, synthetic_fixture):
        """Test 1: fixture has train, validation, test splits."""
        assert "train" in synthetic_fixture
        assert "validation" in synthetic_fixture
        assert "test" in synthetic_fixture

    def test_fixture_has_enough_train_nodes(self, synthetic_fixture):
        """Test 2: train has enough nodes for k=10 MAGIC."""
        train_nodes = set()
        for r in synthetic_fixture["train"]:
            train_nodes.add(r.get("src", ""))
            train_nodes.add(r.get("dst", ""))

        assert len(train_nodes) >= MIN_TRAIN_NODES, \
            f"Train nodes {len(train_nodes)} < MIN_TRAIN_NODES {MIN_TRAIN_NODES}"

    def test_fixture_has_multiple_node_types(self, synthetic_fixture):
        """Fixture has multiple node types."""
        node_types = set()
        for r in synthetic_fixture["train"]:
            node_types.add(r.get("src_type", ""))
            node_types.add(r.get("dst_type", ""))

        assert len(node_types) >= 2, \
            f"Expected at least 2 node types, got {node_types}"

    def test_fixture_has_multiple_edge_types(self, synthetic_fixture):
        """Fixture has multiple edge types including READ/RECV/LOAD."""
        edge_types = set()
        for r in synthetic_fixture["train"]:
            edge_types.add(r.get("edge_type", ""))

        # Must include READ/RECV/LOAD for direction reversal testing
        assert any("READ" in et for et in edge_types), "Missing READ edge type"
        assert len(edge_types) >= 2, f"Expected multiple edge types, got {edge_types}"

    def test_fixture_has_same_timestamp_different_event_index(self, synthetic_fixture):
        """Fixture has same timestamp with different global_event_index."""
        timestamp_to_indices = {}
        for r in synthetic_fixture["train"]:
            ts = r.get("t", 0)
            idx = r.get("global_event_index", 0)
            if ts not in timestamp_to_indices:
                timestamp_to_indices[ts] = []
            timestamp_to_indices[ts].append(idx)

        # Find a timestamp with multiple event indices
        has_multi_index = any(
            len(indices) > 1 for indices in timestamp_to_indices.values()
        )
        assert has_multi_index, \
            "Expected at least one timestamp with multiple global_event_index values"

    def test_fixture_validation_has_unseen_types(self, synthetic_fixture):
        """Validation contains train-unseen types."""
        train_types = set()
        for r in synthetic_fixture["train"]:
            train_types.add(r.get("src_type", ""))
            train_types.add(r.get("dst_type", ""))

        val_types = set()
        for r in synthetic_fixture["validation"]:
            val_types.add(r.get("src_type", ""))
            val_types.add(r.get("dst_type", ""))

        unseen = val_types - train_types
        assert len(unseen) > 0, \
            f"Validation should have unseen types. Train: {train_types}, Val: {val_types}"


# =============================================================================
# Test 3: Full Train→Val→Test E2E
# =============================================================================

class TestFullE2E:
    """Test full synthetic E2E pipeline."""

    def test_full_e2e_produces_predictions(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 3: Full train→val→test synthetic E2E produces predictions."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        assert result is not None
        assert isinstance(result, SmokeResult)
        assert len(result.predictions) > 0
        assert result.threshold_value is not None

    def test_full_e2e_preserves_all_test_nodes(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 8: All canonical test nodes are retained."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Get all test nodes from fixture
        test_nodes = set()
        for r in synthetic_fixture["test"]:
            test_nodes.add(r.get("src", ""))
            test_nodes.add(r.get("dst", ""))

        # All test nodes should be in predictions
        predicted_nodes = set(result.predictions.keys())
        missing = test_nodes - predicted_nodes
        assert len(missing) == 0, \
            f"Missing test nodes in predictions: {missing}"

    def test_e2e_uses_k10(
        self,
        synthetic_fixture,
        ground_truth,
    ):
        """Test E2E uses k=10 for MAGIC scoring."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)
        config = SmokeConfig(
            seed=0,
            k=10,  # Explicit k=10
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=config,
            ground_truth=ground_truth,
        )

        assert result.config.k == 10


# =============================================================================
# Test 4: No Ground-truth Leakage
# =============================================================================

class TestNoGroundTruthLeakage:
    """Test that ground truth does not affect scores/predictions."""

    def test_scores_invariance_to_labels(
        self,
        synthetic_fixture,
        ground_truth,
        altered_ground_truth,
        smoke_config,
    ):
        """Test 4: Scores/predictions invariant to ground truth changes."""
        # Run with original ground truth
        result_a = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Run with altered ground truth
        result_b = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=altered_ground_truth,
        )

        # Scores must be identical
        assert result_a.test_node_scores == result_b.test_node_scores, \
            "Test scores changed when ground truth was altered"

        # Predictions must be identical
        assert result_a.predictions == result_b.predictions, \
            "Predictions changed when ground truth was altered"

        # Threshold must be identical
        assert result_a.threshold_value == result_b.threshold_value, \
            "Threshold changed when ground truth was altered"

        # Metrics may differ
        assert result_a.metrics != result_b.metrics, \
            "Metrics should differ when labels are flipped"

    def test_raw_artifact_no_y_true(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 12: Raw artifact has no y_true."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Read raw artifact files
        raw_dir = result.artifact_dir / "raw_magic"

        # Check validation scores
        val_scores_file = raw_dir / "validation_node_window_scores.csv"
        with open(val_scores_file) as f:
            content = f.read()
        assert "y_true" not in content.lower(), \
            "Raw validation artifact contains y_true"

        # Check test scores
        test_scores_file = raw_dir / "test_node_window_scores.csv"
        with open(test_scores_file) as f:
            content = f.read()
        assert "y_true" not in content.lower(), \
            "Raw test artifact contains y_true"


# =============================================================================
# Test 5: No Future Leakage
# =============================================================================

class TestNoFutureLeakage:
    """Test that future events don't leak into earlier snapshots."""

    def test_snapshot_boundary_respected(
        self,
        synthetic_fixture,
    ):
        """Test 5: Snapshot boundary T1 excludes future events at T2 > T1."""
        from src.baselines.magic.input_adapter import MAGICInputAdapter
        from src.baselines.magic.protocol import build_causal_snapshots

        # Build adapter
        adapter = MAGICInputAdapter(
            dataset=MAGIC_DATASET,
            train_records=synthetic_fixture["train"],
            val_records=synthetic_fixture["validation"],
            test_records=synthetic_fixture["test"],
        )
        train_contract, _, _ = adapter.fit_transform()

        # Get sorted edges
        edges = train_contract.get_sorted_edges()

        # Build snapshot at T1
        snapshots = build_causal_snapshots(edges, [SNAPSHOT_T1])

        assert len(snapshots) == 1
        snapshot = snapshots[0]

        # Verify no future events
        for edge in edges:
            if edge.timestamp > SNAPSHOT_T1:
                # This edge should not be in snapshot
                assert edge.src not in snapshot.nodes, \
                    f"Future event leaked: {edge.src} at {edge.timestamp} > {SNAPSHOT_T1}"
                assert edge.dst not in snapshot.nodes, \
                    f"Future event leaked: {edge.dst} at {edge.timestamp} > {SNAPSHOT_T1}"

    def test_global_event_index_stable(
        self,
        synthetic_fixture,
    ):
        """Test global_event_index mapping is stable across runs."""
        # Run adapter twice
        adapter1 = MAGICInputAdapter(
            dataset=MAGIC_DATASET,
            train_records=synthetic_fixture["train"],
            val_records=synthetic_fixture["validation"],
            test_records=synthetic_fixture["test"],
        )
        train1, _, _ = adapter1.fit_transform()

        adapter2 = MAGICInputAdapter(
            dataset=MAGIC_DATASET,
            train_records=synthetic_fixture["train"],
            val_records=synthetic_fixture["validation"],
            test_records=synthetic_fixture["test"],
        )
        train2, _, _ = adapter2.fit_transform()

        # Compare edges by (timestamp, global_event_index, src, dst)
        def edge_key(e):
            return (e.timestamp, e.global_event_index, e.src, e.dst)

        edges1 = sorted(train1.get_sorted_edges(), key=edge_key)
        edges2 = sorted(train2.get_sorted_edges(), key=edge_key)

        assert len(edges1) == len(edges2)
        for e1, e2 in zip(edges1, edges2):
            assert e1.timestamp == e2.timestamp
            assert e1.global_event_index == e2.global_event_index
            assert e1.src == e2.src
            assert e1.dst == e2.dst


# =============================================================================
# Test 6: Max Node Merge
# =============================================================================

class TestMaxNodeMerge:
    """Test max node merge policy."""

    def test_max_merge_policy(self):
        """Test 6: Max merge policy for node scores."""
        from src.baselines.magic.contracts import SnapshotScores

        # Create two snapshot scores for same node
        scores1 = SnapshotScores(
            snapshot_id="snap1",
            snapshot_end=1000,
            node_scores={"node_a": 1.0, "node_b": 2.0},
        )

        scores2 = SnapshotScores(
            snapshot_id="snap2",
            snapshot_end=2000,
            node_scores={"node_a": 3.0, "node_c": 1.0},
        )

        # Merge
        merged = scores1.merge_with(scores2)

        # node_a should have max(1.0, 3.0) = 3.0
        assert merged.node_scores["node_a"] == 3.0
        # node_b should be preserved
        assert merged.node_scores["node_b"] == 2.0
        # node_c should be added
        assert merged.node_scores["node_c"] == 1.0


# =============================================================================
# Test 7: q=0.999 Threshold
# =============================================================================

class TestQuantileThreshold:
    """Test q=0.999 threshold computation."""

    def test_q_0999_threshold(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 7: q=0.999 threshold produces non-degenerate value."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Threshold should be a valid finite number
        assert result.threshold_value is not None
        assert np.isfinite(result.threshold_value)
        assert result.threshold_value > 0

    def test_threshold_provenance(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 14: Threshold provenance is validation_only."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        assert result.threshold_config.provenance == "validation_only"


# =============================================================================
# Test 9-10: Seed Reproducibility
# =============================================================================

class TestSeedReproducibility:
    """Test seed reproducibility."""

    def test_seed0_identical_checksum(
        self,
        synthetic_fixture,
        ground_truth,
    ):
        """Test 9: seed=0 produces identical checksums."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)
        config1 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )
        config2 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        result1 = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=config1,
            ground_truth=ground_truth,
        )
        result2 = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=config2,
            ground_truth=ground_truth,
        )

        # Checksums should be identical
        assert result1.get_score_checksum() == result2.get_score_checksum(), \
            "seed=0 produced different score checksums"
        assert result1.get_prediction_checksum() == result2.get_prediction_checksum(), \
            "seed=0 produced different prediction checksums"

    def test_seed1_differs_from_seed0(
        self,
        synthetic_fixture,
        ground_truth,
    ):
        """Test 10: seed=1 produces different stochastic trace than seed=0."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)

        config0 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )
        config1 = SmokeConfig(
            seed=1,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        result0 = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=config0,
            ground_truth=ground_truth,
        )
        result1 = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=config1,
            ground_truth=ground_truth,
        )

        # At least one should differ
        checksums_differ = (
            result0.get_score_checksum() != result1.get_score_checksum() or
            result0.get_prediction_checksum() != result1.get_prediction_checksum()
        )
        assert checksums_differ, \
            "seed=1 produced same traces as seed=0 (expected difference)"


# =============================================================================
# Test 11: Artifact Schema
# =============================================================================

class TestArtifactSchema:
    """Test artifact output schema."""

    def test_artifact_schema(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 11: Artifact has correct schema."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        artifact_dir = result.artifact_dir

        # Check required files exist
        required_files = [
            "environment.json",
            "config_resolved.yml",
            "runtime.json",
            "raw_magic/graph_manifest.json",
            "raw_magic/local_global_node_map.csv",
            "raw_magic/validation_node_window_scores.csv",
            "raw_magic/test_node_window_scores.csv",
            "raw_magic/knn_reference_manifest.json",
            "raw_magic/threshold.json",
            "node_scores/node_predictions.csv",
            "node_scores/metrics.json",
        ]

        for rel_path in required_files:
            file_path = artifact_dir / rel_path
            assert file_path.exists(), f"Missing required file: {rel_path}"

        # Verify JSON files are parseable
        for rel_path in ["environment.json", "runtime.json", "raw_magic/graph_manifest.json",
                         "raw_magic/threshold.json", "node_scores/metrics.json"]:
            file_path = artifact_dir / rel_path
            with open(file_path) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{rel_path} is not a JSON object"

    def test_csv_row_counts(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test CSV row counts are correct."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        artifact_dir = result.artifact_dir

        # Check validation scores CSV
        val_csv = artifact_dir / "raw_magic" / "validation_node_window_scores.csv"
        with open(val_csv) as f:
            val_lines = f.readlines()
        assert len(val_lines) >= 2, "Validation scores CSV too short"

        # Check test scores CSV
        test_csv = artifact_dir / "raw_magic" / "test_node_window_scores.csv"
        with open(test_csv) as f:
            test_lines = f.readlines()
        assert len(test_lines) >= 2, "Test scores CSV too short"


# =============================================================================
# Test 15-17: Metadata Fields
# =============================================================================

class TestMetadataFields:
    """Test required metadata fields."""

    def test_is_smoke_true(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 15: is_smoke=true in artifacts."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Check runtime.json
        runtime_file = result.artifact_dir / "runtime.json"
        with open(runtime_file) as f:
            runtime = json.load(f)
        assert runtime.get("is_smoke") == True, "is_smoke should be True"

        # Check metrics.json
        metrics_file = result.artifact_dir / "node_scores" / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)
        assert metrics.get("is_smoke") == True, "is_smoke should be True in metrics"

    def test_dataset_synthetic_magic(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 16: dataset=SYNTHETIC_MAGIC."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Check runtime.json
        runtime_file = result.artifact_dir / "runtime.json"
        with open(runtime_file) as f:
            runtime = json.load(f)
        assert runtime.get("dataset") == MAGIC_DATASET, \
            f"dataset should be {MAGIC_DATASET}"

    def test_backend_synthetic_smoke(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 17: backend=synthetic_smoke."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Check environment.json
        env_file = result.artifact_dir / "environment.json"
        with open(env_file) as f:
            env = json.load(f)
        assert env.get("backend") == MAGIC_BACKEND, \
            f"backend should be {MAGIC_BACKEND}"

        # Check runtime.json
        runtime_file = result.artifact_dir / "runtime.json"
        with open(runtime_file) as f:
            runtime = json.load(f)
        assert runtime.get("backend") == MAGIC_BACKEND, \
            f"backend should be {MAGIC_BACKEND}"


# =============================================================================
# Test 18-22: Cache Identity
# =============================================================================

class TestCacheIdentity:
    """Test cache/resume identity verification."""

    def test_same_identity_reusable(
        self,
        synthetic_fixture,
        ground_truth,
    ):
        """Test 18: Same config identity can be reused."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)

        config1 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )
        config2 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        assert verify_cache_identity(config1, config2), \
            "Same config should have same cache identity"

    def test_seed_change_invalidates(
        self,
        synthetic_fixture,
    ):
        """Test 19: Seed change invalidates cache."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)

        config0 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )
        config1 = SmokeConfig(
            seed=1,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        assert not verify_cache_identity(config0, config1), \
            "Different seed should have different cache identity"

    def test_fixture_change_invalidates(
        self,
        synthetic_fixture,
        synthetic_fixture_seed1,
    ):
        """Test 20: Fixture change invalidates cache."""
        checksum0 = compute_fixture_checksum(synthetic_fixture)
        checksum1 = compute_fixture_checksum(synthetic_fixture_seed1)

        config0 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=checksum0,
        )
        config1 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=checksum1,
        )

        assert not verify_cache_identity(config0, config1), \
            "Different fixture should have different cache identity"

    def test_config_change_invalidates_k(
        self,
        synthetic_fixture,
    ):
        """Test 21: Config change (k) invalidates cache."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)

        config_k10 = SmokeConfig(
            seed=0,
            k=10,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )
        config_k20 = SmokeConfig(
            seed=0,
            k=20,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
        )

        assert not verify_cache_identity(config_k10, config_k20), \
            "Different k should have different cache identity"

    def test_upstream_sha_change_invalidates(
        self,
        synthetic_fixture,
    ):
        """Test 22: Upstream SHA change invalidates cache."""
        fixture_checksum = compute_fixture_checksum(synthetic_fixture)

        config_sha1 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
            upstream_sha=UPSTREAM_SHA_FROZEN,
        )
        config_sha2 = SmokeConfig(
            seed=0,
            k=MAGIC_K_NEIGHBORS,
            q=MAGIC_Q,
            output_dir=Path(tempfile.mkdtemp()),
            fixture_checksum=fixture_checksum,
            upstream_sha="different_sha",
        )

        assert not verify_cache_identity(config_sha1, config_sha2), \
            "Different upstream SHA should have different cache identity"


# =============================================================================
# Test 23-24: Collector Default Excludes Smoke
# =============================================================================

class TestCollectorExclusion:
    """Test that collector default excludes smoke artifacts."""

    def test_collector_default_excludes_smoke(self):
        """Test 23: Collector defaults to skip smoke artifacts."""
        # Import collector
        from src.experiments.collect_results import collect

        # Create a temp directory with a fake smoke run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            run_status_dir = results_dir / "run_status"
            run_status_dir.mkdir(parents=True)

            # Create a run directory with is_smoke=true
            run_dir = root / "SYNTHETIC_MAGIC" / "runs" / "magic" / "seed_0"
            run_dir.mkdir(parents=True)

            # Write runtime.json with is_smoke=true
            runtime = {"is_smoke": True}
            with open(run_dir / "runtime.json", "w") as f:
                json.dump(runtime, f)

            # Write environment.json
            env = {"git_commit": "test"}
            with open(run_dir / "environment.json", "w") as f:
                json.dump(env, f)

            # Write status marker
            smoke_status = {
                "dataset": "SYNTHETIC_MAGIC",
                "status": "completed",
                "config": "/fake/config.yml",
                "seed": 0,
                "artifact_root": str(root),
            }
            with open(run_status_dir / "run_status.json", "w") as f:
                json.dump(smoke_status, f)

            # Collect with default (should skip smoke)
            rows = collect(root, include_smoke=False)
            assert len(rows) == 0, \
                "Collector should skip smoke artifacts by default"

    def test_collector_include_smoke_explicit(self):
        """Test 24: Collector includes smoke with explicit flag."""
        from src.experiments.collect_results import collect

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            run_status_dir = results_dir / "run_status"
            run_status_dir.mkdir(parents=True)

            # Create a run directory with is_smoke=true
            run_dir = root / "SYNTHETIC_MAGIC" / "runs" / "magic" / "seed_0"
            run_dir.mkdir(parents=True)

            # Write runtime.json with is_smoke=true
            runtime = {"is_smoke": True}
            with open(run_dir / "runtime.json", "w") as f:
                json.dump(runtime, f)

            # Write environment.json
            env = {"git_commit": "test"}
            with open(run_dir / "environment.json", "w") as f:
                json.dump(env, f)

            # Write config
            with open(run_dir / "config_resolved.yml", "w") as f:
                f.write("is_smoke: true\n")

            # Write metrics.json
            metrics = {"precision": 0.5, "recall": 0.5}
            node_scores_dir = run_dir / "node_scores"
            node_scores_dir.mkdir()
            with open(node_scores_dir / "metrics.json", "w") as f:
                json.dump(metrics, f)

            # Write status marker
            smoke_status = {
                "dataset": "SYNTHETIC_MAGIC",
                "status": "completed",
                "config": "/fake/config.yml",
                "seed": 0,
                "artifact_root": str(root),
            }
            with open(run_status_dir / "run_status.json", "w") as f:
                json.dump(smoke_status, f)

            # Collect with explicit include
            rows = collect(root, include_smoke=True)
            # Should include smoke run when explicitly requested
            assert len(rows) >= 1, \
                "Collector should include smoke with explicit include_smoke=True"


# =============================================================================
# Test 25-28: No External Dependencies
# =============================================================================

class TestNoExternalDependencies:
    """Test that smoke doesn't require external resources."""

    def test_smoke_no_dgl_import(self):
        """Test 25: Smoke does not require DGL."""
        # DGL should not be imported in smoke path
        # This is implicitly tested by running without DGL
        # We verify the smoke module doesn't import dgl
        smoke_path = SRC_ROOT / "src" / "baselines" / "magic" / "smoke.py"
        with open(smoke_path) as f:
            content = f.read()

        # Should not have dgl imports in smoke module
        assert "import dgl" not in content
        assert "from dgl" not in content

    def test_smoke_no_cuda_required(self):
        """Test 26: Smoke does not require CUDA."""
        # Verify no torch.cuda in smoke module
        smoke_path = SRC_ROOT / "src" / "baselines" / "magic" / "smoke.py"
        with open(smoke_path) as f:
            content = f.read()

        assert "torch.cuda" not in content

    def test_smoke_no_postgresql_required(self):
        """Test 27: Smoke does not require PostgreSQL."""
        # Verify no psycopg2 in smoke module
        smoke_path = SRC_ROOT / "src" / "baselines" / "magic" / "smoke.py"
        with open(smoke_path) as f:
            content = f.read()

        assert "psycopg2" not in content
        assert "postgresql" not in content.lower()

    def test_smoke_no_network_access(self):
        """Test 28: Smoke does not access network."""
        # Verify no requests/urllib in smoke module
        smoke_path = SRC_ROOT / "src" / "baselines" / "magic" / "smoke.py"
        with open(smoke_path) as f:
            content = f.read()

        assert "requests." not in content
        assert "urllib.request" not in content
        assert "http.client" not in content


# =============================================================================
# Test 29: M3-M7 Regression
# =============================================================================

class TestM3M7Regressions:
    """Test that M3-M7 components still work correctly."""

    def test_m3_input_adapter(self, synthetic_fixture):
        """Test M3: Input adapter still works."""
        from src.baselines.magic.input_adapter import MAGICInputAdapter

        adapter = MAGICInputAdapter(
            dataset=MAGIC_DATASET,
            train_records=synthetic_fixture["train"],
            val_records=synthetic_fixture["validation"],
            test_records=synthetic_fixture["test"],
        )
        train_c, val_c, test_c = adapter.fit_transform()

        assert train_c is not None
        assert val_c is not None
        assert test_c is not None
        assert train_c.train_node_count > 0

    def test_m4_protocol(self, synthetic_fixture):
        """Test M4: Protocol still works."""
        from src.baselines.magic.protocol import (
            fit_threshold,
            apply_threshold,
            ThresholdProtocol,
        )

        # Create fake validation scores
        val_scores = {f"node_{i}": float(i) / 10 for i in range(100)}

        # Fit threshold
        threshold_config = fit_threshold(val_scores, method="validation_quantile", q=0.999)
        assert threshold_config.threshold_value is not None

        # Apply threshold
        test_scores = {f"node_{i}": float(i) / 10 for i in range(50)}
        predictions = apply_threshold(test_scores, threshold_config)
        assert len(predictions) == 50

    def test_m5_scorer(self, synthetic_fixture):
        """Test M5: Scorer still works."""
        from src.baselines.magic.scoring import MAGICEntityScorer

        # Create fake embeddings
        train_emb = np.random.randn(50, 64).astype(np.float32)
        train_ids = [f"train_node_{i}" for i in range(50)]
        target_emb = np.random.randn(10, 64).astype(np.float32)
        target_ids = [f"test_node_{i}" for i in range(10)]

        scorer = MAGICEntityScorer(k=10, seed=0)
        scorer.fit(train_emb, train_ids)
        assert scorer.is_fitted

        scores = scorer.score(target_emb, target_ids)
        assert len(scores) == 10

    def test_m6_evaluator(self, synthetic_fixture):
        """Test M6: Evaluator still works."""
        from src.baselines.magic.evaluator import MagicEvaluator

        evaluator = MagicEvaluator(k=10)

        # Create fake scores
        val_scores = {f"node_{i}": float(i) / 10 for i in range(100)}
        test_scores = {f"node_{i}": float(i) / 10 for i in range(50)}

        # Fit threshold
        evaluator.fit_threshold(val_scores)
        assert evaluator.is_threshold_fitted

        # Predict
        predictions = evaluator.predict(test_scores)
        assert len(predictions) == 50

    def test_m7_seed_control(self):
        """Test M7: Seed control still works."""
        from src.baselines.magic.seed import MagicSeedController

        controller = MagicSeedController(seed=0)
        manifest = controller.set_all_seeds()

        assert manifest.seed == 0
        assert manifest.python_seed_set
        assert manifest.numpy_seed_set


# =============================================================================
# Test 13: Final Prediction May Contain y_true
# =============================================================================

class TestFinalPrediction:
    """Test final prediction artifact may contain y_true."""

    def test_final_prediction_with_y_true(
        self,
        synthetic_fixture,
        ground_truth,
        smoke_config,
    ):
        """Test 13: Final prediction artifact may contain y_true."""
        result = run_synthetic_e2e(
            fixture=synthetic_fixture,
            config=smoke_config,
            ground_truth=ground_truth,
        )

        # Read metrics.json - this is the final prediction artifact
        metrics_file = result.artifact_dir / "node_scores" / "metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Final artifact can contain metrics that depend on y_true
        # (precision, recall, etc.) but raw scores must not
        assert "precision" in metrics or "TP" in metrics or "mcc" in metrics, \
            "Final artifact should have metrics that depend on y_true"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
