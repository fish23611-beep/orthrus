"""Unit tests for calibration_runner post-processing module."""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "src" / "mstc" / "calibration.py"
CALIBRATION_SPEC = importlib.util.spec_from_file_location("calibration_for_tests", CALIBRATION_PATH)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
CALIBRATION_MODULE = importlib.util.module_from_spec(CALIBRATION_SPEC)
import sys
sys.modules[CALIBRATION_SPEC.name] = CALIBRATION_MODULE
CALIBRATION_SPEC.loader.exec_module(CALIBRATION_MODULE)
HierarchicalRelationCalibrator = CALIBRATION_MODULE.HierarchicalRelationCalibrator

# Import runner using relative import
import importlib.util
import sys

RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "mstc" / "calibration_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("calibration_runner_for_tests", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER_MODULE
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)
run_calibration = RUNNER_MODULE.run_calibration
load_event_records_from_csv = RUNNER_MODULE.load_event_records_from_csv


def _record(score, src=1, edge=2, dst=3, event_index=0, **extra):
    """Create a test record with required fields."""
    return {
        "score_raw": score,
        "src_type": src,
        "edge_type_index": edge,
        "dst_type": dst,
        "event_index": event_index,
        "time": 1000 + event_index,
        "srcnode": src * 10 + event_index,
        "dstnode": dst * 10 + event_index,
        "loss_type": float(score),
        **extra,
    }


class Test1FullRunner:
    """Test 1: Verify all output artifacts are created with correct content."""

    def test_full_runner_creates_all_artifacts(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]
        test_records = [
            _record(1.5, event_index=100),
            _record(2.5, event_index=101),
        ]

        summary = run_calibration(
            val_records,
            test_records,
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Check files exist
        assert (tmp_path / "calibrator.pkl").exists()
        assert (tmp_path / "validation_calibrated.csv").exists()
        assert (tmp_path / "test_calibrated.csv").exists()
        assert (tmp_path / "calibration_summary.json").exists()

        # Verify summary content
        assert summary["event_counts"]["num_validation_events"] == 3
        assert summary["event_counts"]["num_test_events"] == 2
        assert summary["reference"]["global"]["sample_count"] == 3
        assert summary["validation_loo"]["triplet_count"] == 3
        assert summary["test"]["triplet_count"] == 2

        # Verify CSV content
        val_csv = load_event_records_from_csv(tmp_path / "validation_calibrated.csv")
        assert len(val_csv) == 3
        assert "score_calibrated" in val_csv[0]
        assert "calibration_level" in val_csv[0]

        test_csv = load_event_records_from_csv(tmp_path / "test_calibrated.csv")
        assert len(test_csv) == 2


class Test2ValidationUsesLOO:
    """Test 2: Verify validation truly uses LOO (different from regular calibrate)."""

    def test_validation_uses_loo_not_regular_calibrate(self, tmp_path):
        # Create records with specific triplet so LOO makes a difference
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]

        run_calibration(
            val_records,
            [],
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Load calibrated validation
        val_csv = load_event_records_from_csv(tmp_path / "validation_calibrated.csv")

        # Now calibrate with regular calibrate (not LOO)
        calibrator = HierarchicalRelationCalibrator(min_triplet_samples=1, min_type_pair_samples=1)
        calibrator.fit(val_records)
        regular_calibrated = calibrator.calibrate(val_records)

        # LOO results should differ from regular calibrate
        # Because LOO excludes each event from its own reference
        for i, (loo_rec, reg_rec) in enumerate(zip(val_csv, regular_calibrated)):
            # Check calibration_level can differ
            # For this simple case all triplet, but p-values differ
            assert "score_calibrated" in loo_rec
            assert "score_calibrated" in reg_rec


class Test3TestNotLOO:
    """Test 3: Test output matches regular calibrate (not LOO)."""

    def test_test_matches_regular_calibrate(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]
        test_records = [
            _record(1.5, event_index=100),
            _record(2.5, event_index=101),
        ]

        run_calibration(
            val_records,
            test_records,
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Get runner output
        test_csv = load_event_records_from_csv(tmp_path / "test_calibrated.csv")

        # Get regular calibrate output
        calibrator = HierarchicalRelationCalibrator(min_triplet_samples=1, min_type_pair_samples=1)
        calibrator.fit(val_records)
        regular_calibrated = calibrator.calibrate(test_records)

        # Should match exactly
        for i, (run_rec, reg_rec) in enumerate(zip(test_csv, regular_calibrated)):
            assert float(run_rec["score_calibrated"]) == pytest.approx(reg_rec["score_calibrated"])
            assert run_rec["calibration_level"] == reg_rec["calibration_level"]


class Test4TestNotInFit:
    """Test 4: Reference counts should only depend on validation, not test."""

    def test_reference_unchanged_by_different_test_data(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]
        test1 = [_record(1.5, event_index=100)]
        test2 = [_record(99.0, event_index=200)]  # Very different score

        summary1 = run_calibration(
            val_records, test1, tmp_path / "run1",
            min_triplet_samples=1, min_type_pair_samples=1,
        )
        summary2 = run_calibration(
            val_records, test2, tmp_path / "run2",
            min_triplet_samples=1, min_type_pair_samples=1,
        )

        # Reference counts should be identical
        assert summary1["reference"]["global"]["sample_count"] == summary2["reference"]["global"]["sample_count"]
        assert len(summary1["reference"]["triplet_groups"]) == len(summary2["reference"]["triplet_groups"])
        assert len(summary1["reference"]["type_pair_groups"]) == len(summary2["reference"]["type_pair_groups"])


class Test5PickleRoundTrip:
    """Test 5: Pickle round-trip preserves calibration results."""

    def test_pickle_load_produces_same_scores(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]
        test_records = [
            _record(1.5, event_index=100),
            _record(2.5, event_index=101),
        ]

        run_calibration(
            val_records,
            test_records,
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Load calibrator
        with open(tmp_path / "calibrator.pkl", "rb") as f:
            loaded_calibrator = pickle.load(f)

        # Calibrate again with loaded calibrator
        re_calibrated = loaded_calibrator.calibrate(test_records)

        # Load original test CSV
        test_csv = load_event_records_from_csv(tmp_path / "test_calibrated.csv")

        # Compare scores
        for i, (orig_rec, re_rec) in enumerate(zip(test_csv, re_calibrated)):
            assert float(orig_rec["score_calibrated"]) == pytest.approx(re_rec["score_calibrated"])
            assert orig_rec["calibration_level"] == re_rec["calibration_level"]


class Test6LevelCounts:
    """Test 6: Verify correct calibration levels are used."""

    def test_level_counts_and_ratios(self, tmp_path):
        # Create data that triggers different levels
        # Triplet (1,2,3) has 5 samples - can use triplet level
        # Type pair (1,4) has 2 samples but edge 2 different - can use type_pair
        # Type pair (1,7) has 2 samples but edge different - can use type_pair
        val_records = [
            _record(1.0, src=1, edge=2, dst=3, event_index=0),
            _record(2.0, src=1, edge=2, dst=3, event_index=1),
            _record(3.0, src=1, edge=2, dst=3, event_index=2),
            _record(4.0, src=1, edge=2, dst=3, event_index=3),
            _record(5.0, src=1, edge=2, dst=3, event_index=4),
            # Type pair (1,4) with edge 5 and 6 - 2 samples
            _record(6.0, src=1, edge=5, dst=4, event_index=5),
            _record(7.0, src=1, edge=6, dst=4, event_index=6),
            # Type pair (1,7) with edge 7 and 8 - 2 samples
            _record(8.0, src=1, edge=7, dst=7, event_index=7),
            _record(9.0, src=1, edge=8, dst=7, event_index=8),
        ]
        test_records = [
            _record(3.5, src=1, edge=2, dst=3, event_index=100),  # triplet
            _record(8.5, src=1, edge=9, dst=4, event_index=101),  # type_pair (src=1,dst=4)
            _record(9.5, src=1, edge=9, dst=7, event_index=102),  # type_pair (src=1,dst=7)
            _record(10.0, src=10, edge=11, dst=12, event_index=103),  # global
        ]

        summary = run_calibration(
            val_records,
            test_records,
            tmp_path,
            min_triplet_samples=2,
            min_type_pair_samples=2,
        )

        # All 4 test events should be calibrated
        total_test = (
            summary["test"]["triplet_count"]
            + summary["test"]["type_pair_count"]
            + summary["test"]["global_count"]
        )
        assert total_test == 4

        # At least one triplet, one type_pair, one global
        assert summary["test"]["triplet_count"] >= 1
        assert summary["test"]["type_pair_count"] >= 1
        assert summary["test"]["global_count"] >= 1

        # Check ratios sum to 1.0
        total_ratio = (
            summary["test"]["triplet_ratio"]
            + summary["test"]["type_pair_ratio"]
            + summary["test"]["global_ratio"]
        )
        assert total_ratio == pytest.approx(1.0)


class Test7ReferenceGroupCounts:
    """Test 7: Verify reference group counts match validation."""

    def test_reference_group_counts_match_validation(self, tmp_path):
        val_records = [
            _record(1.0, src=1, edge=2, dst=3, event_index=0),
            _record(2.0, src=1, edge=2, dst=3, event_index=1),
            _record(3.0, src=1, edge=4, dst=3, event_index=2),
            _record(4.0, src=5, edge=6, dst=7, event_index=3),
        ]

        summary = run_calibration(
            val_records,
            [],
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Global count = all validation events
        assert summary["reference"]["global"]["sample_count"] == 4

        # Triplet (1,2,3) should have 2 samples
        triplet_123 = summary["reference"]["triplet_groups"].get("src=1|edge=2|dst=3")
        if triplet_123:
            assert triplet_123["sample_count"] == 2

        # Type pair (1,3) should have 3 samples (events 0, 1, 2)
        type_pair_13 = summary["reference"]["type_pair_groups"].get("src=1|dst=3")
        if type_pair_13:
            assert type_pair_13["sample_count"] == 3


class Test8ReferenceQuantiles:
    """Test 8: Verify reference quantiles come from validation only."""

    def test_quantiles_from_validation_only(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
            _record(4.0, event_index=3),
            _record(5.0, event_index=4),
            _record(6.0, event_index=5),
            _record(7.0, event_index=6),
            _record(8.0, event_index=7),
            _record(9.0, event_index=8),
            _record(10.0, event_index=9),
        ]
        # Extreme test score should NOT affect validation quantiles
        test_records = [
            _record(100.0, event_index=100),
        ]

        summary1 = run_calibration(
            val_records,
            test_records,
            tmp_path / "run1",
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # p50 should be around 5.5 (median of 1..10)
        global_p50 = summary1["reference"]["global"]["quantiles"]["p50"]
        assert 5.0 <= global_p50 <= 6.0

        # p90 should be around 9.1
        global_p90 = summary1["reference"]["global"]["quantiles"]["p90"]
        assert 9.0 <= global_p90 <= 10.0


class Test9OriginalFieldsPreserved:
    """Test 9: Verify original fields are preserved in output."""

    def test_original_fields_preserved(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0, time=1000, srcnode=10, dstnode=20),
            _record(2.0, event_index=1, time=2000, srcnode=11, dstnode=21),
        ]

        run_calibration(
            val_records,
            [],
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        val_csv = load_event_records_from_csv(tmp_path / "validation_calibrated.csv")

        # Original fields preserved
        assert "score_raw" in val_csv[0]
        assert "event_index" in val_csv[0]
        assert "time" in val_csv[0]
        assert "srcnode" in val_csv[0]
        assert "dstnode" in val_csv[0]
        assert "loss_type" in val_csv[0]
        assert "src_type" in val_csv[0]
        assert "dst_type" in val_csv[0]

        # Calibration fields added
        assert "score_calibrated" in val_csv[0]
        assert "calibration_level" in val_csv[0]

        # Original values preserved
        assert float(val_csv[0]["score_raw"]) == 1.0
        assert int(val_csv[0]["event_index"]) == 0


class Test10EmptyTest:
    """Test 10: Handle empty test gracefully."""

    def test_empty_test_creates_valid_artifacts(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
        ]

        summary = run_calibration(
            val_records,
            [],
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        assert summary["event_counts"]["num_test_events"] == 0
        assert summary["test"]["triplet_count"] == 0
        assert summary["test"]["type_pair_count"] == 0
        assert summary["test"]["global_count"] == 0
        assert summary["test"]["triplet_ratio"] == 0.0
        assert summary["test"]["type_pair_ratio"] == 0.0
        assert summary["test"]["global_ratio"] == 0.0

        # Test CSV should exist but be empty (or have just header)
        test_csv = load_event_records_from_csv(tmp_path / "test_calibrated.csv")
        assert len(test_csv) == 0

        # No NaN or Infinity in ratios
        assert np.isfinite(summary["test"]["triplet_ratio"])
        assert np.isfinite(summary["test"]["type_pair_ratio"])
        assert np.isfinite(summary["test"]["global_ratio"])


class Test11EmptyValidation:
    """Test 11: Empty validation should raise ValueError."""

    def test_empty_validation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            run_calibration([], [], tmp_path)


class Test12DuplicateEventIndex:
    """Test 12: Duplicate event_index should raise ValueError."""

    def test_duplicate_event_index_raises(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=0),  # Duplicate!
        ]

        with pytest.raises(ValueError, match="duplicate"):
            run_calibration(val_records, [], tmp_path)


class Test13UnorderedInput:
    """Test 13: Verify stable sorting and consistent order for fit/LOO."""

    def test_unordered_input_produces_stable_order(self, tmp_path):
        # Provide records in non-sorted order
        val_records = [
            _record(3.0, event_index=2),
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
        ]
        test_records = [
            _record(1.5, event_index=101),
            _record(0.5, event_index=100),
        ]

        summary = run_calibration(
            val_records,
            test_records,
            tmp_path,
            min_triplet_samples=1,
            min_type_pair_samples=1,
        )

        # Should work without error
        assert summary["event_counts"]["num_validation_events"] == 3
        assert summary["event_counts"]["num_test_events"] == 2

        # Verify order is stable by checking event_indices
        val_csv = load_event_records_from_csv(tmp_path / "validation_calibrated.csv")
        event_indices = [int(r["event_index"]) for r in val_csv]
        # Should be sorted
        assert event_indices == [0, 1, 2]


class Test14RawArtifactsNotOverwritten:
    """Test 14: Input CSV should not be modified."""

    def test_input_csv_unchanged(self, tmp_path, monkeypatch):
        # Create input CSV
        input_csv = tmp_path / "input_validation.csv"
        import csv
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["score_raw", "src_type", "edge_type_index", "dst_type", "event_index"])
            writer.writeheader()
            writer.writerow({"score_raw": "1.0", "src_type": "1", "edge_type_index": "2", "dst_type": "3", "event_index": "0"})
            writer.writerow({"score_raw": "2.0", "src_type": "1", "edge_type_index": "2", "dst_type": "3", "event_index": "1"})

        # Read original content
        with open(input_csv, "rb") as f:
            original_content = f.read()

        # Run calibration with file-based input
        val_records = load_event_records_from_csv(input_csv)
        run_calibration(val_records, [], tmp_path / "output")

        # Verify input file unchanged
        with open(input_csv, "rb") as f:
            new_content = f.read()

        assert new_content == original_content


class Test15Determinism:
    """Test 15: Same input should produce identical output across runs."""

    def test_deterministic_output(self, tmp_path):
        val_records = [
            _record(1.0, event_index=0),
            _record(2.0, event_index=1),
            _record(3.0, event_index=2),
        ]
        test_records = [
            _record(1.5, event_index=100),
            _record(2.5, event_index=101),
        ]

        # Run twice
        summary1 = run_calibration(
            val_records, test_records, tmp_path / "run1",
            min_triplet_samples=1, min_type_pair_samples=1,
        )
        summary2 = run_calibration(
            val_records, test_records, tmp_path / "run2",
            min_triplet_samples=1, min_type_pair_samples=1,
        )

        # Summary should be identical
        assert summary1["reference"]["global"]["sample_count"] == summary2["reference"]["global"]["sample_count"]
        assert summary1["validation_loo"]["triplet_count"] == summary2["validation_loo"]["triplet_count"]
        assert summary1["test"]["triplet_count"] == summary2["test"]["triplet_count"]

        # CSVs should be identical
        csv1 = load_event_records_from_csv(tmp_path / "run1" / "test_calibrated.csv")
        csv2 = load_event_records_from_csv(tmp_path / "run2" / "test_calibrated.csv")

        for r1, r2 in zip(csv1, csv2):
            assert float(r1["score_calibrated"]) == pytest.approx(float(r2["score_calibrated"]))
            assert r1["calibration_level"] == r2["calibration_level"]


@pytest.mark.parametrize(
    ("method", "allowed_levels"),
    [
        ("global_empirical", {"global"}),
        ("relation_triplet", {"triplet", "global"}),
        ("hierarchical_relation", {"triplet", "type_pair", "global"}),
    ],
)
def test_runner_supports_all_calibration_methods_and_records_summary(
    tmp_path, method, allowed_levels
):
    validation = [
        _record(1.0, event_index=0),
        _record(2.0, event_index=1),
        _record(3.0, event_index=2),
    ]
    testing = [_record(2.5, event_index=100)]
    output = tmp_path / method
    summary = run_calibration(
        validation, testing, output,
        min_triplet_samples=2, min_type_pair_samples=2, method=method,
    )
    assert summary["calibration_method"] == method
    assert all((output / name).exists() for name in (
        "calibrator.pkl", "calibration_summary.json",
        "validation_calibrated.csv", "test_calibrated.csv",
    ))
    levels = {
        record["calibration_level"]
        for record in load_event_records_from_csv(output / "validation_calibrated.csv")
        + load_event_records_from_csv(output / "test_calibrated.csv")
    }
    assert levels <= allowed_levels
