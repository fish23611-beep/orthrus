"""Unit tests for hierarchical empirical event-score calibration."""

import math
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "src" / "mstc" / "calibration.py"
CALIBRATION_SPEC = importlib.util.spec_from_file_location("calibration_for_tests", CALIBRATION_PATH)
assert CALIBRATION_SPEC is not None and CALIBRATION_SPEC.loader is not None
CALIBRATION_MODULE = importlib.util.module_from_spec(CALIBRATION_SPEC)
sys.modules[CALIBRATION_SPEC.name] = CALIBRATION_MODULE
CALIBRATION_SPEC.loader.exec_module(CALIBRATION_MODULE)
HierarchicalRelationCalibrator = CALIBRATION_MODULE.HierarchicalRelationCalibrator


def _record(score, src=1, edge=2, dst=3, **extra):
    return {
        "score_raw": score,
        "src_type": src,
        "edge_type_index": edge,
        "dst_type": dst,
        **extra,
    }


def _fit(records, **kwargs):
    return HierarchicalRelationCalibrator(**kwargs).fit(records)


def test_p_monotonic():
    calibrator = _fit([_record(score) for score in [1.0, 2.0, 3.0]], min_triplet_samples=1)
    result = calibrator.calibrate([_record(score) for score in [1.0, 2.0, 4.0]])
    p_values = [row["empirical_p"] for row in result]
    scores = [row["score_calibrated"] for row in result]
    assert p_values == sorted(p_values, reverse=True)
    assert scores == sorted(scores)


def test_add_one_formula():
    calibrator = _fit([_record(score) for score in [1.0, 2.0, 3.0]], min_triplet_samples=1)
    result = calibrator.calibrate([_record(2.0)])[0]
    assert result["empirical_p"] == pytest.approx(0.75)
    assert result["score_calibrated"] == pytest.approx(-math.log(0.75))


def test_duplicate_scores_use_greater_equal_boundary():
    calibrator = _fit([_record(score) for score in [1.0, 2.0, 2.0, 3.0]], min_triplet_samples=1)
    result = calibrator.calibrate([_record(2.0)])[0]
    assert result["empirical_p"] == pytest.approx(0.8)


def test_triplet_level_when_threshold_is_met():
    records = [_record(1.0), _record(2.0)] + [_record(10.0, src=8, edge=9, dst=10)]
    result = _fit(records, min_triplet_samples=2, min_type_pair_samples=3).calibrate([_record(3.0)])[0]
    assert result["calibration_level"] == "triplet"


def test_triplet_falls_back_to_type_pair():
    records = [_record(1.0, edge=2), _record(2.0, edge=4)]
    result = _fit(records, min_triplet_samples=3, min_type_pair_samples=2).calibrate([_record(3.0, edge=5)])[0]
    assert result["calibration_level"] == "type_pair"


def test_falls_back_to_global():
    records = [_record(1.0), _record(2.0, src=4, edge=5, dst=6)]
    result = _fit(records, min_triplet_samples=2, min_type_pair_samples=2).calibrate([_record(3.0, src=8, edge=9, dst=10)])[0]
    assert result["calibration_level"] == "global"


def test_loo_p_excludes_current_event_once():
    records = [_record(score) for score in [1.0, 2.0, 3.0]]
    result = _fit(records, min_triplet_samples=1).transform_val_with_loo(records)[1]
    assert result["empirical_p"] == pytest.approx(2.0 / 3.0)


def test_loo_duplicate_score_excludes_one_event_not_all_ties():
    records = [_record(score) for score in [1.0, 2.0, 2.0, 3.0]]
    result = _fit(records, min_triplet_samples=1).transform_val_with_loo(records)[1]
    assert result["empirical_p"] == pytest.approx(0.75)


def test_loo_threshold_uses_count_without_self():
    records = [
        _record(1.0, edge=2),
        _record(2.0, edge=2),
        _record(3.0, edge=2),
        _record(4.0, edge=7),
        _record(5.0, edge=8),
    ]
    calibrator = _fit(records, min_triplet_samples=3, min_type_pair_samples=4)
    assert calibrator.calibrate([_record(5.0, edge=2)])[0]["calibration_level"] == "triplet"
    assert calibrator.transform_val_with_loo(records)[0]["calibration_level"] == "type_pair"


def test_extreme_score_has_positive_finite_p_and_score():
    result = _fit([_record(1.0), _record(2.0)], min_triplet_samples=1).calibrate([_record(99.0)])[0]
    assert result["empirical_p"] == pytest.approx(1.0 / 3.0)
    assert result["empirical_p"] > 0.0
    assert np.isfinite(result["score_calibrated"])


def test_unknown_groups_fall_back_to_global_without_keyerror():
    result = _fit([_record(1.0)], min_triplet_samples=1, min_type_pair_samples=1).calibrate([_record(2.0, src=8, edge=9, dst=10)])[0]
    assert result["calibration_level"] == "global"


def test_empty_fit_raises():
    with pytest.raises(ValueError, match="empty"):
        HierarchicalRelationCalibrator().fit([])


def test_unfitted_methods_raise():
    calibrator = HierarchicalRelationCalibrator()
    with pytest.raises(RuntimeError, match="fitted"):
        calibrator.calibrate([_record(1.0)])
    with pytest.raises(RuntimeError, match="fitted"):
        calibrator.transform_val_with_loo([_record(1.0)])


def test_validation_and_constructor_input_validation():
    with pytest.raises(ValueError, match="positive integer"):
        HierarchicalRelationCalibrator(min_triplet_samples=0)
    with pytest.raises(ValueError, match="epsilon"):
        HierarchicalRelationCalibrator(epsilon=float("nan"))
    with pytest.raises(KeyError, match="score_raw"):
        HierarchicalRelationCalibrator().fit([{"src_type": 1, "edge_type_index": 2, "dst_type": 3}])
    with pytest.raises(ValueError, match="finite"):
        HierarchicalRelationCalibrator().fit([_record(float("inf"))])


def test_edge_type_alias_is_accepted():
    records = [{"score_raw": 1.0, "src_type": 1, "edge_type": 2, "dst_type": 3}]
    result = _fit(records, min_triplet_samples=1).calibrate(records)[0]
    assert result["calibration_level"] == "triplet"
