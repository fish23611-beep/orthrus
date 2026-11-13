"""CPU contracts for the isolated E3 analysis; no production artifacts needed."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("e3_analysis", ROOT / "scripts/analyze_e3_postprocessing.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
from mstc.aggregation import NodeScoreAggregator
from mstc.calibration import HierarchicalRelationCalibrator


def records():
    rng = np.random.default_rng(17)
    result = []
    for src_type, edge, dst_type, count in [(0, 0, 1, 8), (0, 1, 1, 3), (2, 2, 0, 1)]:
        for _ in range(count):
            result.append(dict(src_type=src_type, edge_type_index=edge, dst_type=dst_type,
                               score_raw=float(rng.integers(0, 4))))
    return result


def test_vector_calibration_matches_production_loo_ties_and_fallback():
    val = records()
    fitted = HierarchicalRelationCalibrator(min_triplet_samples=3, min_type_pair_samples=4).fit(val)
    frame = pd.DataFrame(val)
    refs = analysis.References(frame, frame.score_raw.to_numpy(), min_triplet=3, min_pair=4)
    actual, levels = refs.query(frame, frame.score_raw.to_numpy(), loo=True)
    expected = fitted.transform_val_with_loo(val)
    np.testing.assert_allclose(actual, [row["score_calibrated"] for row in expected], rtol=0, atol=0)
    assert list(levels) == [row["calibration_level"] for row in expected]
    queries = val + [dict(src_type=9, edge_type_index=9, dst_type=9, score_raw=100.0)]
    test = pd.DataFrame(queries)
    actual, levels = refs.query(test, test.score_raw.to_numpy())
    expected = fitted.calibrate(queries)
    np.testing.assert_allclose(actual, [row["score_calibrated"] for row in expected], rtol=0, atol=0)
    assert list(levels) == [row["calibration_level"] for row in expected]


@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_bounded_aggregation_matches_production_all_k_and_directions(chunk_size):
    rng = np.random.default_rng(42)
    src, dst = rng.integers(0, 13, size=(2, 100))
    scores = rng.integers(0, 20, size=100).astype(float) / 3
    events = [dict(srcnode=int(s), dstnode=int(d), score_calibrated=float(v)) for s, d, v in zip(src, dst, scores)]
    left, right = analysis.TopK(np.arange(13)), analysis.TopK(np.arange(13))
    for start in range(0, 100, chunk_size):
        stop = start + chunk_size
        left.update(src[start:stop], scores[start:stop])
        right.update(dst[start:stop], scores[start:stop])
    for k in analysis.KS:
        expected = NodeScoreAggregator().aggregate_events(events, topk=k)
        np.testing.assert_allclose(left.scores(k, right), [expected[i] for i in range(13)], rtol=0, atol=1e-14)
        expected_src = NodeScoreAggregator().aggregate_events(events, topk=k, include_dst=False)
        np.testing.assert_allclose(left.scores(k), [expected_src[i] for i in range(13)], rtol=0, atol=1e-14)
    assert left.values.shape == (13, 10)
    assert left.counts.sum() == right.counts.sum() == 100


def test_self_loop_multiplicity_and_absent_direction():
    left, right = analysis.TopK([1, 2]), analysis.TopK([1, 2])
    left.update(np.array([1, 1]), np.array([9., 1.]))
    right.update(np.array([1, 2]), np.array([9., 1.]))
    assert left.scores(3, right)[0] == pytest.approx(19 / 3)
    assert left.scores(1)[1] == -np.inf
    with pytest.raises(ValueError, match="outside"):
        left.update(np.array([99]), np.array([1.]))
    with pytest.raises(ValueError, match="Non-finite"):
        left.update(np.array([1]), np.array([np.nan]))


def test_thresholds_separate_per_k_without_test_inputs():
    left, right = analysis.TopK([1, 2]), analysis.TopK([1, 2])
    left.update(np.array([1, 1, 2]), np.array([10., 0., 2.]))
    first = analysis.compute_node_threshold(left.scores(1, right), quantile=.5)
    second = analysis.compute_node_threshold(left.scores(5, right), quantile=.5)
    assert first == 6 and second == 3.5
    unrelated_test = analysis.TopK([1, 2])
    unrelated_test.update(np.array([1]), np.array([1000.]))
    assert analysis.compute_node_threshold(left.scores(1, right), quantile=.5) == first


def test_stream_hash_and_missing_value_rejection(tmp_path):
    import hashlib
    frame = pd.DataFrame([{c: ("triplet" if c == "calibration_level" else 1) for c in analysis.COLUMNS}] * 3)
    path = tmp_path / "events.csv"
    frame.to_csv(path, index=False)
    evidence = {}
    assert sum(len(f) for f in analysis.read_chunks(path, 2, evidence)) == 3
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    frame.loc[0, "score_raw"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing"):
        list(analysis.read_chunks(path, 2, {}))


def test_output_exclusive_and_json_no_nan(tmp_path):
    path = tmp_path / "summary.json"
    analysis.write_json(path, {"undefined": np.nan, "value": np.int64(1)})
    assert json.loads(path.read_text()) == {"undefined": None, "value": 1}
    with pytest.raises(FileExistsError):
        analysis.write_json(path, {})
