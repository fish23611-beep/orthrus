"""Golden compatibility coverage for the unmodified 2cc2ee7 baseline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = Path(__file__).resolve().parent / "baseline_reference"
sys.path.insert(0, str(REFERENCE_DIR))

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from generate_golden import GENERATOR_VERSION, INPUT_SUMMARY, compute_outputs


BASELINE_COMMIT = "2cc2ee714f7a9c288e25209c171c37810942eb2f"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "baseline_2cc2ee7.json"
ATOL = 1e-6
RTOL = 1e-5


@pytest.fixture(scope="module")
def golden():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current_outputs():
    return compute_outputs(REPO_ROOT / "src")


def _assert_close(actual, expected, label):
    np.testing.assert_allclose(
        actual,
        expected,
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"baseline compatibility mismatch for {label}",
    )


def test_fixture_records_audited_baseline_and_inputs(golden):
    assert golden["baseline_commit"] == BASELINE_COMMIT
    assert golden["generator_version"] == GENERATOR_VERSION
    assert golden["input_summary"] == INPUT_SUMMARY
    assert golden["environment"]["python"]
    assert golden["environment"]["torch"]


def test_edge_loss_and_inference_scores_match_golden(golden, current_outputs):
    expected = golden["outputs"]
    _assert_close(
        current_outputs["edge_level_loss"],
        expected["edge_level_loss"],
        "edge-level loss",
    )
    for key in (
        "replay_train_scores",
        "validation_scores",
        "inference_scores",
    ):
        _assert_close(current_outputs[key], expected[key], key)


def test_node_aggregation_and_predictions_match_golden(golden, current_outputs):
    expected = golden["outputs"]
    assert current_outputs["node_scores"].keys() == expected["node_scores"].keys()
    _assert_close(
        list(current_outputs["node_scores"].values()),
        list(expected["node_scores"].values()),
        "node scores",
    )
    _assert_close(
        current_outputs["prediction_threshold"],
        expected["prediction_threshold"],
        "prediction threshold",
    )
    assert current_outputs["edge_predictions"] == expected["edge_predictions"]
    assert current_outputs["node_predictions"] == expected["node_predictions"]


def test_output_shapes_and_stability_summaries_match_golden(golden, current_outputs):
    expected = golden["outputs"]
    assert current_outputs["shapes"] == expected["shapes"]
    assert current_outputs["edge_level_loss"] > 0.0
    assert all(np.isfinite(current_outputs["inference_scores"]))
    for key, value in expected["stability"].items():
        _assert_close(current_outputs["stability"][key], value, key)
