"""Test only smoke configuration mapping; no claim of real-data inference."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_e5_gpu_smoke import configure


def example():
    return {"resolved_config": {
        "_run_dir": "/server/run", "_metadata_dir": "/server/metadata",
        "pipeline": {"mode": "detection_only"}, "model": {"variant": "mstc"},
        "dataset": {"name": "THEIA_E5", "raw_dir": "/data/"},
        "edge_featurization": {"embed_edges": {"_edge_embeds_dir": "/server/edges"}},
        "graph_construction": {"build_graphs": {"_graphs_dir": "/server/graphs"}},
        "detection": {"gnn_training": {"encoder": {"batch_size": 512, "neighbor_budgets": [8, 8, 8]},
                                       "decoder": {"lambda_time": 0.3}},
                      "gnn_testing": {"_edge_losses_dir": "/server/scores"},
                      "evaluation": {"node_evaluation": {}, "deterministic_inference": True}},
        "calibration": {"method": "hierarchical_relation"},
        "node_aggregation": {"method": "topk_mean", "topk": 5},
        "node_threshold": {"quantile": 0.999},
    }}


def test_mapping_preserves_frozen_numerical_configuration(tmp_path):
    manifest = example()
    before = copy.deepcopy(manifest)
    cfg, changes = configure(manifest, tmp_path / "data", tmp_path / "run")
    assert manifest == before
    for key in ["calibration", "node_aggregation", "node_threshold", "model"]:
        assert cfg[key] == before["resolved_config"][key]
    assert cfg["detection"]["gnn_training"]["encoder"] == before["resolved_config"]["detection"]["gnn_training"]["encoder"]
    assert cfg["detection"]["gnn_training"]["decoder"] == before["resolved_config"]["detection"]["gnn_training"]["decoder"]
    assert cfg["detection"]["evaluation"]["deterministic_inference"] is True
    assert cfg["_inference_checkpoint"].endswith("model_epoch_6")
    assert changes


def test_mapping_receipt_reports_effective_paths(tmp_path):
    cfg, changes = configure(example(), tmp_path / "data", tmp_path / "run")
    row = next(r for r in changes if r["field"] == "edge_featurization.embed_edges._edge_embeds_dir")
    assert row["windows"] == cfg["edge_featurization"]["embed_edges"]["_edge_embeds_dir"]
    assert not Path(row["windows"]).is_relative_to(tmp_path / "run/unused_paths")


@pytest.mark.parametrize("section,key,value", [("pipeline", "mode", "full_pipeline"), ("model", "variant", "orthrus_baseline")])
def test_other_pipeline_modes_are_refused(tmp_path, section, key, value):
    manifest = example()
    manifest["resolved_config"][section][key] = value
    with pytest.raises(ValueError, match="Only frozen"):
        configure(manifest, tmp_path / "data", tmp_path / "run")
