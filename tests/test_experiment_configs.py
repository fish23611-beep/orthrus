"""C8-E semantic contracts for the authoritative experiment matrix."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import config as config_module

EXPERIMENTS = Path(__file__).resolve().parents[1] / "config" / "experiments"
REQUIRED = {
    "baseline.yml", "mstc_full.yml",
    "ablation_no_multiscale.yml", "ablation_no_gate.yml", "ablation_no_time.yml",
    "ablation_no_calibration.yml", "ablation_no_topk.yml",
    "multiscale_recent20.yml", "multiscale_recent24.yml", "multiscale_single_window.yml",
    "multiscale_equal.yml", "multiscale_gate.yml",
    "time_type_only.yml", "time_time_only.yml", "time_joint.yml",
    "calibration_max.yml", "calibration_quantile.yml", "calibration_kmeans.yml",
    "calibration_global_p.yml", "calibration_relation.yml", "calibration_hierarchical.yml",
    "backbone_graphtransformer.yml", "backbone_graphsage.yml", "backbone_mlp.yml",
    "efficiency_multiscale.yml", "efficiency_multiscale_time.yml",
}


def _args(path: Path, *, dataset: str = "THEIA_E3", seed: int = 17, artifact_root=None):
    return SimpleNamespace(
        dataset=dataset, model="orthrus", config=str(path), cpu=True,
        from_weights=False, seed=seed, skip_tracing=False, artifact_root=artifact_root,
        resume_checkpoint=None, inference_checkpoint=None,
    )


@pytest.fixture(autouse=True)
def _avoid_task_path_side_effects(monkeypatch):
    monkeypatch.setattr(config_module, "set_task_paths", lambda cfg: None)


def resolve_experiment_config(filename: str, *, dataset: str = "THEIA_E3", seed: int = 17, artifact_root=None):
    """Resolve through the production loader (base + experiment + CLI)."""
    path = EXPERIMENTS / filename
    with contextlib.redirect_stdout(io.StringIO()):
        return config_module.get_yml_cfg(_args(path, dataset=dataset, seed=seed, artifact_root=artifact_root))


def _contract(cfg):
    enc = cfg.detection.gnn_training.encoder
    dec = cfg.detection.gnn_training.decoder
    ms = enc.context.multiscale
    values = {
        "variant": cfg.model.variant,
        "backbone": enc.backbone,
        "neighbor_size": enc.neighbor_size,
        "context.mode": enc.context.mode,
        "multiscale.enabled": ms.enabled,
        "type.enabled": dec.predict_edge_type.enabled,
        "time.enabled": dec.time_gap.enabled,
        "time.lambda": dec.time_gap.lambda_time,
        "calibration": cfg.calibration.method,
        "aggregation": cfg.node_aggregation.method,
        "aggregation.topk": cfg.node_aggregation.topk,
        "aggregation.score": cfg.node_aggregation.score_field,
        "threshold": cfg.node_threshold.method,
        "threshold.quantile": cfg.node_threshold.quantile,
        "dataset_view": cfg.dataset_view.mode,
        "epochs": cfg.detection.gnn_training.num_epochs,
        "learning_rate": cfg.detection.gnn_training.lr,
        "hidden_dim": cfg.detection.gnn_training.node_hid_dim,
        "out_dim": cfg.detection.gnn_training.node_out_dim,
    }
    if enc.context.mode == "multiscale":
        values.update({
            "multiscale.budgets": tuple(ms.neighbor_budgets),
            "multiscale.quantiles": tuple(ms.scale_quantiles),
            "multiscale.candidate_capacity": ms.candidate_capacity,
            "multiscale.share_encoder": ms.share_encoder,
            "multiscale.fusion": ms.fusion,
        })
    return values


def _diff(left, right):
    # Inactive multiscale-only fields are deliberately omitted from the
    # normalized contract when a configuration uses Recent context.
    return {key for key in left.keys() & right.keys() if left[key] != right[key]}


def test_authoritative_files_exist_and_deprecated_gated_alias_is_absent():
    actual = {path.name for path in EXPERIMENTS.glob("*.yml")}
    assert REQUIRED <= actual
    assert "multiscale_gated.yml" not in actual
    assert "time_only.yml" not in actual


def test_every_experiment_yaml_resolves_with_the_production_loader():
    for path in sorted(EXPERIMENTS.glob("*.yml")):
        cfg = resolve_experiment_config(path.name)
        assert cfg.logging.wandb_mode == "disabled"


def test_every_formal_experiment_declares_the_frozen_semantics_version():
    baseline_v1 = {
        "baseline.yml",
        "backbone_graphsage_baseline.yml",
        "backbone_mlp.yml",
    }
    for path in sorted(EXPERIMENTS.glob("*.yml")):
        cfg = resolve_experiment_config(path.name)
        expected = "baseline_v1" if path.name in baseline_v1 else "temporal_v2"
        assert cfg.experiment_identity.semantics_version == expected, path.name


def test_baseline_is_isolated_from_mstc_paths():
    cfg = resolve_experiment_config("baseline.yml")
    enc, dec = cfg.detection.gnn_training.encoder, cfg.detection.gnn_training.decoder
    assert cfg.model.variant == "orthrus_baseline"
    assert enc.context.mode == "recent" and not enc.context.multiscale.enabled
    assert dec.predict_edge_type.enabled and not dec.time_gap.enabled
    # The evaluation dispatcher selects the legacy path solely for this variant;
    # C6 settings therefore cannot make the baseline enter an MSTC code path.
    evaluation_source = (SRC_ROOT / "detection" / "evaluation.py").read_text(encoding="utf-8")
    assert 'cfg.model.variant == \"orthrus_baseline\"' in evaluation_source


def test_full_semantic_contract():
    cfg = resolve_experiment_config("mstc_full.yml")
    enc, dec = cfg.detection.gnn_training.encoder, cfg.detection.gnn_training.decoder
    ms = enc.context.multiscale
    assert cfg.model.variant == "mstc"
    assert enc.backbone == "graph_transformer"
    assert (enc.context.mode, ms.enabled, tuple(ms.neighbor_budgets), ms.share_encoder, ms.fusion) == (
        "multiscale", True, (8, 8, 8), True, "gated"
    )
    assert (dec.predict_edge_type.enabled, dec.time_gap.enabled, dec.time_gap.lambda_time) == (True, True, 0.3)
    assert (cfg.calibration.method, cfg.node_aggregation.method, cfg.node_aggregation.topk,
            cfg.node_threshold.method, cfg.dataset_view.mode) == (
        "hierarchical_relation", "topk_mean", 5, "validation_quantile", "host_network_full"
    )


@pytest.mark.parametrize("filename, allowed", [
    ("ablation_no_multiscale.yml", {"context.mode", "multiscale.enabled"}),
    ("ablation_no_gate.yml", {"multiscale.fusion"}),
    ("ablation_no_time.yml", {"time.lambda"}),
    ("ablation_no_calibration.yml", {"aggregation.score"}),
    ("ablation_no_topk.yml", {"aggregation"}),
])
def test_ablation_resolved_diff_is_limited_to_its_claimed_factor(filename, allowed):
    full = _contract(resolve_experiment_config("mstc_full.yml"))
    ablated = _contract(resolve_experiment_config(filename))
    assert _diff(full, ablated) == allowed


def test_multiscale_budget_controls_are_fair():
    recent20 = resolve_experiment_config("multiscale_recent20.yml")
    recent24 = resolve_experiment_config("multiscale_recent24.yml")
    single = resolve_experiment_config("multiscale_single_window.yml")
    equal = resolve_experiment_config("multiscale_equal.yml")
    gated = resolve_experiment_config("multiscale_gate.yml")

    recent20_enc = recent20.detection.gnn_training.encoder
    recent24_enc = recent24.detection.gnn_training.encoder
    single_enc = single.detection.gnn_training.encoder
    equal_ms = equal.detection.gnn_training.encoder.context.multiscale
    gated_ms = gated.detection.gnn_training.encoder.context.multiscale

    assert (recent20_enc.context.mode, recent20_enc.neighbor_size) == ("recent", 20)
    assert (recent24_enc.context.mode, recent24_enc.neighbor_size) == ("recent", 24)
    assert not recent20_enc.context.multiscale.enabled
    assert not recent24_enc.context.multiscale.enabled

    # Single-window has one Q99-bounded pool, not a Q50 short-scale budget.
    assert single_enc.context.mode == "single_window"
    assert tuple(single_enc.context.multiscale.neighbor_budgets) == (24,)

    assert tuple(equal_ms.neighbor_budgets) == (8, 8, 8)
    assert tuple(gated_ms.neighbor_budgets) == (8, 8, 8)
    assert sum(single_enc.context.multiscale.neighbor_budgets) == 24
    assert sum(equal_ms.neighbor_budgets) == 24
    assert sum(gated_ms.neighbor_budgets) == 24
    assert equal_ms.fusion == "equal"
    assert gated_ms.fusion == "gated"


def test_time_variants_are_real_loss_controls():
    type_only = resolve_experiment_config("time_type_only.yml")
    time_only = resolve_experiment_config("time_time_only.yml")
    joint = resolve_experiment_config("time_joint.yml")
    type_dec = type_only.detection.gnn_training.decoder
    time_dec = time_only.detection.gnn_training.decoder
    joint_dec = joint.detection.gnn_training.decoder
    assert (type_dec.predict_edge_type.enabled, type_dec.time_gap.enabled) == (True, False)
    assert (time_dec.predict_edge_type.enabled, time_dec.time_gap.enabled) == (False, True)
    assert (joint_dec.predict_edge_type.enabled, joint_dec.time_gap.enabled, joint_dec.time_gap.lambda_time) == (True, True, 0.3)


def test_calibration_variants_map_to_supported_methods_or_thresholds():
    expected = {
        "calibration_max.yml": ("hierarchical_relation", "score_raw", "max_validation"),
        "calibration_quantile.yml": ("hierarchical_relation", "score_raw", "validation_quantile"),
        "calibration_kmeans.yml": ("hierarchical_relation", "score_raw", "kmeans"),
        "calibration_global_p.yml": ("global_empirical", "score_calibrated", "validation_quantile"),
        "calibration_relation.yml": ("relation_triplet", "score_calibrated", "validation_quantile"),
        "calibration_hierarchical.yml": ("hierarchical_relation", "score_calibrated", "validation_quantile"),
    }
    for filename, expected_values in expected.items():
        cfg = resolve_experiment_config(filename)
        assert (cfg.calibration.method, cfg.node_aggregation.score_field, cfg.node_threshold.method) == expected_values


def test_backbone_variants_distinguish_graphsage_baseline_and_mstc():
    transformer = resolve_experiment_config("backbone_graphtransformer.yml")
    graphsage = resolve_experiment_config("backbone_graphsage.yml")
    graphsage_baseline = resolve_experiment_config("backbone_graphsage_baseline.yml")
    mlp = resolve_experiment_config("backbone_mlp.yml")
    assert transformer.detection.gnn_training.encoder.backbone == "graph_transformer"
    assert (graphsage.model.variant, graphsage.detection.gnn_training.encoder.backbone) == ("mstc", "graphsage")
    assert (graphsage_baseline.model.variant, graphsage_baseline.detection.gnn_training.encoder.backbone) == ("orthrus_baseline", "graphsage")
    assert (mlp.detection.gnn_training.encoder.backbone, mlp.detection.gnn_training.encoder.context.mode,
            mlp.detection.gnn_training.encoder.context.multiscale.enabled) == ("semantic_mlp", "none", False)


def test_cli_seed_and_artifact_root_remain_authoritative(tmp_path):
    cfg = resolve_experiment_config("mstc_full.yml", seed=23, artifact_root=str(tmp_path / "artifacts"))
    assert cfg._seed == 23
    assert cfg._artifact_root_raw == str(tmp_path / "artifacts")


def test_main_config_is_portable_between_theia_engagements_and_has_no_secrets_or_personal_paths():
    for dataset in ("THEIA_E3", "THEIA_E5"):
        assert resolve_experiment_config("mstc_full.yml", dataset=dataset).dataset.name == dataset
    forbidden = ("password", "api_key", "api-key", "token", "/home/", "d:\\\\")
    for path in EXPERIMENTS.glob("*.yml"):
        source = path.read_text(encoding="utf-8").lower()
        assert not any(value in source for value in forbidden), path.name
