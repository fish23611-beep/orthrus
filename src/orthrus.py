import argparse
import os
import random
import time as time_module
from datetime import datetime, timezone

import torch
import wandb  # noqa: F401 - kept for backward compatibility with existing tests
import numpy as np
from provnet_utils import remove_underscore_keys, log

from graph_construction import (
    build_orthrus_graphs,
)
from edge_featurization import (
    build_feature_word2vec,
    embed_edges_feature_word2vec,
)
from detection import (
    orthrus_gnn_training,
    orthrus_gnn_testing,
    evaluation,
)

from config import (
    get_yml_cfg,
    get_runtime_required_args,
)

from attack_reconstruction import (
    tracing,
)

from pipeline_stages import (
    STANDARD_STAGES,
    VALID_STAGES,
    PREPROCESS_SUBSTAGES,
    parse_stages as _parse_stages,
    check_conflict as _check_conflict,
)

from artifact_paths import resolve_artifact_paths
from run_metadata import dump_environment, dump_config, dump_runtime
from wandb_control import resolve_wandb_mode, init_wandb, wandb_log, wandb_finish


# ---------------------------------------------------------------------------
# Detection-only mode helpers (C2)
# ---------------------------------------------------------------------------

def _is_detection_only_mode(cfg) -> bool:
    """Check if pipeline mode is detection_only."""
    return (
        hasattr(cfg, "pipeline")
        and hasattr(cfg.pipeline, "mode")
        and cfg.pipeline.mode == "detection_only"
    )


def _check_detection_only_prerequisites(stages, cfg) -> list[str]:
    """
    Check that required artifacts exist for detection_only mode.

    Returns list of missing artifacts. Empty list means all OK.
    """
    missing = []

    # Check for preprocess artifacts (required even in detection_only)
    graph_path = cfg.graph_construction.build_graphs._graphs_dir
    if not os.path.isdir(graph_path):
        missing.append(f"Graphs directory: {graph_path}")
    elif not os.listdir(graph_path):
        missing.append(f"Graphs directory is empty: {graph_path}")

    w2v_path = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
    if not os.path.isdir(w2v_path):
        missing.append(f"Word2Vec models: {w2v_path}")

    edge_embeds = cfg.edge_featurization.embed_edges._edge_embeds_dir
    if not os.path.isdir(edge_embeds):
        missing.append(f"Edge embeddings: {edge_embeds}")

    # Train/checkpoint requirements
    if "train" in stages:
        pass  # Train will generate checkpoints
    elif "test" in stages or "evaluate" in stages:
        checkpoint_dir = cfg.detection.gnn_training._trained_models_dir
        if not os.path.isdir(checkpoint_dir):
            missing.append(f"Checkpoints: {checkpoint_dir}")
        elif not os.listdir(checkpoint_dir):
            missing.append(f"Checkpoints directory empty: {checkpoint_dir}")

    if "evaluate" in stages:
        edge_scores_dir = cfg.detection.gnn_testing._edge_losses_dir
        if not os.path.isdir(edge_scores_dir):
            missing.append(f"Edge scores: {edge_scores_dir}")
        elif "test" not in os.listdir(edge_scores_dir):
            missing.append(f"Test split in edge scores: {edge_scores_dir}")

    return missing


# ---------------------------------------------------------------------------
# Artifact prerequisite checks
# ---------------------------------------------------------------------------
def _check_artifact_prerequisites(stages, cfg):
    """
    Check that required artifacts exist when an upstream stage is NOT being run.

    Rules:
    - test selected  AND train NOT selected  鈫?check model checkpoint dir
    - evaluate selected AND test NOT selected 鈫?check test edge-loss output dir
    """
    train_run = "train" in stages
    test_run = "test" in stages

    if "test" in stages and not train_run:
        train_path = cfg.detection.gnn_training._trained_models_dir
        if not isinstance(train_path, (str, os.PathLike)):
            raise FileNotFoundError(
                f"Stage 'test': invalid checkpoint path (not a string/Path): {train_path!r}"
            )
        if not os.path.isdir(train_path):
            raise FileNotFoundError(
                f"Stage 'test' requires trained model weights, but no models found at "
                f"{train_path}. Run 'train' stage first."
            )
        if not os.listdir(train_path):
            raise FileNotFoundError(
                f"Stage 'test' requires trained model weights, but the directory "
                f"{train_path} is empty. Run 'train' stage first."
            )

    if "evaluate" in stages and not test_run:
        edge_losses_path = cfg.detection.gnn_testing._edge_losses_dir
        if not isinstance(edge_losses_path, (str, os.PathLike)):
            raise FileNotFoundError(
                f"Stage 'evaluate': invalid edge-loss path (not a string/Path): {edge_losses_path!r}"
            )
        if not os.path.isdir(edge_losses_path):
            raise FileNotFoundError(
                f"Stage 'evaluate' requires test edge scores from 'test' stage, "
                f"but no output found at {edge_losses_path}. Run 'test' stage first."
            )
        try:
            splits = os.listdir(edge_losses_path)
        except OSError:
            raise FileNotFoundError(
                f"Stage 'evaluate': cannot read edge-loss directory {edge_losses_path}."
            )
        if "test" not in splits:
            raise FileNotFoundError(
                f"Stage 'evaluate' requires test split edge scores, "
                f"but no 'test' split found in {edge_losses_path}. Run 'test' stage first."
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg, args, **kwargs):
    # ------------------------------------------------------------------
    # 0. C2: Detection-only mode validation
    # ------------------------------------------------------------------
    is_detection_only = _is_detection_only_mode(cfg)

    if is_detection_only:
        log("Running in detection_only mode")
        # Parse stages normally
        conflict_warning = _check_conflict(args.stages, args.run_from_training)
        if conflict_warning is not None:
            log(conflict_warning)

        stages = _parse_stages(args.stages, args.run_from_training)

        # Check prerequisites
        missing = _check_detection_only_prerequisites(stages, cfg)
        if missing:
            raise FileNotFoundError(
                f"detection_only mode: required artifacts missing:\n" +
                "\n".join(f"  - {m}" for m in missing) +
                "\n\nRun in full_pipeline mode first, or provide required artifacts."
            )

        # In detection_only mode, force exclude preprocess stages
        stages = [s for s in stages if s != "preprocess"]
        if "preprocess" in stages:
            stages.remove("preprocess")
        log(f"detection_only: running stages {stages} (preprocess skipped)")
    else:
        # ------------------------------------------------------------------
        # 1. Parse stages + resolve tracing
        # ------------------------------------------------------------------
        conflict_warning = _check_conflict(args.stages, args.run_from_training)
        if conflict_warning is not None:
            log(conflict_warning)
            # When both are set, --stages takes precedence; run_from_training is ignored.

        stages = _parse_stages(args.stages, args.run_from_training)

    # Tracing is gated by both the stage list and cfg.pipeline.run_tracing
    do_trace = "trace" in stages and cfg.pipeline.run_tracing

    # ------------------------------------------------------------------
    # 2. Seed
    # ------------------------------------------------------------------
    if cfg.detection.gnn_training.use_seed:
        seed = cfg._seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time_module.time()

    # ------------------------------------------------------------------
    # 3. Preprocess (skip in detection_only mode)
    # ------------------------------------------------------------------
    t_build_graphs = None
    t_embed_nodes = None
    t_embed_edges = None

    if "preprocess" in stages:
        build_orthrus_graphs.main(cfg)
        t_build_graphs = time_module.time()

        build_feature_word2vec.main(cfg)
        t_embed_nodes = time_module.time()

        embed_edges_feature_word2vec.main(cfg)
        t_embed_edges = time_module.time()

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    t_gnn_training = None
    if "train" in stages:
        orthrus_gnn_training.main(cfg)
        torch.cuda.empty_cache()
        t_gnn_training = time_module.time()

    # ------------------------------------------------------------------
    # 5. Test
    # ------------------------------------------------------------------
    t_gnn_testing = None
    if "test" in stages:
        _check_artifact_prerequisites(stages, cfg)
        orthrus_gnn_testing.main(cfg)
        t_gnn_testing = time_module.time()

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    t_evaluation = None
    if "evaluate" in stages:
        _check_artifact_prerequisites(stages, cfg)
        evaluation.main(cfg)
        t_evaluation = time_module.time()

    # ------------------------------------------------------------------
    # 7. Trace
    # ------------------------------------------------------------------
    t_tracing = None
    if do_trace:
        tracing.main(cfg)
        t_tracing = time_module.time()

    # ------------------------------------------------------------------
    # 8. Timing summary
    # ------------------------------------------------------------------
    t_final = time_module.time()

    def _delta(start_t, end_t):
        return round(end_t - start_t, 2) if (start_t is not None and end_t is not None) else 0.0

    # Reference points for each phase boundary
    # train baseline: t0 if no preprocess, else t_embed_edges
    train_baseline = t_embed_edges if ("preprocess" in stages and t_embed_edges is not None) else t0
    # test baseline: t_gnn_training
    test_baseline = t_gnn_training if t_gnn_training is not None else t0
    # eval baseline: t_gnn_testing
    eval_baseline = t_gnn_testing if t_gnn_testing is not None else t0
    # trace baseline: t_evaluation if eval ran, else t_gnn_testing
    trace_baseline = t_evaluation if t_evaluation is not None else eval_baseline

    time_consumption = {
        "time_total": round(t_final - t0, 2),
        "time_build_graphs": _delta(t0, t_build_graphs),
        "time_embed_nodes": _delta(t_build_graphs, t_embed_nodes),
        "time_embed_edges": _delta(t_embed_nodes, t_embed_edges),
        "time_gnn_training": _delta(train_baseline, t_gnn_training),
        "time_gnn_testing": _delta(test_baseline, t_gnn_testing),
        "time_evaluation": _delta(eval_baseline, t_evaluation),
        "time_tracing": _delta(trace_baseline, t_tracing),
    }

    log("==" * 30)
    log("Run finished. Time consumed in each step:")
    for k, v in time_consumption.items():
        log(f"{k}: {v} s")

    log("==" * 30)
    if wandb.run is not None:
        wandb_log(time_consumption)

    return time_consumption


if __name__ == '__main__':
    import wandb

    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    # ------------------------------------------------------------------ #
    # 0. W&B mode resolution
    # ------------------------------------------------------------------ #
    # Pre-create cfg so we can read logging.wandb_mode before full init.
    cfg_pre = get_yml_cfg(args)
    wandb_mode = resolve_wandb_mode(cfg_pre, args)

    # ------------------------------------------------------------------ #
    # 1. Artifact root + run dir resolution
    # ------------------------------------------------------------------ #
    # Parse stages for artifact path resolution
    stages = _parse_stages(args.stages, args.run_from_training)

    # Resolve artifact paths.  create_dirs=True so stage sub-directories are created.
    # If cfg fields are invalid, ValueError is raised;
    # the caller is responsible for providing valid cfg in production.
    cli_root = getattr(args, "artifact_root", None)
    env_root = os.environ.get("ORTHRUS_ARTIFACT_ROOT", None)
    run_dir = resolve_artifact_paths(
        cfg_pre, stages,
        cli_artifact_root=cli_root,
        env_artifact_root=env_root,
        create_dirs=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Write run metadata (environment + resolved config)
    # ------------------------------------------------------------------ #
    cfg_pre._run_start_time = datetime.now(timezone.utc).isoformat()
    dump_environment(run_dir)
    dump_config(cfg_pre, run_dir)

    # ------------------------------------------------------------------ #
    # 3. W&B initialisation
    # ------------------------------------------------------------------ #
    init_wandb(cfg_pre, args, wandb_mode)

    # ------------------------------------------------------------------ #
    # 4. Run pipeline
    # ------------------------------------------------------------------ #
    try:
        timing = main(cfg_pre, args)
        status = "completed"
        error_msg = None
    except Exception as exc:
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # ------------------------------------------------------------------ #
        # 5. Write runtime.json and clean up
        # ------------------------------------------------------------------ #
        dump_runtime(
            cfg_pre,
            run_dir,
            status=status,
            executed_stages=stages,
            timing=timing if "timing" in dir() else {},
            wandb_mode=wandb_mode,
            error_message=error_msg if "error_msg" in dir() else None,
        )
        wandb_finish()
