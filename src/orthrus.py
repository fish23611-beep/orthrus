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
    parse_preprocess_substages as _parse_preprocess_substages,
    check_conflict as _check_conflict,
    check_preprocess_stage_complete,
)

from artifact_paths import resolve_artifact_paths
from run_metadata import dump_environment, dump_config, dump_runtime
from wandb_control import resolve_wandb_mode, init_wandb, wandb_log, wandb_finish


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def _get_memory_usage_mb():
    """Get current process RSS memory in MB using psutil."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _log_memory(label=""):
    """Log current memory usage."""
    rss_mb = _get_memory_usage_mb()
    if rss_mb > 0:
        log(f"[Memory{': ' + label if label else ''}] RSS = {rss_mb:.1f} MB ({rss_mb/1024:.2f} GB)")
    return rss_mb


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

    # Train produces checkpoints; an explicit inference checkpoint replaces the
    # default run checkpoint directory for test/evaluate-only invocations.
    inference_checkpoint = getattr(cfg, "_inference_checkpoint", None)
    has_inference_checkpoint = (
        isinstance(inference_checkpoint, str) and bool(inference_checkpoint)
    )
    if "train" not in stages and ("test" in stages or "evaluate" in stages) and not has_inference_checkpoint:
        checkpoint_dir = cfg.detection.gnn_training._trained_models_dir
        if not os.path.isdir(checkpoint_dir):
            missing.append(f"Checkpoints: {checkpoint_dir}")
        elif not os.listdir(checkpoint_dir):
            missing.append(f"Checkpoints directory empty: {checkpoint_dir}")

    # Test produces the edge scores consumed by evaluate in the same pipeline.
    if "evaluate" in stages and "test" not in stages:
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
    - test selected  AND train NOT selected  -> check model checkpoint dir
    - evaluate selected AND test NOT selected -> check test edge-loss output dir
    """
    train_run = "train" in stages
    test_run = "test" in stages

    inference_checkpoint = getattr(cfg, "_inference_checkpoint", None)
    has_inference_checkpoint = (
        isinstance(inference_checkpoint, str)
        and bool(inference_checkpoint)
    )
    if "test" in stages and not train_run and not has_inference_checkpoint:
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
# Preprocess substage execution
# ---------------------------------------------------------------------------

def _check_preprocess_substage_prerequisites(substage, cfg, completed_substages=()):
    """
    Check prerequisites for running a specific preprocess substage.
    
    Args:
        substage: one of "build_graphs", "embed_nodes", "embed_edges"
        cfg: configuration object
        
        completed_substages: substages successfully completed in this invocation
    Raises:
        FileNotFoundError: if prerequisites are not met
    """
    completed_substages = set(completed_substages)
    if substage == "embed_nodes":
        # embed_nodes needs database access (checked at runtime)
        pass
    elif substage == "embed_edges":
        # embed_edges needs graphs and Word2Vec model
        if ("build_graphs" not in completed_substages and
                not check_preprocess_stage_complete(cfg, "build_graphs")):
            raise FileNotFoundError(
                f"Substage 'embed_edges' requires completed 'build_graphs' stage. "
                f"Run 'build_graphs' first, or full 'preprocess' pipeline."
            )
        if ("embed_nodes" not in completed_substages and
                not check_preprocess_stage_complete(cfg, "embed_nodes")):
            raise FileNotFoundError(
                f"Substage 'embed_edges' requires completed 'embed_nodes' stage. "
                f"Run 'embed_nodes' first, or full 'preprocess' pipeline."
            )


def _run_preprocess_substages(cfg, substages):
    """
    Run specified preprocessing substages with memory tracking.
    
    Args:
        cfg: configuration object
        substages: list of substage names to run
        
    Returns:
        dict with timing information
    """
    timings = {}
    peak_rss = 0.0
    
    completed_substages = set()
    log("=" * 40)
    log("Starting bounded-memory preprocessing")
    _log_memory("preprocess start")
    log("=" * 40)
    
    for substage in substages:
        log(f"\n>>> Starting substage: {substage}")
        t_start = time_module.time()
        rss_start = _get_memory_usage_mb()
        
        try:
            if substage == "build_graphs":
                build_orthrus_graphs.main(cfg)
            elif substage == "embed_nodes":
                _check_preprocess_substage_prerequisites(substage, cfg, completed_substages)
                build_feature_word2vec.main(cfg)
            elif substage == "embed_edges":
                _check_preprocess_substage_prerequisites(substage, cfg, completed_substages)
                embed_edges_feature_word2vec.main(cfg)
            else:
                raise ValueError(f"Unknown substage: {substage}")
            
            t_end = time_module.time()
            completed_substages.add(substage)
            rss_end = _get_memory_usage_mb()
            rss_peak = max(rss_start, rss_end, _get_memory_usage_mb())
            peak_rss = max(peak_rss, rss_peak)
            
            log(f">>> Completed substage: {substage} "
                f"(time: {t_end - t_start:.1f}s, "
                f"RSS: {rss_end:.1f}MB)")
            
            timings[f"time_{substage}"] = round(t_end - t_start, 2)
            timings[f"rss_{substage}_mb"] = round(rss_end, 1)
            timings[f"peak_rss_{substage}_mb"] = round(rss_peak, 1)
            
        except Exception as e:
            log(f">>> FAILED substage: {substage}: {e}")
            raise
        
        # Force garbage collection between stages
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    log("\n" + "=" * 40)
    log("Preprocessing complete")
    _log_memory("preprocess end")
    log(f"Peak RSS during preprocessing: {peak_rss:.1f} MB ({peak_rss/1024:.2f} GB)")
    log("=" * 40)
    
    timings["preprocess_peak_rss_mb"] = round(peak_rss, 1)
    
    return timings


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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    t0 = time_module.time()
    
    # Record start memory
    rss_start = _get_memory_usage_mb()
    if rss_start > 0:
        log(f"[Memory] Process start RSS: {rss_start:.1f} MB ({rss_start/1024:.2f} GB)")

    # ------------------------------------------------------------------
    # 3. Preprocess (skip in detection_only mode)
    # ------------------------------------------------------------------
    t_build_graphs = None
    t_embed_nodes = None
    t_embed_edges = None
    preprocess_timings = {}

    if "preprocess" in stages:
        # Parse preprocess substages from args
        preprocess_substages = _parse_preprocess_substages(
            getattr(args, 'preprocess_substages', None)
        )
        
        log(f"Preprocess substages to run: {preprocess_substages}")
        
        preprocess_timings = _run_preprocess_substages(cfg, preprocess_substages)
        
        # Record timing boundaries
        if "build_graphs" in preprocess_substages:
            t_build_graphs = time_module.time()
        if "embed_nodes" in preprocess_substages:
            t_embed_nodes = time_module.time()
        if "embed_edges" in preprocess_substages:
            t_embed_edges = time_module.time()

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    t_gnn_training = None
    if "train" in stages:
        orthrus_gnn_training.main(cfg)
        if torch.cuda.is_available():
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
    rss_final = _get_memory_usage_mb()

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
    
    # Add memory info to timing
    if rss_start > 0:
        time_consumption["rss_start_mb"] = round(rss_start, 1)
    if rss_final > 0:
        time_consumption["rss_end_mb"] = round(rss_final, 1)
    if preprocess_timings:
        time_consumption.update(preprocess_timings)

    log("==" * 30)
    log("Run finished. Time consumed in each step:")
    for k, v in time_consumption.items():
        log(f"{k}: {v} s" if not k.endswith("_mb") else f"{k}: {v} MB")

    log("==" * 30)
    if wandb.run is not None:
        wandb_log(time_consumption)

    return time_consumption


def run(args):
    """Resolve runtime services and execute the standard ORTHRUS pipeline.

    This is the reusable form of the historical ``__main__`` block. Keeping
    orchestration here ensures alternate CLIs share configuration resolution,
    artifact paths, metadata, W&B handling, and the single pipeline entry.
    """
    cfg_pre = get_yml_cfg(args)
    wandb_mode = resolve_wandb_mode(cfg_pre, args)
    stages = _parse_stages(args.stages, args.run_from_training)

    cli_root = getattr(args, "artifact_root", None)
    env_root = os.environ.get("ORTHRUS_ARTIFACT_ROOT", None)
    run_dir = resolve_artifact_paths(
        cfg_pre, stages,
        cli_artifact_root=cli_root,
        env_artifact_root=env_root,
        create_dirs=True,
    )

    cfg_pre._run_start_time = datetime.now(timezone.utc).isoformat()
    dump_environment(run_dir)
    dump_config(cfg_pre, run_dir)
    init_wandb(cfg_pre, args, wandb_mode)

    try:
        timing = main(cfg_pre, args)
        status = "completed"
        error_msg = None
        return timing
    except Exception as exc:
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
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


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")
    run(args)
