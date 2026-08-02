import argparse
import os
import random
import time as time_module

import torch
import wandb
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


# ---------------------------------------------------------------------------
# Artifact prerequisite checks
# ---------------------------------------------------------------------------
def _check_artifact_prerequisites(stages, cfg):
    """
    Check that required artifacts exist when an upstream stage is NOT being run.

    Rules:
    - test selected  AND train NOT selected  → check model checkpoint dir
    - evaluate selected AND test NOT selected → check test edge-loss output dir
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
    # 1. Parse stages + resolve tracing
    # ------------------------------------------------------------------
    _check_conflict(args.stages, args.run_from_training)

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
    # 3. Preprocess
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
        wandb.log(time_consumption)

    return time_consumption


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)

    exp_name = args.exp if args.exp != "" else \
        args.__dict__["dataset"]
        # "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="orthrus_repo",  # Can be changed
        name=exp_name,
        tags=tags,
    )

    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg, args)

    wandb.finish()
