"""Thin, single-experiment CLI that delegates to the ORTHRUS pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DATASET_DEFAULT_CONFIG
from pipeline_stages import parse_stages


def _existing_path(value: str, *, label: str) -> str:
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"{label} is neither a regular file nor a directory: {path}")
    return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one ORTHRUS experiment through the standard pipeline.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name accepted by ORTHRUS.")
    parser.add_argument("--config", required=True, metavar="PATH", help="Explicit YAML configuration file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--artifact-root", default=None, metavar="PATH",
        help="Scoped experiment artifact root (isolates run artifacts: checkpoints, "
             "edge_scores, node_scores). For matrix runs, this is set by run_matrix "
             "to matrix_artifacts/<config-id>.",
    )
    parser.add_argument(
        "--shared-artifact-root", default=None, metavar="PATH",
        help="Shared preprocessing artifact root (graph_construction, Word2Vec, "
             "edge_embeddings, metadata). When absent, falls back to --artifact-root. "
             "For matrix runs, run_matrix passes the top-level artifact root here "
             "while using --artifact-root for scoped run isolation.",
    )
    parser.add_argument(
        "--stages", default="all",
        help="Pipeline stages: train, test,evaluate, train,test,evaluate, or all (default: all).",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help=("With train, structured checkpoint used for training resume. Without train, "
              "checkpoint file or directory used for inference."),
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of TemporalData windows to load per split (train/val/test). "
             "Default: None (load all windows). "
             "Use 2 for bounded baseline smoke test.",
    )
    return parser


def build_pipeline_args(namespace: argparse.Namespace) -> SimpleNamespace:
    """Validate wrapper input and translate it to the established CLI contract."""
    if namespace.dataset not in DATASET_DEFAULT_CONFIG:
        available = ", ".join(sorted(DATASET_DEFAULT_CONFIG))
        raise ValueError(f"Unknown dataset {namespace.dataset!r}. Available datasets: {available}")

    config = _existing_path(namespace.config, label="Config file")
    if not Path(config).is_file():
        raise ValueError(f"Config file is not a regular file: {config}")
    import yaml
    try:
        with Path(config).open(encoding="utf-8") as handle:
            yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Unable to parse YAML config {config}: {exc}") from exc

    stages = parse_stages(namespace.stages, run_from_training=False)
    checkpoint = None
    if namespace.checkpoint is not None:
        checkpoint = _existing_path(namespace.checkpoint, label="Checkpoint")

    pipeline_args = SimpleNamespace(
        dataset=namespace.dataset,
        model="orthrus",
        config=config,
        seed=namespace.seed,
        artifact_root=namespace.artifact_root,
        shared_artifact_root=getattr(namespace, "shared_artifact_root", None),
        stages=namespace.stages,
        checkpoint=checkpoint,
        cpu=namespace.cpu,
        from_weights=False,
        run_from_training=False,
        skip_tracing=False,
        wandb=False,
        exp="",
        tags="",
        show_attack=0,
        gt_type="orthrus",
        plot_gt=False,
        max_windows_per_split=namespace.max_windows_per_split,
    )
    if checkpoint is not None:
        if "train" in stages:
            pipeline_args.resume_checkpoint = checkpoint
        else:
            pipeline_args.inference_checkpoint = checkpoint
    return pipeline_args


def _run_pipeline(args):
    """Import the heavy pipeline lazily so CLI help stays side-effect free."""
    import orthrus
    return orthrus.run(args)


def main(argv: Sequence[str] | None = None):
    """Parse one experiment invocation and delegate it to ``orthrus.run``."""
    namespace = build_parser().parse_args(argv)
    return _run_pipeline(build_pipeline_args(namespace))


if __name__ == "__main__":
    main()
