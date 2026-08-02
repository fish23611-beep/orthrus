"""
Stage control helpers for the orthrus pipeline.

This module is intentionally free of torch / wandb / psycopg2 / numpy / yacs /
detection / attack_reconstruction imports so it can be unit-tested in isolation.
"""
from __future__ import annotations


STANDARD_STAGES = ["preprocess", "train", "test", "evaluate", "trace"]
VALID_STAGES = set(STANDARD_STAGES)

PREPROCESS_SUBSTAGES = ["build_graphs", "embed_nodes", "embed_edges"]


def parse_stages(stages_str, run_from_training):
    """
    Parse --stages CLI argument and resolve effective stages.

    Returns:
        ordered list of canonical stage names (preprocess/train/test/evaluate/trace)
    """
    if stages_str is None:
        if run_from_training:
            return ["train", "test", "evaluate"]
        return ["preprocess", "train", "test", "evaluate", "trace"]

    stages_str = stages_str.strip()
    if stages_str.lower() == "all":
        return ["preprocess", "train", "test", "evaluate", "trace"]

    raw = [s.strip() for s in stages_str.split(",") if s.strip()]
    if not raw:
        raise ValueError("--stages must specify at least one stage")

    seen = set()
    ordered = []
    for s in raw:
        if s not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{s}'. Valid stages are: {sorted(VALID_STAGES)}"
            )
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def check_conflict(stages_str, run_from_training):
    """Raise ValueError if --stages and --run_from_training conflict."""
    if stages_str is not None and run_from_training:
        raise ValueError(
            "Conflicting arguments: --stages and --run_from_training cannot both be set. "
            "Use --stages alone to control the pipeline; --run_from_training is deprecated "
            "in favour of explicit --stages."
        )
