"""
Stage control helpers for the orthrus pipeline.

This module is intentionally free of torch / wandb / psycopg2 / numpy / yacs /
detection / attack_reconstruction imports so it can be unit-tested in isolation.
"""
from __future__ import annotations


STANDARD_STAGES = ["preprocess", "train", "test", "evaluate", "trace"]
VALID_STAGES = set(STANDARD_STAGES)
DEFAULT_STAGES = ["preprocess", "train", "test", "evaluate"]

PREPROCESS_SUBSTAGES = ["build_graphs", "embed_nodes", "embed_edges"]


def parse_stages(stages_str, run_from_training):
    """
    Parse --stages CLI argument and resolve effective stages.

    When both --stages and --run_from_training are set, --stages takes precedence
    and a deprecation warning is logged (handled in orthrus.main).

    Returns:
        ordered list of canonical stage names (preprocess/train/test/evaluate/trace)
    """
    if stages_str is None:
        if run_from_training:
            return ["train", "test", "evaluate"]
        return list(DEFAULT_STAGES)

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
    """
    Check for conflicting arguments.

    When both --stages and --run_from_training are set, a warning is returned
    (not raised) so orthrus.main can log it and continue.  The --stages value
    takes precedence.

    Returns:
        str | None: deprecation warning message, or None if no conflict.
    """
    if stages_str is not None and run_from_training:
        return (
            "WARNING: --run_from_training is deprecated when --stages is also set. "
            "Ignoring --run_from_training; using explicit --stages. "
            "Run without --stages to use --run_from_training."
        )
    return None
