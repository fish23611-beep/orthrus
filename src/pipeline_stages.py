"""
Stage control helpers for the orthrus pipeline.

This module is intentionally free of torch / wandb / psycopg2 / numpy / yacs /
detection / attack_reconstruction imports so it can be unit-tested in isolation.
"""
from __future__ import annotations

import os


STANDARD_STAGES = ["preprocess", "train", "test", "evaluate", "trace"]
VALID_STAGES = set(STANDARD_STAGES)

# Preprocess substages that can be run independently
PREPROCESS_SUBSTAGES = ["build_graphs", "embed_nodes", "embed_edges"]
VALID_PREPROCESS_SUBSTAGES = set(PREPROCESS_SUBSTAGES)


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


def parse_preprocess_substages(substages_str):
    """
    Parse --preprocess-substages CLI argument.
    
    Args:
        substages_str: comma-separated list of preprocess substages or None
        
    Returns:
        list of preprocess substages to run, or None for default (all)
        
    Raises:
        ValueError: if an invalid substage is specified
    """
    if substages_str is None:
        # Default: run all preprocess substages in order
        return list(PREPROCESS_SUBSTAGES)
    
    substages_str = substages_str.strip()
    if not substages_str:
        return list(PREPROCESS_SUBSTAGES)
    
    raw = [s.strip() for s in substages_str.split(",") if s.strip()]
    if not raw:
        return list(PREPROCESS_SUBSTAGES)
    
    # Validate each substage
    for s in raw:
        if s not in VALID_PREPROCESS_SUBSTAGES:
            raise ValueError(
                f"Invalid preprocess substage '{s}'. Valid substages are: {sorted(VALID_PREPROCESS_SUBSTAGES)}"
            )
    
    return raw


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


# ---------------------------------------------------------------------------
# Artifact completion checks for bounded-memory preprocessing
# ---------------------------------------------------------------------------

def _get_completion_marker_path(graphs_dir, stage):
    """Get path to completion marker for a preprocessing stage."""
    markers = {
        "build_graphs": ".preprocess_build_graphs_complete",
        "embed_nodes": ".preprocess_embed_nodes_complete",
        "embed_edges": ".preprocess_embed_edges_complete",
    }
    return os.path.join(graphs_dir, markers.get(stage, f".preprocess_{stage}_complete"))


def check_preprocess_stage_complete(cfg, stage):
    """
    Check if a preprocessing stage has completed successfully.
    
    Uses completion markers for new artifacts and falls back to checking
    for actual artifacts in legacy scenarios.
    
    Args:
        cfg: configuration object
        stage: one of "build_graphs", "embed_nodes", "embed_edges"
    
    Returns:
        bool: True if stage appears to have completed
    """
    graphs_dir = cfg.graph_construction.build_graphs._graphs_dir
    
    # Check completion marker first (new mechanism)
    marker_path = _get_completion_marker_path(graphs_dir, stage)
    if os.path.isfile(marker_path):
        return True
    
    # Fallback: check for actual artifacts (legacy compatibility)
    if stage == "build_graphs":
        # Check if graphs directory has content
        if not os.path.isdir(graphs_dir):
            return False
        # Count graph files recursively (exclude markers and temp files)
        # Note: frozen version saved graphs WITHOUT .pt suffix (time_interval format)
        # New version also doesn't add .pt suffix (uses atomic rename pattern)
        # So we look for any non-temp file in graph subdirectories
        graph_files = []
        for root, dirs, files in os.walk(graphs_dir):
            for f in files:
                # Graph files may or may not have .pt suffix
                # Exclude completion markers and temp files
                if f.startswith('.preprocess_') or f.endswith('.tmp'):
                    continue
                # Graph subdirectories are named graph_N, files inside are time_intervals
                # or optionally have .pt suffix in some legacy scenarios
                if 'graph_' in root:
                    graph_files.append(f)
        return len(graph_files) > 0
    
    elif stage == "embed_nodes":
        # Check for Word2Vec model
        model_dir = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
        model_path = os.path.join(model_dir, "feature_word2vec.model")
        return os.path.isfile(model_path)
    
    elif stage == "embed_edges":
        # Check for edge embeddings
        edge_embeds_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
        if not os.path.isdir(edge_embeds_dir):
            return False
        # Check for train/val/test subdirectories with content
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(edge_embeds_dir, split)
            if os.path.isdir(split_dir):
                files = [f for f in os.listdir(split_dir) if not f.startswith('.')]
                if files:
                    return True
        return False
    
    return False


def check_all_preprocess_stages_complete(cfg):
    """
    Check if all preprocessing stages have completed.
    
    Args:
        cfg: configuration object
        
    Returns:
        bool: True if all stages appear complete
    """
    return all(
        check_preprocess_stage_complete(cfg, stage)
        for stage in PREPROCESS_SUBSTAGES
    )
