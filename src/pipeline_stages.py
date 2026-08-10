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

def _get_completion_marker_path(artifact_dir, stage):
    """Return the marker written by a preprocessing stage."""
    markers = {
        "build_graphs": ".preprocess_build_graphs_complete",
        "embed_nodes": ".preprocess_embed_nodes_complete",
        "embed_edges": ".preprocess_embed_edges_complete",
        "metadata": ".preprocess_metadata_complete",
    }
    return os.path.join(
        artifact_dir, markers.get(stage, f".preprocess_{stage}_complete")
    )


def _has_visible_file(folder):
    if not os.path.isdir(folder):
        return False
    return any(
        os.path.isfile(os.path.join(folder, name))
        and not name.startswith(".preprocess_")
        and not name.endswith(".tmp")
        for name in os.listdir(folder)
    )


def _stage_artifacts_valid(cfg, stage):
    graphs_dir = cfg.graph_construction.build_graphs._graphs_dir
    if stage == "build_graphs":
        if not os.path.isdir(graphs_dir):
            return False
        expected_folders = []
        dataset = getattr(cfg, "dataset", None)
        if dataset is not None:
            for attr in ("train_files", "val_files", "test_files"):
                values = getattr(dataset, attr, None)
                if isinstance(values, (list, tuple)):
                    expected_folders.extend(values)
        expected_folders = list(dict.fromkeys(expected_folders))
        if expected_folders:
            return all(
                _has_visible_file(os.path.join(graphs_dir, folder))
                for folder in expected_folders
            )
        return any(
            name.startswith("graph_")
            and _has_visible_file(os.path.join(graphs_dir, name))
            for name in os.listdir(graphs_dir)
        )

    if stage == "embed_nodes":
        model_dir = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
        return os.path.isfile(os.path.join(model_dir, "feature_word2vec.model"))

    if stage == "embed_edges":
        edge_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
        return all(
            _has_visible_file(os.path.join(edge_dir, split))
            for split in ("train", "val", "test")
        )

    if stage == "metadata":
        metadata_dir = getattr(cfg, "_metadata_dir", None)
        if not metadata_dir:
            return False
        from mstc.metadata_cache import MetadataCache, metadata_complete
        return metadata_complete(MetadataCache(metadata_dir))

    return False


def check_preprocess_stage_complete(cfg, stage):
    """Require a completion marker plus key artifact validation.

    Marker-free graph/model/edge artifacts retain legacy compatibility, while a
    marker can never make a partial or corrupt stage look complete.
    """
    if stage == "build_graphs":
        artifact_dir = cfg.graph_construction.build_graphs._graphs_dir
    elif stage == "embed_nodes":
        artifact_dir = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
    elif stage == "embed_edges":
        artifact_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
    elif stage == "metadata":
        artifact_dir = getattr(cfg, "_metadata_dir", "")
    else:
        return False

    valid = _stage_artifacts_valid(cfg, stage)
    marker = _get_completion_marker_path(artifact_dir, stage)
    if os.path.isfile(marker):
        return valid
    if stage == "metadata":
        return False
    return valid


def check_all_preprocess_stages_complete(cfg):
    """Require all four markers and validate every stage's key artifacts."""
    artifact_dirs = {
        "build_graphs": cfg.graph_construction.build_graphs._graphs_dir,
        "embed_nodes": cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir,
        "embed_edges": cfg.edge_featurization.embed_edges._edge_embeds_dir,
        "metadata": getattr(cfg, "_metadata_dir", ""),
    }
    return all(
        os.path.isfile(_get_completion_marker_path(artifact_dirs[stage], stage))
        and check_preprocess_stage_complete(cfg, stage)
        for stage in (*PREPROCESS_SUBSTAGES, "metadata")
    )
