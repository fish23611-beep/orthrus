"""
Stage control helpers for the orthrus pipeline.

This module is intentionally free of torch / wandb / psycopg2 / numpy / yacs /
detection / attack_reconstruction imports so it can be unit-tested in isolation.
"""
from __future__ import annotations

import json
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


def _is_verified_empty_day(folder, *, dataset=None, graph_name=None, day=None):
    """A legal empty day is a marker, not a graph file.

    The marker must be structurally valid and identity-matching for the
    (dataset, graph_name, day) tuple that the validator is checking. This
    makes a raw-empty day equivalent to a real graph artifact for the
    purpose of artifact validation, but never equivalent to a missing,
    malformed, or stale marker.
    """
    try:
        from graph_construction.empty_day import is_verified_empty_day
    except ImportError:
        return False
    kwargs = {}
    if dataset is not None:
        kwargs["dataset"] = dataset
    if graph_name is not None:
        kwargs["graph_name"] = graph_name
    if day is not None:
        kwargs["day"] = day
    return is_verified_empty_day(folder, **kwargs)


def _has_real_graph_artifact(folder):
    """True iff ``folder`` contains at least one real graph artifact.

    Hidden preprocess markers (``.preprocess_*``) and in-flight
    ``.tmp`` files are explicitly excluded so they can never be
    miscounted as graph artifacts.
    """
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        if name.startswith(".preprocess_") or name.endswith(".tmp"):
            continue
        if os.path.isfile(os.path.join(folder, name)):
            return True
    return False


def _split_is_valid(folder, *, dataset=None, graph_name=None, day=None):
    """An expected split is valid iff it has real graph files OR a
    valid verified-empty-day marker. Anything else (missing, empty,
    malformed, raw_event_count != 0) is invalid."""
    if not os.path.isdir(folder):
        return False
    if _has_real_graph_artifact(folder):
        return True
    return _is_verified_empty_day(
        folder, dataset=dataset, graph_name=graph_name, day=day
    )


def _day_index_from_graph_name(graph_name):
    """Parse the integer day index from a ``graph_<n>`` folder name."""
    if not isinstance(graph_name, str):
        return None
    if not graph_name.startswith("graph_"):
        return None
    suffix = graph_name[len("graph_"):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _is_valid_embed_edges_marker(artifact_dir):
    """
    Validate embed_edges completion marker is v2 format.

    Old ISO timestamp markers are rejected (schema_version 1 implicit).
    v2 markers must have correct schema_version and temporal_order.

    Returns:
        bool: True if marker is valid v2, False otherwise
    """
    marker_path = os.path.join(artifact_dir, ".preprocess_embed_edges_complete")
    if not os.path.isfile(marker_path):
        return False

    try:
        with open(marker_path, 'r', encoding="utf-8") as f:
            content = f.read().strip()

        # Try parsing as JSON v2 marker
        if content.startswith('{'):
            marker_data = json.loads(content)
            if not isinstance(marker_data, dict):
                return False
            schema_version = marker_data.get("schema_version")
            temporal_order = marker_data.get("temporal_order")
            # Valid v2 marker
            if schema_version == 2 and temporal_order == "nondecreasing":
                return True
            return False

        # Old ISO timestamp marker format (schema_version implicit = 1)
        # These are no longer valid
        return False
    except (json.JSONDecodeError, IOError, UnicodeDecodeError):
        return False


def _stage_artifacts_valid(cfg, stage):
    graphs_dir = cfg.graph_construction.build_graphs._graphs_dir
    if stage == "build_graphs":
        if not os.path.isdir(graphs_dir):
            return False
        expected_folders = []
        dataset = getattr(cfg, "dataset", None)
        train_files = val_files = test_files = ()
        if dataset is not None:
            train_files = getattr(dataset, "train_files", None) or ()
            val_files = getattr(dataset, "val_files", None) or ()
            test_files = getattr(dataset, "test_files", None) or ()
            expected_folders.extend(train_files)
            expected_folders.extend(val_files)
            expected_folders.extend(test_files)
        expected_folders = list(dict.fromkeys(expected_folders))
        if expected_folders:
            dataset_name = getattr(dataset, "name", None)
            for folder in expected_folders:
                day_index = _day_index_from_graph_name(folder)
                if not _split_is_valid(
                    os.path.join(graphs_dir, folder),
                    dataset=dataset_name,
                    graph_name=folder,
                    day=day_index,
                ):
                    return False
            return True
        return any(
            name.startswith("graph_")
            and _has_real_graph_artifact(os.path.join(graphs_dir, name))
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

    For embed_edges, v2 JSON marker with temporal_order="nondecreasing" is required.
    Old ISO timestamp markers are automatically rejected.
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
        # embed_edges uses v2 marker validation
        if stage == "embed_edges":
            return valid and _is_valid_embed_edges_marker(artifact_dir)
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
