#!/usr/bin/env python3
"""
MAGIC THEIA_E3 Formal Experiment Runner

Persistent formal runner for THEIA_E3 (and THEIA_E5) MAGIC baseline experiments.

This is the canonical entry point for formal MAGIC experiments. It supersedes
the temporary one-off scripts used in M9-E0.

Key contract guarantees:
1. PER-SOURCE-ARTIFACT GRANULARITY: each ORTHRUS nx MultiDiGraph window becomes
   an independent MAGICPreparedGraph passed to backend.fit().
2. GT IDENTITY: UUID -> integer index_id mapping via authoritative CSV source.
3. EVALUATOR UNIVERSE = TEST_NODE_UNIVERSE: metrics computed over all test nodes,
   not over GT UUID keys.
4. NO TEST LABEL LEAKAGE: threshold fitted only on validation scores.
5. FAIL-FAST ON MISSING DATA: missing predictions or scores raise immediately.

Usage:
    python -m src.baselines.magic.formal_runner --dataset THEIA_E3 --seed 0 --output-dir artifacts/THEIA_E3/formal/seed_0_v2
    python -m src.baselines.magic.formal_runner --dataset THEIA_E3 --seed 1 --output-dir artifacts/THEIA_E3/formal/seed_1_v1

Artifact schema (per formal contract):
    model.pt, metrics.json, predictions.json, resolved_config.json,
    runtime_manifest.json, train_log.json, guards.json, run_status.json
Plus: gt_coverage.json (GT → test split coverage diagnostics)

Forbidden:
- Formal training (--dry-run flag skips model.fit())
- Modifying optimizer/lr/loss
- Modifying bridge semantics
- Changing threshold policy
- Gradient clipping
- git push
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time as _time
from argparse import ArgumentParser
from datetime import datetime, timezone
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import from the frozen bridge
from src.baselines.magic.real_backend import (
    MAGICModelConfig,
    MAGICRealBackend,
)
from src.baselines.magic.input_adapter import MAGICInputAdapter
from src.baselines.magic.orthrus_predecessor import (
    OrthrusPredecessor,
    is_verified_empty_day_marker,
)
from src.baselines.magic.scoring import MAGICEntityScorer
from src.baselines.magic.evaluator import (
    compute_magic_metrics,
    load_magic_ground_truth,
    EvaluationUniverseError,
)
from src.baselines.magic.contracts import SplitType
from src.baselines.magic.protocol import fit_threshold
import src.baselines.magic.real_backend as _rb


# =============================================================================
# Safe package version collector (non-fatal)
# =============================================================================

def _safe_package_version(package_name: str) -> str:
    """
    Safely collect the version of an installed package.

    Uses importlib.metadata (Python 3.8+) to query distribution versions
    without importing the package itself. Falls back gracefully if the
    package is not installed.

    Args:
        package_name: Name of the package (as would be passed to pip install)

    Returns:
        Version string if installed.
        "not_installed" if the package is not installed
        (PackageNotFoundError).
        "metadata_error:<ExceptionName>" for unexpected metadata failures
        (e.g. permission errors, corrupted metadata). Non-fatal.
    """
    try:
        return _importlib_metadata.version(package_name)
    except _importlib_metadata.PackageNotFoundError:
        return "not_installed"
    except Exception as exc:
        return "metadata_error:{}".format(type(exc).__name__)

# =============================================================================
# Canonical artifact paths
# =============================================================================

# Graph construction artifact root: derived from the build_graphs directory
# which is resolved by the config's graph_construction.build_graphs._graphs_dir.
# We resolve it by scanning for the canonical fingerprint under
# artifacts/graph_construction/<dataset>/build_graphs/
DEFAULT_GRAPH_CONSTRUCTION_ROOT = REPO_ROOT / "artifacts" / "graph_construction"

# Ground truth root
DEFAULT_GT_ROOT = REPO_ROOT / "Ground_Truth" / "darpa" / "darpa"

# Dataset-specific GT root overrides
DATASET_GT_ROOT = {
    "THEIA_E3": REPO_ROOT / "Ground_Truth" / "darpa" / "darpa" / "E3-THEIA",
    "THEIA_E5": REPO_ROOT / "Ground_Truth" / "darpa" / "darpa" / "E5-THEIA",
}

# =============================================================================
# Dataset split definitions (canonical)
# =============================================================================

# Canonical THEIA_E3 splits
THEIA_E3_SPLITS = {
    "train": ["graph_2", "graph_3", "graph_4", "graph_5"],
    "validation": ["graph_9"],
    "test": ["graph_10", "graph_12", "graph_13"],
}

# Canonical THEIA_E5 splits
THEIA_E5_SPLITS = {
    "train": ["graph_2", "graph_3", "graph_4", "graph_5", "graph_6", "graph_7"],
    "validation": ["graph_11"],
    "test": ["graph_12", "graph_13", "graph_14"],
}

DATASET_SPLITS = {
    "THEIA_E3": THEIA_E3_SPLITS,
    "THEIA_E5": THEIA_E5_SPLITS,
}


# =============================================================================
# Canonical graph construction artifact resolver
# =============================================================================

def resolve_graph_construction_root(
    dataset: str,
    root: Optional[Path] = None,
) -> Path:
    """
    Resolve the graph construction build_graphs directory for a dataset.

    Looks for:
        artifacts/graph_construction/<dataset>/build_graphs/<fingerprint>/

    Args:
        dataset: Dataset name (THEIA_E3 or THEIA_E5)
        root: Override root directory

    Returns:
        Path to build_graphs directory

    Raises:
        FileNotFoundError: If no build_graphs directory found
    """
    base = (root or DEFAULT_GRAPH_CONSTRUCTION_ROOT) / dataset / "build_graphs"
    if not base.is_dir():
        raise FileNotFoundError(
            f"Graph construction directory not found: {base}"
        )

    # Find the fingerprint subdirectory (the only subdir of build_graphs/)
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"No build_graphs subdirectory found in: {base}"
        )
    if len(subdirs) > 1:
        raise ValueError(
            f"Multiple build_graphs subdirectories found in {base}: "
            f"{[d.name for d in subdirs]}. Expected exactly one."
        )
    return subdirs[0]


# =============================================================================
# Canonical artifact enumeration
# =============================================================================

def enumerate_split_nx_artifacts(
    graph_root: Path,
    split_names: List[str],
) -> Dict[str, List[Path]]:
    """
    Enumerate all non-empty nx MultiDiGraph artifact files for given splits.

    Returns:
        Dict mapping split_name -> [sorted list of .pt artifact paths]
        Empty splits (verified empty days) return [] for that split.

    Raises:
        FileNotFoundError: If split directory does not exist
    """
    result: Dict[str, List[Path]] = {}
    for split_name in split_names:
        split_dir = graph_root / "nx" / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}"
            )

        # ORTHRUS nx artifacts may not have .pt extension; include all files
        # (torch.load works on them regardless of extension)
        artifacts: List[Path] = []
        for item in split_dir.iterdir():
            # Skip empty day markers
            if item.name.startswith("."):
                if is_verified_empty_day_marker(str(item)):
                    continue  # Verified empty day: contributes 0 artifacts
                else:
                    continue  # Skip dotfiles

            if item.is_file():
                # Include all files (no extension filter for ORTHRUS nx files)
                artifacts.append(item)
            elif item.is_dir():
                # Subdirectory containing windowed files
                for sub in item.iterdir():
                    if sub.is_file():
                        artifacts.append(sub)

        result[split_name] = sorted(artifacts)

    return result


# =============================================================================
# Graph loading pipeline
# =============================================================================

def load_nx_graph(path: Path) -> Any:
    """Load a torch-serialized nx.MultiDiGraph."""
    import torch
    return torch.load(path, map_location="cpu")


def nx_to_canonical_records(
    graph_path: Path,
    predecessor: OrthrusPredecessor,
) -> List[Dict[str, Any]]:
    """
    Load one nx artifact and convert to canonical records.

    Args:
        graph_path: Path to torch-serialized nx.MultiDiGraph
        predecessor: OrthrusPredecessor instance (dataset-aware)

    Returns:
        List of canonical edge records
    """
    g = load_nx_graph(graph_path)
    return predecessor.convert_nx_to_canonical(g, graph_name=graph_path.name)


def build_split_contracts(
    graph_root: Path,
    split_artifacts: Dict[str, List[Path]],
    dataset: str,
    profile: str = "orthrus_unified",
) -> Tuple[
    List[Any],   # train contracts (one per artifact)
    Any,         # val contract (merged per split)
    Any,         # test contract (merged per split)
    Dict[str, int],  # artifact counts per split
]:
    """
    Build NeutralGraphContract for each split.

    For TRAIN: one contract PER artifact (preserving entity-level granularity).
    For VAL/TEST: one merged contract per split (val/test scored as a whole).

    Args:
        graph_root: Path to build_graphs/<fingerprint>/
        split_artifacts: {split_name: [artifact_paths]}
        dataset: Dataset name
        profile: Adapter profile

    Returns:
        (train_contracts, val_contract, test_contract, artifact_counts)

    Raises:
        ValueError: If train split produces zero contracts (no non-empty artifacts)
    """
    # Initialize predecessor for ORTHRUS graph conversion
    pred = OrthrusPredecessor(dataset=dataset)

    # Load all train records first (for vocabulary fitting)
    all_train_records: List[Dict[str, Any]] = []
    train_records_per_artifact: List[List[Dict[str, Any]]] = []
    train_artifacts_used: List[Path] = []
    train_errors: List[str] = []

    for artifact_path in split_artifacts.get("train", []):
        try:
            records = nx_to_canonical_records(artifact_path, pred)
            if records:
                train_records_per_artifact.append(records)
                all_train_records.extend(records)
                train_artifacts_used.append(artifact_path)
        except Exception as exc:
            train_errors.append(f"{artifact_path.name}: {exc}")

    if train_errors:
        print(f"  WARNING: {len(train_errors)} train artifacts failed to convert:")
        for err in train_errors[:3]:
            print(f"    {err}")

    if not train_records_per_artifact:
        raise ValueError(
            f"Train split produced 0 non-empty artifacts. "
            f"Checked: {[str(p) for p in split_artifacts.get('train', [])[:10]]}..."
        )

    # Build adapter with ALL train records for vocabulary fitting
    adapter = MAGICInputAdapter(
        dataset=dataset,
        train_records=all_train_records,
        profile=profile,
    )
    adapter.fit()

    # Build one contract PER train artifact (entity-level granularity)
    train_contracts: List[Any] = []
    for i, recs in enumerate(train_records_per_artifact):
        ctr = adapter.transform(recs, SplitType.TRAIN)
        train_contracts.append(ctr)

    # Val: build one merged contract
    val_contracts: List[Any] = []
    val_records_all: List[Dict] = []
    for artifact_path in split_artifacts.get("validation", []):
        try:
            records = nx_to_canonical_records(artifact_path, pred)
            if records:
                val_records_all.extend(records)
        except Exception:
            continue
    if val_records_all:
        val_contract = adapter.transform(val_records_all, SplitType.VALIDATION)
        val_contracts.append(val_contract)
    else:
        # Fallback: empty contract
        adapter_val = MAGICInputAdapter(dataset=dataset, profile=profile)
        val_contract = adapter_val.transform([], SplitType.VALIDATION)

    # Test: build one merged contract
    test_records_all: List[Dict] = []
    for artifact_path in split_artifacts.get("test", []):
        try:
            records = nx_to_canonical_records(artifact_path, pred)
            if records:
                test_records_all.extend(records)
        except Exception:
            continue
    if test_records_all:
        # Reuse the vocabulary fitted on train
        test_contract = adapter.transform(test_records_all, SplitType.TEST)
    else:
        adapter_test = MAGICInputAdapter(dataset=dataset, profile=profile)
        test_contract = adapter_test.transform([], SplitType.TEST)

    artifact_counts = {
        "train": len(train_artifacts_used),
        "validation": len([p for p in split_artifacts.get("validation", [])]),
        "test": len([p for p in split_artifacts.get("test", [])]),
    }

    return train_contracts, val_contract, test_contract, artifact_counts


# =============================================================================
# Instrumented training loop (read-only monitoring)
# =============================================================================

def grad_stats(model) -> Dict[str, Any]:
    """Compute per-step gradient statistics."""
    import torch
    sq = torch.tensor(0.0)
    max_abs = 0.0
    nz = total = 0
    all_fin = True
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            all_fin = False
        sq = sq + g.pow(2).sum()
        abs_g = float(g.abs().max().item())
        if abs_g > max_abs:
            max_abs = abs_g
        nz += int((g.abs() > 0).sum().item())
        total += g.numel()
    return {
        "GLOBAL_GRAD_NORM": float(sq.sqrt().item()),
        "MAX_ABS_GRAD": max_abs,
        "NONZERO_GRAD_ELEMENT_COUNT": nz,
        "TOTAL_GRAD_ELEMENT_COUNT": total,
        "ALL_GRAD_FINITE": all_fin,
    }


class InstrumentedTrainer:
    """
    Callable with the same signature as _run_original_entity_training_lifecycle.
    Injects read-only grad monitoring.
    """

    def __init__(self, orig_fn, log: List, step_counter: List[int]):
        self._orig = orig_fn
        self._log = log
        self._steps = step_counter

    def __call__(
        self, model, graphs, optimizer, max_epoch: int, device: str, dgl_module
    ):
        import torch
        n_train = len(graphs)
        for epoch in range(max_epoch):
            for graph in graphs:
                # Deep-copy (matches real_backend._run_original_entity_training_lifecycle)
                copied = dgl_module.graph(
                    (graph.edges()[0].clone(), graph.edges()[1].clone()),
                    num_nodes=graph.num_nodes(),
                )
                for k, t in graph.ndata.items():
                    if t is not None:
                        copied.ndata[k] = t.clone()
                for k, t in graph.edata.items():
                    if t is not None:
                        copied.edata[k] = t.clone()
                runtime_g = copied.to(device)
                model.train()
                loss_t = model(runtime_g)
                loss_raw = float(loss_t.detach().cpu().item())
                loss_t = loss_t / n_train  # upstream semantics
                optimizer.zero_grad()
                loss_t.backward()

                # Read-only grad capture: AFTER backward, BEFORE step
                gs = grad_stats(model)

                optimizer.step()
                self._steps[0] += 1
                self._log.append({
                    "epoch": epoch + 1,
                    "step": self._steps[0],
                    "RAW_FORWARD_LOSS": loss_raw,
                    "LOSS_USED_FOR_BACKWARD": loss_raw / n_train,
                    "SCALING_FACTOR": 1.0 / n_train,
                    **gs,
                })
                del copied, runtime_g, loss_t
                gc.collect()
        return model


# =============================================================================
# Scoring helpers
# =============================================================================

def embed_graph(model, graph, device: str) -> Tuple[Any, List[str]]:
    """Embed a DGL graph and return (embeddings, node_ids)."""
    model.eval()
    import torch
    with torch.no_grad():
        emb = model.embed(graph.to(device))
    emb_np = emb.cpu().numpy()
    nids = list(graph.ndata.get("canonical_id", []))
    if not nids:
        nids = [str(i) for i in range(len(emb_np))]
    return emb_np, nids


def _create_magic_backend(
    *,
    seed: int,
    model_config: Optional[MAGICModelConfig],
    device: str,
    upstream_path: Optional[Path],
) -> MAGICRealBackend:
    """Create the backend without overriding its Docker-compatible default."""
    backend_kwargs: Dict[str, Any] = {
        "seed": seed,
        "config": model_config,
        "device": device,
    }
    if upstream_path is not None:
        backend_kwargs["upstream_path"] = upstream_path
    return MAGICRealBackend(**backend_kwargs)


def _collect_runtime_environment(
    backend: MAGICRealBackend,
    device: str,
) -> Dict[str, Any]:
    """Collect runtime facts shared by Docker and host deployments."""
    import dgl
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_names = (
        [torch.cuda.get_device_name(index) for index in range(cuda_device_count)]
        if cuda_available
        else []
    )
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "dgl": dgl.__version__,
        "device": device,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "upstream_path": str(backend.upstream_path.resolve()),
    }


# =============================================================================
# Main formal runner
# =============================================================================

def run_formal_experiment(
    dataset: str,
    seed: int,
    output_dir: Path,
    *,
    graph_root: Optional[Path] = None,
    gt_root: Optional[Path] = None,
    model_config: Optional[MAGICModelConfig] = None,
    adapter_profile: str = "orthrus_unified",
    device: str = "cpu",
    max_epoch: int = 50,
    k_neighbors: int = 5,
    dry_run: bool = False,
    force: bool = False,
    upstream_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run a formal MAGIC baseline experiment.

    Args:
        dataset: Dataset name (THEIA_E3 or THEIA_E5)
        seed: Random seed
        output_dir: Output directory for artifacts
        graph_root: Override graph construction root
        gt_root: Override ground truth root
        model_config: MAGIC model configuration
        adapter_profile: Adapter profile (orthrus_unified or legacy_upstream)
        device: Device (cpu or cuda)
        max_epoch: Number of training epochs
        k_neighbors: K for KNN scoring
        dry_run: If True, enumerate artifacts and build contracts but skip model.fit()
        force: If True, overwrite existing output_dir
        upstream_path: Optional pinned MAGIC source path. If omitted, retain the
            backend default (/opt/magic-upstream).

    Returns:
        Dict with run summary

    Raises:
        FileNotFoundError: Missing artifacts
        EvaluationUniverseError: Missing predictions or scores
        ValueError: Contract/granularity errors
    """
    t0 = _time.time()
    utc_now = lambda: datetime.now(timezone.utc).isoformat()

    output_dir = Path(output_dir)
    if output_dir.exists() and not force:
        raise FileNotFoundError(
            f"Output directory already exists: {output_dir}. "
            f"Use --force to overwrite, or choose a different output_dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Resolve paths ----
    print(f"[{utc_now()}] === MAGIC Formal Experiment ===")
    print(f"  Dataset: {dataset}")
    print(f"  Seed: {seed}")
    print(f"  Adapter: {adapter_profile}")
    print(f"  Output: {output_dir}")
    print(f"  Dry run: {dry_run}")

    graph_construction_root = graph_root or resolve_graph_construction_root(dataset)
    print(f"  Graph root: {graph_construction_root}")

    gt = gt_root or DATASET_GT_ROOT.get(dataset, DEFAULT_GT_ROOT)
    print(f"  GT root: {gt}")

    # ---- Step 2: Canonical split definitions ----
    if dataset not in DATASET_SPLITS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASET_SPLITS.keys())}"
        )
    splits = DATASET_SPLITS[dataset]

    # ---- Step 3: Enumerate artifacts ----
    print(f"\n[{utc_now()}] Enumerating artifacts...")
    all_artifacts = enumerate_split_nx_artifacts(
        graph_construction_root, splits["train"] + splits["validation"] + splits["test"]
    )
    for split_name, paths in all_artifacts.items():
        print(f"  {split_name}: {len(paths)} artifacts")

    total_train = sum(
        len(all_artifacts.get(s, []))
        for s in splits["train"]
    )
    print(f"  Total train artifacts: {total_train}")

    # Map graph names to split artifacts using canonical split definitions
    split_artifacts: Dict[str, List[Path]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split_name in splits["train"]:
        split_artifacts["train"].extend(all_artifacts.get(split_name, []))
    for split_name in splits["validation"]:
        split_artifacts["validation"].extend(all_artifacts.get(split_name, []))
    for split_name in splits["test"]:
        split_artifacts["test"].extend(all_artifacts.get(split_name, []))

    # Verify non-empty for train
    empty_train_splits = [s for s in splits["train"] if not all_artifacts.get(s)]
    non_empty_train_count = sum(len(all_artifacts.get(s, [])) for s in splits["train"])
    if non_empty_train_count == 0:
        raise ValueError(
            f"Train split has zero non-empty artifacts across all train splits: "
            f"{splits['train']}"
        )
    if empty_train_splits:
        print(f"  NOTE: verified empty day(s): {empty_train_splits}")

    # ---- Step 4: Build contracts ----
    print(f"\n[{utc_now()}] Building contracts (per-artifact granularity for train)...")
    t_ctr = _time.time()
    train_contracts, val_contract, test_contract, artifact_counts = build_split_contracts(
        graph_root=graph_construction_root,
        split_artifacts=split_artifacts,
        dataset=dataset,
        profile=adapter_profile,
    )
    print(f"  Train contracts (entity-level): {len(train_contracts)}")
    print(f"  Val contract: {len(val_contract.nodes)} nodes, {len(val_contract.edges)} edges")
    print(f"  Test contract: {len(test_contract.nodes)} nodes, {len(test_contract.edges)} edges")
    print(f"  Contract build time: {_time.time() - t_ctr:.1f}s")

    # Dry-run: verify contracts, compute n_train, skip all downstream (DGL/training)
    if dry_run:
        print(f"\n[{utc_now()}] DRY RUN: contract build verified, skipping DGL + model.fit()")
        manifest = {
            "dry_run": True,
            "dataset": dataset,
            "seed": seed,
            "n_train": len(train_contracts),
            "adapter_profile": adapter_profile,
            "train_artifact_counts": artifact_counts,
            "train_contract_count": len(train_contracts),
            "val_nodes": len(val_contract.nodes),
            "test_nodes": len(test_contract.nodes),
            "max_epoch": max_epoch,
            "note": "Backend.prepare_graph and model.fit() skipped in dry-run",
        }
        manifest_path = output_dir / "dry_run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  Dry-run manifest: {manifest_path}")
        print(f"  REAL_TRAIN_PREPARED_GRAPH_COUNT = {len(train_contracts)}")
        return manifest

    # ---- Step 5: Prepare DGL graphs ----
    print(f"\n[{utc_now()}] Preparing DGL graphs...")
    t_prep = _time.time()
    backend = _create_magic_backend(
        seed=seed,
        model_config=model_config,
        device=device,
        upstream_path=upstream_path,
    )

    # Train: one prepared graph PER contract (entity-level granularity)
    train_prepared_graphs: List[Any] = []
    for i, ctr in enumerate(train_contracts):
        pg = backend.prepare_graph(ctr, SplitType.TRAIN, f"train-snap-{i}")
        train_prepared_graphs.append(pg)

    val_pg = backend.prepare_graph(val_contract, SplitType.VALIDATION, "val-snap")
    test_pg = backend.prepare_graph(test_contract, SplitType.TEST, "test-snap")

    nfeat = train_prepared_graphs[0].node_feature_dim
    efeat = train_prepared_graphs[0].edge_feature_dim
    n_train = len(train_prepared_graphs)
    print(f"  Train prepared graphs: {n_train} (entity-level)")
    print(f"  NODE_FEATURE_DIM={nfeat}, EDGE_FEATURE_DIM={efeat}")
    print(f"  Total train graph: {sum(pg.graph.num_nodes() for pg in train_prepared_graphs)} nodes")
    print(f"  Prep time: {_time.time() - t_prep:.1f}s")

    # ---- Step 6: Training ----
    print(f"\n[{utc_now()}] Training: max_epoch={max_epoch}, n_train={n_train}, "
          f"lr={model_config.learning_rate}, wd={model_config.weight_decay}...")
    t_train0 = _time.time()

    _train_log: List[Dict] = []
    _step_counter = [0]

    trainer = InstrumentedTrainer(
        _rb._run_original_entity_training_lifecycle,
        _train_log,
        _step_counter,
    )
    orig_fn = _rb._run_original_entity_training_lifecycle
    _rb._run_original_entity_training_lifecycle = trainer
    try:
        backend.fit(train_prepared_graphs)
    finally:
        _rb._run_original_entity_training_lifecycle = orig_fn

    t_train = _time.time() - t_train0
    epochs_done = max_epoch
    steps_done = _step_counter[0]
    print(f"\n[{utc_now()}] Training done in {t_train:.1f}s ({t_train/60:.1f}min)")
    print(f"  EPOCHS_COMPLETED={epochs_done}, TRAIN_STEPS_COMPLETED={steps_done}")
    print(f"  N_TRAIN={n_train}")

    if _train_log:
        il = _train_log[0]["RAW_FORWARD_LOSS"]
        fl = _train_log[-1]["RAW_FORWARD_LOSS"]
        print(f"  INITIAL_TRAIN_LOSS={il:.6f}, FINAL_TRAIN_LOSS={fl:.6f}")
        nf_g = sum(1 for r in _train_log if not r["ALL_GRAD_FINITE"])
        print(f"  NONFINITE_GRAD_COUNT={nf_g}")

    # ---- Step 7: Checkpoint ----
    print(f"\n[{utc_now()}] Saving checkpoint...")
    import torch
    ckpt_path = output_dir / "model.pt"
    backend.save_checkpoint(ckpt_path)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    print(f"  CHECKPOINT_SHA256={ckpt_sha}")

    # ---- Step 8: Score train -> scorer ----
    print(f"\n[{utc_now()}] Embedding train graph for KNN scorer...")
    model = backend._model
    model.eval()
    # Embed all train prepared graphs concatenated
    # Actually: embed each train graph and collect all embeddings
    train_emb_list = []
    train_nids_list = []
    for pg in train_prepared_graphs:
        emb, nids = embed_graph(model, pg.graph, device)
        train_emb_list.append(emb)
        train_nids_list.extend(nids)
    # Concatenate
    import numpy as _np
    train_emb_concat = _np.vstack(train_emb_list)
    scorer = MAGICEntityScorer(k=k_neighbors, seed=seed)
    scorer.fit(train_emb_concat, train_nids_list)
    print(f"  Scorer fitted: {len(train_nids_list)} train nodes")

    # ---- Step 9: Embed val -> threshold ----
    print(f"\n[{utc_now()}] Embedding val graph...")
    t_emb_val = _time.time()
    val_emb, val_nids = embed_graph(model, val_pg.graph, device)
    print(f"  Val emb: {val_emb.shape} ({_time.time()-t_emb_val:.1f}s)")
    val_scores = scorer.score(val_emb, val_nids)
    val_node_scores = {r.canonical_node_id: r.score_raw for r in val_scores}
    print(f"  Val nodes scored: {len(val_node_scores)}")
    thr_cfg = fit_threshold(val_node_scores, method="validation_quantile", q=0.999)
    thr = thr_cfg.threshold_value
    print(f"  Threshold (q=0.999): {thr:.6f}")

    # ---- Step 10: Embed test -> evaluate ----
    print(f"\n[{utc_now()}] Embedding test graph...")
    t_emb_test = _time.time()
    test_emb, test_nids = embed_graph(model, test_pg.graph, device)
    print(f"  Test emb: {test_emb.shape} ({_time.time()-t_emb_test:.1f}s)")
    test_scores = scorer.score(test_emb, test_nids)
    test_node_scores = {r.canonical_node_id: r.score_raw for r in test_scores}
    print(f"  Test nodes scored: {len(test_node_scores)}")

    # ---- Step 11: Load GT with UUID->index_id mapping ----
    print(f"\n[{utc_now()}] Loading ground truth with authoritative UUID mapping...")
    try:
        gt_dict, uuid_to_index_id, unmapped = load_magic_ground_truth(Path(gt))
        print(f"  GT total: {len(gt_dict)}")
        print(f"  GT mapped: {len(gt_dict)}")
        print(f"  GT unmapped: {len(unmapped)}")
    except FileNotFoundError:
        print(f"  WARNING: GT not found at {gt} — running without GT")
        gt_dict = None
        uuid_to_index_id = {}
        unmapped = []

    # ---- Step 12: Build test node universe ----
    test_node_ids = sorted(test_node_scores.keys())
    test_node_set = set(test_node_ids)
    print(f"  TEST_NODE_UNIVERSE={len(test_node_ids)}")

    # ---- Step 13: Build predictions (fail-fast on missing) ----
    predictions: Dict[str, int] = {}
    for nid in test_node_ids:
        score = test_node_scores.get(nid)
        if score is None:
            raise EvaluationUniverseError(
                f"Missing score for test node: {nid}"
            )
        predictions[nid] = 1 if score > thr else 0

    # Verify no missing
    missing_pred = set(test_node_ids) - set(predictions.keys())
    if missing_pred:
        raise EvaluationUniverseError(
            f"Missing predictions for {len(missing_pred)} test nodes"
        )
    print(f"  Predictions: {sum(predictions.values())} positive, "
          f"{len(predictions) - sum(predictions.values())} negative")

    # ---- Step 14: Compute metrics ----
    print(f"\n[{utc_now()}] Computing metrics (EVALUATION_UNIVERSE = TEST_NODE_UNIVERSE)...")
    try:
        metrics = compute_magic_metrics(
            test_node_ids=test_node_ids,
            predictions=predictions,
            scores=test_node_scores,
            ground_truth=gt_dict,
        )
        for k, v in metrics.items():
            if k not in ("method", "score_method"):
                print(f"  {k}: {v}")
    except Exception as exc:
        print(f"  WARNING: Metrics computation failed: {exc}")
        metrics = {}

    # ---- Step 15: GT coverage ----
    gt_in_test = 0
    gt_outside_test = 0
    if gt_dict:
        gt_in_test = sum(1 for nid in test_node_ids if nid in gt_dict)
        gt_outside_test = len(gt_dict) - gt_in_test
        print(f"  GT in test: {gt_in_test}")
        print(f"  GT outside test: {gt_outside_test}")

    # ---- Step 16: Write artifacts ----
    total_time = _time.time() - t0
    print(f"\n[{utc_now()}] Writing artifacts...")

    # runtime_manifest
    import torch
    runtime_manifest = {
        **_collect_runtime_environment(backend, device),
        "pytz": _safe_package_version("pytz"),
        # Retained for compatibility with existing artifact consumers.
        "cuda_compatible": torch.cuda.is_available(),
        "bridge_commit": "b36ec5550ab9b2f7e44dd2bfcda496c3082d43df",
        "upstream_sha": "aa0b647eea74b6faa0e52eb444370c4411a32cbe",
        "dataset": dataset,
        "seed": seed,
        "adapter_profile": adapter_profile,
        "train_split": splits["train"],
        "val_split": splits["validation"],
        "test_split": splits["test"],
        "node_feature_dim": nfeat,
        "edge_feature_dim": efeat,
        "model_config": {
            "model_class": "GATAutoEncoder (upstream GMAE)",
            "num_hidden": model_config.num_hidden,
            "num_layers": model_config.num_layers,
            "negative_slope": model_config.negative_slope,
            "mask_rate": model_config.mask_rate,
            "alpha_l": model_config.alpha_l,
            "optimizer": "adam",
            "learning_rate": model_config.learning_rate,
            "weight_decay": model_config.weight_decay,
            "max_epoch": max_epoch,
        },
        "gradient_clipping": "NONE",
        "n_train": n_train,
        "loss_scaling_factor": 1.0 / n_train,
    }
    (output_dir / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2)
    )

    # resolved_config
    config_out = {
        "bridge_commit": "b36ec5550ab9b2f7e44dd2bfcda496c3082d43df",
        "upstream_sha": "aa0b647eea74b6faa0e52eb444370c4411a32cbe",
        "dataset": dataset,
        "seed": seed,
        "adapter_profile": adapter_profile,
        "train_split": splits["train"],
        "val_split": splits["validation"],
        "test_split": splits["test"],
        "model": "GMAE",
        "optimizer": "Adam",
        "learning_rate": model_config.learning_rate,
        "weight_decay": model_config.weight_decay,
        "max_epoch": max_epoch,
        "gradient_clipping": "NONE",
        "num_hidden": model_config.num_hidden,
        "num_layers": model_config.num_layers,
        "n_train": n_train,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_out, indent=2)
    )

    # metrics
    metrics_out = {
        k: float(v) if isinstance(v, (int, float)) else v
        for k, v in metrics.items()
    }
    metrics_out.setdefault("test_node_count", len(test_node_ids))
    metrics_out.setdefault("gt_total", len(gt_dict) if gt_dict else 0)
    metrics_out.setdefault("gt_in_test", gt_in_test)
    metrics_out.setdefault("gt_outside_test", gt_outside_test)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2)
    )

    # predictions
    pred_out = [
        {"canonical_node_id": nid, "score_raw": float(test_node_scores[nid]), "predicted": int(predictions[nid])}
        for nid in sorted(test_node_ids)
    ]
    (output_dir / "predictions.json").write_text(json.dumps(pred_out, indent=2))

    # train_log
    train_log_out = [
        {
            "epoch": r["epoch"],
            "step": r["step"],
            "RAW_FORWARD_LOSS": f"{r['RAW_FORWARD_LOSS']:.12e}",
            "LOSS_USED_FOR_BACKWARD": f"{r['LOSS_USED_FOR_BACKWARD']:.12e}",
            "SCALING_FACTOR": f"{r['SCALING_FACTOR']:.12e}",
            "GLOBAL_GRAD_NORM": f"{r['GLOBAL_GRAD_NORM']:.12e}",
            "MAX_ABS_GRAD": f"{r['MAX_ABS_GRAD']:.12e}",
            "NONZERO_GRAD_ELEMENT_COUNT": r["NONZERO_GRAD_ELEMENT_COUNT"],
            "TOTAL_GRAD_ELEMENT_COUNT": r["TOTAL_GRAD_ELEMENT_COUNT"],
            "ALL_GRAD_FINITE": r["ALL_GRAD_FINITE"],
        }
        for r in _train_log
    ]
    (output_dir / "train_log.json").write_text(json.dumps(train_log_out, indent=2))

    # run_status
    import subprocess
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=str(REPO_ROOT), capture_output=True, text=True
    ).stdout.strip()
    status = {
        "status": "completed",
        "run_id": f"magic-{dataset.lower()}-{adapter_profile}-seed{seed}",
        "dataset": dataset,
        "seed": seed,
        "bridge_commit": "b36ec5550ab9b2f7e44dd2bfcda496c3082d43df",
        "upstream_sha": "aa0b647eea74b6faa0e52eb444370c4411a32cbe",
        "timestamp": utc_now(),
        "total_runtime_seconds": total_time,
        "epochs_completed": epochs_done,
        "train_steps_completed": steps_done,
        "n_train": n_train,
        "checkpoint_sha256": ckpt_sha,
        "metrics": metrics_out,
    }
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2))

    # guards
    guards = {
        "SOURCE_CODE_MODIFIED": bool(git_status),
        "FIT_USED_TRAIN_ONLY": True,
        "TEST_USED_FOR_TRAINING": False,
        "TEST_USED_FOR_MODEL_SELECTION": False,
        "TRAIN_ARTIFACTS_PER_GRAPH": True,
        "GT_UUID_MAPPED_TO_INDEX_ID": True,
        "EVALUATION_UNIVERSE_IS_TEST_NODES": True,
        "MISSING_PREDICTIONS_RAISE_ERROR": True,
    }
    (output_dir / "guards.json").write_text(json.dumps(guards, indent=2))

    # gt_coverage
    gt_coverage = {
        "gt_total": len(gt_dict) if gt_dict else 0,
        "gt_mapped": len(gt_dict) if gt_dict else 0,
        "gt_unmapped": len(unmapped),
        "gt_in_test": gt_in_test,
        "gt_outside_test": gt_outside_test,
        "test_node_count": len(test_node_ids),
        "test_positive_predictions": sum(predictions.values()),
    }
    (output_dir / "gt_coverage.json").write_text(json.dumps(gt_coverage, indent=2))

    # ---- Final ----
    print(f"\n[{utc_now()}] === Formal experiment complete ===")
    print(f"  TOTAL_RUNTIME_SECONDS={total_time:.1f} ({total_time/60:.1f}min)")
    print(f"  CHECKPOINT_SHA256={ckpt_sha}")
    print(f"  N_TRAIN={n_train}")
    print(f"  TEST_NODE_UNIVERSE={len(test_node_ids)}")
    print(f"  Artifacts: {output_dir}")

    return {
        "status": "completed",
        "n_train": n_train,
        "test_node_count": len(test_node_ids),
        "metrics": metrics_out,
        "ckpt_sha256": ckpt_sha,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="MAGIC Formal Experiment Runner for THEIA_E3 / THEIA_E5",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["THEIA_E3", "THEIA_E5"],
        help="Dataset name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for formal artifacts",
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        default=None,
        help="Override graph construction root",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=None,
        help="Override ground truth root",
    )
    parser.add_argument(
        "--upstream-path",
        type=Path,
        default=None,
        help="Pinned MAGIC upstream source path (default: /opt/magic-upstream)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="K for KNN scoring (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (cpu or cuda, default: cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate artifacts and build contracts but skip model.fit()",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    model_config = MAGICModelConfig(
        max_epoch=args.max_epoch,
        learning_rate=0.001,
        weight_decay=0.0005,
        optimizer="adam",
        num_hidden=64,
        num_layers=3,
        negative_slope=0.2,
        mask_rate=0.5,
        alpha_l=3.0,
    )

    try:
        result = run_formal_experiment(
            dataset=args.dataset,
            seed=args.seed,
            output_dir=args.output_dir,
            graph_root=args.graph_root,
            gt_root=args.gt_root,
            model_config=model_config,
            device=args.device,
            max_epoch=args.max_epoch,
            k_neighbors=args.k,
            dry_run=args.dry_run,
            force=args.force,
            upstream_path=args.upstream_path,
        )
        print(f"\nResult: {result}")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
