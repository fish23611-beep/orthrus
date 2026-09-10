"""
MAGIC Synthetic Smoke Harness

本模块实现 M8: MAGIC Synthetic End-to-End Smoke Tests。

设计目标：
1. Bounded CPU synthetic E2E test
2. 验证 M3-M7 完整数据流
3. 支持 seed reproducibility
4. 支持 cache/resume identity
5. 输出符合 canonical artifact contract

禁止：
- 真实 THEIA 数据
- 真实 GAT/GMAE backend
- 伪装成 magic_upstream
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

# =============================================================================
# Constants
# =============================================================================

# Magic constants
MAGIC_K_NEIGHBORS = 10
MAGIC_Q = 0.999
MAGIC_BACKEND = "synthetic_smoke"
MAGIC_DATASET = "SYNTHETIC_MAGIC"

# Upstream SHA (frozen, not used in synthetic mode)
UPSTREAM_SHA_FROZEN = "9003a2ec19632073d55a1d689b828272bc53496685175fa4c88dbb63fdf028ad"

# =============================================================================
# Config Identity / Cache Fingerprint
# =============================================================================

def compute_config_fingerprint(
    seed: int,
    fixture_checksum: str,
    k: int = MAGIC_K_NEIGHBORS,
    q: float = MAGIC_Q,
    upstream_sha: str = UPSTREAM_SHA_FROZEN,
    backend: str = MAGIC_BACKEND,
) -> str:
    """
    Compute deterministic config fingerprint for cache identity.

    Args:
        seed: Random seed
        fixture_checksum: Checksum of input fixture
        k: K for KNN
        q: Quantile value
        upstream_sha: Frozen upstream SHA
        backend: Backend identity

    Returns:
        SHA256 fingerprint
    """
    data = {
        "seed": seed,
        "fixture_checksum": fixture_checksum,
        "k": k,
        "q": q,
        "upstream_sha": upstream_sha,
        "backend": backend,
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def fingerprint_from_config(
    config: "SmokeConfig",
) -> str:
    """Compute fingerprint from smoke config."""
    return compute_config_fingerprint(
        seed=config.seed,
        fixture_checksum=config.fixture_checksum,
        k=config.k,
        q=config.q,
        upstream_sha=config.upstream_sha,
        backend=config.backend,
    )


# =============================================================================
# Config
# =============================================================================

@dataclass
class SmokeConfig:
    """
    Configuration for synthetic smoke test.
    """
    seed: int = 0
    k: int = MAGIC_K_NEIGHBORS
    q: float = MAGIC_Q
    output_dir: Optional[Path] = None
    backend: str = MAGIC_BACKEND
    dataset: str = MAGIC_DATASET
    is_smoke: bool = True
    upstream_sha: str = UPSTREAM_SHA_FROZEN
    fixture_checksum: str = ""

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path("/tmp/magic_smoke")

    @property
    def fingerprint(self) -> str:
        """Compute cache fingerprint."""
        return compute_config_fingerprint(
            seed=self.seed,
            fixture_checksum=self.fixture_checksum,
            k=self.k,
            q=self.q,
            upstream_sha=self.upstream_sha,
            backend=self.backend,
        )


# =============================================================================
# Synthetic Backend
# =============================================================================

class SyntheticBackend:
    """
    Synthetic embedding backend for smoke tests.

    This is NOT a real GAT/GMAE backend.
    It generates deterministic embeddings from fixture data.

    Backend identity: synthetic_smoke
    """

    def __init__(
        self,
        seed: int = 0,
        embedding_dim: int = 64,
    ):
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.backend = MAGIC_BACKEND

    def generate(
        self,
        fixture: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[
        Dict[str, List[List[float]]],  # embeddings
        Dict[str, List[str]],           # node_id lists
    ]:
        """
        Generate synthetic embeddings for fixture.

        Args:
            fixture: Synthetic fixture with train/validation/test

        Returns:
            Tuple of (embeddings_dict, node_id_lists_dict)
        """
        from tests.test_magic.fixtures.synthetic_fixture import (
            SyntheticEmbeddingGenerator,
            make_synthetic_embedding,
            make_anomalous_embedding,
        )

        def get_nodes(records):
            nodes = set()
            for r in records:
                nodes.add(r.get("src", ""))
                nodes.add(r.get("dst", ""))
            return sorted(nodes)

        train_nodes = get_nodes(fixture.get("train", []))
        val_nodes = get_nodes(fixture.get("validation", []))
        test_nodes = get_nodes(fixture.get("test", []))

        # Identify anomalous nodes
        anomalous_nodes = {n for n in test_nodes if "malicious" in n}

        # Generate embeddings
        gen = SyntheticEmbeddingGenerator(seed=self.seed, embedding_dim=self.embedding_dim)

        train_emb = gen.generate_embeddings(train_nodes, anomalous=False)
        val_emb = gen.generate_embeddings(val_nodes, anomalous=False)
        test_emb = []
        for n in test_nodes:
            if n in anomalous_nodes:
                emb = make_anomalous_embedding(n, self.embedding_dim, self.seed)
            else:
                emb = make_synthetic_embedding(n, self.embedding_dim, self.seed)
            test_emb.append(emb)

        return {
            "train": train_emb,
            "validation": val_emb,
            "test": test_emb,
        }, {
            "train": train_nodes,
            "validation": val_nodes,
            "test": test_nodes,
        }


# =============================================================================
# E2E Pipeline
# =============================================================================

def run_synthetic_e2e(
    fixture: Dict[str, List[Dict[str, Any]]],
    config: SmokeConfig,
    ground_truth: Optional[Mapping[str, int]] = None,
) -> "SmokeResult":
    """
    Run synthetic E2E smoke test.

    Data flow:
    1. M3: MAGICInputAdapter (canonical -> neutral graph)
    2. M4: Causal snapshots (validation_quantile threshold)
    3. Synthetic backend (generate embeddings)
    4. M5: MAGICEntityScorer (raw scores)
    5. M6: Validation threshold -> test predictions
    6. Metrics (if ground truth provided)
    7. Write artifacts

    Args:
        fixture: Synthetic fixture
        config: Smoke configuration
        ground_truth: Optional ground truth for metrics

    Returns:
        SmokeResult with predictions, scores, metrics
    """
    from src.baselines.magic.input_adapter import MAGICInputAdapter
    from src.baselines.magic.protocol import (
        fit_threshold,
        apply_threshold,
    )
    from src.baselines.magic.scoring import MAGICEntityScorer
    from src.baselines.magic.evaluator import compute_magic_metrics

    # Step 1: M3 - Input Adapter
    adapter = MAGICInputAdapter(
        dataset=config.dataset,
        train_records=fixture.get("train", []),
        val_records=fixture.get("validation", []),
        test_records=fixture.get("test", []),
    )
    train_contract, val_contract, test_contract = adapter.fit_transform()

    # Get all nodes from contracts
    train_nodes = list(train_contract.nodes.keys())
    val_nodes = list(val_contract.nodes.keys())
    test_nodes = list(test_contract.nodes.keys())

    # Step 2: Generate synthetic embeddings
    backend = SyntheticBackend(seed=config.seed)
    embeddings, node_ids = backend.generate(fixture)

    # Step 3: M5 - Fit scorer
    scorer = MAGICEntityScorer(
        k=config.k,
        seed=config.seed,
    )

    train_embeddings = np.array(embeddings["train"])
    train_node_ids = node_ids["train"]

    scorer.fit(train_embeddings, train_node_ids)

    # Step 4: Score validation
    val_embeddings = np.array(embeddings["validation"])
    val_node_ids = node_ids["validation"]

    val_scores_records = scorer.score(val_embeddings, val_node_ids)
    val_node_scores = {
        r.canonical_node_id: r.score_raw
        for r in val_scores_records
    }

    # Step 5: M4 - Fit threshold on validation
    threshold_config = fit_threshold(
        val_node_scores,
        method="validation_quantile",
        q=config.q,
    )

    # Step 6: Score test
    test_embeddings = np.array(embeddings["test"])
    test_node_ids = node_ids["test"]

    test_scores_records = scorer.score(test_embeddings, test_node_ids)
    test_node_scores = {
        r.canonical_node_id: r.score_raw
        for r in test_scores_records
    }

    # Step 7: Apply threshold
    predictions = apply_threshold(test_node_scores, threshold_config)

    # Step 8: Compute metrics (if ground truth provided)
    # FIXED (FORMAL-F1): use TEST_NODE_UNIVERSE as evaluation universe
    metrics = None
    if ground_truth is not None:
        test_node_ids = sorted(test_node_scores.keys())
        metrics = compute_magic_metrics(
            test_node_ids=test_node_ids,
            predictions=predictions,
            scores=test_node_scores,
            ground_truth=ground_truth,
        )

    # Step 9: Write artifacts
    artifact_dir = _write_artifacts(
        fixture=fixture,
        config=config,
        train_contract=train_contract,
        val_contract=val_contract,
        test_contract=test_contract,
        val_node_scores=val_node_scores,
        test_node_scores=test_node_scores,
        predictions=predictions,
        threshold_config=threshold_config,
        metrics=metrics,
        ground_truth=ground_truth,
    )

    return SmokeResult(
        config=config,
        artifact_dir=artifact_dir,
        val_node_scores=val_node_scores,
        test_node_scores=test_node_scores,
        predictions=predictions,
        threshold_config=threshold_config,
        metrics=metrics,
    )


def _write_artifacts(
    fixture: Dict[str, List[Dict[str, Any]]],
    config: SmokeConfig,
    train_contract,  # NeutralGraphContract
    val_contract,
    test_contract,
    val_node_scores: Dict[str, float],
    test_node_scores: Dict[str, float],
    predictions: Dict[str, int],
    threshold_config,  # ThresholdConfig
    metrics: Optional[Dict[str, Any]],
    ground_truth: Optional[Mapping[str, int]],
) -> Path:
    """Write smoke artifacts to disk."""
    output_dir = Path(config.output_dir)
    smoke_id = f"smoke_seed{config.seed}"
    artifact_dir = output_dir / "SYNTHETIC_MAGIC" / "runs" / "magic" / smoke_id

    # Create directories
    raw_dir = artifact_dir / "raw_magic"
    node_scores_dir = artifact_dir / "node_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    node_scores_dir.mkdir(parents=True, exist_ok=True)

    # Write environment.json
    env = {
        "git_commit": "local_smoke_test",
        "backend": config.backend,
        "is_smoke": True,
        "upstream_sha": config.upstream_sha,
    }
    with open(artifact_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    # Write config_resolved.yml (minimal)
    import yaml
    cfg = {
        "is_smoke": True,
        "backend": config.backend,
        "dataset": config.dataset,
        "seed": config.seed,
        "k": config.k,
        "q": config.q,
    }
    with open(artifact_dir / "config_resolved.yml", "w") as f:
        yaml.dump(cfg, f)

    # Write runtime.json
    runtime = {
        "is_smoke": True,
        "backend": config.backend,
        "dataset": config.dataset,
        "seed": config.seed,
    }
    with open(artifact_dir / "runtime.json", "w") as f:
        json.dump(runtime, f, indent=2)

    # Write graph_manifest.json
    graph_manifest = {
        "train_node_count": train_contract.train_node_count,
        "val_node_count": val_contract.val_node_count,
        "test_node_count": test_contract.test_node_count,
        "train_edge_count": train_contract.train_edge_count,
        "val_edge_count": val_contract.val_edge_count,
        "test_edge_count": test_contract.test_edge_count,
    }
    with open(raw_dir / "graph_manifest.json", "w") as f:
        json.dump(graph_manifest, f, indent=2)

    # Write local_global_node_map.csv
    all_nodes = {}
    for node_id, node in train_contract.nodes.items():
        all_nodes[node_id] = node.node_type
    for node_id, node in val_contract.nodes.items():
        if node_id not in all_nodes:
            all_nodes[node_id] = node.node_type
    for node_id, node in test_contract.nodes.items():
        if node_id not in all_nodes:
            all_nodes[node_id] = node.node_type

    with open(raw_dir / "local_global_node_map.csv", "w") as f:
        f.write("canonical_node_id,node_type,local_index\n")
        for i, (node_id, node_type) in enumerate(sorted(all_nodes.items())):
            f.write(f"{node_id},{node_type},{i}\n")

    # Write validation scores (NO ground truth)
    val_scores_lines = ["canonical_node_id,score_raw"]
    for node_id in sorted(val_node_scores.keys()):
        val_scores_lines.append(f"{node_id},{val_node_scores[node_id]:.6f}")
    with open(raw_dir / "validation_node_window_scores.csv", "w") as f:
        f.write("\n".join(val_scores_lines) + "\n")

    # Write test scores (NO ground truth in raw)
    test_scores_lines = ["canonical_node_id,score_raw"]
    for node_id in sorted(test_node_scores.keys()):
        test_scores_lines.append(f"{node_id},{test_node_scores[node_id]:.6f}")
    with open(raw_dir / "test_node_window_scores.csv", "w") as f:
        f.write("\n".join(test_scores_lines) + "\n")

    # Write knn_reference_manifest.json
    knn_ref = {
        "k": config.k,
        "backend": config.backend,
        "is_smoke": True,
    }
    with open(raw_dir / "knn_reference_manifest.json", "w") as f:
        json.dump(knn_ref, f, indent=2)

    # Write threshold.json
    threshold_json = {
        "method": threshold_config.method,
        "q": threshold_config.q,
        "threshold_value": threshold_config.threshold_value,
        "validation_score_count": threshold_config.validation_score_count,
        "provenance": threshold_config.provenance,
    }
    with open(raw_dir / "threshold.json", "w") as f:
        json.dump(threshold_json, f, indent=2)

    # Write node_predictions.csv (may include y_true in final artifact)
    pred_lines = ["canonical_node_id,node_type,score_raw,threshold,prediction"]
    node_types = {n.node_id: n.node_type for n in list(train_contract.nodes.values()) +
                  list(val_contract.nodes.values()) + list(test_contract.nodes.values())}
    for node_id in sorted(predictions.keys()):
        node_type = node_types.get(node_id, "")
        score = test_node_scores.get(node_id, 0.0)
        pred = predictions.get(node_id, 0)
        pred_lines.append(f"{node_id},{node_type},{score:.6f},{threshold_config.threshold_value},{pred}")
    with open(node_scores_dir / "node_predictions.csv", "w") as f:
        f.write("\n".join(pred_lines) + "\n")

    # Write metrics.json
    if metrics:
        metrics_to_write = {
            "method": "MAGIC",
            "score_method": "knn_distance_ratio",
            "k": config.k,
            "merge_method": "max",
            "threshold_method": "validation_quantile",
            "threshold_quantile": config.q,
            "threshold": threshold_config.threshold_value,
            "threshold_provenance": threshold_config.provenance,
            "validation_score_count": threshold_config.validation_score_count,
            "is_smoke": True,
            "backend": config.backend,
            "dataset": config.dataset,
        }
        metrics_to_write.update(metrics)
        with open(node_scores_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_write, f, indent=2)
    else:
        # Still write minimal metrics without ground truth dependent fields
        metrics_to_write = {
            "method": "MAGIC",
            "score_method": "knn_distance_ratio",
            "k": config.k,
            "threshold_method": "validation_quantile",
            "threshold_quantile": config.q,
            "threshold": threshold_config.threshold_value,
            "threshold_provenance": threshold_config.provenance,
            "is_smoke": True,
            "backend": config.backend,
            "dataset": config.dataset,
        }
        with open(node_scores_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_write, f, indent=2)

    return artifact_dir


# =============================================================================
# Result
# =============================================================================

@dataclass
class SmokeResult:
    """
    Result from synthetic E2E smoke test.
    """
    config: SmokeConfig
    artifact_dir: Path
    val_node_scores: Dict[str, float]
    test_node_scores: Dict[str, float]
    predictions: Dict[str, int]
    threshold_config: "ThresholdConfig"  # From contracts.py
    metrics: Optional[Dict[str, Any]] = None

    @property
    def threshold_value(self) -> float:
        return self.threshold_config.threshold_value

    @property
    def test_node_count(self) -> int:
        return len(self.test_node_scores)

    @property
    def predicted_positive_count(self) -> int:
        return sum(self.predictions.values())

    def get_score_checksum(self) -> str:
        """Compute checksum of test scores for reproducibility verification."""
        data = {
            "seed": self.config.seed,
            "scores": {k: float(v) for k, v in sorted(self.test_node_scores.items())},
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_prediction_checksum(self) -> str:
        """Compute checksum of predictions."""
        data = {
            "seed": self.config.seed,
            "predictions": {k: int(v) for k, v in sorted(self.predictions.items())},
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# =============================================================================
# Cache Verification
# =============================================================================

def verify_cache_identity(
    config_a: SmokeConfig,
    config_b: SmokeConfig,
) -> bool:
    """
    Verify if two configs have the same cache identity.

    Args:
        config_a: First config
        config_b: Second config

    Returns:
        True if same cache identity
    """
    return config_a.fingerprint == config_b.fingerprint


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MAGIC_K_NEIGHBORS",
    "MAGIC_Q",
    "MAGIC_BACKEND",
    "MAGIC_DATASET",
    "UPSTREAM_SHA_FROZEN",
    "SmokeConfig",
    "SyntheticBackend",
    "run_synthetic_e2e",
    "SmokeResult",
    "compute_config_fingerprint",
    "fingerprint_from_config",
    "verify_cache_identity",
]
