"""
MAGIC Synthetic Smoke Test CLI

Usage:
    python src/baselines/magic/run_magic_smoke.py --seed 0
    python src/baselines/magic/run_magic_smoke.py --seed 0 --seed 1
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.test_magic.fixtures.synthetic_fixture import (
    generate_synthetic_fixture,
    generate_ground_truth,
    compute_fixture_checksum,
    MIN_TRAIN_NODES,
)
from src.baselines.magic.smoke import (
    SmokeConfig,
    run_synthetic_e2e,
    MAGIC_K_NEIGHBORS,
    MAGIC_Q,
    MAGIC_BACKEND,
    MAGIC_DATASET,
    UPSTREAM_SHA_FROZEN,
)


def run_smoke(
    seed: int,
    output_dir: Optional[Path] = None,
    train_node_count: int = MIN_TRAIN_NODES,
) -> Path:
    """
    Run a single synthetic smoke test.

    Args:
        seed: Random seed
        output_dir: Output directory (default: temp)
        train_node_count: Number of train nodes

    Returns:
        Path to artifact directory
    """
    # Generate fixture
    fixture = generate_synthetic_fixture(
        seed=seed,
        train_node_count=train_node_count,
    )

    # Compute checksum
    fixture_checksum = compute_fixture_checksum(fixture)

    # Generate ground truth
    ground_truth = generate_ground_truth(fixture)

    # Create config
    config = SmokeConfig(
        seed=seed,
        k=MAGIC_K_NEIGHBORS,
        q=MAGIC_Q,
        output_dir=output_dir or Path(tempfile.mkdtemp()),
        backend=MAGIC_BACKEND,
        dataset=MAGIC_DATASET,
        is_smoke=True,
        upstream_sha=UPSTREAM_SHA_FROZEN,
        fixture_checksum=fixture_checksum,
    )

    # Run E2E
    result = run_synthetic_e2e(
        fixture=fixture,
        config=config,
        ground_truth=ground_truth,
    )

    print(f"Smoke test completed:")
    print(f"  Seed: {seed}")
    print(f"  Artifact dir: {result.artifact_dir}")
    print(f"  Test nodes: {result.test_node_count}")
    print(f"  Predicted positive: {result.predicted_positive_count}")
    print(f"  Threshold: {result.threshold_value:.6f}")

    if result.metrics:
        print(f"  Metrics:")
        for key, value in result.metrics.items():
            if key not in ('method', 'score_method'):
                print(f"    {key}: {value}")

    return result.artifact_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run MAGIC synthetic smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="Random seed(s) for smoke test (default: [0])",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts (default: temp dir)",
    )
    parser.add_argument(
        "--train-nodes",
        type=int,
        default=MIN_TRAIN_NODES,
        help=f"Number of train nodes (default: {MIN_TRAIN_NODES})",
    )

    args = parser.parse_args(argv)

    print(f"MAGIC Synthetic Smoke Test")
    print(f"  Backend: {MAGIC_BACKEND}")
    print(f"  Dataset: {MAGIC_DATASET}")
    print(f"  K: {MAGIC_K_NEIGHBORS}")
    print(f"  Q: {MAGIC_Q}")
    print()

    for seed in args.seed:
        print(f"Running smoke test with seed={seed}...")
        artifact_dir = run_smoke(
            seed=seed,
            output_dir=args.output_dir,
            train_node_count=args.train_nodes,
        )
        print()

    print("All smoke tests completed successfully.")


if __name__ == "__main__":
    main()
