"""Generate the deterministic ORTHRUS baseline compatibility fixture.

Run this script in a separate process with ``--source-root`` pointing at the
checked-out baseline commit.  It uses no database, network, W&B, or real data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, TemporalData


GENERATOR_VERSION = "c1-baseline-v1"
INPUT_SUMMARY = {
    "topology": "three six-edge directed rings (train, val, test)",
    "num_nodes": 6,
    "num_edge_types": 2,
    "node_feature_dim": 4,
    "model_seed": 314159,
    "data_seeds": {"train": 101, "val": 202, "test": 303},
    "dropout": 0.0,
    "node_aggregation": "maximum incident test-edge score",
    "prediction_threshold": "maximum validation-edge score",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_ring(seed: int, time_offset: int, edge_type_offset: int) -> TemporalData:
    _seed_everything(seed)
    num_nodes = INPUT_SUMMARY["num_nodes"]
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)
    t = time_offset + torch.arange(1, num_nodes + 1, dtype=torch.long)
    x_src = torch.randn(num_nodes, INPUT_SUMMARY["node_feature_dim"])
    x_dst = torch.randn(num_nodes, INPUT_SUMMARY["node_feature_dim"])
    edge_classes = (torch.arange(num_nodes) + edge_type_offset) % 2
    edge_type = F.one_hot(edge_classes, num_classes=2).float()
    data = TemporalData(
        src=src,
        dst=dst,
        t=t,
        msg=torch.cat([x_src, x_dst], dim=-1),
        edge_type=edge_type,
    )
    data.x_src = x_src.clone()
    data.x_dst = x_dst.clone()
    data.edge_index = torch.stack([src, dst])
    return data


def _build_model(source_root: Path):
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from data_utils import GraphReindexer
    from decoders import EdgeTypeDecoder
    from encoders import GraphTransformer, OrthrusEncoder
    from model import Orthrus
    from temporal import LastNeighborLoader

    _seed_everything(INPUT_SUMMARY["model_seed"])
    device = torch.device("cpu")
    num_nodes = INPUT_SUMMARY["num_nodes"]
    graph_reindexer = GraphReindexer(num_nodes=num_nodes, device=device)
    graph_encoder = GraphTransformer(
        in_dim=4,
        hid_dim=8,
        out_dim=8,
        edge_dim=2,
        dropout=0.0,
        activation=torch.nn.ReLU(),
        num_heads=2,
    )
    neighbor_loader = LastNeighborLoader(
        num_nodes=num_nodes, size=5, device=device
    )
    encoder = OrthrusEncoder(
        encoder=graph_encoder,
        neighbor_loader=neighbor_loader,
        in_dim=4,
        temporal_dim=4,
        use_node_feats_in_gnn=True,
        graph_reindexer=graph_reindexer,
        edge_features=["edge_type"],
        device=device,
        num_nodes=num_nodes,
        edge_dim=2,
    )

    def cross_entropy(logits, targets, inference=False, **kwargs):
        reduction = "none" if inference else "mean"
        return F.cross_entropy(logits, targets, reduction=reduction)

    decoder = EdgeTypeDecoder(
        in_dim=8,
        num_edge_types=2,
        loss_fn=cross_entropy,
        dropout=0.0,
        num_layers=2,
        activation=torch.nn.ReLU(),
    )
    return Orthrus(
        encoder=encoder,
        decoders=[decoder],
        num_nodes=num_nodes,
        in_dim=4,
        out_dim=8,
        use_contrastive_learning=False,
        device=device,
        graph_reindexer=graph_reindexer,
    ).to(device)


def compute_outputs(source_root: Path) -> dict:
    """Execute baseline model loss, replay/inference, and node aggregation."""
    source_root = Path(source_root)
    train = _make_ring(INPUT_SUMMARY["data_seeds"]["train"], 100, 0)
    val = _make_ring(INPUT_SUMMARY["data_seeds"]["val"], 200, 1)
    test = _make_ring(INPUT_SUMMARY["data_seeds"]["test"], 300, 0)
    full_data = Data(
        msg=torch.cat([train.msg, val.msg, test.msg]),
        t=torch.cat([train.t, val.t, test.t]),
        edge_type=torch.cat([train.edge_type, val.edge_type, test.edge_type]),
    )

    training_model = _build_model(source_root)
    training_model.train()
    edge_level_loss = training_model(train.clone(), full_data)

    inference_model = _build_model(source_root)
    inference_model.eval()
    inference_model.encoder.reset_state()

    def infer(graph):
        with torch.no_grad():
            return inference_model(graph.clone(), full_data, inference=True).cpu()

    replay_scores = infer(train)
    val_scores = infer(val)
    test_scores = infer(test)
    threshold = float(val_scores.max())

    node_to_scores = defaultdict(list)
    for src, dst, score in zip(test.src.tolist(), test.dst.tolist(), test_scores.tolist()):
        node_to_scores[src].append(score)
        node_to_scores[dst].append(score)
    node_scores = {
        str(node): float(max(scores)) for node, scores in sorted(node_to_scores.items())
    }
    node_predictions = {
        node: int(score > threshold) for node, score in node_scores.items()
    }

    return {
        "edge_level_loss": float(edge_level_loss.detach().cpu()),
        "replay_train_scores": replay_scores.tolist(),
        "validation_scores": val_scores.tolist(),
        "inference_scores": test_scores.tolist(),
        "node_scores": node_scores,
        "prediction_threshold": threshold,
        "edge_predictions": (test_scores > threshold).to(torch.int64).tolist(),
        "node_predictions": node_predictions,
        "shapes": {
            "edge_level_loss": list(edge_level_loss.shape),
            "replay_train_scores": list(replay_scores.shape),
            "validation_scores": list(val_scores.shape),
            "inference_scores": list(test_scores.shape),
        },
        "stability": {
            "inference_sum": float(test_scores.sum()),
            "inference_mean": float(test_scores.mean()),
            "node_score_sum": float(sum(node_scores.values())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    args = parser.parse_args()

    payload = {
        "baseline_commit": args.baseline_commit,
        "generator_version": GENERATOR_VERSION,
        "input_summary": INPUT_SUMMARY,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "outputs": compute_outputs(args.source_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
