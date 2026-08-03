"""Smoke tests for the production training loop and testing dispatcher.

The training test uses a tiny model rather than the full ORTHRUS model. It
exercises production batching, backward propagation, optimizer stepping, and
checkpoint persistence. The testing test is deliberately an orchestration
test: the edge-scoring function records calls and performs no file output.
"""

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, TemporalData


@pytest.fixture(scope="module")
def detection_modules():
    """Import production modules with reversible path and network patches."""
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "path", [src_dir, *sys.path]))
        stack.enter_context(patch("nltk.download", return_value=True))
        from detection import orthrus_gnn_testing, orthrus_gnn_training

        yield orthrus_gnn_training, orthrus_gnn_testing


def _make_temporal_data(seed: int) -> TemporalData:
    """Return a small, fully populated TemporalData with four events."""
    generator = torch.Generator().manual_seed(seed)
    num_edges = 4
    src = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    dst = torch.tensor([1, 2, 0, 2], dtype=torch.long)
    edge_type = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 0, 1]), num_classes=2
    ).float()
    x_src = torch.rand((num_edges, 3), generator=generator)
    x_dst = torch.rand((num_edges, 3), generator=generator)
    msg = torch.cat((x_src, x_dst), dim=1)

    return TemporalData(
        src=src,
        dst=dst,
        t=torch.arange(1, num_edges + 1, dtype=torch.long),
        msg=msg,
        edge_type=edge_type,
        edge_feats=edge_type.clone(),
        x_src=x_src,
        x_dst=x_dst,
        node_type=torch.eye(3),
    )


class _TinyTrainingModel(nn.Module):
    """Minimal model satisfying the production training-loop interface."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.encoder = nn.Identity()
        self.graph_reindexer = None
        self.batch_losses = []
        self.backward_calls = 0
        self.weight.register_hook(self._record_backward)

    def _record_backward(self, gradient):
        self.backward_calls += 1
        return gradient

    def forward(self, batch, full_data):
        del full_data
        prediction = self.weight * batch.msg.float().mean()
        loss = (prediction - 1.0).square()
        self.batch_losses.append(float(loss.detach()))
        return loss


def test_training_main_runs_batches_and_saves_each_epoch_checkpoint(
    tmp_path, detection_modules
):
    """Exercise production batching/backward/step and per-epoch saving."""
    orthrus_gnn_training, _ = detection_modules
    epochs = 3
    train_data = [_make_temporal_data(seed=0)]
    full_data = Data(
        msg=train_data[0].msg,
        t=train_data[0].t,
        edge_type=train_data[0].edge_type,
    )
    model = _TinyTrainingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoints_dir = tmp_path / "checkpoints"
    cfg = SimpleNamespace(
        _from_weights=False,
        _test_mode=False,
        detection=SimpleNamespace(
            gnn_training=SimpleNamespace(
                _trained_models_dir=str(checkpoints_dir),
                num_epochs=epochs,
                encoder=SimpleNamespace(batch_size=8),
            )
        ),
    )

    with ExitStack() as stack:
        optimizer_step = stack.enter_context(
            patch.object(optimizer, "step", wraps=optimizer.step)
        )
        load_datasets = stack.enter_context(
            patch.object(
                orthrus_gnn_training,
                "load_all_datasets",
                return_value=(train_data, [], [], full_data, 3),
            )
        )
        build_model = stack.enter_context(
            patch.object(orthrus_gnn_training, "build_model", return_value=model)
        )
        optimizer_factory = stack.enter_context(
            patch.object(
                orthrus_gnn_training,
                "optimizer_factory",
                return_value=optimizer,
            )
        )
        stack.enter_context(
            patch.object(
                orthrus_gnn_training,
                "get_device",
                return_value=torch.device("cpu"),
            )
        )
        stack.enter_context(patch.object(orthrus_gnn_training.wandb, "log"))
        stack.enter_context(patch.object(orthrus_gnn_training, "log"))
        cuda_memory = stack.enter_context(
            patch.object(
                orthrus_gnn_training.torch.cuda,
                "max_memory_allocated",
                return_value=0,
            )
        )

        orthrus_gnn_training.main(cfg)

    assert load_datasets.call_count == 1
    assert build_model.call_count == 1
    assert optimizer_factory.call_count == 1
    assert cuda_memory.call_count == epochs
    assert len(model.batch_losses) == epochs
    assert bool(torch.isfinite(torch.tensor(model.batch_losses)).all())
    assert model.backward_calls == epochs
    assert optimizer_step.call_count == epochs

    expected_checkpoints = [f"model_epoch_{epoch}" for epoch in range(1, epochs + 1)]
    assert sorted(path.name for path in checkpoints_dir.iterdir()) == expected_checkpoints
    for checkpoint in expected_checkpoints:
        state_file = checkpoints_dir / checkpoint / "state_dict.pkl"
        assert state_file.is_file(), f"Missing model state: {state_file}"


class _SyntheticGraph:
    """Identity-distinguishable graph object for dispatcher testing."""

    def __init__(self, name):
        self.name = name
        self.device_moves = []

    def to(self, device=None):
        self.device_moves.append(device)
        return self


def test_testing_main_dispatches_val_then_test_for_checkpoint(
    tmp_path, detection_modules
):
    """Verify production main dispatches val then test for one checkpoint."""
    _, orthrus_gnn_testing = detection_modules
    checkpoint = "model_epoch_7"
    models_dir = tmp_path / "checkpoints"
    edge_scores_dir = tmp_path / "edge_scores"
    val_graph = _SyntheticGraph("val")
    test_graph = _SyntheticGraph("test")
    full_data = object()
    uninitialized_model = object()
    loaded_model = object()
    calls = []
    cfg = SimpleNamespace(
        _from_weights=False,
        detection=SimpleNamespace(
            gnn_training=SimpleNamespace(_trained_models_dir=str(models_dir)),
            gnn_testing=SimpleNamespace(_edge_losses_dir=str(edge_scores_dir)),
        ),
    )

    def record_test(**kwargs):
        calls.append(
            {
                **kwargs,
                "output_dir": Path(kwargs["cfg"].detection.gnn_testing._edge_losses_dir)
                / kwargs["split"]
                / kwargs["model_epoch_file"],
            }
        )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "init_database_connection",
                return_value=(object(), object()),
            )
        )
        stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "gen_nodeid2msg",
                return_value={0: "node-0"},
            )
        )
        stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "load_all_datasets",
                return_value=([], [val_graph], [test_graph], full_data, 3),
            )
        )
        list_models = stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "listdir_sorted",
                return_value=[checkpoint],
            )
        )
        stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "build_model",
                return_value=uninitialized_model,
            )
        )
        load_model = stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "load_model",
                return_value=loaded_model,
            )
        )
        replay = stack.enter_context(
            patch.object(orthrus_gnn_testing, "_replay_train_history")
        )
        stack.enter_context(
            patch.object(orthrus_gnn_testing, "test", side_effect=record_test)
        )
        stack.enter_context(
            patch.object(
                orthrus_gnn_testing,
                "get_device",
                return_value=torch.device("cpu"),
            )
        )
        stack.enter_context(patch.object(orthrus_gnn_testing, "log"))
        empty_cache = stack.enter_context(
            patch.object(orthrus_gnn_testing.torch.cuda, "empty_cache")
        )

        orthrus_gnn_testing.main(cfg)

    list_models.assert_called_once_with(str(models_dir))
    load_model.assert_called_once_with(
        uninitialized_model, os.path.join(str(models_dir), checkpoint)
    )
    assert replay.call_count == 1
    assert empty_cache.call_count == 1
    assert len(calls) == 2
    assert [call["split"] for call in calls] == ["val", "test"]
    assert calls[0]["data"] is val_graph
    assert calls[1]["data"] is test_graph
    assert calls[0]["data"] is not calls[1]["data"]
    assert calls[0]["output_dir"] == edge_scores_dir / "val" / checkpoint
    assert calls[1]["output_dir"] == edge_scores_dir / "test" / checkpoint
    assert calls[0]["output_dir"] != calls[1]["output_dir"]
    assert all(call["model_epoch_file"] == checkpoint for call in calls)
    assert all(call["model"] is loaded_model for call in calls)
    assert all(call["full_data"] is full_data for call in calls)
    assert not edge_scores_dir.exists()
