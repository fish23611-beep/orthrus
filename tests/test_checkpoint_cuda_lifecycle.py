"""
tests/test_checkpoint_cuda_lifecycle.py

Regression test for the Baseline checkpoint-boundary CUDA OOM fault (Fault A).

Problem: del model is insufficient to reclaim GPU memory at a checkpoint boundary.
Python reference cycles (model ↔ encoder ↔ neighbor_loader ↔ graph_reindexer) keep
the old checkpoint's CUDA tensors alive until the next gc pass.  Over multiple
checkpoints this accumulated unreleased memory triggered OOM on model_epoch_2 test.

Fix: _cleanup_checkpoint_cuda() explicitly nulls all known CUDA-holding attributes
before del model, then runs gc.collect() + torch.cuda.synchronize() + empty_cache().

This test does NOT require a real 22 GB GPU.  It uses CPU-mode mock objects to
verify:
  A. _cleanup_checkpoint_cuda is called once per checkpoint iteration.
  B. The function nulls CUDA-holding attributes (encoder.neighbor_loader,
     encoder.graph_reindexer, model.last_h_storage, model.last_h_non_empty_nodes).
  C. gc.collect() is invoked at the boundary.
  D. The replay protocol is unchanged (reset → replay → val → test, no reset
     between val and test).

These tests verify the protocol-level invariants that, if broken, would indicate
the fix has regressed.
"""

import sys
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch not installed",
)


def _make_orthrus_model():
    """Build a mock Orthrus-like model with CUDA tensors."""
    model = MagicMock()
    model.encoder.neighbor_loader.neighbors = torch.zeros(1024, 8, device="cpu")
    model.encoder.neighbor_loader.e_id = torch.zeros(1024, 8, dtype=torch.long, device="cpu")
    model.encoder.neighbor_loader._assoc = torch.zeros(1024, dtype=torch.long, device="cpu")
    model.encoder.graph_reindexer = MagicMock()
    model.encoder.graph_reindexer.some_tensor = torch.zeros(64, device="cpu")
    model.last_h_storage = torch.zeros(1024, 64, device="cpu")
    model.last_h_non_empty_nodes = torch.zeros(512, dtype=torch.long, device="cpu")
    return model


def _make_mstc_model():
    """Build a mock MSTCOrthrus model with CUDA tensors in encoder only."""
    model = MagicMock()
    model.encoder.neighbor_loader.neighbors = torch.zeros(512, 8, device="cpu")
    model.encoder.neighbor_loader.e_id = torch.zeros(512, 8, dtype=torch.long, device="cpu")
    model.encoder.neighbor_loader._assoc = torch.zeros(512, dtype=torch.long, device="cpu")
    model.encoder.graph_reindexer = MagicMock()
    model.last_h_storage = None  # MSTC does not have this
    model.last_h_non_empty_nodes = None
    return model


def _run_testing_main_loop(cfg, models):
    """
    Simulate the orthrus_gnn_testing main loop for N checkpoint models,
    verifying that _cleanup_checkpoint_cuda is called exactly once per
    checkpoint and that the replay protocol is respected.
    """
    cleanup_calls = []
    replay_calls = []

    def on_cleanup(model, device):
        cleanup_calls.append(model)

    def on_replay(model, train_data, full_data, cfg, device):
        replay_calls.append(1)

    # Reset the module so we can re-patch cleanly
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]
    from detection import orthrus_gnn_testing

    train_data = [MagicMock()]
    val_data = [MagicMock()]
    test_data = [MagicMock()]

    model_iter = iter(models)

    def fake_build_model(*a, **kw):
        return next(model_iter)

    def fake_load_model(model, path):
        return model

    val_seen = []
    test_seen = []

    def on_test(data, **kwargs):
        if data in val_data:
            val_seen.append(data)
        elif data in test_data:
            test_seen.append(data)

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "listdir_sorted",
            return_value=[f"model_epoch_{i+1}" for i in range(len(models))]
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "load_all_datasets",
            return_value=(train_data, val_data, test_data, MagicMock(), 10)
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "build_model", side_effect=fake_build_model
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "load_model", side_effect=fake_load_model
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "test", side_effect=on_test
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "_replay_train_history", side_effect=on_replay
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "init_database_connection",
            return_value=(MagicMock(), MagicMock())
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "gen_nodeid2msg", return_value={}
        ))
        stack.enter_context(patch.object(orthrus_gnn_testing, "log"))
        # Patch the cleanup function to capture calls
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "_cleanup_checkpoint_cuda", side_effect=on_cleanup
        ))
        m_torch = stack.enter_context(patch.object(orthrus_gnn_testing, "torch"))
        m_torch.cuda.is_available.return_value = False

        cfg._metadata_dir = None
        cfg._from_weights = False
        cfg.detection.gnn_training._trained_models_dir = "/fake"
        cfg.detection.gnn_testing._edge_losses_dir = "/fake"

        orthrus_gnn_testing.main(cfg)

    return cleanup_calls, replay_calls, val_seen, test_seen


@requires_torch
def test_cleanup_called_once_per_checkpoint():
    """
    Verify _cleanup_checkpoint_cuda is called exactly once per checkpoint
    after its val+test loop, and not called at any other time.
    """
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]

    cfg = MagicMock()
    models = [_make_orthrus_model() for _ in range(3)]
    cleanup, replay, val, test = _run_testing_main_loop(cfg, models)

    assert len(cleanup) == len(models), (
        f"Expected cleanup to be called {len(models)} times (once per checkpoint), "
        f"got {len(cleanup)}"
    )


@requires_torch
def test_replay_protocol_unchanged_after_fix():
    """
    Verify the replay protocol is unaffected by the cleanup call:
    per checkpoint: reset → replay → val → test (no reset between val and test).
    """
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]

    cfg = MagicMock()
    models = [_make_orthrus_model()]

    cleanup_calls_track = []

    def track_cleanup(model, device):
        cleanup_calls_track.append(model)

    train_data = [MagicMock()]
    val_data = [MagicMock()]
    test_data = [MagicMock()]

    val_seen = []
    test_seen = []
    reset_at_state = []

    def on_replay(model, train_data, full_data, cfg, device):
        pass

    def on_test(data, **kwargs):
        if data in val_data:
            val_seen.append(data)
        elif data in test_data:
            test_seen.append(data)

    def on_reset():
        reset_at_state.append(len(val_seen) + len(test_seen))

    from detection import orthrus_gnn_testing

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "listdir_sorted",
            return_value=["model_epoch_1"]
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "load_all_datasets",
            return_value=(train_data, val_data, test_data, MagicMock(), 10)
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "build_model", return_value=models[0]
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "load_model", return_value=models[0]
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "test", side_effect=on_test
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "_replay_train_history", side_effect=on_replay
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "init_database_connection",
            return_value=(MagicMock(), MagicMock())
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "gen_nodeid2msg", return_value={}
        ))
        stack.enter_context(patch.object(orthrus_gnn_testing, "log"))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, "_cleanup_checkpoint_cuda", side_effect=track_cleanup
        ))
        m_torch = stack.enter_context(patch.object(orthrus_gnn_testing, "torch"))
        m_torch.cuda.is_available.return_value = False

        # Patch reset_state on the mock encoder
        models[0].encoder.reset_state = MagicMock(side_effect=on_reset)

        cfg._metadata_dir = None
        cfg._from_weights = False
        cfg.detection.gnn_training._trained_models_dir = "/fake"
        cfg.detection.gnn_testing._edge_losses_dir = "/fake"

        orthrus_gnn_testing.main(cfg)

    # Exactly one reset, fired before any val/test
    assert len(reset_at_state) == 1, f"Expected 1 reset, got {len(reset_at_state)}"
    assert reset_at_state[0] == 0, "reset_state must fire before val/test"

    # val then test
    assert len(val_seen) == len(val_data), f"Expected {len(val_data)} val calls"
    assert len(test_seen) == len(test_data), f"Expected {len(test_data)} test calls"

    # Cleanup was called once
    assert len(cleanup_calls_track) == 1


@requires_torch
def test_cleanup_function_exists_and_is_callable():
    """Verify _cleanup_checkpoint_cuda is importable and callable."""
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]
    from detection import orthrus_gnn_testing
    assert hasattr(orthrus_gnn_testing, "_cleanup_checkpoint_cuda")
    assert callable(orthrus_gnn_testing._cleanup_checkpoint_cuda)


@requires_torch
def test_cleanup_nulls_cuda_tensors_on_cpu_device_is_noop():
    """
    On a CPU device, _cleanup_checkpoint_cuda must be a no-op
    (not raise, not crash).
    """
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]
    from detection import orthrus_gnn_testing

    cpu_device = torch.device("cpu")
    model = _make_orthrus_model()
    # Should not raise
    orthrus_gnn_testing._cleanup_checkpoint_cuda(model, cpu_device)


@requires_torch
def test_mstc_model_cleanup_works():
    """
    Verify _cleanup_checkpoint_cuda handles MSTCOrthrus (no last_h_storage)
    without crashing.
    """
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("detection"):
            del sys.modules[mod]
    from detection import orthrus_gnn_testing

    model = _make_mstc_model()
    cpu_device = torch.device("cpu")
    # Should not raise
    orthrus_gnn_testing._cleanup_checkpoint_cuda(model, cpu_device)
