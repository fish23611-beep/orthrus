"""
tests/test_test_replay_protocol.py

Unit tests for the test-phase replay protocol in orthrus_gnn_testing.py.

Correct protocol per checkpoint:
    1. reset_state()  — clears stale history
    2. replay train   — rebuilds history from scratch
    3. val (no reset) — sees full train history
    4. test (no reset) — sees train + val history

Forbidden:
  - reset/replay between val and test
"""
import sys
import os
from contextlib import ExitStack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, call, patch


torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch not installed",
)

if torch_available:
    from detection import orthrus_gnn_testing


def _make_fake_model(encoder_has_reset=True):
    """Return a mock model whose encoder optionally has reset_state."""
    model = MagicMock()
    if encoder_has_reset:
        model.encoder.reset_state = MagicMock()
    else:
        del model.encoder  # accessing .encoder raises AttributeError
    model.graph_reindexer = MagicMock()
    return model


def _run_with_mocks(orthrus_gnn_testing, cfg, *, fake_model,
                    fake_train_data, fake_val_data, fake_test_data,
                    on_reset=None, on_replay=None, on_test=None):
    """
    Shared helper: patches every external dependency in orthrus_gnn_testing.main
    and runs it with a fake model and fake datasets.

    All exit-stack patches are auto-cleaned when the function returns.
    """
    def fake_load(cfg, required_splits=None):
        assert required_splits == ("train", "val", "test")
        return fake_train_data, fake_val_data, fake_test_data, MagicMock(), 10

    def mock_test(data, **kwargs):
        if on_test:
            on_test(data, **kwargs)

    fake_cursor = MagicMock(name="db_cursor")
    fake_conn   = MagicMock(name="db_connection")

    fake_reset = MagicMock(side_effect=on_reset) if on_reset else MagicMock()
    try:
        fake_model.encoder.reset_state = fake_reset
    except AttributeError:
        pass  # model has no encoder (e.g. non-Orthrus model) — skip

    with ExitStack() as stack:
        # listdir_sorted is imported into orthrus_gnn_testing via `from provnet_utils import *`
        # → must patch where it is USED, not where it is defined
        stack.enter_context(patch(
            "detection.orthrus_gnn_testing.listdir_sorted",
            return_value=["model_epoch_1"]
        ))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'load_all_datasets',    fake_load))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'build_model',          return_value=fake_model))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'load_model',           return_value=fake_model))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'test',                 side_effect=mock_test))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, '_replay_train_history',
            side_effect=on_replay if on_replay else lambda *a, **kw: None))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'init_database_connection',
            return_value=(fake_cursor, fake_conn)))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'gen_nodeid2msg',       return_value={}))
        stack.enter_context(patch.object(
            orthrus_gnn_testing, 'log'))
        m_torch = stack.enter_context(patch.object(
            orthrus_gnn_testing, 'torch'))
        m_torch.cuda.empty_cache = MagicMock()

        cfg._from_weights = False
        cfg.detection.gnn_training._trained_models_dir = "/fake"
        cfg.detection.gnn_testing._edge_losses_dir    = "/fake"

        orthrus_gnn_testing.main(cfg)


@requires_torch
def test_replay_protocol_calls_reset_then_replay_then_val_then_test():
    """
    Per checkpoint:
    1. reset_state() called once
    2. replay train called once
    3. val processed
    4. test processed
    """
    train_data = [MagicMock()]
    val_data   = [MagicMock()]
    test_data  = [MagicMock()]

    reset_calls  = []
    replay_calls = []
    val_calls    = []
    test_calls   = []

    def on_reset():
        reset_calls.append(1)

    def on_replay(*a, **kw):
        replay_calls.append(1)

    def on_test(data, **kwargs):
        if data in val_data:
            val_calls.append(data)
        elif data in test_data:
            test_calls.append(data)

    cfg = MagicMock()
    fake_model = _make_fake_model(encoder_has_reset=True)

    _run_with_mocks(
        orthrus_gnn_testing, cfg,
        fake_model=fake_model,
        fake_train_data=train_data,
        fake_val_data=val_data,
        fake_test_data=test_data,
        on_reset=on_reset,
        on_replay=on_replay,
        on_test=on_test,
    )

    assert len(reset_calls)  == 1, f"Expected 1 reset_state call, got {len(reset_calls)}"
    assert len(replay_calls) == 1, f"Expected 1 replay call, got {len(replay_calls)}"
    assert len(val_calls)   == len(val_data),  f"Expected {len(val_data)} val calls, got {len(val_calls)}"
    assert len(test_calls)  == len(test_data), f"Expected {len(test_data)} test calls, got {len(test_calls)}"


@requires_torch
def test_replay_protocol_no_reset_between_val_and_test():
    """
    reset_state() is called once before replay, never again.
    The val→test transition must NOT trigger a second reset.
    """
    train_data = [MagicMock()]
    val_data   = [MagicMock()]
    test_data  = [MagicMock()]

    # Each entry is the number of val/test calls seen so far when reset fires.
    # If reset fires twice, we expect two different "state" values.
    reset_at_state = []

    def on_reset():
        # Record how many val+test calls had fired when reset was invoked
        reset_at_state.append(len(val_calls) + len(test_calls))

    def on_test(data, **kwargs):
        pass

    cfg = MagicMock()
    fake_model = _make_fake_model(encoder_has_reset=True)

    val_calls  = []
    test_calls = []

    def track_test(data, **kwargs):
        if data in val_data:
            val_calls.append(data)
        elif data in test_data:
            test_calls.append(data)

    _run_with_mocks(
        orthrus_gnn_testing, cfg,
        fake_model=fake_model,
        fake_train_data=train_data,
        fake_val_data=val_data,
        fake_test_data=test_data,
        on_reset=on_reset,
        on_test=track_test,
    )

    # Exactly one reset, fired before val+test started
    assert len(reset_at_state) == 1, \
        f"reset_state() must fire exactly once, got {len(reset_at_state)}"
    assert reset_at_state[0] == 0, \
        f"reset_state() must fire before any val/test call, fired at state {reset_at_state[0]}"


@requires_torch
def test_replay_protocol_without_reset_state_is_ok():
    """
    If the encoder has no reset_state (e.g. non-Orthrus model), the protocol
    must skip reset and continue: replay still runs, val and test still process.
    No AttributeError should escape.
    """
    train_data = [MagicMock()]
    val_data   = [MagicMock()]
    test_data  = [MagicMock()]

    replay_called = []
    test_called   = []

    def on_replay(*a, **kw):
        replay_called.append(1)

    def on_test(data, **kwargs):
        test_called.append(data)

    cfg = MagicMock()
    # Model without encoder.reset_state — accessing .encoder raises AttributeError
    fake_model = MagicMock()
    del fake_model.encoder

    _run_with_mocks(
        orthrus_gnn_testing, cfg,
        fake_model=fake_model,
        fake_train_data=train_data,
        fake_val_data=val_data,
        fake_test_data=test_data,
        on_replay=on_replay,
        on_test=on_test,
    )

    assert len(replay_called) >= 1, \
        "replay_train_history must be called even when reset_state is unavailable"
    assert len(test_called) == len(val_data) + len(test_data), \
        f"Expected {len(val_data)+len(test_data)} test calls, got {len(test_called)}"


@requires_torch
def test_replay_train_history_uses_eval_no_grad_and_restores_graph_to_cpu():
    """Exercise the real replay helper; only its batch loader is replaced."""
    graph = MagicMock(name="train_graph")
    batch = MagicMock(name="train_batch")
    full_data = MagicMock(name="full_data")
    cfg = MagicMock(name="cfg")
    device = torch.device("cpu")

    model = MagicMock(name="model")
    model.graph_reindexer = MagicMock(name="graph_reindexer")
    grad_states = []

    def forward(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return torch.tensor([0.0])

    model.side_effect = forward

    with patch.object(
        orthrus_gnn_testing,
        "batch_loader_factory",
        return_value=[batch],
    ) as batch_loader:
        orthrus_gnn_testing._replay_train_history(
            model, [graph], full_data, cfg, device
        )

    model.eval.assert_called_once_with()
    batch_loader.assert_called_once_with(cfg, graph, model.graph_reindexer)
    model.assert_called_once_with(batch, full_data, inference=True)
    assert grad_states == [False]
    assert graph.to.call_args_list == [
        call(device=device),
        call("cpu"),
    ]
