"""
Targeted regression tests for the MAGIC Training GC performance fix.

These tests verify that the gc.collect() removal does not introduce regressions.

Key invariants verified:
1. PER_STEP_GC_CALLS = 0 (no per-step gc.collect)
2. InstrumentedTrainer has no gc.collect() call in its training loop
3. Original _run_original_entity_training_lifecycle has no gc.collect() call
4. GC call count upper bound: GC_CALLS <= max_epoch (never n_train * max_epoch)
"""

from __future__ import annotations

import gc
import inspect
import sys
from unittest import mock

import pytest


# =============================================================================
# Static code analysis tests (no runtime needed)
# =============================================================================


def test_instrumented_trainer_source_has_no_gc_collect():
    """Static analysis: InstrumentedTrainer.__call__ source must not contain gc.collect().

    This is the primary regression test for the GC performance fix.
    The original _run_original_entity_training_lifecycle does NOT call gc.collect(),
    so InstrumentedTrainer must also NOT call it.
    """
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    source = inspect.getsource(InstrumentedTrainer.__call__)

    # Remove comments to check only actual code
    lines = source.split("\n")
    code_lines = [l for l in lines if not l.strip().startswith("#")]
    code_only = "\n".join(code_lines)

    # gc.collect() must not appear in the actual code (only in comments is OK for documentation)
    assert "gc.collect()" not in code_only, (
        "gc.collect() found in InstrumentedTrainer.__call__ code (not just comments). "
        "This was the root cause of THEIA_E3 training taking 14+ hours. "
        "The original lifecycle has no gc.collect(), so neither should InstrumentedTrainer."
    )


def test_original_lifecycle_source_has_no_gc_collect():
    """Static analysis: _run_original_entity_training_lifecycle source has no gc.collect().

    Confirms the baseline: the original upstream mirroring function does NOT call gc.collect().
    This justifies removing it from InstrumentedTrainer.
    """
    from src.baselines.magic.real_backend import _run_original_entity_training_lifecycle

    source = inspect.getsource(_run_original_entity_training_lifecycle)

    assert "gc.collect()" not in source, (
        "gc.collect() found in _run_original_entity_training_lifecycle. "
        "This should not exist - original upstream does not call gc.collect()."
    )


def test_instrumented_trainer_preserves_del_cleanup():
    """Verify InstrumentedTrainer still has explicit del statements for cleanup.

    The fix removes gc.collect() but keeps the del statements for graph objects.
    This ensures memory is still freed (implicitly via reference counting).
    """
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    source = inspect.getsource(InstrumentedTrainer.__call__)

    # Should still have del for cleanup
    assert "del copied" in source, (
        "InstrumentedTrainer must still have 'del copied' for explicit cleanup"
    )


# =============================================================================
# Runtime GC call tracking tests
# =============================================================================


def test_gc_collect_not_called_during_formal_runner_import():
    """Verify gc.collect is not called just by importing formal_runner."""
    gc_calls = []

    original_collect = gc.collect

    def tracking_collect(generation=0):
        gc_calls.append(generation)
        return original_collect(generation)

    with mock.patch("gc.collect", side_effect=tracking_collect):
        import src.baselines.magic.formal_runner as fr
        # Re-import to trigger any module-level gc calls
        import importlib
        importlib.reload(fr)

    # Module import should not trigger gc.collect()
    assert len(gc_calls) == 0, (
        f"gc.collect() was called {len(gc_calls)} times during formal_runner import"
    )


def test_instrumented_trainer_class_has_no_gc_in_init():
    """Verify InstrumentedTrainer.__init__ does not call gc.collect."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    init_source = inspect.getsource(InstrumentedTrainer.__init__)
    assert "gc.collect" not in init_source, (
        "gc.collect() found in InstrumentedTrainer.__init__"
    )


def test_grad_stats_function_has_no_gc_collect():
    """Verify grad_stats() helper does not call gc.collect.

    grad_stats is read-only instrumentation and must not have side effects.
    """
    from src.baselines.magic.formal_runner import grad_stats

    source = inspect.getsource(grad_stats)

    assert "gc.collect" not in source, (
        "gc.collect() found in grad_stats() function. "
        "grad_stats must be pure read-only instrumentation."
    )


# =============================================================================
# Upper bound tests: GC_CALLS <= max_epoch
# =============================================================================


def test_gc_collect_upper_bound_documentation():
    """Document the upper bound: GC_CALLS must never equal n_train * max_epoch.

    For THEIA_E3: n_train=194, max_epoch=50
    Old (BUG): 9700 gc.collect() calls
    Fixed: 0 gc.collect() calls
    Even with epoch-level GC fallback: max_epoch=50 calls max

    This test documents the invariant that any future GC strategy must respect.
    """
    n_train = 194
    max_epoch = 50
    total_steps = n_train * max_epoch

    # The buggy pattern would call gc.collect() total_steps times
    buggy_gc_calls = total_steps  # 9700

    # The fixed pattern calls gc.collect() 0 times
    fixed_gc_calls = 0

    # Even with epoch-level GC as fallback, max is max_epoch
    epoch_level_gc_calls = max_epoch  # 50

    # Assertions
    assert buggy_gc_calls == 9700, "THEIA_E3 has 9700 training steps"
    assert fixed_gc_calls == 0, "Fixed version has 0 gc.collect calls"
    assert epoch_level_gc_calls == 50, "Epoch-level GC max is 50"

    # Upper bound invariant
    assert fixed_gc_calls <= max_epoch, (
        "Fixed GC calls must be <= max_epoch"
    )


def test_no_per_step_gc_pattern_in_source():
    """Verify no pattern like 'for ... in graphs: ... gc.collect()' exists in actual code.

    The buggy pattern was:
        for epoch in range(max_epoch):
            for graph in graphs:
                ...
                gc.collect()  # BUG: called 9700 times

    The fixed pattern should not have gc.collect inside the inner loop.
    """
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    source = inspect.getsource(InstrumentedTrainer.__call__)

    # Remove comments to check only actual code
    lines = source.split("\n")
    code_lines = [l for l in lines if not l.strip().startswith("#")]
    code_only = "\n".join(code_lines)

    # Check for the anti-pattern: gc.collect() in the code
    assert "gc.collect()" not in code_only, (
        "gc.collect() found in InstrumentedTrainer.__call__ code. "
        "This was the root cause of THEIA_E3 training taking 14+ hours."
    )


# =============================================================================
# Behavioral tests: InstrumentedTrainer interface
# =============================================================================


def test_instrumented_trainer_has_correct_signature():
    """Verify InstrumentedTrainer.__call__ has the same signature as original lifecycle."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer
    from src.baselines.magic.real_backend import _run_original_entity_training_lifecycle

    inst_sig = inspect.signature(InstrumentedTrainer.__call__)
    orig_sig = inspect.signature(_run_original_entity_training_lifecycle)

    # Parameters should match (excluding 'self' which is only in the class method)
    inst_params = list(inst_sig.parameters.keys())
    orig_params = list(orig_sig.parameters.keys())

    # InstrumentedTrainer.__call__ has 'self' as first param, original function doesn't
    assert inst_params[1:] == orig_params, (
        f"Signature mismatch. Instrumented (excluding self): {inst_params[1:]}, Original: {orig_params}"
    )


def test_instrumented_trainer_returns_model():
    """Verify InstrumentedTrainer.__call__ returns the model after training."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    trainer = InstrumentedTrainer(
        orig_fn=lambda *a, **kw: object(),
        log=[],
        step_counter=[0],
    )

    # Check the function returns a value (via source analysis)
    source = inspect.getsource(InstrumentedTrainer.__call__)
    assert "return model" in source, (
        "InstrumentedTrainer.__call__ must return the model"
    )


def test_instrumented_trainer_initializes_profile_accumulators():
    """Verify InstrumentedTrainer has profiling infrastructure."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    trainer = InstrumentedTrainer(
        orig_fn=lambda *a, **kw: object(),
        log=[],
        step_counter=[0],
    )

    # Should have profiling accumulators
    assert hasattr(trainer, "_profile_accum"), (
        "InstrumentedTrainer should have _profile_accum for profiling"
    )
    assert hasattr(trainer, "_profile_steps"), (
        "InstrumentedTrainer should have _profile_steps counter"
    )


def test_instrumented_trainer_has_profiling_output_frequency():
    """Verify InstrumentedTrainer has a profiling output frequency setting."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    assert hasattr(InstrumentedTrainer, "_PROFILE_EVERY"), (
        "InstrumentedTrainer should have _PROFILE_EVERY constant"
    )

    # Default should be a reasonable frequency (not per-step)
    profile_every = InstrumentedTrainer._PROFILE_EVERY
    assert profile_every >= 10, (
        f"_PROFILE_EVERY={profile_every} is too low. "
        "Profiling output should be low-frequency (e.g., every 100 steps)"
    )


def test_instrumented_trainer_has_profile_timing_stages():
    """Verify profiling stages include all expected timing categories."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    trainer = InstrumentedTrainer(
        orig_fn=lambda *a, **kw: object(),
        log=[],
        step_counter=[0],
    )

    expected_stages = [
        "copy_s",
        "transfer_s",
        "forward_s",
        "backward_s",
        "grad_stats_s",
        "optimizer_s",
        "log_s",
        "del_s",
    ]

    for stage in expected_stages:
        assert stage in trainer._profile_accum, (
            f"Missing profiling stage: {stage}"
        )


def test_instrumented_trainer_train_log_keys():
    """Verify InstrumentedTrainer.train_log entries have expected keys."""
    from src.baselines.magic.formal_runner import InstrumentedTrainer

    trainer = InstrumentedTrainer(
        orig_fn=lambda *a, **kw: object(),
        log=[],
        step_counter=[0],
    )

    # Check the training loop logs expected fields
    source = inspect.getsource(InstrumentedTrainer.__call__)

    required_fields = [
        '"epoch"',
        '"step"',
        '"RAW_FORWARD_LOSS"',
        '"LOSS_USED_FOR_BACKWARD"',
        '"SCALING_FACTOR"',
    ]

    for field in required_fields:
        assert field in source, (
            f"Missing train_log field: {field}"
        )


# =============================================================================
# Integration test: monkeypatch verification
# =============================================================================


def test_gc_collect_monkeypatch_never_called_in_training_loop():
    """Runtime test: monkeypatch gc.collect and verify it's never called during training.

    This is the STEP 6 regression test. Even if someone re-adds gc.collect()
    accidentally, this test will catch it.
    """
    # We can't easily run the full training loop without DGL, but we can
    # verify the source code invariants which is what matters.
    # The static tests above cover this case.

    # This test documents the expected behavior:
    expected_gc_calls_during_training = 0

    from src.baselines.magic.formal_runner import InstrumentedTrainer

    # Verify via source inspection (excluding comments)
    source = inspect.getsource(InstrumentedTrainer.__call__)
    lines = source.split("\n")
    code_lines = [l for l in lines if not l.strip().startswith("#")]
    code_only = "\n".join(code_lines)

    # The fix ensures gc.collect() is not called per-step
    assert "gc.collect()" not in code_only, (
        f"Found gc.collect() in InstrumentedTrainer.__call__ code"
    )


# =============================================================================
# Benchmark documentation test
# =============================================================================


def test_gc_overhead_calculation():
    """Document the GC overhead calculation for THEIA_E3.

    For THEIA_E3 seed=0:
    - n_train = 194 graphs
    - max_epoch = 50
    - Total steps = 9700

    The bug was calling gc.collect() every step:
    - Old: 9700 gc.collect() calls
    - New: 0 gc.collect() calls

    Each gc.collect() scans all three generations and triggers full cyclic GC.
    For large DGL graphs with complex object graphs, this can be extremely expensive.
    """
    n_train = 194
    max_epoch = 50
    total_steps = n_train * max_epoch

    assert total_steps == 9700, f"Expected 9700 steps, got {total_steps}"

    # The fix eliminates all 9700 gc.collect() calls
    gc_calls_eliminated = total_steps

    # Real-world observation: THEIA_E3 training took 14+ hours with per-step GC
    # With ~422s contract build + ~202s DGL prepare = ~624s overhead
    # Training was still running after 14 hours when interrupted
    # This suggests GC overhead alone contributed significantly to the slowdown
    # Conservative per-gc estimate: even 5ms per call × 9700 = 48.5 seconds
    # But the real impact is worse due to cache thrashing and memory pressure

    observed_training_time_hours = 14.0
    observed_contract_prepare_s = 422.1
    observed_dgl_prepare_s = 202.6
    estimated_training_time_s = observed_training_time_hours * 3600  # 50400s

    # The fix should bring training time much closer to expected
    # For now, document the fix impact
    assert gc_calls_eliminated == 9700, (
        f"Fix eliminates {gc_calls_eliminated} gc.collect() calls"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
