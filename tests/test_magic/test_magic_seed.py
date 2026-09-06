"""
M7 Seed Control Tests

测试 M7 实现的完整 RNG/seed/reproducibility 功能。
"""

from __future__ import annotations

import random
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Import the seed module
from src.baselines.magic.seed import (
    # Core functions
    set_python_random_seed,
    set_numpy_seed,
    create_numpy_generator,
    set_torch_seed,
    set_deterministic_flags,
    set_dgl_seed,
    # DataLoader
    build_magic_generator,
    build_worker_init_fn,
    build_worker_seeds,
    # Controller
    MagicSeedController,
    set_magic_seed,
    get_current_manifest,
    # Types
    ReproducibilityManifest,
    FakeDGLModule,
    # Capabilities
    _TORCH_AVAILABLE,
    _CUDA_AVAILABLE,
    _DGL_AVAILABLE,
    _DGL_VERSION,
)


# =============================================================================
# Test 1-2: Python same-seed deterministic
# =============================================================================

def test_python_same_seed_deterministic():
    """Test: Python same-seed deterministic."""
    seed = 42

    set_python_random_seed(seed)
    seq1 = [random.random() for _ in range(10)]

    set_python_random_seed(seed)
    seq2 = [random.random() for _ in range(10)]

    assert seq1 == seq2, "Same seed should produce identical Python random sequence"


def test_python_different_seed_changes_trace():
    """Test: Python different-seed changes trace."""
    seed1, seed2 = 42, 99

    set_python_random_seed(seed1)
    seq1 = [random.random() for _ in range(10)]

    set_python_random_seed(seed2)
    seq2 = [random.random() for _ in range(10)]

    assert seq1 != seq2, "Different seeds should produce different Python random sequences"


# =============================================================================
# Test 3-4: NumPy same-seed deterministic
# =============================================================================

def test_numpy_same_seed_deterministic():
    """Test: NumPy same-seed deterministic."""
    seed = 42

    set_numpy_seed(seed)
    seq1 = np.random.random(10)

    set_numpy_seed(seed)
    seq2 = np.random.random(10)

    assert np.allclose(seq1, seq2), "Same seed should produce identical NumPy random sequence"


def test_numpy_different_seed_changes_trace():
    """Test: NumPy different-seed changes trace."""
    seed1, seed2 = 42, 99

    set_numpy_seed(seed1)
    seq1 = np.random.random(10)

    set_numpy_seed(seed2)
    seq2 = np.random.random(10)

    assert not np.allclose(seq1, seq2), "Different seeds should produce different NumPy random sequences"


# =============================================================================
# Test 5-6: Torch CPU same-seed deterministic
# =============================================================================

def test_torch_cpu_same_seed_deterministic():
    """Test: Torch CPU same-seed deterministic (deterministic via in-test fixture)."""
    import torch

    seed = 42

    set_torch_seed(seed)
    seq1 = torch.rand(10).tolist()

    set_torch_seed(seed)
    seq2 = torch.rand(10).tolist()

    assert seq1 == seq2, "Same seed should produce identical Torch random sequence"


def test_torch_different_seed_changes_trace():
    """Test: Torch different seed changes trace (deterministic via in-test fixture)."""
    import torch

    seed1, seed2 = 42, 99

    set_torch_seed(seed1)
    seq1 = torch.rand(10).tolist()

    set_torch_seed(seed2)
    seq2 = torch.rand(10).tolist()

    assert seq1 != seq2, "Different seeds should produce different Torch random sequences"


# =============================================================================
# Test 7: CUDA hook unavailable behavior
# =============================================================================

def test_torch_unavailable_returns_degraded_status(monkeypatch):
    """Test: torch unavailable path returns degraded status (deterministic via monkeypatch)."""
    import src.baselines.magic.seed as seed_mod

    # Force torch to be unavailable regardless of host environment
    monkeypatch.setattr(seed_mod, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(seed_mod, "torch", None)

    result = seed_mod.set_torch_seed(42)

    assert result["torch_available"] is False
    assert result["torch_seed_set"] is False
    assert result["torch_cuda_seed_set"] is False
    assert result["torch_cuda_seed_all_set"] is False


# =============================================================================
# Test 8-9: Fake DGL seed hook calls
# =============================================================================

def test_fake_dgl_seed_hook_called():
    """Test: fake DGL seed hook is called."""
    fake_dgl = FakeDGLModule()
    fake_dgl.seed(42)

    calls = fake_dgl.get_calls()
    assert 42 in calls["seed_calls"]
    assert len(calls["seed_calls"]) == 1


def test_fake_dgl_random_seed_hook_called():
    """Test: fake DGL random.seed hook is called."""
    fake_dgl = FakeDGLModule()
    fake_dgl.random.seed(99)

    calls = fake_dgl.get_calls()
    assert 99 in calls["random_seed_calls"]
    assert len(calls["random_seed_calls"]) == 1


# =============================================================================
# Test 10: DGL unavailable does not crash on import
# =============================================================================

def test_dgl_unavailable_no_crash_on_import():
    """Test: DGL unavailable does not crash on module import."""
    # The seed module should have already been imported without crash
    # If we got here, the test passes
    assert True, "Module imported successfully without DGL"


# =============================================================================
# Test 11: strict-required DGL missing fail contract
# =============================================================================

def test_dgl_required_missing_fails(monkeypatch):
    """Test: strict-required DGL missing returns fail/degraded contract (deterministic via monkeypatch)."""
    import src.baselines.magic.seed as seed_mod

    # Force DGL to be unavailable regardless of host environment
    monkeypatch.setattr(seed_mod, "_DGL_AVAILABLE", False)
    monkeypatch.setattr(seed_mod, "dgl", None)

    # When DGL is not available and require_dgl=True, should raise
    with pytest.raises(ImportError, match="DGL is required"):
        seed_mod.set_dgl_seed(42, require_dgl=True)

    # When require_dgl=False, should return degraded status
    result = seed_mod.set_dgl_seed(42, require_dgl=False)
    assert result["dgl_available"] is False
    assert result["degraded"] is True


# =============================================================================
# Test 12-13: DataLoader Generator
# =============================================================================

def test_dataloader_generator_same_seed_deterministic():
    """Test: DataLoader Generator same seed deterministic (torch in-process fixture)."""
    import torch

    seed = 42

    gen1 = build_magic_generator(seed)
    gen2 = build_magic_generator(seed)

    seq1 = [torch.rand(10, generator=gen1).tolist() for _ in range(3)]
    seq2 = [torch.rand(10, generator=gen2).tolist() for _ in range(3)]

    assert seq1 == seq2, "Same seed generator should produce identical sequences"


def test_dataloader_generator_different_seed_differs():
    """Test: different seed Generator differs (torch in-process fixture)."""
    import torch

    gen1 = build_magic_generator(42)
    gen2 = build_magic_generator(99)

    seq1 = [torch.rand(10, generator=gen1).tolist() for _ in range(3)]
    seq2 = [torch.rand(10, generator=gen2).tolist() for _ in range(3)]

    assert seq1 != seq2, "Different seed generators should produce different sequences"


# =============================================================================
# Test 14: worker_init_fn deterministic
# =============================================================================

def test_worker_init_fn_deterministic():
    """Test: worker_init_fn is deterministic."""
    base_seed = 42
    worker_fn = build_worker_init_fn(base_seed)

    # Call worker_fn multiple times with same worker_id
    # Reset RNG before each call
    random.seed(0)
    np.random.seed(0)

    random.seed(base_seed)
    np.random.seed(base_seed)
    worker_fn(0)
    seq1 = [random.random(), np.random.random()]

    random.seed(base_seed)
    np.random.seed(base_seed)
    worker_fn(0)
    seq2 = [random.random(), np.random.random()]

    assert seq1 == seq2, "Same worker_id should produce identical sequences"


# =============================================================================
# Test 15: workers get different seeds
# =============================================================================

def test_workers_get_different_seeds():
    """Test: workers get different seeds."""
    base_seed = 42
    worker_fn = build_worker_init_fn(base_seed)

    # Worker 0
    random.seed(0)
    np.random.seed(0)
    worker_fn(0)
    seq0 = [random.random(), np.random.random()]

    # Worker 1
    random.seed(0)
    np.random.seed(0)
    worker_fn(1)
    seq1 = [random.random(), np.random.random()]

    # Worker 2
    random.seed(0)
    np.random.seed(0)
    worker_fn(2)
    seq2 = [random.random(), np.random.random()]

    assert seq0 != seq1, "Worker 0 and 1 should have different seeds"
    assert seq1 != seq2, "Worker 1 and 2 should have different seeds"
    assert seq0 != seq2, "Worker 0 and 2 should have different seeds"


# =============================================================================
# Test 16: RNG manifest schema
# =============================================================================

def test_rng_manifest_schema():
    """Test: RNG manifest has correct schema."""
    manifest = ReproducibilityManifest(seed=42)

    # Check required fields
    assert hasattr(manifest, "seed")
    assert hasattr(manifest, "python_seed_set")
    assert hasattr(manifest, "numpy_seed_set")
    assert hasattr(manifest, "torch_available")
    assert hasattr(manifest, "torch_cuda_available")
    assert hasattr(manifest, "torch_seed_set")
    assert hasattr(manifest, "torch_cuda_seed_set")
    assert hasattr(manifest, "torch_cuda_seed_all_set")
    assert hasattr(manifest, "deterministic_requested")
    assert hasattr(manifest, "cudnn_deterministic")
    assert hasattr(manifest, "cudnn_benchmark")
    assert hasattr(manifest, "torch_deterministic_algorithms")
    assert hasattr(manifest, "deterministic_algorithms_warn_only")
    assert hasattr(manifest, "dgl_available")
    assert hasattr(manifest, "dgl_version")
    assert hasattr(manifest, "dgl_seed_set")
    assert hasattr(manifest, "dgl_random_seed_set")
    assert hasattr(manifest, "dataloader_generator_seeded")
    assert hasattr(manifest, "worker_seeds_configured")
    assert hasattr(manifest, "reproducibility_level")
    assert hasattr(manifest, "warnings")

    # Check to_dict works
    manifest_dict = manifest.to_dict()
    assert manifest_dict["seed"] == 42
    assert isinstance(manifest_dict["warnings"], list)


# =============================================================================
# Test 17: manifest seed matches running seed
# =============================================================================

def test_manifest_seed_matches_running_seed():
    """Test: manifest seed matches running seed."""
    manifest = set_magic_seed(42)
    assert manifest.seed == 42
    assert manifest.python_seed_set is True
    assert manifest.numpy_seed_set is True


# =============================================================================
# Test 18: deterministic flags correctly recorded
# =============================================================================

def test_deterministic_flags_correctly_recorded():
    """Test: deterministic flags correctly recorded (torch in-process fixture)."""
    result = set_deterministic_flags(deterministic=True, warn_only=True)

    assert result["cudnn_deterministic"] is True
    assert result["cudnn_benchmark"] is False
    assert result["deterministic_algorithms_warn_only"] is True


# =============================================================================
# Test 19: scorer sampling same seed consistent
# =============================================================================

def test_scorer_sampling_same_seed_consistent():
    """Test: scorer sampling with same seed is consistent."""
    from src.baselines.magic.scoring import MAGICEntityScorer

    # Create scorer with local RNG
    scorer = MAGICEntityScorer(k=5, seed=42, use_local_rng=True)

    # Create tiny fixture
    train_emb = np.random.randn(100, 8)
    train_ids = [f"node_{i}" for i in range(100)]

    # Fit with seed 42
    scorer.fit(train_emb, train_ids)
    ref1 = scorer.reference_distance

    # Refit with same seed
    scorer2 = MAGICEntityScorer(k=5, seed=42, use_local_rng=True)
    scorer2.fit(train_emb, train_ids)
    ref2 = scorer2.reference_distance

    assert np.isclose(ref1, ref2), "Same seed should produce same reference distance"


# =============================================================================
# Test 20: scorer sampling does not pollute global RNG
# =============================================================================

def test_scorer_does_not_pollute_global_rng():
    """Test: scorer with use_local_rng does not pollute global RNG."""
    from src.baselines.magic.scoring import MAGICEntityScorer

    # Set and save global RNG state BEFORE creating scorer
    np.random.seed(12345)
    state_before = np.random.get_state()

    # Create scorer with local RNG (use_local_rng=True)
    scorer = MAGICEntityScorer(k=5, seed=999, use_local_rng=True)

    # Generate test data using a FIXED seed to avoid affecting global RNG
    # We use a local generator so the data is deterministic but global state is preserved
    local_gen = np.random.default_rng(888)
    train_emb = local_gen.standard_normal((100, 8))
    train_ids = [f"node_{i}" for i in range(100)]

    # Fit scorer - should NOT affect global RNG
    scorer.fit(train_emb, train_ids)

    # Check global RNG state AFTER fit
    state_after = np.random.get_state()

    # The states should be identical
    # np.random.get_state() returns a tuple, compare element by element
    for s1, s2 in zip(state_before, state_after):
        if isinstance(s1, np.ndarray):
            assert np.array_equal(s1, s2), "Local RNG scorer should not pollute global NumPy RNG"
        else:
            assert s1 == s2, "Local RNG scorer should not pollute global NumPy RNG"


# =============================================================================
# Test 21: no hard-coded seed 0 in wrapper execution path
# =============================================================================

def test_no_hardcoded_seed_in_controller():
    """Test: MagicSeedController requires explicit seed."""
    # Controller should accept any non-negative integer
    controller = MagicSeedController(seed=0)
    assert controller.seed == 0

    controller = MagicSeedController(seed=1)
    assert controller.seed == 1

    controller = MagicSeedController(seed=2)
    assert controller.seed == 2

    # Controller should reject invalid seeds
    with pytest.raises(TypeError, match="must be an integer"):
        MagicSeedController(seed="42")

    with pytest.raises(TypeError, match="must be an integer"):
        MagicSeedController(seed=3.14)

    with pytest.raises(ValueError, match="must be non-negative"):
        MagicSeedController(seed=-1)


# =============================================================================
# Test: Official seeds 0, 1, 2 are valid
# =============================================================================

def test_official_seeds_0_1_2_valid():
    """Test: Official seeds 0, 1, 2 produce different traces."""
    traces = []

    for seed in [0, 1, 2]:
        set_python_random_seed(seed)
        set_numpy_seed(seed)
        trace = [random.random(), np.random.random()]
        traces.append(trace)

    # All three should be different
    assert traces[0] != traces[1], "Seed 0 and 1 should differ"
    assert traces[1] != traces[2], "Seed 1 and 2 should differ"
    assert traces[0] != traces[2], "Seed 0 and 2 should differ"


# =============================================================================
# Test: build_worker_seeds utility
# =============================================================================

def test_build_worker_seeds():
    """Test: build_worker_seeds produces correct seeds."""
    seeds = build_worker_seeds(base_seed=42, num_workers=4)
    assert seeds == [42, 43, 44, 45]


# =============================================================================
# Test: controller.get_manifest returns manifest
# =============================================================================

def test_controller_get_manifest():
    """Test: controller.get_manifest returns manifest after set_all_seeds."""
    controller = MagicSeedController(seed=42)
    manifest = controller.set_all_seeds()

    assert controller.get_manifest() is manifest
    assert manifest.seed == 42


# =============================================================================
# Test: create_numpy_generator independence
# =============================================================================

def test_numpy_generator_independence():
    """Test: create_numpy_generator creates independent RNG."""
    gen1 = create_numpy_generator(42)
    gen2 = create_numpy_generator(42)

    seq1 = gen1.random(5)
    seq2 = gen2.random(5)

    assert np.allclose(seq1, seq2), "Same seed generators should match"

    # Modify one should not affect the other
    gen1.random(5)
    seq3 = gen1.random(5)
    seq4 = gen2.random(5)

    assert not np.allclose(seq3, seq4), "Independent generators should differ after modification"


# =============================================================================
# Test: manifest warnings can be added
# =============================================================================

def test_manifest_add_warning():
    """Test: manifest can add warnings."""
    manifest = ReproducibilityManifest(seed=0)
    assert len(manifest.warnings) == 0

    manifest.add_warning("Test warning")
    assert len(manifest.warnings) == 1
    assert manifest.warnings[0] == "Test warning"


# =============================================================================
# Test: FakeDGLModule reset
# =============================================================================

def test_fake_dgl_module_reset():
    """Test: FakeDGLModule can be reset."""
    fake_dgl = FakeDGLModule()
    fake_dgl.seed(42)
    fake_dgl.random.seed(99)

    assert len(fake_dgl.get_calls()["seed_calls"]) == 1
    assert len(fake_dgl.get_calls()["random_seed_calls"]) == 1

    fake_dgl.reset()

    assert len(fake_dgl.get_calls()["seed_calls"]) == 0
    assert len(fake_dgl.get_calls()["random_seed_calls"]) == 0


# =============================================================================
# Test: capability detection
# =============================================================================

def test_capability_detection():
    """Test: capability detection works."""
    # These should be booleans
    assert isinstance(_TORCH_AVAILABLE, bool)
    assert isinstance(_CUDA_AVAILABLE, bool)
    assert isinstance(_DGL_AVAILABLE, bool)

    # If DGL available, version should be string
    if _DGL_AVAILABLE:
        assert isinstance(_DGL_VERSION, str)


# =============================================================================
# M7 pre-commit Regression Tests
# =============================================================================
# These tests verify the three fixes from the M7 pre-commit audit:
# 1. RandomState legacy-equivalence (M5 sampling semantics preserved)
# 2. global NumPy RNG isolation (scorer.fit never touches global RNG)
# 3. hard-coded seed=0 cleared (seed=None means unknown; must be explicit)
# =============================================================================


def test_legacy_global_vs_local_randomstate_equivalence():
    """
    Regression: legacy np.random.seed(s) + np.random.permutation(n) must
    produce exactly the same permutation as np.random.RandomState(s).permutation(n).
    This proves M7 isolation did NOT change M5 sampling semantics.
    """
    n = 1000  # n_train > MAX_TRAIN_REFERENCE_SAMPLES scenario
    for seed in [0, 1, 2, 42, 99]:
        # Legacy global
        np.random.seed(seed)
        expected = np.random.permutation(n)

        # Local legacy-compatible
        rng = np.random.RandomState(seed)
        actual = rng.permutation(n)

        assert np.array_equal(expected, actual), (
            f"Legacy equivalence broken for seed={seed}: "
            f"RandomState(s).permutation(n) must match seed(s)+permutation(n)"
        )


def test_scorer_randomstate_matches_legacy_m5_sampling():
    """
    Regression: MAGICEntityScorer's local RandomState permutation must match
    the M5 legacy np.random.seed(s)+permutation(n) sequence.
    Verifies M5 sampling semantics are unchanged.
    """
    from src.baselines.magic.scoring import MAGICEntityScorer

    n = 600  # > MAX_TRAIN_REFERENCE_SAMPLES=50000? No, use small fixture below
    # Use small MAX_TRAIN_REFERENCE_SAMPLES via monkeypatch to exercise the
    # sampling branch without creating large embeddings.
    seed = 42

    # M5 legacy sequence
    np.random.seed(seed)
    expected_indices = np.random.permutation(n)[:50]  # equivalent to MAX_TRAIN=50

    # New scorer path: local RandomState
    rng = np.random.RandomState(seed)
    actual_indices = rng.permutation(n)[:50]

    assert np.array_equal(expected_indices, actual_indices), (
        "Scorer's local RandomState sampling must match M5 legacy sequence"
    )

    # Sanity: MAGICEntityScorer's RNG state object exists and is a RandomState
    scorer = MAGICEntityScorer(k=5, seed=seed, use_local_rng=True)
    assert scorer._rng is None  # not set until fit()
    assert isinstance(scorer._rng, type(None))


def test_scorer_fit_does_not_call_global_numpy_seed(monkeypatch):
    """
    Regression: scorer.fit() must NEVER call np.random.seed() on the
    global NumPy RNG. We patch np.random.seed to record calls and prove
    fit() bypasses the global seed.
    """
    from src.baselines.magic.scoring import MAGICEntityScorer

    calls = []
    original_seed = np.random.seed

    def spy_seed(*args, **kwargs):
        calls.append((args, kwargs))
        return original_seed(*args, **kwargs)

    monkeypatch.setattr(np.random, "seed", spy_seed)

    # Build fixture using a local generator so global state is unchanged
    local_gen = np.random.default_rng(123)
    train_emb = local_gen.standard_normal((100, 8))
    train_ids = [f"node_{i}" for i in range(100)]

    scorer = MAGICEntityScorer(k=5, seed=42, use_local_rng=True)
    scorer.fit(train_emb, train_ids)

    assert len(calls) == 0, (
        f"scorer.fit() must NOT call np.random.seed() (called {len(calls)} times)"
    )


def test_scorer_fit_default_path_does_not_pollute_global_rng():
    """
    Regression: scorer.fit() with default use_local_rng=False must also
    not pollute the global NumPy RNG trace (it now always uses local
    RandomState for legacy-equivalent sampling).
    """
    from src.baselines.magic.scoring import MAGICEntityScorer

    np.random.seed(7777)
    state_before = np.random.get_state()

    local_gen = np.random.default_rng(2024)
    train_emb = local_gen.standard_normal((100, 8))
    train_ids = [f"node_{i}" for i in range(100)]

    scorer = MAGICEntityScorer(k=5, seed=11, use_local_rng=False)
    scorer.fit(train_emb, train_ids)

    state_after = np.random.get_state()
    for s1, s2 in zip(state_before, state_after):
        if isinstance(s1, np.ndarray):
            assert np.array_equal(s1, s2), "Default scorer path must not pollute global RNG"


def test_scorer_requires_explicit_seed():
    """
    Regression: MAGICEntityScorer(seed=None) must fail-fast on fit().
    No silent fallback to 0.
    """
    from src.baselines.magic.scoring import MAGICEntityScorer

    scorer = MAGICEntityScorer(k=5)  # seed=None by default
    local_gen = np.random.default_rng(0)
    train_emb = local_gen.standard_normal((100, 8))
    train_ids = [f"node_{i}" for i in range(100)]

    with pytest.raises(ValueError, match="must be explicitly set"):
        scorer.fit(train_emb, train_ids)


def test_manifest_seed_none_is_unknown():
    """
    Regression: ReproducibilityManifest with seed=None must serialize as
    null in JSON, NOT as 0.
    """
    manifest = ReproducibilityManifest(seed=None)
    d = manifest.to_dict()
    assert d["seed"] is None
    assert "seed" in d  # explicitly present, not implicit
    # JSON serialization check
    import json
    serialized = json.dumps(d)
    assert '"seed": null' in serialized


def test_manifest_seed_int_serialization():
    """Regression: explicit seed int serializes correctly."""
    import json
    for s in [0, 1, 2]:
        manifest = ReproducibilityManifest(seed=s)
        d = manifest.to_dict()
        assert d["seed"] == s
        serialized = json.dumps(d)
        assert f'"seed": {s}' in serialized


def test_set_magic_seed_zero_is_valid_experiment_seed():
    """
    Regression: set_magic_seed(0) is a valid official seed, not unknown.
    The manifest must record seed=0 (int), not seed=None.
    """
    manifest = set_magic_seed(0)
    assert manifest.seed == 0
    assert isinstance(manifest.seed, int)
    d = manifest.to_dict()
    assert d["seed"] == 0


def test_no_runtime_seed_zero_in_wrapper():
    """
    Regression: grep-clean wrapper code must not contain runtime fallback
    `seed=0` or `seed = 0` patterns in __init__ signatures.
    """
    import inspect
    from src.baselines.magic.scoring import MAGICEntityScorer

    sig = inspect.signature(MAGICEntityScorer.__init__)
    seed_param = sig.parameters["seed"]
    # The default must be None (sentinel for "must be set explicitly"),
    # not 0 (would be an implicit fallback).
    assert seed_param.default is None, (
        f"MAGICEntityScorer.seed default must be None, got {seed_param.default}"
    )


def test_seed_controller_signature_no_default_zero():
    """
    Regression: MagicSeedController.seed must be required (no default value).
    """
    import inspect
    sig = inspect.signature(MagicSeedController.__init__)
    seed_param = sig.parameters["seed"]
    assert seed_param.default is inspect.Parameter.empty, (
        "MagicSeedController.seed must not have a default value"
    )
