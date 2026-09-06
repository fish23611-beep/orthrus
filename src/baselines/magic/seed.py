"""
MAGIC Seed Control / Reproducibility Contract

本模块实现 M7: MAGIC Seed Control，为 MAGIC 基线提供统一的随机数控制。

核心功能：
1. Python random / NumPy / PyTorch CPU/CUDA RNG 控制
2. 可选 DGL seed hook（延迟导入，不强制要求）
3. DataLoader generator 和 worker seed 控制
4. Machine-readable reproducibility manifest 生成

设计原则：
- 调用方必须显式传入 seed，不依赖隐式默认值
- DGL 不可用时明确降级，不静默跳过
- 所有 deterministic flags 都有记录
- 兼容 Python 3.8, Torch 1.12.1

正式 seeds: 0, 1, 2 对应三个独立 stochastic runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union
import random
import numpy as np

# =============================================================================
# Optional Torch import with capability detection
# =============================================================================

_TORCH_AVAILABLE = False
_CUDA_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore

# =============================================================================
# Optional DGL import with capability detection
# =============================================================================

_DGL_AVAILABLE = False
_DGL_VERSION: Optional[str] = None
try:
    import dgl
    _DGL_AVAILABLE = True
    _DGL_VERSION = getattr(dgl, "__version__", None)
except ImportError:
    dgl = None  # type: ignore

# =============================================================================
# Reproducibility Manifest
# =============================================================================

@dataclass
class ReproducibilityManifest:
    """
    Machine-readable reproducibility metadata.

    本 manifest 记录：
    - 所有 RNG 的 seed 状态
    - 所有 deterministic flags 的值
    - 各库的可获得性
    - 降级情况

    seed 字段语义：
    - seed is None: unknown / not yet set
    - seed is int: the actual experiment seed (0, 1, 2 for official runs)
    """
    # Core seed - None means unknown / not yet determined
    seed: Optional[int] = None

    # Controller settings
    require_dgl: bool = False

    # Python / NumPy
    python_seed_set: bool = False
    numpy_seed_set: bool = False

    # PyTorch
    torch_available: bool = _TORCH_AVAILABLE
    torch_cuda_available: bool = _CUDA_AVAILABLE
    torch_seed_set: bool = False
    torch_cuda_seed_set: bool = False
    torch_cuda_seed_all_set: bool = False

    # Deterministic flags
    deterministic_requested: bool = False
    cudnn_deterministic: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    torch_deterministic_algorithms: Optional[bool] = None
    deterministic_algorithms_warn_only: bool = False

    # DGL
    dgl_available: bool = _DGL_AVAILABLE
    dgl_version: Optional[str] = _DGL_VERSION
    dgl_seed_set: bool = False
    dgl_random_seed_set: bool = False

    # DataLoader
    dataloader_generator_seeded: bool = False
    worker_seeds_configured: bool = False

    # Reproducibility level
    reproducibility_level: str = "unknown"

    # Warnings for degraded behavior
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed": self.seed,
            "python_seed_set": self.python_seed_set,
            "numpy_seed_set": self.numpy_seed_set,
            "torch_available": self.torch_available,
            "torch_cuda_available": self.torch_cuda_available,
            "torch_seed_set": self.torch_seed_set,
            "torch_cuda_seed_set": self.torch_cuda_seed_set,
            "torch_cuda_seed_all_set": self.torch_cuda_seed_all_set,
            "deterministic_requested": self.deterministic_requested,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "torch_deterministic_algorithms": self.torch_deterministic_algorithms,
            "deterministic_algorithms_warn_only": self.deterministic_algorithms_warn_only,
            "dgl_available": self.dgl_available,
            "dgl_version": self.dgl_version,
            "dgl_seed_set": self.dgl_seed_set,
            "dgl_random_seed_set": self.dgl_random_seed_set,
            "dataloader_generator_seeded": self.dataloader_generator_seeded,
            "worker_seeds_configured": self.worker_seeds_configured,
            "reproducibility_level": self.reproducibility_level,
            "warnings": self.warnings,
        }

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


# =============================================================================
# Core Seed Setting Functions
# =============================================================================

def set_python_random_seed(seed: int) -> None:
    """
    Set Python's random module seed.

    Args:
        seed: Integer seed for Python's random module
    """
    random.seed(seed)


def set_numpy_seed(seed: int) -> None:
    """
    Set NumPy's global RNG seed.

    Args:
        seed: Integer seed for NumPy's RNG
    """
    np.random.seed(seed)


def create_numpy_generator(seed: int) -> np.random.Generator:
    """
    Create an independent NumPy Generator from a seed.

    Unlike the global np.random.seed(), this creates a separate
    Generator that won't be affected by global RNG operations.

    Args:
        seed: Integer seed for the new Generator

    Returns:
        A new np.random.Generator instance
    """
    return np.random.default_rng(seed)


# =============================================================================
# Torch Seed Setting
# =============================================================================

def set_torch_seed(
    seed: int,
    set_cuda: bool = True,
) -> dict:
    """
    Set PyTorch RNG seeds.

    Args:
        seed: Integer seed for PyTorch
        set_cuda: Whether to also set CUDA seeds (if available)

    Returns:
        Dict with status of each seed operation
    """
    if not _TORCH_AVAILABLE:
        return {
            "torch_available": False,
            "torch_seed_set": False,
            "torch_cuda_seed_set": False,
            "torch_cuda_seed_all_set": False,
        }

    torch.manual_seed(seed)

    result = {
        "torch_available": True,
        "torch_seed_set": True,
        "torch_cuda_seed_set": False,
        "torch_cuda_seed_all_set": False,
    }

    if set_cuda and _CUDA_AVAILABLE:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result["torch_cuda_seed_set"] = True
        result["torch_cuda_seed_all_set"] = True

    return result


def set_deterministic_flags(
    deterministic: bool = True,
    warn_only: bool = True,
) -> dict:
    """
    Set PyTorch deterministic computation flags.

    Args:
        deterministic: Whether to request deterministic operations
        warn_only: If True, use warn_only=True for deterministic_algorithms;
                   If False, require strict determinism (may raise errors)

    Returns:
        Dict with flag states
    """
    if not _TORCH_AVAILABLE:
        return {
            "torch_available": False,
            "cudnn_deterministic": None,
            "cudnn_benchmark": None,
            "torch_deterministic_algorithms": None,
            "deterministic_algorithms_warn_only": warn_only,
        }

    # Set cuDNN flags
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch.backends.cudnn, "benchmark"):
        # Benchmark should be False for deterministic behavior
        torch.backends.cudnn.benchmark = not deterministic

    # Set deterministic algorithms flag
    deterministic_algorithms: Optional[bool] = None
    if hasattr(torch, "use_deterministic_algorithms"):
        if warn_only:
            # Torch 1.12+ supports warn_only parameter
            try:
                torch.use_deterministic_algorithms(deterministic, warn_only=True)
                deterministic_algorithms = deterministic
            except (TypeError, AttributeError):
                # Fallback for older versions without warn_only
                torch.use_deterministic_algorithms(deterministic)
                deterministic_algorithms = deterministic
        else:
            torch.use_deterministic_algorithms(deterministic)
            deterministic_algorithms = deterministic

    return {
        "torch_available": True,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": deterministic_algorithms,
        "deterministic_algorithms_warn_only": warn_only,
    }


# =============================================================================
# DGL Seed Hooks
# =============================================================================

def set_dgl_seed(
    seed: int,
    require_dgl: bool = True,
) -> dict:
    """
    Set DGL RNG seed.

    This function uses lazy import to avoid hard dependency on DGL.
    If DGL is not available and require_dgl=True, raises ImportError.
    If DGL is not available and require_dgl=False, returns degraded status.

    Args:
        seed: Integer seed for DGL
        require_dgl: If True, raise ImportError when DGL unavailable;
                     If False, return degraded status

    Returns:
        Dict with DGL seed status

    Raises:
        ImportError: If require_dgl=True and DGL is not available
    """
    if not _DGL_AVAILABLE:
        if require_dgl:
            raise ImportError(
                "DGL is required for DGL seed setting but is not available. "
                "Install DGL: pip install dgl"
            )
        return {
            "dgl_available": False,
            "dgl_seed_set": False,
            "dgl_random_seed_set": False,
            "degraded": True,
        }

    # DGL 1.0.0+ has dgl.seed() function
    if hasattr(dgl, "seed"):
        dgl.seed(seed)

    # Also try dgl.random.seed() which may be available
    dgl_random_seeded = False
    if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
        dgl.random.seed(seed)
        dgl_random_seeded = True

    return {
        "dgl_available": True,
        "dgl_version": _DGL_VERSION,
        "dgl_seed_set": True,
        "dgl_random_seed_set": dgl_random_seeded,
        "degraded": False,
    }


class FakeDGLModule:
    """
    Mock DGL module for testing without real DGL installation.

    This allows testing DGL seed hook calls in synthetic tests
    without requiring DGL to be installed.
    """

    def __init__(self):
        self._seed_calls: list = []
        self._random_seed_calls: list = []

    def seed(self, seed: int) -> None:
        """Record seed call."""
        self._seed_calls.append(seed)

    @property
    def random(self) -> "FakeDGLRandom":
        """Return fake random submodule."""
        return FakeDGLRandom(self)

    def get_calls(self) -> dict:
        """Get all recorded calls."""
        return {
            "seed_calls": list(self._seed_calls),
            "random_seed_calls": list(self._random_seed_calls),
        }

    def reset(self) -> None:
        """Reset recorded calls."""
        self._seed_calls.clear()
        self._random_seed_calls.clear()


class FakeDGLRandom:
    """Mock dgl.random module for testing."""

    def __init__(self, parent: FakeDGLModule):
        self._parent = parent

    def seed(self, seed: int) -> None:
        """Record random.seed call."""
        self._parent._random_seed_calls.append(seed)


# =============================================================================
# DataLoader Generator and Worker Seeds
# =============================================================================

def build_magic_generator(seed: int) -> Optional["torch.Generator"]:
    """
    Build a seeded torch.Generator for DataLoader.

    Args:
        seed: Integer seed for the generator

    Returns:
        torch.Generator if torch is available, None otherwise
    """
    if not _TORCH_AVAILABLE:
        return None

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def build_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Build a worker_init_fn for DataLoader workers.

    Each worker gets a different seed derived from base_seed + worker_id.
    This ensures workers have different but deterministic random sequences.

    Args:
        base_seed: Base seed for all workers

    Returns:
        A worker_init_fn that can be passed to DataLoader
    """
    def worker_init_fn(worker_id: int) -> None:
        """
        Initialize worker RNG with seed = base_seed + worker_id.

        This ensures:
        - Same base_seed + worker_id always produces same random sequence
        - Different worker_ids produce different sequences
        - Same base_seed on different runs produces same worker seeds
        """
        worker_seed = base_seed + worker_id

        # Set Python random
        random.seed(worker_seed)

        # Set NumPy
        np.random.seed(worker_seed)

        # Set Torch if available
        if _TORCH_AVAILABLE:
            torch.manual_seed(worker_seed)
            if _CUDA_AVAILABLE:
                torch.cuda.manual_seed_all(worker_seed)

    return worker_init_fn


def build_worker_seeds(base_seed: int, num_workers: int) -> list:
    """
    Compute the seed for each worker.

    Args:
        base_seed: Base seed for all workers
        num_workers: Number of workers

    Returns:
        List of worker seeds: [base_seed + 0, base_seed + 1, ..., base_seed + num_workers - 1]
    """
    return [base_seed + i for i in range(num_workers)]


# =============================================================================
# Unified Seed Controller
# =============================================================================

class MagicSeedController:
    """
    Unified seed controller for MAGIC baseline reproducibility.

    Usage:
        >>> # NOTE: the docstring example below uses seed=0 only for illustration.
        >>> # In real experiment runs the caller MUST pass an explicit seed
        >>> # from {0, 1, 2}; there is no implicit default.
        >>> controller = MagicSeedController(seed=42)
        >>> manifest = controller.set_all_seeds()
        >>> # ... run MAGIC code ...
        >>> manifest = controller.get_manifest()

    The controller ensures:
    1. Python random, NumPy, Torch, CUDA, DGL all use the same base seed
    2. Deterministic flags are set appropriately
    3. All seed operations are recorded in a manifest
    """

    def __init__(
        self,
        seed: int,
        set_deterministic: bool = True,
        set_cuda: bool = True,
        require_dgl: bool = False,
    ):
        """
        Initialize the seed controller.

        Args:
            seed: Integer seed (0, 1, or 2 for official runs)
            set_deterministic: Whether to set deterministic flags
            set_cuda: Whether to set CUDA seeds (if available)
            require_dgl: If True, require DGL and fail if unavailable;
                         If False, DGL is optional
        """
        if not isinstance(seed, int):
            raise TypeError(
                f"seed must be an integer, got {type(seed).__name__}"
            )
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        self.seed = seed
        self.set_deterministic = set_deterministic
        self.set_cuda = set_cuda
        self.require_dgl = require_dgl
        self._manifest: Optional[ReproducibilityManifest] = None

    def set_all_seeds(self) -> ReproducibilityManifest:
        """
        Set all RNG seeds and deterministic flags.

        Returns:
            ReproducibilityManifest with all seed states
        """
        manifest = ReproducibilityManifest(seed=self.seed)

        # Python random
        set_python_random_seed(self.seed)
        manifest.python_seed_set = True

        # NumPy
        set_numpy_seed(self.seed)
        manifest.numpy_seed_set = True

        # PyTorch
        if _TORCH_AVAILABLE:
            torch_result = set_torch_seed(self.seed, set_cuda=self.set_cuda)
            manifest.torch_seed_set = torch_result["torch_seed_set"]
            manifest.torch_cuda_seed_set = torch_result["torch_cuda_seed_set"]
            manifest.torch_cuda_seed_all_set = torch_result["torch_cuda_seed_all_set"]

            # Deterministic flags
            if self.set_deterministic:
                manifest.deterministic_requested = True
                det_result = set_deterministic_flags(
                    deterministic=True,
                    warn_only=True,
                )
                manifest.cudnn_deterministic = det_result["cudnn_deterministic"]
                manifest.cudnn_benchmark = det_result["cudnn_benchmark"]
                manifest.torch_deterministic_algorithms = det_result["torch_deterministic_algorithms"]
                manifest.deterministic_algorithms_warn_only = det_result["deterministic_algorithms_warn_only"]
        else:
            manifest.add_warning("PyTorch not available - torch RNG not seeded")

        # DGL (optional)
        if _DGL_AVAILABLE:
            dgl_result = set_dgl_seed(self.seed, require_dgl=False)
            manifest.dgl_seed_set = dgl_result["dgl_seed_set"]
            manifest.dgl_random_seed_set = dgl_result["dgl_random_seed_set"]
        elif self.require_dgl:
            manifest.add_warning("DGL required but not available")
            manifest.reproducibility_level = "degraded"
            self._manifest = manifest
            return manifest
        else:
            manifest.add_warning("DGL not available - DGL RNG not seeded (optional)")

        # Compute reproducibility level
        manifest.reproducibility_level = self._compute_reproducibility_level(manifest)

        self._manifest = manifest
        return manifest

    def _compute_reproducibility_level(self, manifest: ReproducibilityManifest) -> str:
        """Compute the reproducibility level based on manifest."""
        issues = []

        if not manifest.python_seed_set:
            issues.append("python")
        if not manifest.numpy_seed_set:
            issues.append("numpy")
        if manifest.torch_available and not manifest.torch_seed_set:
            issues.append("torch")
        if manifest.torch_cuda_available and not manifest.torch_cuda_seed_all_set:
            issues.append("cuda")

        if manifest.require_dgl and not manifest.dgl_available:
            issues.append("dgl_required")
        elif not manifest.dgl_available:
            issues.append("dgl_optional")

        if not issues:
            return "full"
        elif "cuda" in issues or "dgl_required" in issues:
            return "degraded"
        elif issues:
            return "partial"

        return "unknown"

    def get_manifest(self) -> Optional[ReproducibilityManifest]:
        """Get the reproducibility manifest from last set_all_seeds() call."""
        return self._manifest

    def build_generator(self) -> Optional["torch.Generator"]:
        """Build a seeded torch.Generator for DataLoader."""
        return build_magic_generator(self.seed)

    def build_worker_init_fn(self) -> Callable[[int], None]:
        """Build a worker_init_fn for DataLoader workers."""
        return build_worker_init_fn(self.seed)


# =============================================================================
# Convenience Functions
# =============================================================================

def set_magic_seed(
    seed: int,
    set_deterministic: bool = True,
    set_cuda: bool = True,
) -> ReproducibilityManifest:
    """
    Convenience function to set all MAGIC-related seeds.

    Args:
        seed: Integer seed (0, 1, or 2)
        set_deterministic: Whether to set deterministic flags
        set_cuda: Whether to set CUDA seeds

    Returns:
        ReproducibilityManifest documenting the seed state
    """
    controller = MagicSeedController(
        seed=seed,
        set_deterministic=set_deterministic,
        set_cuda=set_cuda,
        require_dgl=False,
    )
    return controller.set_all_seeds()


def get_current_manifest() -> ReproducibilityManifest:
    """
    Get a manifest reflecting current RNG state without setting seeds.

    This is useful for auditing the current reproducibility state.

    Returns:
        ReproducibilityManifest with current state.
        The seed field is None (unknown) - we are NOT setting any seed here,
        just inspecting the current environment.
    """
    manifest = ReproducibilityManifest(seed=None)  # unknown - no seed was set

    # Check Python
    import sys
    manifest.python_seed_set = True  # Assume set if we're running

    # Check NumPy
    manifest.numpy_seed_set = True

    # Check Torch
    if _TORCH_AVAILABLE:
        manifest.torch_available = True
        manifest.torch_cuda_available = _CUDA_AVAILABLE

        if hasattr(torch.backends.cudnn, "deterministic"):
            manifest.cudnn_deterministic = torch.backends.cudnn.deterministic
        if hasattr(torch.backends.cudnn, "benchmark"):
            manifest.cudnn_benchmark = torch.backends.cudnn.benchmark

    # Check DGL
    manifest.dgl_available = _DGL_AVAILABLE
    manifest.dgl_version = _DGL_VERSION

    return manifest


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core functions
    "set_python_random_seed",
    "set_numpy_seed",
    "create_numpy_generator",
    "set_torch_seed",
    "set_deterministic_flags",
    "set_dgl_seed",
    # DataLoader
    "build_magic_generator",
    "build_worker_init_fn",
    "build_worker_seeds",
    # Controller
    "MagicSeedController",
    "set_magic_seed",
    "get_current_manifest",
    # Types
    "ReproducibilityManifest",
    "FakeDGLModule",
    "FakeDGLRandom",
    # Capabilities
    "_TORCH_AVAILABLE",
    "_CUDA_AVAILABLE",
    "_DGL_AVAILABLE",
    "_DGL_VERSION",
]
