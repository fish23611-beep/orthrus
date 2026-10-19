"""Opt-in, scoped inference reproducibility; no calibration/precision changes."""

from contextlib import contextmanager
import hashlib
import os
import random

import numpy as np
import torch


RNG_STAGES = ("construction", "replay", "val", "test")


def enabled(cfg):
    evaluation = getattr(getattr(cfg, "detection", None), "evaluation", None)
    return getattr(evaluation, "deterministic_inference", False) is True


def stage_seed(cfg, stage):
    # No policy, checkpoint path, process hash or call count in this boundary.
    seed = int(getattr(cfg, "_seed", 0))
    digest = hashlib.sha256(f"p4-ns-v1:{seed}:{stage}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def backend_metadata():
    """Read flags without initializing CUDA or changing precision settings."""
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cuda_build_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
        "cpu_threads": torch.get_num_threads(),
    }


def inference_metadata(cfg):
    return {
        **backend_metadata(),
        "enabled": enabled(cfg),
        "base_seed": int(getattr(cfg, "_seed", 0) or 0),
        "rng_boundary_version": "p4-ns-v1" if enabled(cfg) else None,
        "stage_seeds": {stage: stage_seed(cfg, stage) for stage in RNG_STAGES}
        if enabled(cfg) else {},
    }


@contextmanager
def deterministic_inference(cfg, device):
    """Enforce deterministic algorithms and restore process flags on all exits.

    CUDA callers must export CUBLAS_WORKSPACE_CONFIG before launching Python.
    Unsupported operators propagate PyTorch's error; there is no CPU fallback.
    """
    if not enabled(cfg):
        yield
        return
    if torch.device(device).type == "cuda" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "deterministic_inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) in the outer launcher BEFORE importing torch/CUDA initialization"
        )
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]


@contextmanager
def inference_rng(cfg, device, stage):
    """Isolate Python/NumPy/torch RNGs; preserve temporal model/history state."""
    if not enabled(cfg):
        yield
        return
    if stage not in RNG_STAGES:
        raise ValueError(f"Unknown inference RNG stage: {stage}")
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            seed = stage_seed(cfg, stage)
            random.seed(seed)
            np.random.seed(seed)
            # Seed only the CPU and the selected CUDA generator, never other GPUs.
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
