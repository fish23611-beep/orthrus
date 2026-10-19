"""Bounded native-Windows import, working-set and CUDA compatibility probe."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MIB = 1024 ** 2


def native_memory():
    """Independent Windows API cross-check of psutil's byte counters."""
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"rss_bytes": counters.WorkingSetSize, "peak_bytes": counters.PeakWorkingSetSize}


def memory_probe():
    import psutil
    from runtime.memory_metrics import current_rss_mb, peak_rss_mb

    def sample():
        info = psutil.Process().memory_info()
        current, peak = current_rss_mb(), peak_rss_mb()
        native = native_memory()
        direct_peak = getattr(info, "peak_rss", None)
        if direct_peak is None:
            direct_peak = info.peak_wset
        assert current is not None and peak is not None and 0 < current <= peak
        assert abs(current - info.rss / MIB) < 4
        assert abs(peak - direct_peak / MIB) < 4
        assert abs(current - native["rss_bytes"] / MIB) < 4
        assert abs(peak - native["peak_bytes"] / MIB) < 4
        return {"current_mib": current, "peak_mib": peak,
                "psutil_rss_bytes": info.rss, "psutil_peak_bytes": direct_peak,
                "native": native}

    before = sample()
    block = bytearray(32 * MIB)
    for offset in range(0, len(block), 4096):
        block[offset] = 1
    during = sample()
    assert during["current_mib"] >= before["current_mib"] + 16
    del block
    gc.collect()
    after = sample()
    assert after["peak_mib"] >= during["peak_mib"]
    return {"unit": "MiB", "allocation_bytes": 32 * MIB,
            "before": before, "allocated": during, "released": after}


def gpu_probe():
    import torch
    from torch_geometric.nn import SAGEConv, TransformerConv

    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.cuda.reset_peak_memory_stats()
    matrix = torch.randn(128, 128, device="cuda", requires_grad=True)
    (matrix @ matrix.T).square().mean().backward()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], device="cuda")
    results = {}
    for cls in (TransformerConv, SAGEConv):
        layer = cls(8, 4).cuda()
        x = torch.randn(4, 8, device="cuda", requires_grad=True)
        output = layer(x, edge_index)
        output.square().mean().backward()
        assert output.shape == (4, 4) and torch.isfinite(output).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        results[cls.__name__] = "forward_backward_passed"
    torch.cuda.synchronize()
    return {"device": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "matrix_backward": "passed", **results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This probe requires native Windows")
    modules = [
        "mstc.multiscale_encoder", "mstc.multiscale_sampler", "mstc.history_store",
        "mstc.time_gap", "mstc.calibration", "mstc.aggregation", "mstc.thresholding",
        "detection.checkpoint_policy", "detection.orthrus_gnn_training",
        "detection.orthrus_gnn_testing", "detection.evaluation", "data_utils",
        "runtime.runtime_profiler", "graph_construction.build_orthrus_graphs",
        "edge_featurization.embed_edges_feature_word2vec", "orthrus",
    ]
    for name in modules:
        importlib.import_module(name)
    import igraph
    assert igraph.Graph.Ring(4).vcount() == 4
    payload = {
        "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "torch-geometric", "numpy", "scipy", "psutil", "igraph", "texttable"]},
        "imports": modules, "memory": memory_probe(), "gpu": gpu_probe(),
        "scope": "Synthetic compatibility only; no real THEIA data or Linux execution",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
