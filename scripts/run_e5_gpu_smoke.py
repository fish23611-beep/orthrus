"""Run genuine migrated THEIA inputs through frozen production inference only."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e5_smoke_transfer import digest, no_symlinks, validate_manifest


def configure(manifest, data_root, run_root):
    """Map I/O paths, retaining every non-path algorithm parameter."""
    source = copy.deepcopy(manifest["resolved_config"])
    path_changes = []
    def map_paths(value, keys=()):
        if isinstance(value, dict):
            return {k: map_paths(v, (*keys, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [map_paths(v, (*keys, str(i))) for i, v in enumerate(value)]
        if isinstance(value, str) and value.startswith("/"):
            target = str(run_root / "unused_paths" / "_".join(keys))
            path_changes.append({"field": ".".join(keys), "server": value, "windows": target})
            return target
        return value
    cfg = map_paths(source)
    cfg.update(_run_dir=str(run_root), _metadata_dir=str(data_root / "artifacts/THEIA_E5/metadata"),
               _use_cpu=False, _seed=0, _test_mode=False, _max_windows_per_split=2,
               _shared_artifact_readonly=True, _from_weights=False,
               _inference_checkpoint=str(data_root / "artifacts/THEIA_E5/checkpoints/model_epoch_6"),
               _stages=["test", "evaluate"], _is_smoke=True)
    cfg["dataset"]["raw_dir"] = str(data_root / "datasets/unused_raw")
    cfg["edge_featurization"]["embed_edges"]["_edge_embeds_dir"] = str(data_root / "datasets/THEIA_E5/edge_embeds")
    cfg["graph_construction"]["build_graphs"]["_graphs_dir"] = str(data_root / "artifacts/THEIA_E5/graphs")
    cfg["detection"]["gnn_training"]["_trained_models_dir"] = str(data_root / "artifacts/THEIA_E5/checkpoints")
    cfg["detection"]["gnn_testing"]["_edge_losses_dir"] = str(run_root / "edge_scores")
    cfg["detection"]["evaluation"]["_evaluation_results_dir"] = str(run_root / "evaluation_results")
    cfg["detection"]["evaluation"]["node_evaluation"]["_precision_recall_dir"] = str(run_root / "node_scores")
    if cfg["pipeline"]["mode"] != "detection_only" or cfg["model"]["variant"] != "mstc":
        raise ValueError("Only frozen detection_only MSTC inference is allowed")
    for change in path_changes:
        effective = cfg
        for part in change["field"].split("."):
            effective = effective[int(part)] if isinstance(effective, list) else effective[part]
        change["windows"] = effective
    return cfg, path_changes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="After preflight, run replay + two validation/test windows")
    parser.add_argument("--probe-batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if os.name != "nt":
        raise RuntimeError("This entry point is for native Windows only")
    if args.probe_batch_size <= 0:
        raise ValueError("Probe batch size must be positive")
    run = args.run_root.resolve()
    if run.exists():
        raise FileExistsError("Use a new run root; existing experiments are never overwritten")
    run.mkdir(parents=True)
    result = {"status": "RUNNING", "stages": {}, "mode": "full" if args.full else "preflight",
              "real_data": True, "training": "NOT_RUN", "script_sha256": digest(__file__),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "entrypoint": "production factory/load_training_checkpoint, then orthrus.main(test,evaluate)",
              "checkpoint_selection_scope": "Only externally selected model_epoch_6; not a six-checkpoint selection rerun"}
    started = time.perf_counter()
    def save():
        (run / "smoke_result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    def stage(name, details=None):
        result["stages"][name] = {"status": "PASS", "elapsed_total_seconds": time.perf_counter() - started,
                                  **(details or {})}
        print("SMOKE_STAGE " + name + " PASS", flush=True)
        save()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["NLTK_DATA"] = str(ROOT / "artifacts/nltk_data") + os.pathsep + os.environ.get("NLTK_DATA", "")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = validate_manifest(manifest)
        data_root = args.data_root.resolve()
        if not no_symlinks(args.data_root.absolute()):
            raise ValueError("Data root must not traverse a link")
        for row in rows:
            path = data_root / row["relative_path"]
            if not no_symlinks(path) or not path.is_file() or digest(path) != row["sha256"]:
                raise ValueError("Input integrity failure: " + str(path))
        result["input_identity"] = [{"path": r["relative_path"], "sha256": r["sha256"]} for r in rows]
        result["manifest_sha256"] = digest(args.manifest)
        cp = next(row for row in rows if row["kind"] == "checkpoint")
        result["checkpoint_sha256"] = cp["sha256"]
        stage("input_integrity", {"files": len(rows)})

        import numpy as np
        import psutil
        import torch
        import yaml
        from yacs.config import CfgNode
        import data_utils
        import factory
        from detection import orthrus_gnn_testing as testing
        from detection.inference_stability import deterministic_inference, inference_rng, backend_metadata
        from mstc.time_gap import TimeGapStatistics
        from runtime.runtime_profiler import configure_profiler, finish_profiler
        from runtime.memory_metrics import peak_rss_mb

        raw, path_changes = configure(manifest, data_root, run)
        cfg = CfgNode(raw)
        cfg._ground_truth_dir = str(ROOT / "Ground_Truth/darpa")
        result["path_mapping"] = path_changes
        result["batch_sizes"] = {"Linux": manifest["resolved_config"]["detection"]["gnn_training"]["encoder"]["batch_size"],
                                  "Windows_full": cfg.detection.gnn_training.encoder.batch_size,
                                  "isolated_probe": args.probe_batch_size}
        result["versions"] = {name: importlib.metadata.version(name) for name in ["torch", "torch-geometric", "numpy", "psutil"]}
        result["python"] = sys.version
        result["cuda_build"] = torch.version.cuda
        result["available_ram_before_bytes"] = psutil.virtual_memory().available
        (run / "metadata").mkdir()
        source_stats = data_root / "artifacts/THEIA_E5/run_metadata/time_statistics.json"
        shutil.copyfile(source_stats, run / "metadata/time_statistics.json")
        (run / "config_resolved.windows.yml").write_text(cfg.dump(), encoding="utf-8")
        (run / "path_mapping.json").write_text(json.dumps(path_changes, indent=2), encoding="utf-8")
        stage("configuration")
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        # Match G3 flags before the first CUDA initialization.
        linux_env = json.loads((data_root / "artifacts/THEIA_E5/linux_reference/environment.json").read_text())
        result["linux_environment"] = linux_env
        torch.set_num_threads(linux_env["inference_stability"]["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = linux_env["inference_stability"]["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = linux_env["inference_stability"]["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(linux_env["inference_stability"]["float32_matmul_precision"])
        stats = TimeGapStatistics.load(run / "metadata/time_statistics.json")
        train, val, test, full, max_nodes = data_utils.load_all_datasets(cfg)
        contracts = {}
        for name, collection in [("train", train), ("val", val), ("test", test)]:
            if len(collection) != 2:
                raise ValueError("Exactly two genuine windows required for each split")
            contracts[name] = []
            for index, graph in enumerate(collection):
                if not bool((graph.t[1:] >= graph.t[:-1]).all()):
                    raise ValueError("Nonchronological real window")
                contracts[name].append({"index": index, "events": graph.num_events,
                                        "first_ns": int(graph.t[0]), "last_ns": int(graph.t[-1]),
                                        "fields": {key: {"shape": list(graph[key].shape), "dtype": str(graph[key].dtype)}
                                                   for key in graph.keys() if isinstance(graph[key], torch.Tensor)}})
        stage("real_loader", {"windows": contracts, "max_nodes": max_nodes})
        sample = train[0]
        cpu_model = factory.build_model(sample, torch.device("cpu"), cfg, max_nodes, stats)
        data_utils.load_training_checkpoint(cpu_model, str(data_root / cp["relative_path"]), map_location="cpu", restore_rng=False)
        stage("checkpoint_cpu", {"parameters": sum(p.numel() for p in cpu_model.parameters())})
        del cpu_model
        gc.collect()

        with deterministic_inference(cfg, "cuda"):
            free, total = torch.cuda.mem_get_info()
            result["gpu_free_before_bytes"], result["gpu_total_bytes"] = free, total
            result["gpu"] = torch.cuda.get_device_name()
            if free < 3 * 1024 ** 3:
                raise RuntimeError("Insufficient free VRAM for conservative probe (requires 3 GiB)")
            torch.cuda.reset_peak_memory_stats()
            probe_cfg = cfg.clone()
            probe_cfg.detection.gnn_training.encoder.batch_size = args.probe_batch_size
            with inference_rng(cfg, "cuda", "construction"):
                model = factory.build_model(sample, torch.device("cuda"), probe_cfg, max_nodes, stats)
                data_utils.load_training_checkpoint(model, str(data_root / cp["relative_path"]), map_location="cpu", restore_rng=False)
            model.eval(); model.reset_state()
            graph = train[0].to("cuda")
            batch = next(iter(factory.batch_loader_factory(probe_cfg, graph, model.graph_reindexer)))
            with inference_rng(cfg, "cuda", "replay"), torch.no_grad():
                outputs = model(batch, full, inference=True)
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError("Nonfinite real probe output: " + key)
            torch.cuda.synchronize()
            probe_peak = torch.cuda.max_memory_allocated()
            stage("real_gpu_probe", {"events": batch.edge_index.shape[1], "peak_allocated_bytes": probe_peak,
                                     "backend": backend_metadata()})
            testing._cleanup_checkpoint_cuda(model, torch.device("cuda"))
            graph.to("cpu")
            del outputs, batch, graph, model, probe_cfg, train, val, test, full, sample, stats
            gc.collect(); torch.cuda.empty_cache()
            result["probe_peak_gpu_allocated_bytes"] = probe_peak
            if args.full:
                # The isolated probe used 128; the full run retains G3's 512.
                configure_profiler(enabled=False, basic_timing=True, run_root=run)
                from mstc.experiment_utils import dump_environment
                dump_environment(cfg, run)
                import orthrus
                from types import SimpleNamespace
                stage_start = time.perf_counter()
                timings = orthrus.main(cfg, SimpleNamespace(stages="test,evaluate", run_from_training=False))
                torch.cuda.synchronize()
                finish_profiler(status="completed")
                stage("real_two_windows", {"seconds": time.perf_counter()-stage_start, "timings": timings})
                result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            result["backend"] = backend_metadata()
        result["peak_ram_mib"] = peak_rss_mb()
        result["seconds"] = time.perf_counter() - started
        result["status"] = "PASS"
        result["exit_code"] = 0
        result["real_two_window_inference"] = "PASS" if args.full else "NOT_RUN"
        save()
        return 0
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}", exit_code=1,
                      seconds=time.perf_counter()-started)
        traceback.print_exc()
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
