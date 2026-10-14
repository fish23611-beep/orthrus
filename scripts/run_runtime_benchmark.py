#!/usr/bin/env python3
"""Reproducible, isolated E3/E5 runtime benchmark launcher.

Dry-run is the default.  Measured windows are executed after all causal-prefix
windows, but aggregation excludes both the prefix and configured warm-up epochs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "src" / "experiments" / "run_experiment.py"
PROFILE_ENV = "ORTHRUS_PERF_PROFILE"
PROFILE_SAMPLE_ENV = "ORTHRUS_PERF_PROFILE_SAMPLE_EVERY_N_BATCHES"


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("THEIA_E3", "THEIA_E5"))
    parser.add_argument(
        "--config", default=str(ROOT / "config/experiments/mstc_full.yml")
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--shared-artifact-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", required=True, type=int, choices=(2, 10, 20))
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--selection-manifest")
    parser.add_argument(
        "--bootstrap-inventory",
        action="store_true",
        help="Run a bounded diagnostic inventory; never publishes a formal baseline.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--profile-mode", choices=("on", "off"), required=True)
    parser.add_argument("--basic-timing", choices=("on", "off"), default="on")
    parser.add_argument("--torch-trace-batches", type=int, default=0)
    parser.add_argument("--gpu-sample-ms", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def validate_config_semantics(config_path):
    def merge(base, override):
        result = dict(base)
        for key, value in override.items():
            result[key] = (
                merge(result.get(key, {}), value)
                if isinstance(value, dict) and isinstance(result.get(key), dict)
                else value
            )
        return result

    base = yaml.safe_load((ROOT / "config/orthrus.yml").read_text())
    overlay = yaml.safe_load(Path(config_path).read_text())
    cfg = merge(base, overlay)
    training = cfg["detection"]["gnn_training"]
    multi = training["encoder"]["context"]["multiscale"]
    expected = {
        "batch_size": (training["encoder"]["batch_size"], 512),
        "epochs": (training["num_epochs"], 6),
        "neighbor_budgets": (multi["neighbor_budgets"], [8, 8, 8]),
        "lambda_time": (training["decoder"]["time_gap"]["lambda_time"], 0.3),
        "share_encoder": (multi["share_encoder"], True),
        "fusion": (multi["fusion"], "gated"),
        "scale_embedding": (multi["use_scale_embedding"], True),
        "corpus_scope": (cfg["semantic_features"]["corpus_scope"], "train_only"),
        "model_selection": (cfg["model_selection"]["method"], "min_val_objective"),
    }
    bad = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if bad:
        raise ValueError(f"benchmark config violates semantic guard: {bad}")
    return int(training["num_epochs"])


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_isolated_roots(args) -> tuple[Path, Path]:
    try:
        shared = Path(args.shared_artifact_root).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"shared artifact root does not exist: {args.shared_artifact_root}"
        ) from exc
    if not shared.is_dir():
        raise ValueError(f"shared artifact root is not a directory: {shared}")
    try:
        next(shared.iterdir())
    except StopIteration as exc:
        raise ValueError(f"shared artifact root is empty: {shared}") from exc
    except OSError as exc:
        raise ValueError(f"shared artifact root is not readable: {shared}") from exc

    output = Path(args.artifact_root).expanduser().resolve(strict=False)
    if output.exists():
        raise ValueError(f"benchmark artifact root already exists: {output}")
    if (
        output == shared
        or _is_relative_to(output, shared)
        or _is_relative_to(shared, output)
    ):
        raise ValueError(
            "benchmark artifact root and shared artifact root overlap after "
            f"symlink resolution: output={output}, shared={shared}"
        )
    return output, shared


def resolve_benchmark_config(args, shared: Path):
    """Resolve the effective benchmark config without writing to shared."""
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import config as config_module

    namespace = SimpleNamespace(
        dataset=args.dataset,
        model="orthrus",
        config=str(Path(args.config).resolve()),
        cpu=False,
        from_weights=False,
        seed=args.seed,
        skip_tracing=False,
        artifact_root=str(shared.parent / ".benchmark-preflight-unused"),
        shared_artifact_root=str(shared),
        shared_artifact_readonly=True,
        max_windows_per_split=args.window_start + args.windows,
    )
    # Ground truth does not participate in preprocessing task hashes.  Avoid
    # making a detection-only input preflight depend on an unrelated submodule;
    # the real child process still performs the normal resolver check.
    original_ground_truth_resolver = config_module._resolve_ground_truth_dir
    config_module._resolve_ground_truth_dir = (
        lambda cfg=None: str(ROOT / "Ground_Truth" / "darpa")
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = config_module.get_yml_cfg(namespace)
    finally:
        config_module._resolve_ground_truth_dir = original_ground_truth_resolver

    return cfg


def resolve_expected_shared_inputs(args, shared: Path) -> tuple[Path, Path]:
    """Resolve exact task-hash inputs without creating shared directories."""
    cfg = resolve_benchmark_config(args, shared)
    model = Path(
        cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
    ) / "feature_word2vec.model"
    edge_root = Path(cfg.edge_featurization.embed_edges._edge_embeds_dir)
    return model, edge_root


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_window_manifest(
    args, *, required: bool, expected_num_epochs: int | None = None
) -> dict[str, Any] | None:
    """Validate a frozen planner manifest and report its provenance honestly."""
    if not args.selection_manifest:
        if required:
            raise ValueError("formal execute requires --selection-manifest")
        return None

    path = Path(args.selection_manifest).expanduser().resolve(strict=True)
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"selection manifest is not valid JSON: {path}") from exc
    if not isinstance(plan, dict):
        raise ValueError("selection manifest must be a JSON object")

    required_fields = {
        "schema_version", "dataset", "source", "source_sha256",
        "source_sha256_scope", "warmup_epochs_excluded", "eligible_epochs",
        "selected_start", "selected_end_exclusive", "window_count",
    }
    missing = sorted(required_fields - set(plan))
    if missing:
        raise ValueError(f"selection manifest missing required fields: {missing}")
    if plan["schema_version"] != 2:
        raise ValueError(
            f"unsupported selection manifest schema_version: {plan['schema_version']!r}"
        )
    if plan["dataset"] != args.dataset:
        raise ValueError("selection manifest does not match dataset")
    if not isinstance(plan["source"], str) or not plan["source"].strip():
        raise ValueError("selection manifest source must be a non-empty path")
    if not _is_sha256(plan["source_sha256"]):
        raise ValueError("selection manifest source_sha256 must be 64 hexadecimal characters")
    if plan["source_sha256_scope"] != "content_sha256_of_batches_csv_file":
        raise ValueError("selection manifest source_sha256_scope is unsupported")

    integer_fields = (
        "warmup_epochs_excluded", "selected_start",
        "selected_end_exclusive", "window_count",
    )
    if any(
        not isinstance(plan[field], int) or isinstance(plan[field], bool)
        for field in integer_fields
    ):
        raise ValueError("manifest warmup/start/end/count fields must be integers")
    start = plan["selected_start"]
    end = plan["selected_end_exclusive"]
    count = plan["window_count"]
    if start < 0 or count <= 0 or end != start + count:
        raise ValueError("selection manifest start/end/count are inconsistent")
    if start != args.window_start or count != args.windows:
        raise ValueError("selection manifest start/end/count do not match benchmark request")
    if plan["warmup_epochs_excluded"] != args.warmup_epochs:
        raise ValueError("selection manifest warmup rule does not match benchmark request")

    eligible_epochs = plan["eligible_epochs"]
    if (
        not isinstance(eligible_epochs, list)
        or not eligible_epochs
        or any(not isinstance(epoch, int) or isinstance(epoch, bool) for epoch in eligible_epochs)
        or len(set(eligible_epochs)) != len(eligible_epochs)
        or any(epoch <= args.warmup_epochs for epoch in eligible_epochs)
    ):
        raise ValueError("selection manifest eligible_epochs are invalid")

    expected_eligible_epochs = list(
        range(args.warmup_epochs + 1, max(eligible_epochs) + 1)
    )
    if eligible_epochs != expected_eligible_epochs:
        raise ValueError(
            "selection manifest eligible_epochs must be ordered and contiguous "
            "after the excluded warmup epochs"
        )
    if expected_num_epochs is not None:
        protocol_epochs = list(
            range(args.warmup_epochs + 1, expected_num_epochs + 1)
        )
        if eligible_epochs != protocol_epochs:
            raise ValueError(
                "selection manifest eligible_epochs do not match benchmark epochs"
            )


    source = Path(plan["source"]).expanduser()
    if not source.is_absolute():
        source = path.parent / source
    source = source.resolve(strict=False)
    if not source.is_file():
        raise ValueError(
            f"selection manifest source Batch CSV does not exist: {source}"
        )
    actual_source_sha256 = _sha256_file(source)
    if actual_source_sha256 != plan["source_sha256"].lower():
        raise ValueError("selection manifest source SHA256 verification failed")

    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"epoch", "window_index"} <= set(reader.fieldnames or []):
                raise ValueError(
                    "selection manifest source Batch CSV lacks epoch/window_index"
                )
            source_pairs = set()
            for position, row in enumerate(reader):
                try:
                    epoch = int(row["epoch"])
                    window_index = int(row["window_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "selection manifest source Batch CSV has invalid "
                        f"epoch/window_index at row {position + 2}"
                    ) from exc
                if epoch > args.warmup_epochs:
                    source_pairs.add((epoch, window_index))
    except OSError as exc:
        raise ValueError(
            f"selection manifest source Batch CSV cannot be read: {source}"
        ) from exc
    source_epochs = sorted({epoch for epoch, _ in source_pairs})
    if source_epochs != eligible_epochs:
        raise ValueError(
            "selection manifest eligible_epochs do not match source Batch CSV"
        )
    required_source_pairs = {
        (epoch, window_index)
        for epoch in eligible_epochs
        for window_index in range(start, end)
    }
    if not required_source_pairs <= source_pairs:
        raise ValueError(
            "selection manifest source Batch CSV does not cover every selected "
            "eligible epoch/window"
        )
    source_verified = True
    source_status = "verified_against_available_source_file"

    return {
        "path": str(path),
        "manifest_sha256": _sha256_file(path),
        "schema_version": plan["schema_version"],
        "dataset": plan["dataset"],
        "selected_window_indices": list(range(start, end)),
        "selected_start": start,
        "selected_end_exclusive": end,
        "window_count": count,
        "warmup_epochs_excluded": plan["warmup_epochs_excluded"],
        "eligible_epochs": list(eligible_epochs),
        "source": str(source),
        "source_sha256": plan["source_sha256"].lower(),
        "source_sha256_scope": plan["source_sha256_scope"],
        "source_content_verified": source_verified,
        "source_verification_status": source_status,
        "source_observed_epoch_window_count": len(source_pairs),
    }


def resolved_config_sha256(args, shared: Path) -> str:
    """Hash the canonical, redacted effective CfgNode used by the child run."""
    from run_metadata import _cfg_to_dict, _redact_database_password

    cfg = resolve_benchmark_config(args, shared)
    plain = _redact_database_password(_cfg_to_dict(cfg))
    canonical = yaml.safe_dump(
        plain, allow_unicode=True, default_flow_style=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_identity() -> dict[str, Any]:
    result: dict[str, Any] = {"git_sha": None, "git_dirty": None}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=False, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=False, timeout=10,
        )
        if head.returncode == 0:
            result["git_sha"] = head.stdout.strip()
        if status.returncode == 0:
            result["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _software_identity() -> dict[str, Any]:
    result = {"torch_version": None, "cuda_runtime_version": None}
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_runtime_version"] = torch.version.cuda
    except (ImportError, OSError):
        pass
    return result


def _gpu_identity(*, require_resolved: bool = False) -> dict[str, Any]:
    """Resolve logical cuda:0 to one UUID without guessing container mappings."""
    result = {
        "gpu_uuid": None,
        "gpu_uuids": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpu_identity_scope": "selected logical cuda:0 plus physical inventory",
        "gpu_mapping_validation_status": "not_runtime_verified",
        "gpu_identity_runtime_verified": False,
    }
    executable = shutil.which("nvidia-smi")
    if not executable:
        if require_resolved:
            raise ValueError("formal execute cannot resolve a unique GPU UUID: nvidia-smi missing")
        return result
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if completed.returncode != 0:
            raise ValueError("nvidia-smi GPU inventory query failed")
        inventory = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                inventory.append((parts[0], parts[1]))
        indices = [index for index, _ in inventory]
        uuids = [uuid for _, uuid in inventory]
        result["gpu_uuids"] = uuids
        if not inventory or len(indices) != len(set(indices)) or len(uuids) != len(set(uuids)):
            raise ValueError("nvidia-smi GPU inventory is empty or ambiguous")

        cuda_visible = result["cuda_visible_devices"]
        nvidia_visible = result["nvidia_visible_devices"]
        cuda_tokens = (
            [token.strip() for token in cuda_visible.split(",") if token.strip()]
            if cuda_visible not in (None, "") else []
        )
        nvidia_tokens = (
            [token.strip() for token in nvidia_visible.split(",") if token.strip()]
            if nvidia_visible not in (None, "", "all", "void", "none") else []
        )
        if len(cuda_tokens) > 1:
            raise ValueError("formal execute requires exactly one CUDA_VISIBLE_DEVICES selector")
        if any(token.startswith("MIG-") for token in cuda_tokens + nvidia_tokens):
            raise ValueError("MIG device mapping is not supported by this benchmark runner")

        selector = cuda_tokens[0] if cuda_tokens else None
        target_uuid = None
        mapping_status = None
        if selector and selector.startswith("GPU-"):
            target_uuid = selector
            mapping_status = "validated_cuda_visible_uuid_against_nvidia_smi"
        elif selector and selector.isdigit():
            if len(nvidia_tokens) == 1 and nvidia_tokens[0].startswith("GPU-"):
                if selector != "0":
                    raise ValueError("container ordinal mapping is ambiguous for nonzero selector")
                target_uuid = nvidia_tokens[0]
                mapping_status = "validated_container_ordinal_zero_via_nvidia_visible_uuid"
            elif nvidia_tokens:
                raise ValueError("container GPU index mapping is ambiguous")
            elif len(inventory) > 1:
                raise ValueError(
                    "numeric CUDA_VISIBLE_DEVICES cannot be mapped reliably to "
                    "nvidia-smi on a multi-GPU host; use a GPU UUID"
                )
            else:
                target_uuid = next(
                    (uuid for index, uuid in inventory if index == selector), None
                )
                mapping_status = "validated_numeric_selector_against_nvidia_smi_index"
        elif selector:
            raise ValueError("CUDA_VISIBLE_DEVICES selector format is unsupported")
        elif len(nvidia_tokens) == 1 and nvidia_tokens[0].startswith("GPU-"):
            target_uuid = nvidia_tokens[0]
            mapping_status = "validated_single_nvidia_visible_uuid"
        elif nvidia_tokens:
            raise ValueError("NVIDIA_VISIBLE_DEVICES mapping is ambiguous")
        elif len(inventory) == 1:
            target_uuid = inventory[0][1]
            mapping_status = "validated_single_inventory_gpu"
        else:
            raise ValueError("formal execute cannot resolve a unique GPU UUID")

        if target_uuid not in uuids:
            raise ValueError("selected GPU UUID does not match nvidia-smi inventory")
        result["gpu_uuid"] = target_uuid
        result["gpu_mapping_validation_status"] = mapping_status
        result["gpu_identity_runtime_verified"] = True
    except (OSError, subprocess.SubprocessError, ValueError):
        if require_resolved:
            raise
    return result


def experiment_identity(
    args, shared: Path, *, manifest_validation=None, gpu_identity=None
) -> dict[str, Any]:
    manifest = manifest_validation
    result = {
        **_git_identity(),
        **_software_identity(),
        **(gpu_identity if gpu_identity is not None else _gpu_identity()),
        "resolved_config_sha256": resolved_config_sha256(args, shared),
        "resolved_config_hash_scope": "canonical redacted effective CfgNode",
        "dataset": args.dataset,
        "seed": args.seed,
        "window_manifest_sha256": manifest["manifest_sha256"] if manifest else None,
    }
    if manifest:
        result.update({
            "window_manifest_schema_version": manifest["schema_version"],
            "window_manifest_source": manifest["source"],
            "window_manifest_source_sha256": manifest["source_sha256"],
            "window_manifest_source_sha256_scope": manifest["source_sha256_scope"],
            "window_manifest_source_content_verified": manifest["source_content_verified"],
            "window_manifest_source_verification_status": manifest["source_verification_status"],
        })
    return result


def validate_shared_training_inputs(args, shared: Path) -> dict[str, Any]:
    model, edge_root = resolve_expected_shared_inputs(args, shared)
    missing = []
    if not model.is_file():
        missing.append(str(model))
    split_counts = {}
    for split in ("train", "val", "test"):
        split_dir = edge_root / split
        split_counts[split] = (
            sum(1 for path in split_dir.iterdir() if path.is_file())
            if split_dir.is_dir()
            else 0
        )
        if not split_counts[split]:
            missing.append(str(split_dir))
    for path in (model, edge_root):
        resolved = path.resolve(strict=False)
        if not _is_relative_to(resolved, shared):
            raise ValueError(
                "resolved shared input escapes shared artifact root: "
                f"path={resolved}, shared={shared}"
            )
    if missing:
        raise ValueError(
            "shared artifact root lacks exact task-hash training inputs: "
            + ", ".join(missing)
        )
    return {
        "word2vec_model": str(model.relative_to(shared)),
        "edge_embed_root": str(edge_root.relative_to(shared)),
        "split_file_counts": split_counts,
    }


def validate(args):
    epochs = validate_config_semantics(args.config)
    bootstrap_inventory = bool(getattr(args, "bootstrap_inventory", False))
    if args.seed != 0:
        raise ValueError("formal E3/E5 benchmark requires --seed 0")
    if args.window_start < 0 or args.torch_trace_batches < 0:
        raise ValueError("window and trace counts must be non-negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= epochs:
        raise ValueError(
            f"warmup epochs must be in [0, {epochs - 1}], got {args.warmup_epochs}"
        )
    if args.gpu_sample_ms <= 0:
        raise ValueError("gpu sample interval must be positive")
    if bootstrap_inventory:
        if args.windows not in (2, 10):
            raise ValueError("bootstrap inventory allows only 2 or 10 windows")
        if args.window_start != 0:
            raise ValueError("bootstrap inventory requires --window-start 0")
        if args.selection_manifest:
            raise ValueError("bootstrap inventory must not use a selection manifest")
        if args.profile_mode != "on":
            raise ValueError("bootstrap inventory requires --profile-mode on")
        if args.torch_trace_batches:
            raise ValueError("bootstrap inventory does not allow Torch trace")
        manifest = None
    else:
        if args.windows != 20 and args.window_start != 0:
            raise ValueError(
                "non-zero --window-start requires the 20-window manifest case"
            )
        manifest = validate_window_manifest(
            args,
            required=bool(args.execute or args.windows == 20),
            expected_num_epochs=epochs,
        )
    if args.profile_mode == "off" and args.torch_trace_batches:
        raise ValueError("torch trace requires --profile-mode on")
    roots = resolve_isolated_roots(args)
    validate_shared_training_inputs(args, roots[1])
    args._benchmark_num_epochs = epochs
    args._validated_manifest = manifest
    args._benchmark_protocol = (
        "bootstrap_inventory" if bootstrap_inventory else "formal"
    )
    return roots


def build_command(args, *, roots=None):
    output, shared = roots or resolve_isolated_roots(args)
    end = args.window_start + args.windows
    command = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "--dataset",
        args.dataset,
        "--config",
        str(Path(args.config).resolve()),
        "--seed",
        str(args.seed),
        "--artifact-root",
        str(output),
        "--shared-artifact-root",
        str(shared),
        "--shared-artifact-readonly",
        "--stages",
        "train",
        "--max-windows-per-split",
        str(end),
    ]
    if args.basic_timing == "on":
        command.append("--benchmark-basic-timing")
    if args.profile_mode == "on":
        command += ["--perf-profile", "--perf-profile-sample-every-n-batches", "1"]
        if args.torch_trace_batches:
            command += [
                "--perf-profile-torch-trace-batches",
                str(args.torch_trace_batches),
                "--perf-profile-torch-trace-window-start",
                str(args.window_start),
            ]
    return command


def controlled_child_environment(parent=None):
    environment = dict(os.environ if parent is None else parent)
    environment.pop(PROFILE_ENV, None)
    environment.pop(PROFILE_SAMPLE_ENV, None)
    return environment


def shared_tree_snapshot(root: Path) -> dict[str, Any]:
    """Metadata-only tree fingerprint; does not read large artifact contents."""
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        if path.is_symlink():
            kind = "symlink"
            extra = os.readlink(path)
        elif path.is_dir():
            kind = "directory"
            extra = ""
        elif path.is_file():
            kind = "file"
            extra = str(stat.st_size)
        else:
            kind = "other"
            extra = ""
        entries.append(
            (relative, kind, extra, stat.st_mode, stat.st_size, stat.st_mtime_ns)
        )
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return {
        "entry_count": len(entries),
        "metadata_fingerprint_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "fingerprint_scope": "metadata_only:path,type,mode,size,mtime_ns",
        "content_sha256": None,
        "content_hash_status": "not_computed",
    }


def start_gpu_sampler(path, interval_ms, gpu_uuid=None):
    if not gpu_uuid:
        raise ValueError("GPU UUID is required; sampler may not fall back to all devices")
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ValueError("nvidia-smi is required for bound GPU sampling")
    handle = path.open("w", encoding="utf-8")
    command = [
        executable,
        f"--id={gpu_uuid}",
        "--query-gpu=timestamp,index,name,utilization.gpu,"
        "utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
        "--format=csv",
        "-lms",
        str(interval_ms),
    ]
    try:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT
        )
    except Exception:
        handle.close()
        raise
    return process, handle


def _latest(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.rglob(pattern), key=lambda path: path.stat().st_mtime_ns)
    return candidates[-1] if candidates else None


def _load_measurement_rows(root: Path, profile_mode: str, basic_timing: str):
    if profile_mode == "on":
        path = _latest(root, "profiling/perf_batches.csv")
        kind = "detailed_profile"
    elif basic_timing == "on":
        path = _latest(root, "benchmark_timing/basic_batches.csv")
        kind = "basic_timing"
    else:
        path = None
        kind = "pure_wall"
    if path is None:
        return kind, None, []
    with path.open(newline="", encoding="utf-8") as handle:
        return kind, path, list(csv.DictReader(handle))


def validate_execution_evidence(root: Path, args, *, num_epochs: int) -> dict[str, Any]:
    """Fail closed unless runtime receipt and mode-specific Batch evidence agree."""
    errors: list[dict[str, str]] = []

    def reject(code: str, message: str) -> None:
        if code not in {item["code"] for item in errors}:
            errors.append({"code": code, "message": message})

    runtime_path = _latest(root, "runtime.json")
    runtime = None
    if runtime_path is None:
        reject("runtime_receipt_missing", "runtime.json is missing")
    else:
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            reject("runtime_receipt_invalid", f"runtime.json cannot be parsed: {exc}")

    if isinstance(runtime, dict):
        if runtime.get("status") != "completed":
            reject("runtime_not_completed", "runtime status is not completed")
        if "train" not in runtime.get("executed_stages", []):
            reject("training_stage_not_completed", "runtime does not record train stage completion")
        training = runtime.get("training")
        receipt = training.get("execution_receipt") if isinstance(training, dict) else None
    else:
        receipt = None

    if not isinstance(receipt, dict):
        reject("execution_receipt_missing", "training execution receipt is missing")
        receipt_windows = []
    else:
        if receipt.get("schema_version") != 1 or receipt.get("status") != "completed":
            reject("execution_receipt_invalid", "execution receipt schema/status is invalid")
        if receipt.get("start_epoch") != 1 or receipt.get("end_epoch") != num_epochs:
            reject("execution_receipt_epoch_range_mismatch", "receipt epoch range is incomplete")
        receipt_windows = receipt.get("windows")
        if not isinstance(receipt_windows, list):
            reject("execution_receipt_invalid", "receipt windows must be a list")
            receipt_windows = []

    execution_window_indices = set(range(args.window_start + args.windows))
    measured_window_indices = set(
        range(args.window_start, args.window_start + args.windows)
    )
    all_epochs = set(range(1, num_epochs + 1))
    post_warmup_epochs = set(range(args.warmup_epochs + 1, num_epochs + 1))
    expected_receipt_pairs = {
        (epoch, window_index)
        for epoch in all_epochs
        for window_index in execution_window_indices
    }
    expected_measured_pairs = {
        (epoch, window_index)
        for epoch in post_warmup_epochs
        for window_index in measured_window_indices
    }

    receipt_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for position, item in enumerate(receipt_windows):
        if not isinstance(item, dict):
            reject("execution_receipt_invalid", f"receipt window {position} is not an object")
            continue
        try:
            epoch = int(item["epoch"])
            window_index = int(item["window_index"])
            event_count = int(item["event_count"])
            expected_batches = int(item["expected_batch_count"])
        except (KeyError, TypeError, ValueError):
            reject("execution_receipt_invalid", f"receipt window {position} has invalid fields")
            continue
        pair = (epoch, window_index)
        if pair in receipt_by_pair:
            reject("duplicate_receipt_window", f"receipt repeats epoch/window {pair}")
            continue
        if event_count < 0 or expected_batches < 0:
            reject("execution_receipt_invalid", f"receipt has negative counts for {pair}")
            continue
        receipt_by_pair[pair] = {
            "event_count": event_count,
            "expected_batch_count": expected_batches,
        }

    receipt_pairs = set(receipt_by_pair)
    if receipt_pairs != expected_receipt_pairs:
        reject(
            "receipt_window_coverage_mismatch",
            "execution receipt does not exactly cover every expected epoch/window",
        )

    source_kind, source_path, rows = _load_measurement_rows(
        root, args.profile_mode, args.basic_timing
    )
    batch_rows_required = args.profile_mode == "on" or args.basic_timing == "on"
    profile_path = _latest(root, "profiling/perf_profile.json")
    profile_payload = None
    if args.profile_mode == "on":
        if profile_path is None:
            reject("detailed_profile_missing", "detailed perf_profile.json is missing")
        else:
            try:
                profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                reject("detailed_profile_invalid", f"detailed profile cannot be parsed: {exc}")
            if not isinstance(profile_payload, dict):
                reject("detailed_profile_invalid", "detailed profile must be a JSON object")
            elif (
                profile_payload.get("enabled") is not True
                or profile_payload.get("status") != "completed"
            ):
                reject("detailed_profile_invalid", "detailed profile is not completed/enabled")
        if args.torch_trace_batches:
            trace_path = _latest(root, "profiling/torch_trace.json")
            if trace_path is None:
                reject("torch_trace_missing", "requested independent Torch trace is missing")
    elif profile_path is not None:
        reject("unexpected_detailed_profile", "profile-off run produced a detailed profile")

    if batch_rows_required and source_path is None:
        reject("measurement_rows_missing", f"{source_kind} Batch CSV is missing")
    elif batch_rows_required and not rows:
        reject("measurement_rows_empty", f"{source_kind} Batch CSV has no records")
    elif not batch_rows_required and source_path is not None:
        reject("unexpected_measurement_rows", "pure-off run unexpectedly produced Batch timing rows")

    parsed_rows: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        try:
            parsed_rows.append({
                "batch_index": int(row["batch_index"]),
                "epoch": int(row["epoch"]),
                "window_index": int(row["window_index"]),
                "batch_events": int(row["batch_events"]),
                "total_seconds": float(row["total_seconds"]),
            })
        except (KeyError, TypeError, ValueError):
            reject("measurement_row_invalid", f"Batch row {position} has invalid required fields")

    for position, row in enumerate(parsed_rows):
        if row["batch_events"] <= 0:
            reject(
                "measurement_row_invalid",
                f"Batch row {position} must contain a positive event count",
            )
        if not math.isfinite(row["total_seconds"]) or row["total_seconds"] <= 0:
            reject(
                "measurement_row_invalid",
                f"Batch row {position} must contain a finite positive duration",
            )

    if parsed_rows:
        batch_indices = [row["batch_index"] for row in parsed_rows]
        if len(batch_indices) != len(set(batch_indices)):
            reject("duplicate_batch_index", "Batch CSV contains duplicate batch_index values")
        if sorted(batch_indices) != list(range(1, len(batch_indices) + 1)):
            reject("batch_index_sequence_invalid", "Batch indices are not one complete sequence")

        actual_pairs = {(row["epoch"], row["window_index"]) for row in parsed_rows}
        expected_batch_pairs = {
            pair for pair, item in receipt_by_pair.items()
            if item["expected_batch_count"] > 0
        }
        unexpected_pairs = actual_pairs - expected_receipt_pairs
        if unexpected_pairs:
            reject(
                "unexpected_batch_epoch_window",
                f"Batch CSV contains unexpected epoch/window pairs: {sorted(unexpected_pairs)}",
            )
        if actual_pairs != expected_batch_pairs:
            reject(
                "batch_window_coverage_mismatch",
                "Batch CSV does not exactly cover receipt epoch/windows with expected batches",
            )

        batches_by_pair = Counter(
            (row["epoch"], row["window_index"]) for row in parsed_rows
        )
        events_by_pair: dict[tuple[int, int], int] = defaultdict(int)
        for row in parsed_rows:
            pair = (row["epoch"], row["window_index"])
            events_by_pair[pair] += row["batch_events"]
        for pair, receipt_item in receipt_by_pair.items():
            if batches_by_pair[pair] != receipt_item["expected_batch_count"]:
                reject(
                    "batch_count_mismatch",
                    f"Batch count for {pair} differs from execution receipt",
                )
            if events_by_pair[pair] != receipt_item["event_count"]:
                reject(
                    "batch_event_count_mismatch",
                    f"Batch events for {pair} differ from execution receipt",
                )

    observed_measured_pairs = {
        (row["epoch"], row["window_index"])
        for row in parsed_rows
        if row["epoch"] in post_warmup_epochs
        and row["window_index"] in measured_window_indices
    }
    if batch_rows_required and observed_measured_pairs != expected_measured_pairs:
        reject(
            "measured_window_coverage_mismatch",
            "post-warmup measured epoch/window coverage is incomplete",
        )

    return {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "error_codes": [item["code"] for item in errors],
        "runtime_path": str(runtime_path) if runtime_path else None,
        "measurement_source": source_kind,
        "measurement_path": str(source_path) if source_path else None,
        "batch_rows_required": batch_rows_required,
        "batch_row_count": len(parsed_rows),
        "expected_execution_window_indices": sorted(execution_window_indices),
        "expected_measured_window_indices": sorted(measured_window_indices),
        "expected_post_warmup_epochs": sorted(post_warmup_epochs),
        "receipt_pair_count": len(receipt_pairs),
        "expected_receipt_pair_count": len(expected_receipt_pairs),
        "observed_measured_pair_count": len(observed_measured_pairs),
        "expected_measured_pair_count": len(expected_measured_pairs),
    }


def aggregate(
    root, start, count, *, profile_mode, basic_timing, warmup_epochs,
    torch_trace_batches=0,
):
    runtime_path = _latest(root, "runtime.json")
    runtime_payload = (
        json.loads(runtime_path.read_text()) if runtime_path is not None else None
    )
    source_kind, source_path, rows = _load_measurement_rows(
        root, profile_mode, basic_timing
    )
    parsed = [
        row
        for row in rows
        if row.get("epoch") not in ("", None)
        and row.get("window_index") not in ("", None)
    ]
    warmup = [row for row in parsed if int(row["epoch"]) <= warmup_epochs]
    post_warmup = [row for row in parsed if int(row["epoch"]) > warmup_epochs]
    measured = [
        row
        for row in post_warmup
        if start <= int(row["window_index"]) < start + count
    ]
    causal_replay = [
        row for row in post_warmup if int(row["window_index"]) < start
    ]
    excluded_other = [
        row
        for row in post_warmup
        if int(row["window_index"]) >= start + count
    ]
    result = {
        "measurement_source": source_kind,
        "measurement_rows_found": source_path is not None,
        "source": str(source_path) if source_path is not None else None,
        "runtime": runtime_payload,
        "warmup_epochs": warmup_epochs,
        "warmup_batch_count": len(warmup),
        "causal_replay_batch_count": len(causal_replay),
        "excluded_other_batch_count": len(excluded_other),
        "measured_batch_count": len(measured),
        "measured_window_start": start,
        "measured_window_count": count,
        "measured_distinct_window_indices": sorted(
            {int(row["window_index"]) for row in measured}
        ),
        "measured_observed_window_count": len(
            {int(row["window_index"]) for row in measured}
        ),
        "measured_epoch_window_count": len(
            {(int(row["epoch"]), int(row["window_index"])) for row in measured}
        ),
        "stages": {},
        "statistics_status": (
            "diagnostic_trace_not_throughput"
            if torch_trace_batches
            else (
                "published"
                if measured
                else "execution_complete_no_batch_throughput"
            )
        ),
    }
    time_keys = sorted(
        {key for row in measured for key in row if key.startswith("time.")}
        | {"total_seconds"}
    )
    for key in time_keys:
        values = [
            float(row[key])
            for row in measured
            if row.get(key) not in ("", None)
        ]
        if values:
            result["stages"][key] = {
                "total_seconds": sum(values),
                "p50_seconds": percentile(values, 0.50),
                "p95_seconds": percentile(values, 0.95),
                "count": len(values),
            }
    events = sum(int(row.get("batch_events", 0) or 0) for row in measured)
    seconds = sum(float(row.get("total_seconds", 0) or 0) for row in measured)
    identity = [
        {
            "batch_index": int(row["batch_index"]),
            "epoch": int(row["epoch"]),
            "window_index": int(row["window_index"]),
            "batch_events": int(row.get("batch_events", 0) or 0),
        }
        for row in measured
    ]
    batch_metadata_sha256 = None
    if identity:
        identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        batch_metadata_sha256 = hashlib.sha256(identity_json.encode()).hexdigest()
    publish_throughput = not torch_trace_batches and bool(measured)
    result.update({
        "events": events if measured else None,
        "events_per_second": events / seconds if publish_throughput and seconds else None,
        "batches_per_second": len(measured) / seconds if publish_throughput and seconds else None,
        "batch_metadata_sha256": batch_metadata_sha256,
        # Backward-compatible alias. This is metadata-only, never content identity.
        "measurement_identity_sha256": batch_metadata_sha256,
        "measurement_identity_scope": (
            "batch_metadata_only:batch_index,epoch,window_index,batch_events"
            if batch_metadata_sha256 else "unavailable:no_batch_rows"
        ),
        "measurement_identity_is_content_hash": False,
    })
    profile_path = _latest(root, "profiling/perf_profile.json")
    if profile_path is not None:
        profile = json.loads(profile_path.read_text())
        result["memory"] = profile.get("memory")
        result["observations"] = profile.get("observations")
        result["timing_semantics"] = {
            "cpu": "perf_counter inclusive/self",
            "cuda": "CUDA Event interval; not kernel sum",
            "launch_gap": "torch_trace.json only",
        }
    elif basic_timing == "on":
        result["timing_semantics"] = {
            "cpu": "minimal host perf_counter Batch wall",
            "cuda": "no CUDA Event and no forced synchronization",
        }
    return result


def main(argv=None):
    args = parse_args(argv)
    roots = validate(args)
    bootstrap_inventory = args._benchmark_protocol == "bootstrap_inventory"
    output, shared = roots
    command = build_command(args, roots=roots)
    manifest_validation = getattr(args, "_validated_manifest", None)
    static_gpu_identity = _gpu_identity(require_resolved=False)
    if not args.execute:
        static_status = static_gpu_identity["gpu_mapping_validation_status"]
        static_gpu_identity["gpu_mapping_validation_status"] = (
            f"dry_run_static_candidate:{static_status}"
        )
        static_gpu_identity["gpu_identity_runtime_verified"] = False
    identity = experiment_identity(
        args, shared,
        manifest_validation=manifest_validation,
        gpu_identity=static_gpu_identity,
    )
    inherited_profile_env = {
        PROFILE_ENV: os.environ.get(PROFILE_ENV),
        PROFILE_SAMPLE_ENV: os.environ.get(PROFILE_SAMPLE_ENV),
    }
    request = {
        "result_schema_version": 2,
        "command": command,
        "dataset": args.dataset,
        "seed": args.seed,
        "windows": args.windows,
        "window_start": args.window_start,
        "warmup_epochs": args.warmup_epochs,
        "requested_profile_mode": args.profile_mode,
        "requested_basic_timing": args.basic_timing,
        "run_protocol": (
            "bootstrap_inventory_diagnostic"
            if bootstrap_inventory
            else (
                "torch_trace_diagnostic"
                if args.torch_trace_batches
                else "formal_throughput_benchmark"
            )
        ),
        "protocol_schema_version": 1,
        "formal_baseline_eligible": False,
        "selection_manifest_required": not bootstrap_inventory,
        "bootstrap_inventory": bootstrap_inventory,
        "formal_manifest_source_verified_at_publish": None,
        "shared_artifact_root": str(shared),
        "shared_input_preflight": validate_shared_training_inputs(args, shared),
        "artifact_root": str(output),
        "inherited_profile_environment": inherited_profile_env,
        "child_profile_environment": {
            PROFILE_ENV: None,
            PROFILE_SAMPLE_ENV: None,
        },
        "dry_run": not args.execute,
        "experiment_identity": identity,
    }
    if not args.execute:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return 0

    # Formal execution must resolve one UUID before starting either the sampler
    # or training child. A rejected preflight writes only a diagnostic result
    # under the isolated output root.
    gpu_preflight_started = time.perf_counter()
    try:
        strict_gpu_identity = _gpu_identity(require_resolved=True)
    except Exception as exc:
        before = shared_tree_snapshot(shared)
        output.mkdir(parents=True, exist_ok=False)
        after = shared_tree_snapshot(shared)
        gpu_error = f"{type(exc).__name__}: {exc}"
        measurement_source = (
            "detailed_profile"
            if args.profile_mode == "on"
            else ("basic_timing" if args.basic_timing == "on" else "pure_wall")
        )
        evidence = {
            "schema_version": 1,
            "valid": False,
            "errors": [{
                "code": "gpu_identity_unresolved",
                "message": gpu_error,
            }],
            "error_codes": ["gpu_identity_unresolved"],
            "measurement_source": measurement_source,
            "batch_rows_required": (
                args.profile_mode == "on" or args.basic_timing == "on"
            ),
            "batch_row_count": 0,
        }
        request.update({
            "status": "failed",
            "failure_reasons": [f"gpu_identity_unresolved:{gpu_error}"],
            "returncode": 84,
            "child_returncode": None,
            "child_wall_seconds": None,
            "runner_wall_seconds": time.perf_counter() - gpu_preflight_started,
            "pure_profiler_off_control": (
                args.profile_mode == "off" and args.basic_timing == "off"
            ),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "effective_profile_mode": "not_started",
            "profile_mode_matches_request": False,
            "shared_snapshot_before": before,
            "shared_snapshot_after": after,
            "shared_root_unchanged": before == after,
            "execution_evidence": evidence,
            "aggregate_published": False,
            "diagnostic_statistics_published": False,
            "statistics_published": False,
            "aggregate": {
                "measurement_source": measurement_source,
                "statistics_status": "invalid_not_published",
                "events": None,
                "events_per_second": None,
                "batches_per_second": None,
                "batch_metadata_sha256": None,
                "measurement_identity_sha256": None,
                "measurement_identity_scope": "unavailable:gpu_identity_unresolved",
                "measurement_identity_is_content_hash": False,
                "stages": {},
            },
        })
        (output / "benchmark_result.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2)
        )
        return 84

    identity = experiment_identity(
        args, shared,
        manifest_validation=manifest_validation,
        gpu_identity=strict_gpu_identity,
    )
    request["experiment_identity"] = identity

    before = shared_tree_snapshot(shared)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        sampler = start_gpu_sampler(
            output / "gpu_utilization.csv", args.gpu_sample_ms, identity["gpu_uuid"]
        )
    except Exception as exc:
        after = shared_tree_snapshot(shared)
        sampler_error = f"{type(exc).__name__}: {exc}"
        evidence = {
            "schema_version": 1,
            "valid": False,
            "errors": [{
                "code": "gpu_sampler_start_failed",
                "message": sampler_error,
            }],
            "error_codes": ["gpu_sampler_start_failed"],
            "measurement_source": (
                "detailed_profile"
                if args.profile_mode == "on"
                else ("basic_timing" if args.basic_timing == "on" else "pure_wall")
            ),
            "batch_rows_required": (
                args.profile_mode == "on" or args.basic_timing == "on"
            ),
            "batch_row_count": 0,
        }
        request.update({
            "status": "failed",
            "failure_reasons": [f"gpu_sampler_start_failed:{sampler_error}"],
            "returncode": 89,
            "child_returncode": None,
            "child_wall_seconds": None,
            "runner_wall_seconds": time.perf_counter() - started,
            "pure_profiler_off_control": (
                args.profile_mode == "off" and args.basic_timing == "off"
            ),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "effective_profile_mode": "not_started",
            "profile_mode_matches_request": False,
            "shared_snapshot_before": before,
            "shared_snapshot_after": after,
            "shared_root_unchanged": before == after,
            "execution_evidence": evidence,
            "aggregate_published": False,
            "diagnostic_statistics_published": False,
            "statistics_published": False,
            "aggregate": {
                "measurement_source": evidence["measurement_source"],
                "statistics_status": "invalid_not_published",
                "events": None,
                "events_per_second": None,
                "batches_per_second": None,
                "batch_metadata_sha256": None,
                "measurement_identity_sha256": None,
                "measurement_identity_scope": "unavailable:gpu_sampler_start_failed",
                "measurement_identity_is_content_hash": False,
                "stages": {},
            },
        })
        (output / "benchmark_result.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2)
        )
        return 89

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            env=controlled_child_environment(),
        )
    finally:
        if sampler:
            process, handle = sampler
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            handle.close()
    child_wall_seconds = time.perf_counter() - started
    after = shared_tree_snapshot(shared)
    effective_profile_mode = (
        "on" if _latest(output, "profiling/perf_profile.json") else "off"
    )
    profile_mode_matches = effective_profile_mode == args.profile_mode
    shared_unchanged = before == after
    evidence = validate_execution_evidence(
        output, args, num_epochs=args._benchmark_num_epochs
    )

    final_returncode = completed.returncode
    formal_manifest_source_verified_at_publish = None
    if not bootstrap_inventory:
        try:
            source_path = Path(manifest_validation["source"])
            formal_manifest_source_verified_at_publish = (
                manifest_validation.get("source_content_verified") is True
                and source_path.is_file()
                and _sha256_file(source_path)
                == manifest_validation["source_sha256"]
            )
        except (KeyError, OSError, TypeError):
            formal_manifest_source_verified_at_publish = False

    failure_reasons = []
    if completed.returncode != 0:
        failure_reasons.append(f"child_returncode:{completed.returncode}")
    if not profile_mode_matches:
        failure_reasons.append("profile_mode_mismatch")
        if final_returncode == 0:
            final_returncode = 86
    if not shared_unchanged:
        failure_reasons.append("shared_root_metadata_changed")
        if final_returncode == 0:
            final_returncode = 87
    if not evidence["valid"]:
        failure_reasons.extend(
            f"execution_evidence:{code}" for code in evidence["error_codes"]
        )
        if final_returncode == 0:
            final_returncode = 88

    if formal_manifest_source_verified_at_publish is False:
        failure_reasons.append("formal_manifest_source_not_verified_at_publish")
        if final_returncode == 0:
            final_returncode = 90

    publish_statistics = (
        final_returncode == 0 and evidence["valid"] and profile_mode_matches
        and shared_unchanged
    )
    if publish_statistics:
        aggregate_payload = aggregate(
            output,
            args.window_start,
            args.windows,
            profile_mode=args.profile_mode,
            basic_timing=args.basic_timing,
            warmup_epochs=args.warmup_epochs,
            torch_trace_batches=args.torch_trace_batches,
        )
        if bootstrap_inventory:
            aggregate_payload["statistics_status"] = (
                "diagnostic_bootstrap_inventory_not_formal_baseline"
            )
            aggregate_payload["events_per_second"] = None
            aggregate_payload["batches_per_second"] = None
    else:
        aggregate_payload = {
            "measurement_source": evidence["measurement_source"],
            "statistics_status": "invalid_not_published",
            "events": None,
            "events_per_second": None,
            "batches_per_second": None,
            "batch_metadata_sha256": None,
            "measurement_identity_sha256": None,
            "measurement_identity_scope": "unavailable:execution_evidence_invalid",
            "measurement_identity_is_content_hash": False,
            "stages": {},
        }

    throughput_statistics_published = (
        publish_statistics
        and not bootstrap_inventory
        and not args.torch_trace_batches
        and aggregate_payload.get("measured_batch_count", 0) > 0
    )
    diagnostic_statistics_published = (
        publish_statistics
        and (bootstrap_inventory or bool(args.torch_trace_batches))
    )
    request.update({
        "status": "completed" if final_returncode == 0 else "failed",
        "failure_reasons": failure_reasons,
        "returncode": final_returncode,
        "child_returncode": completed.returncode,
        "child_wall_seconds": child_wall_seconds,
        "runner_wall_seconds": time.perf_counter() - started,
        "pure_profiler_off_control": (
            args.profile_mode == "off" and args.basic_timing == "off"
        ),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "effective_profile_mode": effective_profile_mode,
        "profile_mode_matches_request": profile_mode_matches,
        "shared_snapshot_before": before,
        "shared_snapshot_after": after,
        "shared_root_unchanged": shared_unchanged,
        "execution_evidence": evidence,
        "aggregate_published": publish_statistics,
        "statistics_published": throughput_statistics_published,
        "diagnostic_statistics_published": diagnostic_statistics_published,
        "formal_baseline_eligible": throughput_statistics_published,
        "bootstrap_inventory": bootstrap_inventory,
        "aggregate": aggregate_payload,
        "formal_manifest_source_verified_at_publish": (
            formal_manifest_source_verified_at_publish
        ),
    })
    (output / "benchmark_result.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2)
    )
    return final_returncode


if __name__ == "__main__":
    raise SystemExit(main())
