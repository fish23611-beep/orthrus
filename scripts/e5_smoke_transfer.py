"""Audit G3 evidence, plan minimal inputs, and transfer without changing sources.

No subcommand trains, evaluates, launches CUDA, or executes recorded commands.
Audit/plan need PyYAML; bundle/verify/deploy use only the Python standard library.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid

BASELINE = "027962c372df3ad2d3e5ffd0a09351e06c75e409"
REQUIRED_EVIDENCE = ["preflight.json", "commands.json", "execution.json",
                     "checkpoint_identity_before.json", "source_sha256.json",
                     "all_checkpoints.yml", "val_select_then_test_best.yml",
                     "g3_equivalence_report.json"]
LIMIT = 5 * 1024 ** 3
SENSITIVE_KEY = re.compile(r"(?i)(?:^|_)(?:password|passwd|secret|api_key|access_token|private_key|credential)(?:$|_)")
METADATA_NAMES = ["node_metadata.pkl", "uuid_to_node_id.pkl", "node_id_to_uuid.pkl",
                  "ground_truth_nodes.pkl", "attack_to_nodes.pkl", "time_to_malicious_nodes.pkl",
                  "relation_mapping.json", "dataset_manifest.json", "nodeid2msg.pkl",
                  ".preprocess_metadata_complete"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b""):
            value.update(block)
    return value.hexdigest()


def redact(value):
    if isinstance(value, dict):
        return {str(k): "<REDACTED>" if SENSITIVE_KEY.search(str(k)) else redact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)((?:password|passwd|api[_-]?key|access[_-]?token)\s*[=:]\s*)[^\s,;]+", r"\1<REDACTED>", value)
        if "PRIVATE KEY-----" in value:
            return "<REDACTED_PRIVATE_KEY>"
    return value


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_document(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def walk_values(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_values(item, prefix + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_values(item, prefix + f"[{index}]")
    else:
        yield prefix, value


def nested(value, path, default=None):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def no_symlinks(path):
    path = Path(path).absolute()
    for item in [path, *path.parents]:
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return False  # Also reject Windows junctions/reparse points.
    return True


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    if any(re.search(r'[<>:"|?*\x00-\x1f]', part) or part.endswith((".", " "))
           or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
           for part in path.parts):
        raise ValueError(f"Invalid Windows path: {value!r}")
    if str(path) != value or not path.parts:
        raise ValueError(f"Noncanonical archive path: {value!r}")
    return path


def source_path(value, repo):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing source path in resolved configuration")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).absolute()


def audit(args):
    repo, g3 = args.repo.resolve(), args.g3.resolve()
    if args.output.exists():
        raise FileExistsError("Audit output must be a new directory")
    if args.output.resolve().is_relative_to(g3) or args.output.resolve().is_relative_to(repo):
        raise ValueError("Export output must be outside the frozen repository and G3")
    args.output.mkdir(parents=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", str(repo), "diff", BASELINE, "--name-only", "--", "src"], text=True).splitlines()
    evidence, references, blocked = {}, [], []
    for name in REQUIRED_EVIDENCE:
        path = g3 / name
        if not path.is_file() or not no_symlinks(path):
            evidence[name] = {"status": "BLOCKED", "source": str(path)}
            blocked.append(str(path))
            continue
        raw = load_document(path)
        evidence[name] = {"status": "PASS", "source": str(path), "sha256": digest(path), "content": redact(raw)}
        for key, value in walk_values(raw):
            if isinstance(value, str) and value.startswith("/") and "\n" not in value and not SENSITIVE_KEY.search(key):
                p = Path(value)
                references.append({"document": name, "key": key, "path": value,
                                   "exists": p.exists(), "directory": p.is_dir(), "is_symlink": not no_symlinks(p)})
    configs = []
    if args.resolve_commands:
        sys.path.insert(0, str(repo / "src"))
        os.chdir(repo)
        from experiments.run_experiment import build_parser, build_pipeline_args
        from config import get_yml_cfg
        from artifact_paths import resolve_artifact_paths
        import yaml
        for recorded in load_document(g3 / "commands.json"):
            argv = recorded["argv"]
            runtime = build_pipeline_args(build_parser().parse_args(argv[2:]))
            if not runtime.shared_artifact_readonly or runtime.stages != "test,evaluate":
                raise ValueError("Only recorded readonly test/evaluate commands may be resolved")
            cfg = get_yml_cfg(runtime)
            resolve_artifact_paths(cfg, ["test", "evaluate"], cli_artifact_root=runtime.artifact_root, create_dirs=False)
            full = yaml.safe_load(cfg.dump())
            for key in ["_seed", "_test_mode", "_max_windows_per_split", "_run_dir", "_metadata_dir", "_shared_artifact_readonly"]:
                full[key] = getattr(cfg, key)
            output = args.output / ("resolved_runtime_" + recorded["policy"] + ".json")
            dump(output, redact(full))
            configs.append({"source": str(output), "sha256": digest(output),
                            "derived_from_argv": argv, "resolution_only_no_execution": True,
                            "seed": full["_seed"], "window_limit": full["_max_windows_per_split"]})
    # Inspect metadata only, never recursively enumerate the original event data.
    count = 0
    for directory, subdirs, files in os.walk(g3, followlinks=False):
        subdirs[:] = [d for d in subdirs if not Path(directory, d).is_symlink()]
        for name in files:
            path = Path(directory, name)
            if path.suffix not in {".json", ".yml", ".yaml"} or not no_symlinks(path):
                continue
            count += 1
            if count > 10000:
                raise ValueError("G3 metadata inventory exceeds 10000 documents; narrow scope")
            if path.stat().st_size > 8 * 1024 ** 2:
                continue
            try:
                value = load_document(path)
            except (ValueError, UnicodeError):
                continue
            if (nested(value, "dataset.name") == "THEIA_E5"
                    and nested(value, "edge_featurization.embed_edges._edge_embeds_dir")
                    and nested(value, "detection.gnn_training._trained_models_dir")):
                configs.append({"source": str(path), "sha256": digest(path),
                                "seed": value.get("_seed"), "window_limit": value.get("_max_windows_per_split"),
                                "edge_embeds": nested(value, "edge_featurization.embed_edges._edge_embeds_dir"),
                                "checkpoint_root": nested(value, "detection.gnn_training._trained_models_dir"),
                                "run_root": value.get("_run_dir"), "metadata_root": value.get("_metadata_dir")})
    result = {"status": "PASS" if not blocked and head == BASELINE and not changed else "BLOCKED",
              "repo": str(repo), "head": head, "baseline": BASELINE, "changed_source_files": changed,
              "g3": str(g3), "evidence": evidence, "missing": blocked,
              "referenced_paths": references, "resolved_config_candidates": configs,
              "note": "Reads existing evidence only; does not rerun G3 or infer that its gate passed."}
    expected = load_document(g3 / "source_sha256.json").get("files", {})
    result["source_sha256_verification"] = {name: digest(repo / name) == sha for name, sha in expected.items()}
    if not expected or not all(result["source_sha256_verification"].values()):
        result["status"] = "BLOCKED"
    dump(args.output / "audit.json", result)
    print(json.dumps({"status": result["status"], "resolved_config_candidates": configs, "missing": blocked}, indent=2))
    return 0 if result["status"] == "PASS" else 2


def plan(args):
    repo, g3 = args.repo.resolve(), args.g3.resolve()
    if args.output.resolve().is_relative_to(repo) or args.output.resolve().is_relative_to(g3):
        raise ValueError("Manifest output must be outside the frozen repository and G3")
    cfg = load_document(args.config)
    problems = []
    if nested(cfg, "dataset.name") != "THEIA_E5" or nested(cfg, "model.variant") != "mstc":
        problems.append("Resolved configuration must identify THEIA_E5 and MSTC")
    if cfg.get("_seed") != 0 or cfg.get("_max_windows_per_split") != 2 or cfg.get("_test_mode") is not False:
        problems.append("Require recorded seed=0, max_windows_per_split=2, test_mode=false")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != BASELINE:
        problems.append("Server repository HEAD differs from frozen baseline")
    changed = subprocess.check_output(["git", "-C", str(repo), "diff", BASELINE, "--name-only", "--", "src"], text=True).splitlines()
    if changed:
        problems.append("Server source differs from frozen baseline: " + ", ".join(changed))
    declared_roots = []
    for key in ["edge_featurization.embed_edges._edge_embeds_dir", "_metadata_dir",
                "detection.gnn_training._trained_models_dir", "graph_construction.build_graphs._graphs_dir",
                "detection.gnn_testing._edge_losses_dir", "detection.evaluation._evaluation_results_dir", "_run_dir"]:
        raw = nested(cfg, key)
        if raw:
            declared_roots.append(source_path(raw, repo).resolve())
    declared_roots.append(g3)
    rows = []
    def add(path, relative, kind, purpose, required=True, sensitive=False, depends=()):
        path = Path(path).absolute()
        original = path
        symbolic = not no_symlinks(path)
        resolved = path.resolve()
        approved = any(resolved.is_relative_to(root) for root in declared_roots)
        if approved:
            path = resolved
        safe_relative(relative)
        exists = path.is_file()
        symlink = symbolic
        size = path.stat().st_size if exists else None
        row = {"source_absolute": str(path), "source_alias": str(original),
               "symlink_within_declared_input_scope": approved and no_symlinks(path),
               "source_relative_to_repo": str(path.relative_to(repo)) if path.is_relative_to(repo) else None,
               "relative_path": relative, "windows_target": str(PureWindowsPath(args.windows_root) / PurePosixPath(relative)),
               "kind": kind, "purpose": purpose, "required": required, "exists": exists,
               "is_symlink": symlink, "bytes": size, "mib": size / 1024 ** 2 if size is not None else None,
               "gib": size / 1024 ** 3 if size is not None else None,
               "sha256": digest(path) if exists and approved and no_symlinks(path) else None,
               "requires_path_mapping": path.name != PurePosixPath(relative).name,
               "potentially_sensitive": sensitive,
               "depends_on": list(depends) or ({
                   "checkpoint": ["resolved_config", "time_statistics", "train replay windows"],
                   "temporal_window": ["resolved_config.dataset_view", "resolved_config.encoder.edge_features"],
                   "metadata": ["original node identity mapping"],
                   "linux_reference": ["same checkpoint", "same window identities", "recorded Linux runtime"],
               }.get(kind, []))}
        rows.append(row)
        if required and (not exists or not approved or not no_symlinks(path)):
            problems.append(f"Missing or symbolic required input: {path}")
    embeds = source_path(nested(cfg, "edge_featurization.embed_edges._edge_embeds_dir"), repo)
    for split in ("train", "val", "test"):
        directory = embeds / split
        files = sorted(p for p in directory.iterdir() if p.is_file())[:2] if directory.is_dir() else []
        if len(files) != 2:
            problems.append(f"Expected two real {split} windows under {directory}")
        for path in files:
            add(path, f"datasets/THEIA_E5/edge_embeds/{split}/{path.name.replace(':', '%3A')}", "temporal_window",
                "train history replay" if split == "train" else "validation calibration" if split == "val" else "real test inference")
    cp = source_path(nested(cfg, "detection.gnn_training._trained_models_dir"), repo) / "model_epoch_6"
    explicit = cfg.get("_inference_checkpoint")
    if explicit:
        cp = source_path(explicit, repo)
        if "model_epoch_6" not in cp.parts:
            problems.append("Explicit checkpoint is not identified as model_epoch_6")
    checkpoint = cp if cp.is_file() else cp / ("checkpoint.pt" if (cp / "checkpoint.pt").is_file() else "state_dict.pkl")
    add(checkpoint, f"artifacts/THEIA_E5/checkpoints/model_epoch_6/{checkpoint.name}", "checkpoint",
        "the exact inference checkpoint chosen by the production loader")
    run_root = source_path(cfg.get("_run_dir") or cfg.get("_metadata_dir"), repo)
    stats = run_root / "metadata/time_statistics.json" if cfg.get("_run_dir") else run_root / "time_statistics.json"
    add(stats, "artifacts/THEIA_E5/run_metadata/time_statistics.json", "time_statistics",
        "frozen train-only TimeGap and multiscale boundaries")
    metadata = source_path(cfg.get("_metadata_dir"), repo)
    # MSTC consumers: labelling.get_ground_truth, get_GP_of_each_attack and
    # testing._load_nodeid2msg. Other caches are not consumed on this path.
    names = ["ground_truth_nodes.pkl", "attack_to_nodes.pkl"]
    if nested(cfg, "testing.include_node_messages", True):
        names.append("nodeid2msg.pkl" if (metadata / "nodeid2msg.pkl").is_file() else "node_metadata.pkl")
    for name in names:
        add(metadata / name, f"artifacts/THEIA_E5/metadata/{name}", "metadata", "cache-only messages/ground truth/metrics", True, True)
    # The standard orchestrator requires genuine graph availability even
    # though the MSTC evaluator does not compute baseline TW labels.
    graph_root = source_path(nested(cfg, "graph_construction.build_graphs._graphs_dir"), repo)
    test_windows = [r for r in rows if r["kind"] == "temporal_window" and "/test/" in r["relative_path"]]
    for row in test_windows:
        graph_name = Path(row["source_absolute"]).name.removesuffix(".TemporalData.simple")
        matches = [graph_root / folder / graph_name for folder in nested(cfg, "dataset.test_files", [])
                   if (graph_root / folder / graph_name).is_file()]
        if len(matches) != 1:
            problems.append("Cannot identify the genuine test graph for " + graph_name)
        else:
            path = matches[0]
            add(path, f"artifacts/THEIA_E5/graphs/{path.parent.name}/{path.name.replace(':','%3A')}",
                "source_graph", "genuine standard detection_only graph prerequisite", True, True)
    scores = source_path(nested(cfg, "detection.gnn_testing._edge_losses_dir"), repo)
    for split in ("val", "test"):
        directory = scores / split / "model_epoch_6"
        references = sorted(directory.glob("*.csv")) if directory.is_dir() else []
        if len(references) != 2:
            problems.append(f"Need G3 saved two-window {split} reference CSVs: {directory}")
        for path in references:
            add(path, f"artifacts/THEIA_E5/linux_reference/edge_scores/{split}/model_epoch_6/{path.name.replace(':','%3A')}",
                "linux_reference", "compare event identities, scores and time losses without rerunning G3", True, True)
    evaluation = source_path(nested(cfg, "detection.evaluation._evaluation_results_dir"), repo)
    for path in sorted((evaluation / "calibration/model_epoch_6").glob("*")):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            add(path, f"artifacts/THEIA_E5/linux_reference/calibration/model_epoch_6/{path.name}",
                "linux_reference", "existing calibrated events, node scores and thresholds", True, True)
    for name in ["metrics.json", "node_predictions.csv", "event_predictions.csv"]:
        path = run_root / "node_scores" / name
        # event_predictions duplicates calibrated CSVs; never copy twice.
        if path.is_file() and name == "metrics.json":
            add(path, "artifacts/THEIA_E5/linux_reference/node_scores/" + name,
                "linux_reference", "canonical selected-checkpoint metrics and predictions", True, True)
    for path in [g3 / "all_checkpoints_derived_threshold.json", run_root / "environment.json"]:
        if path.is_file():
            add(path, "artifacts/THEIA_E5/linux_reference/" + path.name,
                "linux_reference", "recorded G3 threshold or backend environment", True)
    for name in REQUIRED_EVIDENCE:
        path = g3 / name
        if not path.is_file():
            problems.append(f"Missing G3 evidence: {path}")
    total = sum(row["bytes"] or 0 for row in rows if row["exists"])
    result = {"schema_version": 1, "status": "BLOCKED" if problems else "PASS",
              "created_utc": datetime.now(timezone.utc).isoformat(), "server_commit": head,
              "g3": str(g3), "server_repo": str(repo), "declared_input_roots": [str(p) for p in declared_roots],
              "resolved_config_source": str(args.config.resolve()),
              "resolved_config_sha256": digest(args.config), "resolved_config": redact(cfg),
              "windows_root": args.windows_root, "files": rows, "file_count": len(rows),
              "total_bytes": total, "total_mib": total / 1024 ** 2, "total_gib": total / 1024 ** 3,
              "large_transfer_requires_confirmation": total > LIMIT, "problems": problems,
              "omitted": ["raw events/PostgreSQL", "other five checkpoints", "regenerable compact sidecar",
                          "Word2Vec retraining assets (features already embedded)", "historical replay state (rebuilt from train)"],
              "scope": "Explicit single-checkpoint inference; not six-checkpoint selection equivalence"}
    dump(args.output, result)
    print(json.dumps({key: result[key] for key in ["status", "file_count", "total_bytes", "total_gib", "problems"]}, indent=2))
    return 0 if not problems else 2


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise ValueError("Manifest is not a completed input audit")
    paths = set()
    for row in manifest["files"]:
        relative = str(safe_relative(row["relative_path"]))
        if relative.casefold() in paths:
            raise ValueError("Duplicate or case-colliding target path")
        paths.add(relative.casefold())
        if row["required"] and (not row["exists"] or (row["is_symlink"] and not row.get("symlink_within_declared_input_scope"))):
            raise ValueError("Required file missing or symbolic")
        if row["exists"] and (not isinstance(row["bytes"], int) or row["bytes"] < 0
                              or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256") or "")):
            raise ValueError("Invalid size or SHA256")
    rows = [row for row in manifest["files"] if row["exists"]]
    if sum(row["bytes"] for row in rows) > LIMIT:
        raise ValueError("BLOCKED: exceeds 5 GiB; requires confirmation and a revised plan")
    return rows


def bundle(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = validate_manifest(manifest)
    # Deliberately no force/override: obtain explicit approval before a >5 GiB transfer.
    if sum(row["bytes"] for row in rows) > LIMIT:
        raise ValueError("BLOCKED: exceeds 5 GiB; refine the plan and obtain confirmation first")
    if args.output.exists():
        raise FileExistsError("Choose a new export directory; existing exports are preserved")
    for protected in [manifest.get("server_repo"), manifest.get("g3")]:
        if protected and args.output.resolve().is_relative_to(Path(protected).resolve()):
            raise ValueError("Export must not write into the repository or G3")
    args.output.mkdir(parents=True)
    for row in rows:
        source = Path(row["source_absolute"])
        if not no_symlinks(source) or source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
            raise ValueError(f"Source identity changed: {source}")
        if source.name.lower() in {".env", ".env.local", "id_rsa", "id_ed25519"} or ".ssh" in source.parts:
            raise ValueError("Credential file is forbidden")
    # Store only the redacted resolved config; never copy raw YAML or environment files.
    manifest = redact(manifest)
    dump(args.output / "manifest.json", manifest)
    archive = args.output / "inputs.tar.gz"
    with tarfile.open(archive, "x:gz", dereference=False) as handle:
        handle.add(args.output / "manifest.json", arcname="manifest.json", recursive=False)
        for row in rows:
            source = Path(row["source_absolute"])
            info = handle.gettarinfo(str(source), arcname="payload/" + row["relative_path"])
            if not info.isfile() or not no_symlinks(source):
                raise ValueError("Only regular files are allowed")
            with source.open("rb") as stream:
                handle.addfile(info, stream)
    verify_archive(archive, manifest)
    checksums = [f"{digest(archive)}  inputs.tar.gz", f"{digest(args.output/'manifest.json')}  manifest.json"]
    (args.output / "sha256sums.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
    lines = ["# THEIA_E5 minimal input manifest", "", f"Total: {sum(r['bytes'] for r in rows)} bytes", "",
             "| Target relative path | Bytes | SHA256 |", "| --- | ---: | --- |"]
    lines.extend(f"| {r['relative_path']} | {r['bytes']} | {r['sha256']} |" for r in rows)
    (args.output / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(__file__, args.output / "e5_smoke_transfer.py")
    (args.output / "verify_windows.ps1").write_text(
        "param([string]$Python='F:\\venvs\\mstc-pids\\Scripts\\python.exe')\n"
        "$ErrorActionPreference='Stop'\n"
        f"& $Python (Join-Path $PSScriptRoot 'e5_smoke_transfer.py') deploy --archive (Join-Path $PSScriptRoot 'inputs.tar.gz') --manifest (Join-Path $PSScriptRoot 'manifest.json') --sha256 '{digest(archive)}' --target 'F:\\Paper\\Data\\MSTC-Data'\n"
        "if ($LASTEXITCODE -ne 0) { throw 'Deployment verification failed' }\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest(archive)}, indent=2))
    return 0


def verify_archive(archive, manifest):
    rows = validate_manifest(manifest)
    expected = {"payload/" + row["relative_path"]: row for row in rows}
    seen = set()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            safe_relative(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError("Archive contains duplicate, link or special member")
            seen.add(member.name)
            if member.name == "manifest.json":
                if member.size > 16 * 1024 ** 2:
                    raise ValueError("Embedded manifest too large")
                if json.load(handle.extractfile(member)) != manifest:
                    raise ValueError("Embedded manifest differs from supplied manifest")
                continue
            if member.name not in expected or member.size != expected[member.name]["bytes"]:
                raise ValueError("Unlisted member or wrong size")
            value = hashlib.sha256()
            with handle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(4 * 1024 ** 2), b""):
                    value.update(block)
            if value.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Payload checksum mismatch: {member.name}")
    if seen != set(expected) | {"manifest.json"}:
        raise ValueError("Archive lacks declared files")
    return rows


def deploy(args):
    if digest(args.archive) != args.sha256.lower():
        raise ValueError("Archive SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = verify_archive(args.archive, manifest)  # Validate everything before writing payloads.
    if args.command == "verify":
        print(json.dumps({"status": "PASS", "verified_files": len(rows)}))
        return 0
    target = args.target.absolute()
    if not no_symlinks(target):
        raise ValueError("Deployment root traverses a symlink")
    target.mkdir(parents=True, exist_ok=True)
    copied, reused = [], []
    # Existing conflicting files are rejected before any new file is written.
    for row in rows:
        destination = target / row["relative_path"]
        if not no_symlinks(destination):
            raise ValueError("Target traverses a symlink")
        if destination.exists() and (not destination.is_file() or digest(destination) != row["sha256"]):
            raise FileExistsError(f"Conflicting existing file is preserved: {destination}")
    with tarfile.open(args.archive, "r:gz") as handle:
        for row in rows:
            destination = target / row["relative_path"]
            if destination.is_file():
                reused.append(str(destination))
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
            with handle.extractfile("payload/" + row["relative_path"]) as source, temporary.open("xb") as stream:
                shutil.copyfileobj(source, stream, 4 * 1024 ** 2)
            if digest(temporary) != row["sha256"]:
                raise ValueError("Extracted bytes changed; temporary file retained for diagnosis")
            if os.name == "nt":
                temporary.rename(destination)  # Windows rename refuses to replace an existing file.
            else:
                os.link(temporary, destination)  # POSIX no-replace publication.
                temporary.unlink()
            copied.append(str(destination))
    result = {"status": "PASS", "target": str(target), "copied": copied, "reused": reused,
              "verified_files": len(rows), "real_gpu_smoke": "NOT_RUN"}
    receipt = target / ("deployment-" + uuid.uuid4().hex + ".json")
    dump(receipt, result)
    print(json.dumps(result, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in [("audit", audit), ("plan", plan)]:
        p = commands.add_parser(name)
        p.add_argument("--repo", type=Path, required=True)
        p.add_argument("--g3", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "plan":
            p.add_argument("--config", type=Path, required=True, help="Actual resolved G3 config discovered by audit")
            p.add_argument("--windows-root", default="F:\\Paper\\Data\\MSTC-Data")
        else:
            p.add_argument("--resolve-commands", action="store_true", help="Resolve recorded readonly argv without executing stages")
        p.set_defaults(run=function)
    p = commands.add_parser("bundle")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(run=bundle)
    for name in ["verify", "deploy"]:
        p = commands.add_parser(name)
        p.add_argument("--archive", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--sha256", required=True)
        if name == "deploy":
            p.add_argument("--target", type=Path, required=True)
        p.set_defaults(run=deploy)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
