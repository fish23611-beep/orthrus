from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import plan_runtime_windows
import run_runtime_benchmark


def _tree_content_snapshot(root: Path):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = ("symlink", path.readlink().as_posix())
        elif path.is_file():
            result[relative] = (
                "file",
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        elif path.is_dir():
            result[relative] = ("directory",)
    return result


def _args(tmp_path, **overrides):
    shared = tmp_path / "shared"
    shared.mkdir(exist_ok=True)
    (shared / "input.marker").write_text("immutable-input")
    values = dict(
        dataset="THEIA_E3",
        config=str(ROOT / "config/experiments/mstc_full.yml"),
        artifact_root=str(tmp_path / "output"),
        shared_artifact_root=str(shared),
        seed=0,
        windows=2,
        window_start=0,
        selection_manifest=None,
        bootstrap_inventory=False,
        selection_inventory=False,
        warmup_epochs=1,
        profile_mode="on",
        basic_timing="on",
        torch_trace_batches=0,
        torch_trace_warmup_batches=1,
        gpu_sample_ms=200,
        execute=False,
    )
    values.update(overrides)
    args = SimpleNamespace(**values)
    # Populate the exact config-derived task-hash paths for ordinary fixtures.
    # Overrides that intentionally point at a missing/incomplete root remain
    # untouched so fail-fast tests exercise the real preflight.
    if Path(args.shared_artifact_root) == shared:
        model, edge_embeds = run_runtime_benchmark.resolve_expected_shared_inputs(
            args, shared.resolve()
        )
        model.parent.mkdir(parents=True)
        model.write_bytes(b"model")
        for split in ("train", "val", "test"):
            split_dir = edge_embeds / split
            split_dir.mkdir(parents=True)
            (split_dir / "window.pt").write_bytes(split.encode())
    return args


def _valid_gpu_telemetry_payload():
    return {
        "schema_version": 1,
        "valid": True,
        "errors": [],
        "error_codes": [],
        "statistics_status": "published",
        "utilization_statistics": {
            "sample_count": 2,
            "utilization_gpu_mean_percent": 50.0,
        },
    }


def _mock_valid_gpu_sampler(monkeypatch):
    monkeypatch.setattr(
        run_runtime_benchmark, "start_gpu_sampler", lambda *args: None
    )
    monkeypatch.setattr(
        run_runtime_benchmark,
        "stop_gpu_sampler",
        lambda sampler: {
            "sampler_started": True,
            "unexpected_exit": False,
            "monitor_error": None,
        },
    )
    monkeypatch.setattr(
        run_runtime_benchmark,
        "validate_gpu_telemetry",
        lambda *args, **kwargs: _valid_gpu_telemetry_payload(),
    )


def _write_gpu_csv(path: Path, timestamps, *, gpu_uuid="GPU-test", corrupt_row=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(run_runtime_benchmark.GPU_QUERY_FIELDS)
        for timestamp in timestamps:
            writer.writerow([
                datetime.fromtimestamp(timestamp).strftime(
                    "%Y/%m/%d %H:%M:%S.%f"
                )[:-3],
                "0",
                gpu_uuid,
                "Test GPU",
                "50",
                "25",
                "1024",
                "100",
                "1200",
                "5000",
            ])
        if corrupt_row is not None:
            writer.writerow(corrupt_row)


def _healthy_sampler_status():
    return {
        "sampler_started": True,
        "unexpected_exit": False,
        "unexpected_exit_returncode": None,
        "unexpected_exit_epoch_seconds": None,
        "shutdown_returncode": -15,
        "termination_requested": True,
        "kill_required": False,
        "monitor_error": None,
    }


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_index",
                "epoch",
                "window_index",
                "batch_events",
                "history_valid_slots",
                "total_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _complete_rows(*, epochs=range(1, 7), window_end=2, events=5):
    rows = []
    batch_index = 1
    for epoch in epochs:
        for window_index in range(window_end):
            rows.append({
                "batch_index": batch_index,
                "epoch": epoch,
                "window_index": window_index,
                "batch_events": events,
                "history_valid_slots": window_index,
                "total_seconds": 0.5,
            })
            batch_index += 1
    return rows


def _write_runtime_receipt(root: Path, *, epochs=range(1, 7), window_end=2,
                           events=5, batches_per_window=1):
    windows = [
        {
            "epoch": epoch,
            "window_index": window_index,
            "window_id": window_index,
            "event_count": events,
            "expected_batch_count": batches_per_window,
        }
        for epoch in epochs
        for window_index in range(window_end)
    ]
    payload = {
        "status": "completed",
        "executed_stages": ["train"],
        "training": {
            "execution_receipt": {
                "schema_version": 1,
                "status": "completed",
                "start_epoch": 1,
                "end_epoch": 6,
                "batch_size": 512,
                "windows": windows,
            }
        },
    }
    (root / "runtime.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "runtime.json").write_text(json.dumps(payload))


def _write_inventory_result(path: Path, source: Path, *, dataset="THEIA_E3",
                            warmup=1):
    with source.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    payload = {
        "result_schema_version": 3,
        "status": "completed",
        "returncode": 0,
        "run_protocol": "selection_inventory_diagnostic",
        "selection_inventory": True,
        "formal_baseline_eligible": False,
        "statistics_published": False,
        "diagnostic_statistics_published": True,
        "dataset": dataset,
        "warmup_epochs": warmup,
        "requested_profile_mode": "off",
        "requested_basic_timing": "on",
        "execution_evidence": {
            "valid": True,
            "measurement_source": "basic_timing",
            "measurement_path": str(source.resolve()),
            "measurement_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "batch_row_count": row_count,
        },
    }
    path.write_text(json.dumps(payload))
    return payload


def _write_valid_manifest(path: Path, *, dataset="THEIA_E3", start=0, count=2,
                          warmup=1, source_exists=True):
    source = path.with_suffix(".csv")
    if source_exists:
        _write_rows(source, _complete_rows(window_end=start + count))
    source_sha256 = hashlib.sha256(
        source.read_bytes() if source_exists else b"frozen-source"
    ).hexdigest()
    inventory_result = path.with_name(path.stem + "-inventory-result.json")
    if source_exists:
        _write_inventory_result(
            inventory_result, source, dataset=dataset, warmup=warmup
        )
        inventory_result_sha256 = hashlib.sha256(
            inventory_result.read_bytes()
        ).hexdigest()
    else:
        inventory_result_sha256 = hashlib.sha256(b"missing-result").hexdigest()
    payload = {
        "schema_version": 2,
        "dataset": dataset,
        "selection_rule": "test fixture",
        "source": str(source),
        "source_sha256": source_sha256,
        "source_sha256_scope": "content_sha256_of_batches_csv_file",
        "inventory_result": str(inventory_result),
        "inventory_result_sha256": inventory_result_sha256,
        "inventory_result_sha256_scope": (
            "content_sha256_of_benchmark_result_json_file"
        ),
        "warmup_epochs_excluded": warmup,
        "eligible_epochs": [2, 3, 4, 5, 6],
        "selected_start": start,
        "selected_end_exclusive": start + count,
        "window_count": count,
    }
    path.write_text(json.dumps(payload))
    return payload


def _ensure_train_window_count(args, count):
    shared = Path(args.shared_artifact_root).resolve()
    _, edge_embeds = run_runtime_benchmark.resolve_expected_shared_inputs(
        args, shared
    )
    train_dir = edge_embeds / "train"
    existing = sum(1 for path in train_dir.iterdir() if path.is_file())
    for index in range(existing, count):
        (train_dir / f"window-{index:03d}.simple").write_bytes(b"train")


def test_benchmark_command_preserves_config_dual_roots_and_seed(tmp_path):
    args = _args(tmp_path)
    roots = run_runtime_benchmark.validate(args)
    command = run_runtime_benchmark.build_command(args, roots=roots)
    assert command[command.index("--max-windows-per-split") + 1] == "2"
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--artifact-root") + 1] == str(
        (tmp_path / "output").resolve()
    )
    assert command[command.index("--shared-artifact-root") + 1] == str(
        (tmp_path / "shared").resolve()
    )
    assert "--shared-artifact-readonly" in command
    assert "--benchmark-basic-timing" in command
    assert "--perf-profile" in command
    assert "--cpu" not in command


def test_trace_command_forwards_bounded_warmup_and_active_batches(tmp_path):
    args = _args(
        tmp_path,
        torch_trace_batches=2,
        torch_trace_warmup_batches=3,
    )
    roots = run_runtime_benchmark.validate(args)
    command = run_runtime_benchmark.build_command(args, roots=roots)

    assert command[command.index("--perf-profile-torch-trace-batches") + 1] == "2"
    assert command[
        command.index("--perf-profile-torch-trace-warmup-batches") + 1
    ] == "3"


def test_twenty_window_case_requires_matching_selection_manifest(tmp_path):
    args = _args(tmp_path, windows=20, window_start=4)
    with pytest.raises(ValueError, match="selection-manifest"):
        run_runtime_benchmark.validate(args)
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest, start=4, count=20)
    args.selection_manifest = str(manifest)
    roots = run_runtime_benchmark.validate(args)
    command = run_runtime_benchmark.build_command(args, roots=roots)
    assert command[command.index("--max-windows-per-split") + 1] == "24"

    args.dataset = "THEIA_E5"
    with pytest.raises(ValueError, match="does not match dataset"):
        run_runtime_benchmark.validate(args)


def test_missing_or_empty_shared_root_fails_fast_without_output(tmp_path):
    missing = tmp_path / "missing"
    args = _args(tmp_path, shared_artifact_root=str(missing))
    with pytest.raises(ValueError, match="does not exist"):
        run_runtime_benchmark.validate(args)
    assert not (tmp_path / "output").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    args.shared_artifact_root = str(empty)
    with pytest.raises(ValueError, match="empty"):
        run_runtime_benchmark.validate(args)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "unrelated").write_text("x")
    args.shared_artifact_root = str(incomplete)
    with pytest.raises(ValueError, match="lacks exact task-hash"):
        run_runtime_benchmark.validate(args)
    assert not (tmp_path / "output").exists()


def test_same_nested_existing_and_symlink_overlap_roots_are_rejected(tmp_path, make_path_alias):
    args = _args(tmp_path)
    shared = Path(args.shared_artifact_root)

    args.artifact_root = str(shared)
    with pytest.raises(ValueError, match="already exists|overlap"):
        run_runtime_benchmark.validate(args)

    args.artifact_root = str(shared / "nested-output")
    with pytest.raises(ValueError, match="overlap"):
        run_runtime_benchmark.validate(args)

    link = make_path_alias(tmp_path / "shared-link", shared, target_is_directory=True)
    args.artifact_root = str(link / "via-symlink")
    with pytest.raises(ValueError, match="overlap"):
        run_runtime_benchmark.validate(args)

    existing = tmp_path / "existing-output"
    existing.mkdir()
    args.artifact_root = str(existing)
    with pytest.raises(ValueError, match="already exists"):
        run_runtime_benchmark.validate(args)


def test_profile_off_removes_conflicting_parent_environment(tmp_path):
    parent = {
        "KEEP_ME": "yes",
        run_runtime_benchmark.PROFILE_ENV: "1",
        run_runtime_benchmark.PROFILE_SAMPLE_ENV: "1",
    }
    child = run_runtime_benchmark.controlled_child_environment(parent)
    assert child["KEEP_ME"] == "yes"
    assert run_runtime_benchmark.PROFILE_ENV not in child
    assert run_runtime_benchmark.PROFILE_SAMPLE_ENV not in child

    args = _args(tmp_path, profile_mode="off")
    command = run_runtime_benchmark.build_command(
        args, roots=run_runtime_benchmark.validate(args)
    )
    assert "--perf-profile" not in command
    assert "--benchmark-basic-timing" in command


def test_warmup_and_causal_replay_are_excluded_and_on_off_match(tmp_path):
    rows = [
        {"batch_index": 1, "epoch": 1, "window_index": 1, "batch_events": 999, "total_seconds": 9.0},
        {"batch_index": 2, "epoch": 2, "window_index": 0, "batch_events": 10, "total_seconds": 1.0},
        {"batch_index": 3, "epoch": 2, "window_index": 1, "batch_events": 20, "total_seconds": 2.0},
        {"batch_index": 4, "epoch": 2, "window_index": 2, "batch_events": 30, "total_seconds": 3.0},
    ]
    on_root = tmp_path / "on"
    off_root = tmp_path / "off"
    _write_rows(on_root / "profiling/perf_batches.csv", rows)
    _write_rows(off_root / "benchmark_timing/basic_batches.csv", rows)

    on = run_runtime_benchmark.aggregate(
        on_root, 1, 2, profile_mode="on", basic_timing="on", warmup_epochs=1
    )
    off = run_runtime_benchmark.aggregate(
        off_root, 1, 2, profile_mode="off", basic_timing="on", warmup_epochs=1
    )
    assert on["warmup_batch_count"] == 1
    assert on["causal_replay_batch_count"] == 1
    assert on["measured_batch_count"] == 2
    assert on["measured_distinct_window_indices"] == [1, 2]
    assert on["measured_observed_window_count"] == 2
    assert on["events"] == 50
    assert on["stages"]["total_seconds"]["total_seconds"] == 5.0
    assert on["measurement_identity_sha256"] == off["measurement_identity_sha256"]
    assert on["measurement_source"] == "detailed_profile"
    assert off["measurement_source"] == "basic_timing"


def test_pure_off_is_separate_from_basic_timing(tmp_path):
    result = run_runtime_benchmark.aggregate(
        tmp_path,
        0,
        2,
        profile_mode="off",
        basic_timing="off",
        warmup_epochs=1,
    )
    assert result["measurement_source"] == "pure_wall"
    assert result["measurement_rows_found"] is False
    assert result["measured_batch_count"] == 0


def test_dry_run_has_no_output_or_shared_side_effect(tmp_path, capsys):
    args = _args(tmp_path)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--basic-timing", "on",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["requested_profile_mode"] == "off"
    identity = payload["experiment_identity"]
    assert identity["git_sha"]
    assert len(identity["resolved_config_sha256"]) == 64
    assert identity["dataset"] == "THEIA_E3"
    assert identity["seed"] == 0
    assert identity["window_manifest_sha256"] is None
    assert not Path(args.artifact_root).exists()
    assert _tree_content_snapshot(shared) == before


def test_shared_snapshot_names_metadata_and_content_scope_honestly(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "input.bin").write_bytes(b"input")
    snapshot = run_runtime_benchmark.shared_tree_snapshot(shared)
    assert len(snapshot["metadata_fingerprint_sha256"]) == 64
    assert snapshot["fingerprint_scope"].startswith("metadata_only:")
    assert snapshot["content_sha256"] is None
    assert snapshot["content_hash_status"] == "not_computed"


def test_e3_e5_manifests_have_content_identity(tmp_path):
    manifest = tmp_path / "e3-selection.json"
    _write_valid_manifest(manifest, start=4, count=20)
    args = _args(
        tmp_path, windows=20, window_start=4, selection_manifest=str(manifest)
    )
    _, shared = run_runtime_benchmark.validate(args)
    identity = run_runtime_benchmark.experiment_identity(
        args, shared, manifest_validation=args._validated_manifest
    )
    assert identity["window_manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_gpu_identity_resolves_selected_logical_device(monkeypatch):
    monkeypatch.setattr(run_runtime_benchmark.shutil, "which", lambda name: "/nvidia-smi")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-three")
    monkeypatch.setattr(
        run_runtime_benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="1, GPU-one\n3, GPU-three\n"
        ),
    )
    identity = run_runtime_benchmark._gpu_identity(require_resolved=True)
    assert identity["gpu_uuid"] == "GPU-three"
    assert identity["gpu_uuids"] == ["GPU-one", "GPU-three"]
    assert identity["gpu_identity_runtime_verified"] is True


def test_execute_keeps_shared_file_list_and_content_hash_unchanged(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="off")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    shared = Path(args.shared_artifact_root)
    nested = shared / "nested"
    nested.mkdir()
    (nested / "input.bin").write_bytes(b"preprocessed")
    before = _tree_content_snapshot(shared)

    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_rows(
            output / "benchmark_timing/basic_batches.csv",
            _complete_rows(),
        )
        _write_runtime_receipt(output)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    _mock_valid_gpu_sampler(monkeypatch)
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity",
        lambda **kwargs: {
            "gpu_uuid": "GPU-test", "gpu_uuids": ["GPU-test"],
            "cuda_visible_devices": "GPU-test", "nvidia_visible_devices": None,
            "cuda_device_order": None, "gpu_identity_scope": "test",
            "gpu_mapping_validation_status": "validated_test_mapping",
            "gpu_identity_runtime_verified": bool(kwargs.get("require_resolved")),
        },
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    assert _tree_content_snapshot(shared) == before
    result = json.loads((Path(args.artifact_root) / "benchmark_result.json").read_text())
    assert result["shared_root_unchanged"] is True
    assert result["gpu_statistics_published"] is True
    assert result["gpu_telemetry"]["statistics_status"] == (
        "published_formal_benchmark"
    )
    assert result["effective_profile_mode"] == "off"
    assert result["profile_mode_matches_request"] is True


def test_readonly_config_path_resolution_creates_nothing_under_shared(
    tmp_path, monkeypatch
):
    import config as config_module

    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "existing.txt").write_text("keep")
    before = _tree_content_snapshot(shared)
    monkeypatch.setattr(
        config_module, "_resolve_ground_truth_dir", lambda cfg=None: str(tmp_path / "gt")
    )
    args = SimpleNamespace(
        dataset="THEIA_E3",
        model="orthrus",
        config=str(ROOT / "config/experiments/mstc_full.yml"),
        cpu=True,
        from_weights=False,
        seed=0,
        skip_tracing=False,
        artifact_root=str(tmp_path / "output"),
        shared_artifact_root=str(shared),
        shared_artifact_readonly=True,
        max_windows_per_split=2,
    )
    cfg = config_module.get_yml_cfg(args)
    assert cfg._shared_artifact_readonly is True
    assert _tree_content_snapshot(shared) == before
    assert not (tmp_path / "output").exists()


def test_readonly_compact_sidecar_publish_is_noop(tmp_path, monkeypatch):
    from data_utils import _CompactIndex

    index = _CompactIndex(specs=(), total_events=0)
    shared = tmp_path / "shared"
    edge_embeds = shared / "edge_embeds"
    edge_embeds.mkdir(parents=True)
    cfg = SimpleNamespace(
        _shared_artifact_readonly=True,
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(edge_embeds))
        ),
    )
    # Make every condition after the read-only guard eligible.  Without that
    # guard this call would create ``_compact_index_v2`` (and then attempt to
    # publish tensors), so the assertion cannot pass merely because the test
    # configuration omitted the persistent root.
    monkeypatch.setattr(_CompactIndex, "has_node_table", lambda self: True)
    index._write_persistent(cfg, "host_network_full", {})
    assert not (edge_embeds / index.PERSISTENT_DIR_NAME).exists()


def test_window_planner_excludes_warmup_and_deduplicates_epochs(tmp_path):
    source = tmp_path / "batches.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_index",
                "epoch",
                "window_index",
                "batch_events",
                "history_valid_slots",
                "total_seconds",
            ],
        )
        writer.writeheader()
        batch_index = 1
        for epoch in (1, 2, 3):
            for window in range(25):
                writer.writerow({
                    "batch_index": batch_index,
                    "epoch": epoch,
                    "window_index": window,
                    "batch_events": 100 + window,
                    "history_valid_slots": window * 10,
                    "total_seconds": (
                        100.0 if epoch == 1 else 1.0 + abs(window - 12) / 100
                    ),
                })
                batch_index += 1
    inventory_result = tmp_path / "inventory-result.json"
    _write_inventory_result(inventory_result, source)
    output = tmp_path / "plan.json"
    assert plan_runtime_windows.main([
        "--dataset", "THEIA_E3",
        "--batches-csv", str(source),
        "--inventory-result", str(inventory_result),
        "--window-count", "20",
        "--warmup-epochs", "1",
        "--expected-epochs", "3",
        "--output", str(output),
    ]) == 0
    plan = json.loads(output.read_text())
    assert plan["schema_version"] == 2
    assert plan["dataset"] == "THEIA_E3"
    assert plan["eligible_epochs"] == [2, 3]
    assert plan["warmup_epochs_excluded"] == 1
    assert plan["window_count"] == 20
    assert set(plan["selected_feature_means"]) == {
        "events", "history", "batch_seconds"
    }
    assert plan["window_features"]["0"]["events"] == 100
    assert len(plan["fast_windows"]) == 3
    assert len(plan["slow_windows"]) == 3


def _run_planner(tmp_path, source, *, output_name="plan.json"):
    inventory_result = tmp_path / "inventory-result.json"
    _write_inventory_result(inventory_result, source)
    output = tmp_path / output_name
    plan_runtime_windows.main([
        "--dataset", "THEIA_E3",
        "--batches-csv", str(source),
        "--inventory-result", str(inventory_result),
        "--window-count", "20",
        "--warmup-epochs", "1",
        "--output", str(output),
    ])
    return output


def test_window_planner_is_deterministic_and_uses_no_test_inputs(tmp_path):
    source = tmp_path / "inventory.csv"
    _write_rows(source, _complete_rows(window_end=25))
    first = _run_planner(tmp_path, source, output_name="first.json")
    second = _run_planner(tmp_path, source, output_name="second.json")
    assert first.read_bytes() == second.read_bytes()
    plan = json.loads(first.read_text())
    assert not any(key.startswith("test") for key in plan)
    assert set(plan["selected_feature_means"]) == {
        "events", "history", "batch_seconds"
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_history", "missing required fields"),
        ("nonfinite_time", "invalid record"),
        ("duplicate_batch", "duplicate batch_index"),
        ("missing_batch", "batch_index sequence is incomplete"),
    ],
)
def test_window_planner_rejects_invalid_batch_records(tmp_path, mutation, message):
    rows = _complete_rows(window_end=25)
    if mutation == "nonfinite_time":
        rows[0]["total_seconds"] = "nan"
    elif mutation == "duplicate_batch":
        rows[1]["batch_index"] = rows[0]["batch_index"]
    elif mutation == "missing_batch":
        rows.pop(1)
    source = tmp_path / "inventory.csv"
    if mutation == "missing_history":
        with source.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "batch_index", "epoch", "window_index", "batch_events",
                "total_seconds",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([
                {key: row[key] for key in fieldnames} for row in rows
            ])
    else:
        _write_rows(source, rows)
    inventory_result = tmp_path / "inventory-result.json"
    _write_inventory_result(inventory_result, source)
    with pytest.raises(ValueError, match=message):
        plan_runtime_windows.main([
            "--dataset", "THEIA_E3",
            "--batches-csv", str(source),
            "--inventory-result", str(inventory_result),
            "--window-count", "20",
            "--warmup-epochs", "1",
            "--output", str(tmp_path / "plan.json"),
        ])
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize(
    ("profile_mode", "basic_timing", "present", "missing_code"),
    [
        ("off", "on", "none", "measurement_rows_missing"),
        ("on", "on", "profile_only", "measurement_rows_missing"),
        ("on", "on", "rows_only", "detailed_profile_missing"),
    ],
)
def test_measurement_artifacts_are_fail_closed(
    tmp_path, profile_mode, basic_timing, present, missing_code
):
    args = _args(
        tmp_path, profile_mode=profile_mode, basic_timing=basic_timing
    )
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    if present in {"profile_only", "both"}:
        profile = root / "profiling/perf_profile.json"
        profile.parent.mkdir(parents=True)
        profile.write_text(json.dumps({"enabled": True, "status": "completed"}))
    if present in {"rows_only", "both"}:
        _write_rows(root / "profiling/perf_batches.csv", _complete_rows())
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert missing_code in evidence["error_codes"]


def test_empty_basic_csv_is_not_valid_measurement(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    _write_rows(root / "benchmark_timing/basic_batches.csv", [])
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "measurement_rows_empty" in evidence["error_codes"]


def test_twenty_window_measurement_rejects_nineteen_observed(tmp_path):
    args = _args(tmp_path, windows=20, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root, window_end=20)
    rows = [
        row for row in _complete_rows(window_end=20)
        if row["window_index"] != 19
    ]
    _write_rows(root / "benchmark_timing/basic_batches.csv", rows)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "batch_window_coverage_mismatch" in evidence["error_codes"]


def test_missing_post_warmup_epoch_window_is_rejected(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    rows = [
        row for row in _complete_rows()
        if not (row["epoch"] == 4 and row["window_index"] == 1)
    ]
    _write_rows(root / "benchmark_timing/basic_batches.csv", rows)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "batch_window_coverage_mismatch" in evidence["error_codes"]


@pytest.mark.parametrize("mutation", ["duplicate_batch", "unexpected_window"])
def test_duplicate_or_unexpected_batch_rows_are_rejected(tmp_path, mutation):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    rows = _complete_rows()
    if mutation == "duplicate_batch":
        rows.append(dict(rows[0]))
        expected = "duplicate_batch_index"
    else:
        extra = dict(rows[-1])
        extra.update(batch_index=999, window_index=2)
        rows.append(extra)
        expected = "unexpected_batch_epoch_window"
    _write_rows(root / "benchmark_timing/basic_batches.csv", rows)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert expected in evidence["error_codes"]


def test_pure_off_uses_receipt_without_batch_csv(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="off")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is True
    assert evidence["measurement_source"] == "pure_wall"
    assert evidence["batch_rows_required"] is False


def test_pure_off_rejects_incomplete_receipt(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="off")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root, epochs=range(1, 6))
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "receipt_window_coverage_mismatch" in evidence["error_codes"]


def test_multiple_batches_per_window_are_validated_without_one_row_assumption(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root, batches_per_window=2)
    rows = []
    batch_index = 1
    for epoch in range(1, 7):
        for window_index in range(2):
            for events in (2, 3):
                rows.append({
                    "batch_index": batch_index,
                    "epoch": epoch,
                    "window_index": window_index,
                    "batch_events": events,
                    "total_seconds": 0.25,
                })
                batch_index += 1
    _write_rows(root / "benchmark_timing/basic_batches.csv", rows)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is True
    assert evidence["batch_row_count"] == 24


@pytest.mark.parametrize(
    ("field", "value"),
    [("batch_events", 0), ("total_seconds", 0), ("total_seconds", float("nan"))],
)
def test_nonpositive_or_nonfinite_batch_measurement_is_rejected(
    tmp_path, field, value
):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    rows = _complete_rows()
    rows[0][field] = value
    _write_rows(root / "benchmark_timing/basic_batches.csv", rows)
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "measurement_row_invalid" in evidence["error_codes"]


def _mock_gpu_inventory(monkeypatch, output):
    monkeypatch.setattr(run_runtime_benchmark.shutil, "which", lambda name: "/nvidia-smi")
    monkeypatch.setattr(
        run_runtime_benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output),
    )


def test_execute_gpu_uuid_unresolved_is_rejected(monkeypatch):
    _mock_gpu_inventory(monkeypatch, "0, GPU-zero\n1, GPU-one\n")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="unique GPU UUID"):
        run_runtime_benchmark._gpu_identity(require_resolved=True)


def test_execute_gpu_multi_device_mapping_is_rejected(monkeypatch):
    _mock_gpu_inventory(monkeypatch, "1, GPU-one\n3, GPU-three\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    with pytest.raises(ValueError, match="exactly one"):
        run_runtime_benchmark._gpu_identity(require_resolved=True)


def test_execute_gpu_numeric_selector_on_multi_gpu_is_rejected(monkeypatch):
    _mock_gpu_inventory(monkeypatch, "1, GPU-one\n3, GPU-three\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="cannot be mapped reliably"):
        run_runtime_benchmark._gpu_identity(require_resolved=True)


def test_execute_gpu_selector_inventory_mismatch_is_rejected(monkeypatch):
    _mock_gpu_inventory(monkeypatch, "0, GPU-zero\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-not-present")
    with pytest.raises(ValueError, match="does not match"):
        run_runtime_benchmark._gpu_identity(require_resolved=True)


def test_gpu_sampler_never_falls_back_to_all_devices(tmp_path, monkeypatch):
    monkeypatch.setattr(run_runtime_benchmark.shutil, "which", lambda name: "/nvidia-smi")
    with pytest.raises(ValueError, match="GPU UUID"):
        run_runtime_benchmark.start_gpu_sampler(tmp_path / "gpu.csv", 200, None)
    assert not (tmp_path / "gpu.csv").exists()


def test_gpu_sampler_is_explicitly_bound_to_uuid_and_separates_stderr(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["stdout"] = kwargs["stdout"]
        captured["stderr"] = kwargs["stderr"]
        return FakeProcess()

    monkeypatch.setattr(run_runtime_benchmark.shutil, "which", lambda name: "/nvidia-smi")
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "Popen", fake_popen)
    sampler = run_runtime_benchmark.start_gpu_sampler(
        tmp_path / "gpu.csv", 200, "GPU-test"
    )
    status = run_runtime_benchmark.stop_gpu_sampler(sampler)
    assert isinstance(sampler.process, FakeProcess)
    assert "--id=GPU-test" in captured["command"]
    assert "--format=csv,nounits" in captured["command"]
    assert captured["stdout"] is not captured["stderr"]
    assert sampler.stderr_path.name == "gpu_sampler.stderr.log"
    assert captured["stdout"].closed is True
    assert captured["stderr"].closed is True
    assert status["unexpected_exit"] is False
    assert status["shutdown_returncode"] == -15


def test_gpu_sampler_start_failure_closes_output_handles(tmp_path, monkeypatch):
    observed = {}

    def fail_popen(command, **kwargs):
        observed["stdout"] = kwargs["stdout"]
        observed["stderr"] = kwargs["stderr"]
        raise OSError("cannot start")

    monkeypatch.setattr(run_runtime_benchmark.shutil, "which", lambda name: "/nvidia-smi")
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "Popen", fail_popen)
    with pytest.raises(OSError, match="cannot start"):
        run_runtime_benchmark.start_gpu_sampler(
            tmp_path / "gpu.csv", 200, "GPU-test"
        )
    assert observed["stdout"].closed is True
    assert observed["stderr"].closed is True


def test_gpu_telemetry_normal_run_covers_child_lifecycle(tmp_path):
    base = datetime(2026, 1, 1, 0, 0, 0).timestamp()
    csv_path = tmp_path / "gpu_utilization.csv"
    stderr_path = tmp_path / "gpu_sampler.stderr.log"
    _write_gpu_csv(csv_path, [base + step * 0.2 for step in range(26)])
    stderr_path.write_text("")
    result = run_runtime_benchmark.validate_gpu_telemetry(
        csv_path,
        stderr_path,
        gpu_uuid="GPU-test",
        interval_ms=200,
        child_started_epoch_seconds=base + 0.1,
        child_ended_epoch_seconds=base + 4.9,
        sampler_status=_healthy_sampler_status(),
    )
    assert result["valid"] is True
    assert result["sample_count"] == 26
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["max_sample_gap_seconds"] == pytest.approx(0.2, abs=0.002)
    assert result["utilization_statistics"]["utilization_gpu_mean_percent"] == 50.0


@pytest.mark.parametrize(
    ("returncode", "expected_codes"),
    [
        (0, {"gpu_sampler_exited_early"}),
        (7, {"gpu_sampler_exited_early", "gpu_sampler_abnormal_exit"}),
    ],
)
def test_gpu_telemetry_rejects_early_and_abnormal_exit(
    tmp_path, returncode, expected_codes
):
    base = datetime(2026, 1, 1, 0, 0, 0).timestamp()
    csv_path = tmp_path / "gpu_utilization.csv"
    stderr_path = tmp_path / "gpu_sampler.stderr.log"
    _write_gpu_csv(csv_path, [base + step * 0.2 for step in range(11)])
    stderr_path.write_text("")
    status = _healthy_sampler_status()
    status.update(
        unexpected_exit=True,
        unexpected_exit_returncode=returncode,
        unexpected_exit_epoch_seconds=base + 1.0,
        termination_requested=False,
        shutdown_returncode=returncode,
    )
    result = run_runtime_benchmark.validate_gpu_telemetry(
        csv_path,
        stderr_path,
        gpu_uuid="GPU-test",
        interval_ms=200,
        child_started_epoch_seconds=base,
        child_ended_epoch_seconds=base + 2.0,
        sampler_status=status,
    )
    assert result["valid"] is False
    assert expected_codes <= set(result["error_codes"])
    assert result["utilization_statistics"] is None


def test_gpu_telemetry_rejects_corrupt_output(tmp_path):
    base = datetime(2026, 1, 1, 0, 0, 0).timestamp()
    csv_path = tmp_path / "gpu_utilization.csv"
    stderr_path = tmp_path / "gpu_sampler.stderr.log"
    _write_gpu_csv(
        csv_path,
        [base, base + 0.2],
        corrupt_row=["Unable to determine the device handle"],
    )
    stderr_path.write_text("")
    result = run_runtime_benchmark.validate_gpu_telemetry(
        csv_path,
        stderr_path,
        gpu_uuid="GPU-test",
        interval_ms=200,
        child_started_epoch_seconds=base,
        child_ended_epoch_seconds=base + 0.4,
        sampler_status=_healthy_sampler_status(),
    )
    assert result["valid"] is False
    assert "gpu_sample_row_invalid" in result["error_codes"]
    assert result["utilization_statistics"] is None


def test_gpu_telemetry_rejects_sampling_interruption(tmp_path):
    base = datetime(2026, 1, 1, 0, 0, 0).timestamp()
    csv_path = tmp_path / "gpu_utilization.csv"
    stderr_path = tmp_path / "gpu_sampler.stderr.log"
    _write_gpu_csv(csv_path, [base, base + 0.2, base + 3.0, base + 3.2])
    stderr_path.write_text("")
    result = run_runtime_benchmark.validate_gpu_telemetry(
        csv_path,
        stderr_path,
        gpu_uuid="GPU-test",
        interval_ms=200,
        child_started_epoch_seconds=base,
        child_ended_epoch_seconds=base + 3.2,
        sampler_status=_healthy_sampler_status(),
    )
    assert result["valid"] is False
    assert "gpu_sample_gap_excessive" in result["error_codes"]


def test_gpu_telemetry_rejects_insufficient_coverage(tmp_path):
    base = datetime(2026, 1, 1, 0, 0, 0).timestamp()
    csv_path = tmp_path / "gpu_utilization.csv"
    stderr_path = tmp_path / "gpu_sampler.stderr.log"
    _write_gpu_csv(csv_path, [base + step * 0.2 for step in range(6)])
    stderr_path.write_text("")
    result = run_runtime_benchmark.validate_gpu_telemetry(
        csv_path,
        stderr_path,
        gpu_uuid="GPU-test",
        interval_ms=200,
        child_started_epoch_seconds=base,
        child_ended_epoch_seconds=base + 10.0,
        sampler_status=_healthy_sampler_status(),
    )
    assert result["valid"] is False
    assert "gpu_coverage_end_early" in result["error_codes"]
    assert "gpu_coverage_insufficient" in result["error_codes"]
    assert result["coverage_ratio"] < 0.2
    assert result["utilization_statistics"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_source_sha", "missing required fields"),
        ("bad_relation", "start/end/count"),
        ("dataset", "dataset"),
        ("source_mismatch", "source SHA256"),
    ],
)
def test_manifest_strong_validation(tmp_path, mutation, message):
    args = _args(tmp_path)
    manifest = tmp_path / "manifest.json"
    payload = _write_valid_manifest(manifest)
    if mutation == "missing_source_sha":
        payload.pop("source_sha256")
    elif mutation == "bad_relation":
        payload["selected_end_exclusive"] = 99
    elif mutation == "dataset":
        payload["dataset"] = "THEIA_E5"
    else:
        payload["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    args.selection_manifest = str(manifest)
    with pytest.raises(ValueError, match=message):
        run_runtime_benchmark.validate_window_manifest(args, required=True)


def test_formal_manifest_missing_source_is_rejected(tmp_path):
    args = _args(tmp_path)
    manifest = tmp_path / "manifest.json"
    _write_valid_manifest(manifest, source_exists=False)
    args.selection_manifest = str(manifest)
    with pytest.raises(ValueError, match="source Batch CSV does not exist"):
        run_runtime_benchmark.validate_window_manifest(args, required=True)


def test_manifest_eligible_epochs_must_be_ordered_and_contiguous(tmp_path):
    args = _args(tmp_path)
    manifest = tmp_path / "manifest.json"
    payload = _write_valid_manifest(manifest)
    payload["eligible_epochs"] = [2, 4, 3, 5, 6]
    manifest.write_text(json.dumps(payload))
    args.selection_manifest = str(manifest)
    with pytest.raises(ValueError, match="ordered and contiguous"):
        run_runtime_benchmark.validate_window_manifest(args, required=True)


def test_manifest_eligible_epochs_must_match_six_epoch_protocol(tmp_path):
    args = _args(tmp_path, execute=True)
    manifest = tmp_path / "manifest.json"
    payload = _write_valid_manifest(manifest)
    source = manifest.with_suffix(".csv")
    _write_rows(source, _complete_rows(epochs=range(1, 4)))
    payload["eligible_epochs"] = [2, 3]
    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))
    args.selection_manifest = str(manifest)
    with pytest.raises(ValueError, match="do not match benchmark epochs"):
        run_runtime_benchmark.validate(args)


def test_manifest_source_must_cover_every_selected_eligible_epoch_window(tmp_path):
    args = _args(tmp_path)
    manifest = tmp_path / "manifest.json"
    payload = _write_valid_manifest(manifest)
    source = manifest.with_suffix(".csv")
    rows = [
        row for row in _complete_rows()
        if not (row["epoch"] == 6 and row["window_index"] == 1)
    ]
    _write_rows(source, rows)
    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    inventory_result = Path(payload["inventory_result"])
    _write_inventory_result(inventory_result, source)
    payload["inventory_result_sha256"] = hashlib.sha256(
        inventory_result.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload))
    args.selection_manifest = str(manifest)
    with pytest.raises(ValueError, match="does not cover every selected"):
        run_runtime_benchmark.validate_window_manifest(args, required=True)


def test_batch_metadata_identity_is_scoped_and_pure_off_has_no_empty_digest(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    result = run_runtime_benchmark.aggregate(
        root, 0, 2, profile_mode="off", basic_timing="off", warmup_epochs=1
    )
    assert result["measurement_identity_sha256"] is None
    assert result["batch_metadata_sha256"] is None
    assert result["measurement_identity_scope"] == "unavailable:no_batch_rows"
    assert result["statistics_status"] == "execution_complete_no_batch_throughput"
    assert result["events"] is None


def test_complete_detailed_evidence_is_accepted(tmp_path):
    args = _args(tmp_path, profile_mode="on", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    _write_rows(root / "profiling/perf_batches.csv", _complete_rows())
    profile = root / "profiling/perf_profile.json"
    profile.write_text(json.dumps({"enabled": True, "status": "completed"}))
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is True
    assert evidence["measurement_source"] == "detailed_profile"
    assert evidence["observed_measured_pair_count"] == 10


def test_detailed_profile_must_be_completed_json_object(tmp_path):
    args = _args(tmp_path, profile_mode="on", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    _write_rows(root / "profiling/perf_batches.csv", _complete_rows())
    profile = root / "profiling/perf_profile.json"
    profile.write_text(json.dumps([]))
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "detailed_profile_invalid" in evidence["error_codes"]


def test_requested_torch_trace_requires_separate_trace_artifact(tmp_path):
    args = _args(
        tmp_path, profile_mode="on", basic_timing="on", torch_trace_batches=1
    )
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    _write_rows(root / "profiling/perf_batches.csv", _complete_rows())
    profile = root / "profiling/perf_profile.json"
    profile.write_text(json.dumps({"enabled": True, "status": "completed"}))
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "torch_trace_missing" in evidence["error_codes"]


def test_trace_run_is_not_published_as_throughput(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _write_rows(root / "profiling/perf_batches.csv", _complete_rows())
    result = run_runtime_benchmark.aggregate(
        root, 0, 2, profile_mode="on", basic_timing="on", warmup_epochs=1,
        torch_trace_batches=1,
    )
    assert result["statistics_status"] == "diagnostic_trace_not_throughput"
    assert result["events_per_second"] is None
    assert result["batches_per_second"] is None


def test_incomplete_execute_writes_failed_result_without_statistics(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output)
        rows = [
            row for row in _complete_rows()
            if not (row["epoch"] == 6 and row["window_index"] == 1)
        ]
        _write_rows(output / "benchmark_timing/basic_batches.csv", rows)
        return SimpleNamespace(returncode=0)

    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    _mock_valid_gpu_sampler(monkeypatch)
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity", lambda **kwargs: resolved_gpu
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 88
    result_path = Path(args.artifact_root) / "benchmark_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["child_returncode"] == 0
    assert result["statistics_published"] is False
    assert result["aggregate_published"] is False
    assert result["aggregate"]["statistics_status"] == "invalid_not_published"
    assert result["aggregate"]["events"] is None
    assert any(
        reason.startswith("execution_evidence:")
        for reason in result["failure_reasons"]
    )
    assert _tree_content_snapshot(shared) == before


def test_gpu_sampler_start_failure_writes_failed_result(tmp_path, monkeypatch):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity", lambda **kwargs: resolved_gpu
    )
    monkeypatch.setattr(
        run_runtime_benchmark,
        "start_gpu_sampler",
        lambda *args: (_ for _ in ()).throw(OSError("sampler unavailable")),
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 89
    result_path = Path(args.artifact_root) / "benchmark_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["child_returncode"] is None
    assert result["returncode"] == 89
    assert result["statistics_published"] is False
    assert result["aggregate_published"] is False
    assert result["execution_evidence"]["error_codes"] == [
        "gpu_sampler_start_failed"
    ]
    assert result["aggregate"]["statistics_status"] == "invalid_not_published"
    assert result["aggregate"]["events_per_second"] is None
    assert _tree_content_snapshot(shared) == before


def test_formal_statistics_rejected_when_gpu_telemetry_is_invalid(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="on", basic_timing="on")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output)
        _write_rows(output / "profiling/perf_batches.csv", _complete_rows())
        (output / "profiling/perf_profile.json").write_text(
            json.dumps({"enabled": True, "status": "completed"})
        )
        return SimpleNamespace(returncode=0)

    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    _mock_valid_gpu_sampler(monkeypatch)
    monkeypatch.setattr(
        run_runtime_benchmark,
        "validate_gpu_telemetry",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "valid": False,
            "errors": [{
                "code": "gpu_coverage_insufficient",
                "message": "fault-injected incomplete coverage",
            }],
            "error_codes": ["gpu_coverage_insufficient"],
            "statistics_status": "invalid_not_published",
            "utilization_statistics": None,
        },
    )
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity", lambda **kwargs: resolved_gpu
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "on",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 91
    output = Path(args.artifact_root)
    result = json.loads((output / "benchmark_result.json").read_text())
    assert result["status"] == "failed"
    assert result["child_returncode"] == 0
    assert result["gpu_telemetry"]["valid"] is False
    assert result["gpu_telemetry"]["utilization_statistics"] is None
    assert result["aggregate_published"] is False
    assert result["statistics_published"] is False
    assert result["gpu_statistics_published"] is False
    assert result["formal_baseline_eligible"] is False
    assert result["aggregate"]["statistics_status"] == "invalid_not_published"
    assert "gpu_telemetry:gpu_coverage_insufficient" in result["failure_reasons"]
    assert (output / "runtime.json").is_file()
    assert (output / "profiling/perf_batches.csv").is_file()
    assert (output / "profiling/perf_profile.json").is_file()


def test_complete_pure_off_execute_uses_receipt_without_publishing_throughput(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="off", basic_timing="off")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output)
        return SimpleNamespace(returncode=0)

    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    _mock_valid_gpu_sampler(monkeypatch)
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity", lambda **kwargs: resolved_gpu
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "off",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    result_path = Path(args.artifact_root) / "benchmark_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["status"] == "completed"
    assert result["execution_evidence"]["valid"] is True
    assert result["execution_evidence"]["batch_rows_required"] is False
    assert result["aggregate_published"] is True
    assert result["statistics_published"] is False
    assert result["aggregate"]["statistics_status"] == (
        "execution_complete_no_batch_throughput"
    )
    assert result["aggregate"]["events"] is None
    assert result["aggregate"]["events_per_second"] is None
    assert result["aggregate"]["measurement_identity_sha256"] is None
    assert result["formal_manifest_source_verified_at_publish"] is True
    assert _tree_content_snapshot(shared) == before


def test_gpu_identity_preflight_failure_writes_status_without_starting_run(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    static_identity = {
        "gpu_uuid": None,
        "gpu_uuids": ["GPU-zero", "GPU-one"],
        "cuda_visible_devices": None,
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "not_runtime_verified",
        "gpu_identity_runtime_verified": False,
    }

    def fake_gpu_identity(*, require_resolved=False):
        if require_resolved:
            raise ValueError("formal execute cannot resolve a unique GPU UUID")
        return dict(static_identity)

    monkeypatch.setattr(run_runtime_benchmark, "_gpu_identity", fake_gpu_identity)
    monkeypatch.setattr(
        run_runtime_benchmark,
        "start_gpu_sampler",
        lambda *args: pytest.fail("sampler must not start after GPU preflight failure"),
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 84
    result_path = Path(args.artifact_root) / "benchmark_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert result["returncode"] == 84
    assert result["child_returncode"] is None
    assert result["experiment_identity"]["gpu_uuid"] is None
    assert result["experiment_identity"]["gpu_identity_runtime_verified"] is False
    assert result["execution_evidence"]["error_codes"] == [
        "gpu_identity_unresolved"
    ]
    assert result["statistics_published"] is False
    assert result["aggregate_published"] is False
    assert result["aggregate"]["events_per_second"] is None
    assert _tree_content_snapshot(shared) == before


def test_formal_execute_without_manifest_is_rejected(tmp_path):
    args = _args(tmp_path, execute=True)
    with pytest.raises(ValueError, match="formal execute requires --selection-manifest"):
        run_runtime_benchmark.validate(args)
    assert not Path(args.artifact_root).exists()


def test_selection_inventory_accepts_more_than_twenty_train_only_windows(tmp_path):
    args = _args(
        tmp_path,
        windows=40,
        profile_mode="off",
        basic_timing="on",
        selection_inventory=True,
    )
    _ensure_train_window_count(args, 40)
    roots = run_runtime_benchmark.validate(args)
    command = run_runtime_benchmark.build_command(args, roots=roots)
    assert args._benchmark_protocol == "selection_inventory"
    assert args._validated_manifest is None
    assert command[command.index("--max-windows-per-split") + 1] == "40"
    assert command[command.index("--stages") + 1] == "train"
    assert "--benchmark-basic-timing" in command
    assert "--benchmark-inventory-history-stats" in command
    assert "--perf-profile" not in command
    assert "--shared-artifact-readonly" in command


def test_selection_inventory_rejects_insufficient_train_windows(tmp_path):
    args = _args(
        tmp_path,
        windows=40,
        profile_mode="off",
        selection_inventory=True,
    )
    with pytest.raises(ValueError, match="more windows than the shared train split"):
        run_runtime_benchmark.validate(args)


def test_selection_inventory_execute_is_diagnostic_only(tmp_path, monkeypatch):
    args = _args(
        tmp_path,
        windows=21,
        profile_mode="off",
        basic_timing="on",
        selection_inventory=True,
    )
    _ensure_train_window_count(args, 21)
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output, window_end=21)
        _write_rows(
            output / "benchmark_timing/basic_batches.csv",
            _complete_rows(window_end=21),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_runtime_benchmark,
        "_gpu_identity",
        lambda **kwargs: {
            "gpu_uuid": "GPU-test",
            "gpu_uuids": ["GPU-test"],
            "cuda_visible_devices": "GPU-test",
            "nvidia_visible_devices": None,
            "cuda_device_order": None,
            "gpu_identity_scope": "test",
            "gpu_mapping_validation_status": "validated_test_mapping",
            "gpu_identity_runtime_verified": kwargs.get("require_resolved", False),
        },
    )
    _mock_valid_gpu_sampler(monkeypatch)
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "21",
        "--profile-mode", "off",
        "--basic-timing", "on",
        "--selection-inventory",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    result = json.loads(
        (Path(args.artifact_root) / "benchmark_result.json").read_text()
    )
    assert result["run_protocol"] == "selection_inventory_diagnostic"
    assert result["selection_inventory"] is True
    assert result["selection_manifest_required"] is False
    assert result["formal_baseline_eligible"] is False
    assert result["statistics_published"] is False
    assert result["diagnostic_statistics_published"] is True
    assert result["execution_evidence"]["valid"] is True
    assert result["execution_evidence"]["measurement_sha256"]
    assert result["aggregate"]["statistics_status"] == (
        "diagnostic_selection_inventory_not_formal_baseline"
    )
    assert result["aggregate"]["events_per_second"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"windows": 20}, "only 2 or 10"),
        ({"window_start": 1}, "window-start 0"),
        ({"profile_mode": "off"}, "profile-mode on"),
        ({"torch_trace_batches": 1}, "does not allow Torch trace"),
    ],
)
def test_invalid_bootstrap_parameters_are_rejected(tmp_path, overrides, message):
    args = _args(tmp_path, bootstrap_inventory=True, **overrides)
    with pytest.raises(ValueError, match=message):
        run_runtime_benchmark.validate(args)
    assert not Path(args.artifact_root).exists()


def test_bootstrap_rejects_selection_manifest(tmp_path):
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    args = _args(
        tmp_path,
        bootstrap_inventory=True,
        selection_manifest=str(manifest),
    )
    with pytest.raises(ValueError, match="must not use a selection manifest"):
        run_runtime_benchmark.validate(args)


@pytest.mark.parametrize(
    ("dataset", "windows"),
    [
        ("THEIA_E3", 2),
        ("THEIA_E3", 10),
        ("THEIA_E5", 2),
        ("THEIA_E5", 10),
    ],
)
def test_bootstrap_uses_common_e3_e5_bounded_command_path(
    tmp_path, dataset, windows
):
    args = _args(
        tmp_path,
        dataset=dataset,
        windows=windows,
        bootstrap_inventory=True,
    )
    roots = run_runtime_benchmark.validate(args)
    command = run_runtime_benchmark.build_command(args, roots=roots)
    assert args._benchmark_protocol == "bootstrap_inventory"
    assert args._validated_manifest is None
    assert command[command.index("--dataset") + 1] == dataset
    assert command[command.index("--max-windows-per-split") + 1] == str(windows)
    assert "--perf-profile" in command
    assert "--shared-artifact-readonly" in command


def test_bootstrap_execute_is_diagnostic_and_still_requires_strict_gpu(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, bootstrap_inventory=True)
    shared = Path(args.shared_artifact_root)
    before = _tree_content_snapshot(shared)
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output)
        _write_rows(output / "profiling/perf_batches.csv", _complete_rows())
        profile = output / "profiling/perf_profile.json"
        profile.write_text(json.dumps({"enabled": True, "status": "completed"}))
        return SimpleNamespace(returncode=0)

    gpu_calls = []
    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }

    def fake_gpu_identity(*, require_resolved=False):
        gpu_calls.append(require_resolved)
        result = dict(resolved_gpu)
        result["gpu_identity_runtime_verified"] = require_resolved
        return result

    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(run_runtime_benchmark, "_gpu_identity", fake_gpu_identity)
    _mock_valid_gpu_sampler(monkeypatch)
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--window-start", "0",
        "--profile-mode", "on",
        "--basic-timing", "on",
        "--bootstrap-inventory",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    result_path = Path(args.artifact_root) / "benchmark_result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text())
    assert gpu_calls == [False, True]
    assert result["status"] == "completed"
    assert result["run_protocol"] == "bootstrap_inventory_diagnostic"
    assert result["protocol_schema_version"] == 2
    assert result["bootstrap_inventory"] is True
    assert result["selection_manifest_required"] is False
    assert result["formal_manifest_source_verified_at_publish"] is None
    assert result["formal_baseline_eligible"] is False
    assert result["statistics_published"] is False
    assert result["diagnostic_statistics_published"] is True
    assert result["gpu_statistics_published"] is False
    assert result["gpu_telemetry"]["statistics_status"] == (
        "diagnostic_only_not_formal_baseline"
    )
    assert result["aggregate_published"] is True
    assert result["execution_evidence"]["valid"] is True
    assert result["aggregate"]["statistics_status"] == (
        "diagnostic_bootstrap_inventory_not_formal_baseline"
    )
    assert result["aggregate"]["events_per_second"] is None
    assert result["aggregate"]["batches_per_second"] is None
    assert result["experiment_identity"]["window_manifest_sha256"] is None
    assert (Path(args.artifact_root) / "profiling/perf_batches.csv").is_file()
    assert _tree_content_snapshot(shared) == before


def test_bootstrap_dry_run_stays_static_and_writes_no_result(
    tmp_path, monkeypatch, capsys
):
    args = _args(tmp_path, bootstrap_inventory=True)
    original_run = run_runtime_benchmark.subprocess.run
    gpu_calls = []

    def fake_run(command, **kwargs):
        if "--artifact-root" in command:
            pytest.fail("bootstrap dry-run must not start the training child")
        return original_run(command, **kwargs)

    def fake_gpu_identity(*, require_resolved=False):
        gpu_calls.append(require_resolved)
        return {
            "gpu_uuid": None,
            "gpu_uuids": [],
            "cuda_visible_devices": None,
            "nvidia_visible_devices": None,
            "cuda_device_order": None,
            "gpu_identity_scope": "test",
            "gpu_mapping_validation_status": "not_runtime_verified",
            "gpu_identity_runtime_verified": False,
        }

    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(run_runtime_benchmark, "_gpu_identity", fake_gpu_identity)
    monkeypatch.setattr(
        run_runtime_benchmark,
        "start_gpu_sampler",
        lambda *args: pytest.fail("bootstrap dry-run must not start GPU sampling"),
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "on",
        "--bootstrap-inventory",
    ]
    assert run_runtime_benchmark.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert gpu_calls == [False]
    assert payload["dry_run"] is True
    assert payload["run_protocol"] == "bootstrap_inventory_diagnostic"
    assert payload["formal_baseline_eligible"] is False
    assert payload["experiment_identity"]["gpu_identity_runtime_verified"] is False
    assert not Path(args.artifact_root).exists()


def test_bootstrap_fails_closed_when_gpu_uuid_cannot_be_resolved(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, bootstrap_inventory=True)

    def fake_gpu_identity(*, require_resolved=False):
        if require_resolved:
            raise ValueError("cannot resolve unique GPU UUID")
        return {
            "gpu_uuid": None,
            "gpu_uuids": ["GPU-zero", "GPU-one"],
            "cuda_visible_devices": None,
            "nvidia_visible_devices": None,
            "cuda_device_order": None,
            "gpu_identity_scope": "test",
            "gpu_mapping_validation_status": "ambiguous",
            "gpu_identity_runtime_verified": False,
        }

    monkeypatch.setattr(run_runtime_benchmark, "_gpu_identity", fake_gpu_identity)
    monkeypatch.setattr(
        run_runtime_benchmark,
        "start_gpu_sampler",
        lambda *args: pytest.fail("GPU sampler must not start"),
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "on",
        "--bootstrap-inventory",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 84
    result = json.loads(
        (Path(args.artifact_root) / "benchmark_result.json").read_text()
    )
    assert result["status"] == "failed"
    assert result["bootstrap_inventory"] is True
    assert result["execution_evidence"]["error_codes"] == [
        "gpu_identity_unresolved"
    ]
    assert result["statistics_published"] is False
    assert result["diagnostic_statistics_published"] is False


def test_prefix_windows_execute_but_only_selected_windows_are_aggregated(tmp_path):
    args = _args(
        tmp_path,
        windows=20,
        window_start=4,
        profile_mode="off",
        basic_timing="on",
    )
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root, window_end=24)
    _write_rows(
        root / "benchmark_timing/basic_batches.csv",
        _complete_rows(window_end=24),
    )
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is True
    assert evidence["expected_execution_window_indices"] == list(range(24))
    assert evidence["expected_measured_window_indices"] == list(range(4, 24))
    assert evidence["receipt_pair_count"] == 6 * 24
    assert evidence["observed_measured_pair_count"] == 5 * 20

    result = run_runtime_benchmark.aggregate(
        root,
        4,
        20,
        profile_mode="off",
        basic_timing="on",
        warmup_epochs=1,
    )
    assert result["warmup_batch_count"] == 24
    assert result["causal_replay_batch_count"] == 5 * 4
    assert result["measured_batch_count"] == 5 * 20
    assert result["measured_distinct_window_indices"] == list(range(4, 24))
    assert result["excluded_other_batch_count"] == 0


def test_execution_evidence_rejects_non_unique_window_identity(tmp_path):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    root = tmp_path / "run"
    root.mkdir()
    _write_runtime_receipt(root)
    runtime_path = root / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    windows = runtime["training"]["execution_receipt"]["windows"]
    for item in windows:
        if item["window_index"] == 1:
            item["window_id"] = 0
    runtime_path.write_text(json.dumps(runtime))
    _write_rows(root / "benchmark_timing/basic_batches.csv", _complete_rows())
    evidence = run_runtime_benchmark.validate_execution_evidence(
        root, args, num_epochs=6
    )
    assert evidence["valid"] is False
    assert "receipt_window_identity_mismatch" in evidence["error_codes"]


def test_ten_window_bootstrap_inventory_cannot_plan_twenty_windows(tmp_path):
    source = tmp_path / "bootstrap_batches.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_index",
                "epoch",
                "window_index",
                "batch_events",
                "history_valid_slots",
                "total_seconds",
            ],
        )
        writer.writeheader()
        batch_index = 1
        for epoch in range(1, 7):
            for window_index in range(10):
                writer.writerow({
                    "batch_index": batch_index,
                    "epoch": epoch,
                    "window_index": window_index,
                    "batch_events": 100 + window_index,
                    "history_valid_slots": window_index * 10,
                    "total_seconds": 1.0,
                })
                batch_index += 1
    inventory_result = tmp_path / "inventory-result.json"
    _write_inventory_result(inventory_result, source, dataset="THEIA_E5")
    with pytest.raises(ValueError, match="fewer windows than requested"):
        plan_runtime_windows.main([
            "--dataset", "THEIA_E5",
            "--batches-csv", str(source),
            "--inventory-result", str(inventory_result),
            "--window-count", "20",
            "--warmup-epochs", "1",
            "--output", str(tmp_path / "manifest.json"),
        ])
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("gap", "indices must be contiguous"),
        ("missing_epoch", "does not cover every eligible epoch"),
    ],
)
def test_window_planner_rejects_incomplete_inventory(tmp_path, mutation, message):
    source = tmp_path / "inventory.csv"
    rows = []
    for epoch in range(1, 4):
        for window_index in range(25):
            if mutation == "gap" and window_index == 12:
                continue
            if mutation == "missing_epoch" and epoch == 3 and window_index == 12:
                continue
            rows.append({
                "batch_index": len(rows) + 1,
                "epoch": epoch,
                "window_index": window_index,
                "batch_events": 100,
                "history_valid_slots": 10,
                "total_seconds": 1.0,
            })
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    inventory_result = tmp_path / "inventory-result.json"
    _write_inventory_result(inventory_result, source)
    output = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match=message):
        plan_runtime_windows.main([
            "--dataset", "THEIA_E3",
            "--batches-csv", str(source),
            "--inventory-result", str(inventory_result),
            "--window-count", "20",
            "--warmup-epochs", "1",
            "--expected-epochs", "3",
            "--output", str(output),
        ])
    assert not output.exists()


def test_formal_source_change_during_run_blocks_statistics_publication(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, profile_mode="off", basic_timing="on")
    manifest = tmp_path / "selection.json"
    _write_valid_manifest(manifest)
    source = manifest.with_suffix(".csv")
    original_run = run_runtime_benchmark.subprocess.run

    def fake_run(command, **kwargs):
        if "--artifact-root" not in command:
            return original_run(command, **kwargs)
        output = Path(command[command.index("--artifact-root") + 1])
        _write_runtime_receipt(output)
        _write_rows(output / "benchmark_timing/basic_batches.csv", _complete_rows())
        source.write_bytes(source.read_bytes() + b"\n")
        return SimpleNamespace(returncode=0)

    resolved_gpu = {
        "gpu_uuid": "GPU-test",
        "gpu_uuids": ["GPU-test"],
        "cuda_visible_devices": "GPU-test",
        "nvidia_visible_devices": None,
        "cuda_device_order": None,
        "gpu_identity_scope": "test",
        "gpu_mapping_validation_status": "validated_test_mapping",
        "gpu_identity_runtime_verified": True,
    }
    monkeypatch.setattr(run_runtime_benchmark.subprocess, "run", fake_run)
    _mock_valid_gpu_sampler(monkeypatch)
    monkeypatch.setattr(
        run_runtime_benchmark, "_gpu_identity", lambda **kwargs: resolved_gpu
    )
    argv = [
        "--dataset", args.dataset,
        "--config", args.config,
        "--artifact-root", args.artifact_root,
        "--shared-artifact-root", args.shared_artifact_root,
        "--windows", "2",
        "--profile-mode", "off",
        "--selection-manifest", str(manifest),
        "--basic-timing", "on",
        "--execute",
    ]
    assert run_runtime_benchmark.main(argv) == 90
    result = json.loads(
        (Path(args.artifact_root) / "benchmark_result.json").read_text()
    )
    assert result["status"] == "failed"
    assert result["child_returncode"] == 0
    assert result["formal_manifest_source_verified_at_publish"] is False
    assert result["aggregate_published"] is False
    assert result["statistics_published"] is False
    assert result["diagnostic_statistics_published"] is False
    assert result["aggregate"]["statistics_status"] == "invalid_not_published"
    assert result["aggregate"]["events_per_second"] is None
    assert "formal_manifest_source_not_verified_at_publish" in result[
        "failure_reasons"
    ]
