"""Synthetic transfer-tool tests; these are not THEIA experiment evidence."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location("e5_transfer_tests", Path(__file__).resolve().parents[1] / "scripts/e5_smoke_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def fixture_manifest(tmp_path, count=1):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index in range(count):
        path = source / f"fixture{index}.bin"
        path.write_bytes(f"synthetic transfer fixture {index}".encode())
        rows.append({"source_absolute": str(path), "relative_path": f"datasets/THEIA_E5/fixture{index}.bin",
                     "bytes": path.stat().st_size, "sha256": transfer.digest(path),
                     "required": True, "exists": True, "is_symlink": False})
    value = {"schema_version": 1, "status": "PASS", "files": rows,
             "resolved_config": {"database": {"password": "DO_NOT_PACKAGE"}, "model": {"variant": "mstc"}}}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def build(tmp_path, count=1):
    plan, original = fixture_manifest(tmp_path, count)
    folder = tmp_path / "package"
    assert transfer.bundle(NS(manifest=plan, output=folder)) == 0
    archive = folder / "inputs.tar.gz"
    manifest = folder / "manifest.json"
    return original, NS(command="deploy", archive=archive, manifest=manifest,
                        sha256=transfer.digest(archive), target=tmp_path / "target")


def test_round_trip_and_repeat_reuse_without_overwrite(tmp_path):
    original, args = build(tmp_path)
    assert "DO_NOT_PACKAGE" not in args.manifest.read_text(encoding="utf-8")
    assert transfer.deploy(args) == 0
    target = args.target / original["files"][0]["relative_path"]
    before = target.stat().st_mtime_ns
    assert transfer.digest(target) == original["files"][0]["sha256"]
    assert transfer.deploy(args) == 0
    assert target.stat().st_mtime_ns == before
    assert (args.archive.parent / "sha256sums.txt").is_file()
    assert (args.archive.parent / "verify_windows.ps1").is_file()


def test_verify_does_not_deploy(tmp_path):
    _, args = build(tmp_path)
    args.command = "verify"
    assert transfer.deploy(args) == 0
    assert not args.target.exists()


def test_existing_conflict_fails_before_any_payload_write(tmp_path):
    original, args = build(tmp_path, count=2)
    conflict = args.target / original["files"][1]["relative_path"]
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"existing experiment data")
    with pytest.raises(FileExistsError, match="preserved"):
        transfer.deploy(args)
    assert conflict.read_bytes() == b"existing experiment data"
    assert not (args.target / original["files"][0]["relative_path"]).exists()


def test_wrong_archive_checksum_prevents_all_writes(tmp_path):
    _, args = build(tmp_path)
    args.sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        transfer.deploy(args)
    assert not args.target.exists()


def test_changed_source_is_rejected_before_archive_creation(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    Path(manifest["files"][0]["source_absolute"]).write_bytes(b"changed source")
    folder = tmp_path / "package"
    with pytest.raises(ValueError, match="Source identity changed"):
        transfer.bundle(NS(manifest=path, output=folder))
    assert not (folder / "inputs.tar.gz").exists()


def test_five_gib_cap_is_enforced_before_source_read(tmp_path):
    path, manifest = fixture_manifest(tmp_path)
    manifest["files"][0]["bytes"] = transfer.LIMIT + 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    folder = tmp_path / "package"
    with pytest.raises(ValueError, match="5 GiB"):
        transfer.bundle(NS(manifest=path, output=folder))
    assert not folder.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\escape", "a:name", "CON", "aux.txt", "a/COM1.bin", "a. ", "a//b"])
def test_invalid_windows_or_traversal_paths_are_rejected(name):
    with pytest.raises(ValueError):
        transfer.safe_relative(name)


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE])
def test_special_archive_members_are_rejected(tmp_path, member_type):
    _, manifest = fixture_manifest(tmp_path)
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("payload/link")
        member.type = member_type
        member.linkname = "../../outside"
        handle.addfile(member)
    with pytest.raises(ValueError, match="special member"):
        transfer.verify_archive(archive, transfer.redact(manifest))


def test_case_colliding_targets_are_rejected(tmp_path):
    _, manifest = fixture_manifest(tmp_path)
    row = copy.deepcopy(manifest["files"][0])
    row["relative_path"] = row["relative_path"].upper()
    manifest["files"].append(row)
    with pytest.raises(ValueError, match="case-colliding"):
        transfer.validate_manifest(manifest)


def test_missing_required_file_cannot_be_packaged(tmp_path):
    _, manifest = fixture_manifest(tmp_path)
    manifest["files"][0]["exists"] = False
    with pytest.raises(ValueError, match="missing"):
        transfer.validate_manifest(manifest)


def test_embedded_manifest_must_match_external_manifest(tmp_path):
    _, args = build(tmp_path)
    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    value["server_commit"] = "different"
    with pytest.raises(ValueError, match="Embedded manifest differs"):
        transfer.verify_archive(args.archive, value)


def test_windows_junction_in_target_is_rejected(tmp_path, make_path_alias):
    _, args = build(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = make_path_alias(args.target, outside, target_is_directory=True)
    args.target = alias
    with pytest.raises(ValueError, match="root traverses a symlink"):
        transfer.deploy(args)
    assert list(outside.iterdir()) == []


def test_credentials_are_redacted_without_changing_algorithm_fields():
    result = transfer.redact({"database_password": "secret", "api_key": "secret",
                              "neighbor_budgets": [8, 8, 8], "lambda_time": 0.3,
                              "command": "tool password=secret"})
    assert "secret" not in json.dumps(result)
    assert result["neighbor_budgets"] == [8, 8, 8]
    assert result["lambda_time"] == 0.3
