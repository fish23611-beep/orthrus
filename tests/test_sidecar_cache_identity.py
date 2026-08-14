"""
Correctness contract for the compact-history persistent sidecar cache.

Covers:
  1. semantic_config_fingerprint gate (baseline / MSTC cannot mis-share)
  2. source_metadata_fingerprint uses relative paths (no basename collision)
  3. sidecar is structurally absent from source-window discovery
  4. window count and global_event_index are stable across sidecar runs
  5. metadata-scan and compact-build source-load phases are distinguished
  6. second load_all_datasets still does NOT touch source artifacts in
     the compact-build phase (the metadata-scan phase is a separate
     count and is documented to be additive)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_utils


def _encoder():
    return SimpleNamespace(
        edge_features="edge_type,msg",
        use_node_type_in_node_feats=True,
    )


def _cfg(root, *, predict_edge_type=False, num_node_types=3, num_edge_types=4):
    decoder = SimpleNamespace(
        used_methods="predict_edge_type" if predict_edge_type else "custom",
    )
    training = SimpleNamespace(encoder=_encoder(), decoder=decoder)
    return SimpleNamespace(
        _test_mode=False,
        _max_windows_per_split=None,
        dataset=SimpleNamespace(
            name="synthetic",
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
        ),
        dataset_view=SimpleNamespace(mode="host_network_full"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(root)),
            embed_nodes=SimpleNamespace(used_method="only_type", emb_dim=8),
        ),
        detection=SimpleNamespace(gnn_training=training),
    )


def _raw_window(base: int, events: int = 3) -> TemporalData:
    src = torch.arange(base, base + events, dtype=torch.long)
    dst = src + 10
    t = torch.arange(base * 100, base * 100 + events, dtype=torch.long)
    src_type = torch.nn.functional.one_hot(src.remainder(3), num_classes=3)
    dst_type = torch.nn.functional.one_hot(dst.remainder(3), num_classes=3)
    edge_type = torch.nn.functional.one_hot(
        torch.arange(events).remainder(4), num_classes=4
    )
    return TemporalData(
        src=src,
        dst=dst,
        t=t,
        msg=torch.cat([src_type, edge_type, dst_type], dim=-1).float(),
    )


def _artifacts(root, counts=(3, 2, 4), name_prefix="window_"):
    """Write the given (train, val, test) source-window counts."""
    for split_index, (split, count) in enumerate(zip(("train", "val", "test"), counts)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for window_index in range(count):
            torch.save(
                _raw_window(split_index * 100 + window_index * 10),
                directory / f"{name_prefix}{window_index:02d}.pt",
            )


def _manifest(sidecar_root: Path) -> dict:
    files = list(sidecar_root.glob("manifest__*.json"))
    assert files, "expected a manifest to be written"
    with open(files[0], "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 1. semantic_config_fingerprint invalidation
# ---------------------------------------------------------------------------

def test_persistent_sidecar_invalidation_on_semantic_config_change():
    """Changing a semantic-affecting cfg field must invalidate the sidecar."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root)
        cfg1 = _cfg(root, predict_edge_type=False)
        _, _, _, full1, _ = data_utils.load_all_datasets(cfg1)
        sidecar_root = root / data_utils._CompactIndex.PERSISTENT_DIR_NAME
        assert sidecar_root.is_dir()
        manifest1 = _manifest(sidecar_root)
        assert manifest1.get("completed") is True
        sem_fp1 = manifest1["semantic_config_fingerprint_sha256"]
        assert sem_fp1, "semantic_config_fingerprint must be recorded"

        # Same source artifacts, but flip predict_edge_type: this changes
        # the msg reconstruction contract (msg_dim differs) so the sidecar
        # MUST be invalidated and rebuilt.
        cfg2 = _cfg(root, predict_edge_type=True)
        _, _, _, full2, _ = data_utils.load_all_datasets(cfg2)
        manifest2 = _manifest(sidecar_root)
        sem_fp2 = manifest2["semantic_config_fingerprint_sha256"]
        assert sem_fp1 != sem_fp2, (
            "semantic_config_fingerprint_sha256 must change when "
            "predict_edge_type flips; the sidecar was reused incorrectly"
        )
        tel2_obj = full2._compact_index._telemetry
        tel2 = tel2_obj.as_dict() if hasattr(tel2_obj, "as_dict") else tel2_obj
        assert tel2["persistent_cache_hit"] is False
        # A rebuild is observable: the new run did a full source scan in the
        # compact-build phase and produced a fresh manifest with a different
        # semantic-config fingerprint.
        assert tel2["compact_build_source_load_count"] > 0
        assert tel2["index_build_count"] == 1, (
            "The fresh compact index for the second run must have built "
            "exactly once (cache miss path)."
        )
        print(
            "PASSED: semantic-config change invalidates the sidecar "
            f"(sem_fp1={sem_fp1[:8]} -> sem_fp2={sem_fp2[:8]})"
        )


def test_persistent_sidecar_hit_on_unchanged_semantic_config():
    """Re-running with the same cfg (incl. any non-semantic training noise)
    must hit the sidecar and produce an event-identical compact index."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root)
        cfg = _cfg(root)
        _, _, _, full1, _ = data_utils.load_all_datasets(cfg)
        sidecar_root = root / data_utils._CompactIndex.PERSISTENT_DIR_NAME
        manifest = _manifest(sidecar_root)
        fp_built = manifest["semantic_config_fingerprint_sha256"]

        # Re-run with the same cfg; the sidecar must be a hit.
        _, _, _, full2, _ = data_utils.load_all_datasets(cfg)
        tel2_obj = full2._compact_index._telemetry
        tel2 = tel2_obj.as_dict() if hasattr(tel2_obj, "as_dict") else tel2_obj
        assert tel2["persistent_cache_hit"] is True
        assert tel2["compact_build_source_load_count"] == 0, (
            "On a cache hit the compact-build phase must not re-touch any "
            "source artifact."
        )

        # The fp used for validation must match the recorded one.
        assert (
            tel2["sidecar_semantic_config_fingerprint_sha256"] == fp_built
        )

        # Event-by-event equality of the compact tensors (no semantic drift).
        ci1 = full1._compact_index
        ci2 = full2._compact_index
        assert ci1._src.tolist() == ci2._src.tolist()
        assert ci1._dst.tolist() == ci2._dst.tolist()
        assert ci1._t.tolist() == ci2._t.tolist()
        assert ci1._edge_type_index.tolist() == ci2._edge_type_index.tolist()
        print("PASSED: identical cfg keeps the cache hit and identical tensors")


# ---------------------------------------------------------------------------
# 2. source-metadata fingerprint uses relative paths
# ---------------------------------------------------------------------------

def test_source_metadata_fingerprint_distinguishes_basename_collision():
    """Two artifacts with the same basename in different splits must
    produce different source_metadata_fingerprint_sha256 values.

    Production layouts store windows under <edge_embeds>/<split>/<file>
    so we synthesise a layout with the same basename repeated in
    ``train`` and ``val``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Two splits share the basename ``window_00.pt``.
        (root / "train").mkdir(parents=True)
        (root / "val").mkdir(parents=True)
        (root / "test").mkdir(parents=True)
        for split in ("train", "val", "test"):
            torch.save(_raw_window(0), root / split / "window_00.pt")

        ci = data_utils._CompactIndex(
            specs=(),
            total_events=0,
        )
        # Attach a single artifact from train then val; fingerprint must
        # change as the relative_path differs.
        spec_train = data_utils._WindowSpec(
            path=str(root / "train" / "window_00.pt"),
            split_name="train",
            split_index=0,
            split_window_index=0,
        )
        ci._specs = (spec_train,)
        fp_train = ci._compute_source_metadata_fingerprint()["sha256"]

        spec_val = data_utils._WindowSpec(
            path=str(root / "val" / "window_00.pt"),
            split_name="val",
            split_index=1,
            split_window_index=0,
        )
        ci._specs = (spec_val,)
        fp_val = ci._compute_source_metadata_fingerprint()["sha256"]

        assert fp_train != fp_val, (
            "Source-metadata fingerprint must use relative paths so "
            "basename collisions across splits are distinguishable"
        )
        # And it must NOT depend on absolute path ordering: order-only
        # reordering of two different splits must still produce distinct
        # fingerprints because the relative path differs.
        print(
            "PASSED: source-metadata fingerprint is relative-path stable "
            f"(train={fp_train[:8]} val={fp_val[:8]})"
        )


# ---------------------------------------------------------------------------
# 3. sidecar is excluded from source-window discovery
# ---------------------------------------------------------------------------

def test_sidecar_excluded_from_source_window_discovery():
    """Even if the sidecar directory contains a file whose name matches a
    source-window filename, source-window discovery must not include it."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, counts=(2, 1, 1))
        sidecar = root / data_utils._CompactIndex.PERSISTENT_DIR_NAME
        sidecar.mkdir(parents=True, exist_ok=True)
        # Plant a decoy file with a name that LOOKS like a source window.
        decoy = sidecar / "window_00.pt"
        torch.save(_raw_window(999), decoy)

        cfg = _cfg(root)
        train, val, test, full, _ = data_utils.load_all_datasets(cfg)

        # Total window count is exactly 2+1+1, not inflated by the decoy.
        assert len(train) == 2
        assert len(val) == 1
        assert len(test) == 1
        # And no spec path is allowed to point inside the sidecar.
        for spec in full._specs:
            assert data_utils._CompactIndex.PERSISTENT_DIR_NAME not in spec.path, (
                f"Source window spec path leaked into the sidecar: {spec.path}"
            )
        print("PASSED: sidecar is excluded from source-window discovery")


# ---------------------------------------------------------------------------
# 4. window count and global_event_index stability across sidecar runs
# ---------------------------------------------------------------------------

def test_window_count_and_global_event_index_stable_across_sidecar_runs():
    """Two cold runs (no sidecar) and a warm run (sidecar hit) must
    produce identical window counts, ordering, and global event indices."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, counts=(3, 2, 4))
        cfg = _cfg(root)

        _, _, _, full_a, _ = data_utils.load_all_datasets(cfg)
        # Remove sidecar to force a second cold start.
        sidecar = root / data_utils._CompactIndex.PERSISTENT_DIR_NAME
        for child in sidecar.iterdir():
            if child.is_file():
                child.unlink()
        sidecar.rmdir()

        _, _, _, full_b, _ = data_utils.load_all_datasets(cfg)
        # Now warm hit.
        _, _, _, full_c, _ = data_utils.load_all_datasets(cfg)

        def _ids(full):
            return [
                (spec.global_window_id, spec.global_offset, spec.num_events)
                for spec in full._specs
            ]

        ids_a = _ids(full_a)
        ids_b = _ids(full_b)
        ids_c = _ids(full_c)
        assert ids_a == ids_b == ids_c, (
            "Window ordering / offsets / counts differ between runs"
        )

        # global_event_index field must also be identical.
        for f in (full_a, full_b, full_c):
            assert int(f.num_events) > 0
        ids_a_gei = full_a.get_event_values(
            "global_event_index", torch.arange(full_a.num_events)
        ).tolist()
        ids_b_gei = full_b.get_event_values(
            "global_event_index", torch.arange(full_b.num_events)
        ).tolist()
        ids_c_gei = full_c.get_event_values(
            "global_event_index", torch.arange(full_c.num_events)
        ).tolist()
        assert ids_a_gei == ids_b_gei == ids_c_gei
        print(
            "PASSED: cold-cold-warm runs produce identical window / "
            "global_event_index state"
        )


# ---------------------------------------------------------------------------
# 5. metadata scan and compact build are distinguished
# ---------------------------------------------------------------------------

def test_metadata_scan_vs_compact_build_distinguished_in_telemetry():
    """On a cold run both phases must be > 0 and their sum must equal the
    total source-artifact load count."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, counts=(3, 2, 4))
        cfg = _cfg(root)
        _, _, _, full, _ = data_utils.load_all_datasets(cfg)
        tel = full._compact_index.telemetry

        # The metadata phase is in _scan_lazy_collections: it touches every
        # artifact to learn offsets and field metadata.
        # The compact-build phase is in _CompactIndex.build: it touches
        # every artifact to copy fields into the compact tensors / node
        # table.  Both must run on a cold start and they must sum to the
        # total.
        assert tel["metadata_source_load_count"] == 3 + 2 + 4
        assert tel["compact_build_source_load_count"] == 3 + 2 + 4
        assert tel["total_source_artifact_load_count"] == (
            tel["metadata_source_load_count"]
            + tel["compact_build_source_load_count"]
        )
        # source_artifact_scan_count is the back-compat alias that exposes
        # the compact-build phase only.
        assert tel["source_artifact_scan_count"] == tel["compact_build_source_load_count"]
        print(
            "PASSED: cold-run telemetry distinguishes metadata-scan "
            f"({tel['metadata_source_load_count']}) and compact-build "
            f"({tel['compact_build_source_load_count']}) phases; total "
            f"= {tel['total_source_artifact_load_count']}"
        )


def test_second_load_hit_keeps_compact_build_phase_at_zero():
    """On a sidecar cache hit the compact-build phase must not load any
    source artifact (the previous build is restored from disk).

    The metadata-scan phase is allowed to re-run because it is needed to
    learn per-window offsets and field metadata for BoundedFullData
    regardless of whether the sidecar is valid; the truthful answer is
    exposed via ``metadata_source_load_count``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, counts=(2, 1, 1))
        cfg = _cfg(root)
        _, _, _, full1, _ = data_utils.load_all_datasets(cfg)
        cold_meta = full1._compact_index.telemetry["metadata_source_load_count"]
        cold_build = full1._compact_index.telemetry["compact_build_source_load_count"]
        assert cold_meta > 0
        assert cold_build > 0

        _, _, _, full2, _ = data_utils.load_all_datasets(cfg)
        tel = full2._compact_index.telemetry
        assert tel["persistent_cache_hit"] is True
        # The compact-build phase is the one that costs the per-event copy;
        # the sidecar is exactly designed to skip it.
        assert tel["compact_build_source_load_count"] == 0
        # Total source load on a cache hit is exactly the metadata phase.
        assert (
            tel["total_source_artifact_load_count"]
            == tel["metadata_source_load_count"]
        )
        print(
            "PASSED: warm-hit run does compact-build=0 and only re-runs "
            "the metadata phase (which is required for BoundedFullData)"
        )


# ---------------------------------------------------------------------------
# 6. source-metadata fingerprint naming is accurate in the manifest
# ---------------------------------------------------------------------------

def test_sidecar_manifest_records_fingerprints_with_accurate_names():
    """The sidecar manifest must record both fingerprints under names that
    make the metadata-vs-content distinction clear."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, counts=(2, 1, 1))
        cfg = _cfg(root)
        _, _, _, _, _ = data_utils.load_all_datasets(cfg)
        sidecar = root / data_utils._CompactIndex.PERSISTENT_DIR_NAME
        manifest = _manifest(sidecar)
        assert "source_metadata_fingerprint_sha256" in manifest
        assert "semantic_config_fingerprint_sha256" in manifest
        # The legacy alias must NOT be present in a v3 manifest.
        assert "fingerprint_sha256" not in manifest
        # Both are 64 hex chars (SHA-256).
        for key in (
            "source_metadata_fingerprint_sha256",
            "semantic_config_fingerprint_sha256",
        ):
            value = manifest[key]
            assert isinstance(value, str) and len(value) == 64
            int(value, 16)
        print("PASSED: sidecar manifest records both fingerprints with the right names")
