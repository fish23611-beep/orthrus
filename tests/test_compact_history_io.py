"""
Microbenchmark for compact history index I/O amplification.

This test proves that repeated history lookups do NOT trigger
repeated full artifact loads after the compact index is built.
"""

from __future__ import annotations

import tempfile
import torch
from torch_geometric.data import TemporalData
from pathlib import Path
from types import SimpleNamespace

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_all_datasets


def _cfg(root, limit=None):
    encoder = SimpleNamespace(
        edge_features="edge_type,msg",
        use_node_type_in_node_feats=True,
    )
    training = SimpleNamespace(
        encoder=encoder,
        decoder=SimpleNamespace(used_methods="custom"),
    )
    return SimpleNamespace(
        _test_mode=False,
        _max_windows_per_split=limit,
        dataset=SimpleNamespace(
            name="synthetic",
            num_node_types=3,
            num_edge_types=4,
        ),
        dataset_view=SimpleNamespace(mode="host_network_full"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(root)),
            embed_nodes=SimpleNamespace(used_method="only_type"),
        ),
        detection=SimpleNamespace(gnn_training=training),
    )


def _artifacts(root, n_windows=10, events_per_window=20):
    """Create synthetic artifacts with known content."""
    for split_index, (split, count) in enumerate(zip(("train", "val", "test"), (n_windows, 2, 2))):
        directory = root / split
        directory.mkdir(parents=True)
        for window_index in range(count):
            base_idx = window_index * events_per_window
            src = torch.arange(base_idx, base_idx + events_per_window, dtype=torch.long)
            dst = src + 100
            t = torch.arange(events_per_window, dtype=torch.long) + window_index * 1000
            src_type = torch.nn.functional.one_hot(src.remainder(3), num_classes=3)
            dst_type = torch.nn.functional.one_hot(dst.remainder(3), num_classes=3)
            edge_type = torch.nn.functional.one_hot(
                torch.arange(events_per_window).remainder(4), num_classes=4
            )
            data = TemporalData(
                src=src,
                dst=dst,
                t=t,
                msg=torch.cat([src_type, edge_type, dst_type], dim=-1).float(),
            )
            torch.save(data, directory / f"window_{window_index:02d}.pt")


def test_repeated_lookup_no_io_amplification():
    """
    Prove that repeated history lookups after compact index build
    do NOT trigger additional full-window artifact loads.

    The compact index serves src/dst/t/edge_type/edge_type_index directly,
    so lookup amplification is bounded. Source-artifact scan during the
    initial build is one-time and tracked separately.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, n_windows=10, events_per_window=20)
        cfg = _cfg(root)

        # Load with compact index
        train_d, val_d, test_d, full_data, max_node = load_all_datasets(cfg)

        # Initial telemetry snapshot
        initial_tel = full_data._compact_index.telemetry
        initial_fallback = initial_tel["fallback_full_window_load_count"]
        initial_source_scans = initial_tel["source_artifact_scan_count"]
        initial_compact_lookups = initial_tel["compact_lookup_count"]

        # Simulate 100 repeated lookups across different windows
        # This simulates the pattern that caused the original thrashing
        event_ids = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

        for _ in range(10):  # 10 rounds × 10 events = 100 total lookups
            # Lookup edge_type (compact field)
            _ = full_data.get_event_values("edge_type", event_ids)
            # Lookup t (compact field)
            _ = full_data.get_event_values("t", event_ids)
            # Lookup src (compact field)
            _ = full_data.get_event_values("src", event_ids)

        # After all lookups, check telemetry
        final_tel = full_data._compact_index.telemetry
        final_fallback = final_tel["fallback_full_window_load_count"]
        final_source_scans = final_tel["source_artifact_scan_count"]
        final_compact_lookups = final_tel["compact_lookup_count"]

        # Verify: no additional artifact loads occurred (fallback remains 0)
        assert final_fallback == initial_fallback, (
            f"IO AMPLIFICATION DETECTED: {final_fallback - initial_fallback} "
            f"additional fallback loads after 100 lookups"
        )

        # Source scans happen ONLY during the initial build.
        assert final_source_scans == initial_source_scans, (
            "Source scan count changed after build (should be one-time)"
        )
        assert final_source_scans > 0, "Source scan count must be > 0 after build"

        # Verify: all lookups were served from compact index
        expected_lookups = 100 * 3  # 100 events × 3 fields
        actual_new_lookups = final_compact_lookups - initial_compact_lookups
        assert actual_new_lookups == expected_lookups, (
            f"Expected {expected_lookups} compact lookups, got {actual_new_lookups}"
        )

        print(f"PASSED: {expected_lookups} lookups, 0 fallback loads")
        print(f"  Source scans (one-time): {final_source_scans}")
        print(f"  Fallback loads (post-build): {final_fallback}")
        print(f"  Compact lookups: {final_compact_lookups}")


def test_history_lookup_telemetry():
    """Verify history access telemetry is correctly populated."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, n_windows=5, events_per_window=10)
        cfg = _cfg(root)

        train_d, val_d, test_d, full_data, max_node = load_all_datasets(cfg)

        # Verify telemetry structure
        tel = full_data._compact_index.telemetry
        assert "history_lookup_calls" in tel
        assert "history_lookup_events" in tel
        assert "field_lookup_counts" in tel
        assert "compact_lookup_count" in tel
        assert "node_lookup_count" in tel
        assert "msg_lookup_count" in tel
        assert "fallback_full_window_load_count" in tel
        assert "source_artifact_scan_count" in tel
        assert "index_build_seconds" in tel
        assert "compact_bytes" in tel
        assert "node_table_bytes" in tel

        # Initial state
        assert tel["history_lookup_calls"] == 0
        assert tel["compact_lookup_count"] == 0

        # Perform lookups
        event_ids = torch.arange(10)
        full_data.get_event_values("edge_type", event_ids)
        full_data.get_event_values("t", event_ids)
        full_data.get_event_values("src", event_ids)
        full_data.get_event_values("dst", event_ids)

        tel = full_data._compact_index.telemetry
        assert tel["history_lookup_calls"] == 4
        assert tel["history_lookup_events"] == 40
        assert tel["field_lookup_counts"]["edge_type"] == 10
        assert tel["field_lookup_counts"]["t"] == 10
        assert tel["field_lookup_counts"]["src"] == 10
        assert tel["field_lookup_counts"]["dst"] == 10

        print("PASSED: History access telemetry correctly populated")


def test_cross_window_lookup_efficiency():
    """Test that cross-window lookups are still efficient."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Create 10 windows, each with 20 events
        _artifacts(root, n_windows=10, events_per_window=20)
        cfg = _cfg(root)

        train_d, val_d, test_d, full_data, max_node = load_all_datasets(cfg)

        # Get events from different windows
        # Events 0-19: window 0
        # Events 20-39: window 1
        # etc.
        cross_window_ids = torch.tensor([0, 19, 20, 39, 40, 59, 80, 99, 100, 119, 140, 159, 180, 199])

        initial_tel = full_data._compact_index.telemetry
        initial_fallback = initial_tel["fallback_full_window_load_count"]

        # Lookup all cross-window events
        _ = full_data.get_event_values("t", cross_window_ids)

        final_tel = full_data._compact_index.telemetry
        final_fallback = final_tel["fallback_full_window_load_count"]

        # No additional loads should have occurred
        assert final_fallback == initial_fallback, (
            f"Cross-window lookup triggered {final_fallback - initial_fallback} fallback loads"
        )

        print(f"PASSED: Cross-window lookup of {len(cross_window_ids)} events, 0 fallback loads")


def test_persistent_sidecar_avoids_rescan_on_second_load():
    """
    Second load_all_datasets() on the same artifacts must skip the streaming
    source-artifact scan by restoring from the on-disk sidecar.
    """
    import json

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, n_windows=4, events_per_window=8)
        cfg = _cfg(root)

        # First load: must scan source artifacts and write a sidecar.
        _, _, _, full_data1, _ = load_all_datasets(cfg)
        first_tel = full_data1._compact_index.telemetry
        assert first_tel["source_artifact_scan_count"] > 0, (
            "First load must scan source artifacts"
        )

        # Confirm sidecar was written.
        sidecar_root = root / full_data1._compact_index.PERSISTENT_DIR_NAME
        assert sidecar_root.is_dir(), "Persistent sidecar directory must exist"
        manifest_files = list(sidecar_root.glob("manifest__*.json"))
        assert manifest_files, "Manifest must be written"
        with open(manifest_files[0], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest.get("completed") is True
        assert manifest.get("schema_version") == 3
        # v3 manifest records distinct source-metadata and semantic-config
        # fingerprints; both must be present for cache-hit validation.
        assert "source_metadata_fingerprint_sha256" in manifest
        assert "semantic_config_fingerprint_sha256" in manifest

        # Second load: sidecar must be a hit, no source scan.
        _, _, _, full_data2, _ = load_all_datasets(cfg)
        second_tel = full_data2._compact_index.telemetry
        second_scan_count = second_tel["source_artifact_scan_count"]
        assert second_scan_count == 0, (
            f"Sidecar cache hit was expected but source scan count = "
            f"{second_scan_count}"
        )
        assert second_tel["persistent_cache_hit"] is True
        assert second_tel["persistent_cache_hit_count"] == 1

        # Telemetry values must match across the two loads (no semantic drift).
        ci1 = full_data1._compact_index
        ci2 = full_data2._compact_index
        assert ci1._src.tolist() == ci2._src.tolist(), (
            "Persistent sidecar must produce an event-identical compact index"
        )

        print("PASSED: persistent sidecar skipped source scan on second load")


def test_persistent_sidecar_invalidates_when_source_changes():
    """
    Touching a source artifact must invalidate the sidecar and force a rebuild.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, n_windows=3, events_per_window=6)
        cfg = _cfg(root)

        # First load — warm the sidecar.
        _, _, _, full_data1, _ = load_all_datasets(cfg)

        # Mutate one source window's mtime.
        train_window = next((root / "train").glob("window_*.pt"))
        import os
        new_mtime = os.stat(train_window).st_mtime_ns + 10_000_000
        os.utime(train_window, ns=(new_mtime, new_mtime))

        # Second load — must miss cache, do a fresh scan.
        _, _, _, full_data2, _ = load_all_datasets(cfg)
        tel2 = full_data2._compact_index.telemetry
        assert tel2["persistent_cache_hit"] is False
        assert tel2["source_artifact_scan_count"] > 0, (
            "Source mtime change must invalidate sidecar and trigger a rescan"
        )
        print("PASSED: persistent sidecar invalidated after source mtime change")


if __name__ == "__main__":
    print("=" * 60)
    print("Compact History Index I/O Amplification Benchmark")
    print("=" * 60)

    print("\n1. Test: Repeated lookup no IO amplification")
    test_repeated_lookup_no_io_amplification()

    print("\n2. Test: History lookup telemetry")
    test_history_lookup_telemetry()

    print("\n3. Test: Cross-window lookup efficiency")
    test_cross_window_lookup_efficiency()

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS PASSED")
    print("=" * 60)
