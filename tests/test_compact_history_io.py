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
    do NOT trigger additional artifact loads.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _artifacts(root, n_windows=10, events_per_window=20)
        cfg = _cfg(root)

        # Load with compact index
        train_d, val_d, test_d, full_data, max_node = load_all_datasets(cfg)

        # Initial telemetry snapshot
        initial_tel = full_data._compact_index.telemetry
        initial_load_count = initial_tel["full_artifact_load_count"]
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
        final_load_count = final_tel["full_artifact_load_count"]
        final_compact_lookups = final_tel["compact_lookup_count"]

        # Verify: no additional artifact loads occurred
        assert final_load_count == initial_load_count, (
            f"IO AMPLIFICATION DETECTED: {final_load_count - initial_load_count} "
            f"additional artifact loads after 100 lookups"
        )

        # Verify: all lookups were served from compact index
        expected_lookups = 100 * 3  # 100 events × 3 fields
        actual_new_lookups = final_compact_lookups - initial_compact_lookups
        assert actual_new_lookups == expected_lookups, (
            f"Expected {expected_lookups} compact lookups, got {actual_new_lookups}"
        )

        print(f"PASSED: {expected_lookups} lookups, 0 additional artifact loads")
        print(f"  Initial load count: {initial_load_count}")
        print(f"  Final load count: {final_load_count}")
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
        assert "full_artifact_load_count" in tel
        assert "index_build_seconds" in tel
        assert "compact_bytes" in tel

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
        initial_loads = initial_tel["full_artifact_load_count"]

        # Lookup all cross-window events
        _ = full_data.get_event_values("t", cross_window_ids)

        final_tel = full_data._compact_index.telemetry
        final_loads = final_tel["full_artifact_load_count"]

        # No additional loads should have occurred
        assert final_loads == initial_loads, (
            f"Cross-window lookup triggered {final_loads - initial_loads} artifact loads"
        )

        print(f"PASSED: Cross-window lookup of {len(cross_window_ids)} events, 0 additional loads")


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
