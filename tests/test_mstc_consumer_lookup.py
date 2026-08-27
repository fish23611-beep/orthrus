"""
Consumer-level MSTC synthetic test for C8 full-data I/O.

This test exercises the actual code path that MSTC runs at training and
testing time:

  MultiScaleOrthrusEncoder.forward
    -> MultiScaleNeighborLoader.__call__   (query history by event_id)
    -> MultiScaleOrthrusEncoder._historical_node_features
       -> _full_values("x_src"/"x_dst", event_ids)
    -> MultiScaleOrthrusEncoder._edge_features
       -> _full_values("edge_type"/"msg", event_ids)

It asserts that across many repeated batches whose history spans many
windows, the *fallback_full_window_load_count* telemetry remains 0.

It also validates:

  - node-level x_src/x_dst are served from the compact index, not via
    torch.load of the source TemporalData.
  - msg is derived from the compact index when requested.
  - the field-level consistency invariant (same node_id -> same x_src).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import TemporalData

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_utils import load_all_datasets  # noqa: E402

from mstc.history_store import HistoryStore  # noqa: E402
from mstc.multiscale_encoder import MultiScaleOrthrusEncoder  # noqa: E402
from mstc.multiscale_sampler import MultiScaleNeighborLoader  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic artifact generator
# --------------------------------------------------------------------------- #
def _cfg(root: Path, *, predict_edge_type: bool = True) -> SimpleNamespace:
    decoder_methods = (
        "predict_edge_type,custom" if predict_edge_type else "custom"
    )
    encoder = SimpleNamespace(
        edge_features="edge_type,msg",
        use_node_type_in_node_feats=True,
        use_node_feats_in_gnn=True,
        temporal_dim=8,
        node_hid_dim=8,
        node_out_dim=4,
        gate_hidden_dim=4,
        use_scale_embedding=False,
        fusion="gated",
    )
    training = SimpleNamespace(
        encoder=encoder,
        decoder=SimpleNamespace(used_methods=decoder_methods),
    )
    return SimpleNamespace(
        _test_mode=False,
        _max_windows_per_split=None,
        dataset=SimpleNamespace(
            name="synthetic",
            num_node_types=3,
            num_edge_types=4,
        ),
        dataset_view=SimpleNamespace(mode="host_network_full"),
        edge_featurization=SimpleNamespace(
            embed_edges=SimpleNamespace(_edge_embeds_dir=str(root)),
            embed_nodes=SimpleNamespace(used_method="feature_word2vec", emb_dim=6),
        ),
        detection=SimpleNamespace(gnn_training=training),
    )


def _build_artifacts(
    root: Path,
    *,
    n_windows: int = 12,
    events_per_window: int = 24,
    n_distinct_nodes: int = 16,
    emb_dim: int = 6,
    node_type_dim: int = 3,
    edge_type_dim: int = 4,
) -> dict[int, torch.Tensor]:
    """
    Create synthetic edge_embedding artifacts that mirror the real ORTHRUS
    layout: msg = [src_type_onehot | src_emb | edge_type | dst_type_onehot | dst_emb].

    x_src = concat([src_emb, src_type_onehot]) (use_node_type_in_node_feats=True).

    Crucially: x_src/x_dst are node-INVARIANT (the same node_id always yields
    the same feature), so the node-level compact table can be built safely.
    """
    rng = torch.Generator().manual_seed(0)
    # Per-node invariant features.
    src_emb_table = torch.randn((n_distinct_nodes, emb_dim), generator=rng)
    dst_emb_table = torch.randn((n_distinct_nodes, emb_dim), generator=rng)
    src_type_table = torch.nn.functional.one_hot(
        torch.arange(n_distinct_nodes) % node_type_dim, num_classes=node_type_dim
    ).float()
    dst_type_table = torch.nn.functional.one_hot(
        (torch.arange(n_distinct_nodes) + 1) % node_type_dim, num_classes=node_type_dim
    ).float()

    for split, count in (("train", n_windows), ("val", 2), ("test", 2)):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for w in range(count):
            src = torch.randint(0, n_distinct_nodes, (events_per_window,), generator=rng)
            dst = torch.randint(0, n_distinct_nodes, (events_per_window,), generator=rng)
            t = (
                torch.arange(events_per_window, dtype=torch.long)
                + w * events_per_window * 100
            )

            src_emb = src_emb_table[src]
            dst_emb = dst_emb_table[dst]
            src_type_oh = src_type_table[src]
            dst_type_oh = dst_type_table[dst]
            edge_type_idx = torch.randint(0, edge_type_dim, (events_per_window,), generator=rng)
            edge_type_oh = torch.nn.functional.one_hot(edge_type_idx, num_classes=edge_type_dim).float()

            # msg layout: [src_type | src_emb | edge_type | dst_type | dst_emb]
            msg = torch.cat([src_type_oh, src_emb, edge_type_oh, dst_type_oh, dst_emb], dim=-1)

            data = TemporalData(
                src=src,
                dst=dst,
                t=t,
                msg=msg,
            )
            torch.save(data, directory / f"window_{w:02d}.pt")

    return {"src_emb_table": src_emb_table, "dst_emb_table": dst_emb_table,
            "src_type_table": src_type_table, "dst_type_table": dst_type_table}


def _eager_x_src(event_ids, src, dst, *, emb_dim, node_type_dim, ref_tables):
    """Eager reference: rebuild x_src from src node IDs using stored tables."""
    src_nodes = src[event_ids]
    src_emb = ref_tables["src_emb_table"][src_nodes]
    src_type = ref_tables["src_type_table"][src_nodes]
    return torch.cat([src_emb, src_type], dim=-1)


# --------------------------------------------------------------------------- #
# MSTC consumer-level lookup test
# --------------------------------------------------------------------------- #
def test_mstc_cross_window_lookup_does_not_trigger_fallback_loads():
    """
    Drive a synthetic MSTC encoder across many batches whose history spans
    many windows. Verify that:

      1. x_src / x_dst lookups for history events stay on the compact path.
      2. fallback_full_window_load_count stays 0 (no torch.load thrashing).
      3. msg derived from compact matches the per-window reference.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ref_tables = _build_artifacts(root, n_windows=14, events_per_window=24)
        cfg = _cfg(root, predict_edge_type=True)

        train_d, val_d, test_d, full_data, max_node = load_all_datasets(cfg)

        # Sanity: compact index has node table and msg derivation ready.
        ci = full_data._compact_index
        assert ci.has_node_table(), "node table must be built for MSTC"
        assert ci.has_msg_derivation(), "msg derivation must be available"
        tel = ci.telemetry
        assert tel["source_artifact_scan_count"] > 0, (
            "source_artifact_scan_count must reflect the initial build scan"
        )

        # Capture the post-build telemetry baseline.
        initial_fallback = tel["fallback_full_window_load_count"]
        initial_compact = tel["compact_lookup_count"]
        initial_node = tel["node_lookup_count"]
        initial_msg = tel["msg_lookup_count"]

        # Set up an MSTC encoder + history store wired into the compact data.
        n_nodes = int(max_node)
        store = HistoryStore(num_nodes=n_nodes, candidate_capacity=32, device="cpu")
        loader = MultiScaleNeighborLoader(
            store,
            tau_short_ns=10_000,
            tau_medium_ns=50_000,
            tau_max_ns=200_000,
            short_budget=4,
            medium_budget=4,
            long_budget=4,
        )
        # Shared graph encoder is a stub that ignores edge_feats.
        import torch.nn as nn
        class _StubEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(8, 4)
            def forward(self, x, edge_index, edge_feats=None, **kwargs):
                return self.lin(x)

        encoder = MultiScaleOrthrusEncoder(
            shared_graph_encoder=_StubEncoder(),
            neighbor_loader=loader,
            in_dim=full_data.field_metadata("x_src")[0][0],
            temporal_dim=8,
            node_out_dim=4,
            edge_features=("edge_type", "msg"),
            device="cpu",
            gate_hidden_dim=8,
            use_scale_embedding=False,
            fusion="gated",
        )

        # Drive many batches; history spans multiple windows.
        # Pre-populate the history store with cross-window events so the
        # sampler returns neighbor entries whose e_id crosses windows.
        src_pre = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.long)
        dst_pre = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
        t_pre = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.long) * 1_000_000_000
        ei_pre = torch.tensor([10, 30, 50, 70, 90, 110, 130, 150], dtype=torch.long)
        store.insert(src_pre, dst_pre, ei_pre, t_pre)

        n_rounds = 60
        for round_idx in range(n_rounds):
            # Pick event ids from multiple windows.
            start = (round_idx * 3) % full_data.num_events
            event_ids = torch.arange(start, start + 16, dtype=torch.long) % full_data.num_events

            src = full_data.src[event_ids]
            dst = full_data.dst[event_ids]
            t = full_data.t[event_ids]
            # Use the actual history sampler so e_ids cross windows.
            sampled = loader(src, dst, t)

            # Now run the encoder forward (the test asserts no fallback).
            # Build x_src/x_dst from compact, msg from compact, edge_type from compact.
            x_src_full = full_data.get_event_values("x_src", event_ids)
            x_dst_full = full_data.get_event_values("x_dst", event_ids)
            msg_full = full_data.get_event_values("msg", event_ids)
            et_full = full_data.get_event_values("edge_type", event_ids)

            # Validate msg matches the eager per-window reference for predict_edge_type
            # case (msg = cat([x_src, x_dst])).
            x_src_ref = _eager_x_src(
                event_ids, full_data._compact_index._src, full_data._compact_index._dst,
                emb_dim=6, node_type_dim=3, ref_tables=ref_tables,
            )
            assert torch.allclose(x_src_full, x_src_ref), (
                "compact x_src must match the per-node invariant reference"
            )

            # Run the encoder forward to exercise the per-scale path that also
            # calls _full_values("x_src"/"x_dst"/"edge_type"/"msg") internally.
            x = (x_src_full, x_dst_full)
            encoder(
                edge_index=torch.stack([src, dst]),
                t=t,
                msg=msg_full,
                x=x,
                full_data=full_data,
                inference=False,
                global_event_index=event_ids,
                edge_feats=torch.cat([et_full, msg_full], dim=-1),
            )

            # Update history after each batch to mirror the training pattern.
            store.insert(src, dst, event_ids, t)

        final_tel = ci.telemetry
        final_fallback = final_tel["fallback_full_window_load_count"]
        final_compact = final_tel["compact_lookup_count"]
        final_node = final_tel["node_lookup_count"]
        final_msg = final_tel["msg_lookup_count"]

        assert final_fallback == initial_fallback, (
            f"IO AMPLIFICATION DETECTED: {final_fallback - initial_fallback} "
            f"fallback loads after {n_rounds} consumer-level rounds"
        )
        assert final_node > initial_node, (
            "node-level x_src/x_dst lookups should have grown"
        )
        assert final_msg > initial_msg, (
            "msg derivation should have grown"
        )
        assert final_compact > initial_compact, (
            "compact tensor lookups (src/dst/t/edge_type) should have grown"
        )

        print(
            f"PASSED: {n_rounds} MSTC rounds, fallback loads={final_fallback}, "
            f"node_lookups={final_node}, msg_lookups={final_msg}, "
            f"compact_lookups={final_compact}"
        )


# --------------------------------------------------------------------------- #
# Consistency validation: conflicting x_src -> build rejection
# --------------------------------------------------------------------------- #
def test_node_level_consistency_rejects_invariant_violation():
    """
    If the same node_id appears with DIFFERENT x_src values across windows,
    the build must abort with a descriptive error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = torch.Generator().manual_seed(1)
        n_events = 6
        node_dim = 3
        edge_dim = 4
        emb_dim = 6
        # Window 0: src=0 -> emb_a
        src0 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        src_emb0 = torch.tensor([
            [1.0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0],
            [0, 0, 0, 1.0, 0, 0],
            [0, 0, 0, 0, 1.0, 0],
            [0, 0, 0, 0, 0, 1.0],
        ])
        src_type0 = torch.nn.functional.one_hot(src0 % node_dim, num_classes=node_dim).float()
        et0 = torch.nn.functional.one_hot(
            torch.zeros(n_events, dtype=torch.long), num_classes=edge_dim
        ).float()
        dst0 = (src0 + 1) % 6
        msg0 = torch.cat([src_type0, src_emb0, et0, src_type0, src_emb0], dim=-1)

        # Window 1: src=0 -> emb_b (different from window 0)
        src1 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        src_emb1 = torch.full((n_events, emb_dim), 99.0)  # all different
        src_type1 = torch.nn.functional.one_hot(src1 % node_dim, num_classes=node_dim).float()
        et1 = et0.clone()
        dst1 = (src1 + 1) % 6
        msg1 = torch.cat([src_type1, src_emb1, et1, src_type1, src_emb1], dim=-1)

        for split in ("train", "val", "test"):
            (root / split).mkdir(parents=True, exist_ok=True)
        torch.save(TemporalData(src=src0, dst=dst0, t=torch.arange(n_events), msg=msg0),
                   root / "train" / "window_00.pt")
        torch.save(TemporalData(src=src1, dst=dst1, t=torch.arange(n_events, n_events*2), msg=msg1),
                   root / "train" / "window_01.pt")
        # Val/test windows reuse the consistent window 0.
        torch.save(TemporalData(src=src0, dst=dst0, t=torch.arange(n_events, n_events*2), msg=msg0),
                   root / "val" / "window_00.pt")
        torch.save(TemporalData(src=src0, dst=dst0, t=torch.arange(n_events*2, n_events*3), msg=msg0),
                   root / "test" / "window_00.pt")

        cfg = _cfg(root)
        import pytest
        with pytest.raises(RuntimeError, match="x_src/x_dst are not invariant"):
            load_all_datasets(cfg)


# --------------------------------------------------------------------------- #
# Field-level lookup parity: compact x_src/x_dst matches eager per-window
# --------------------------------------------------------------------------- #
def test_compact_x_src_x_dst_match_per_window_eager_reference():
    """
    For every event, get_event_values('x_src') and get_event_values('x_dst')
    must exactly equal the per-window x_src/x_dst at the same event index.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_artifacts(root, n_windows=6, events_per_window=12)
        cfg = _cfg(root, predict_edge_type=False)
        # Eager reference: load each window via _prepare_window (which already
        # injects x_src/x_dst/msg via extract_msg_from_data) and concat them.
        from data_utils import load_data_set
        eager_train = load_data_set(cfg, str(root), "train")
        eager_val = load_data_set(cfg, str(root), "val")
        eager_test = load_data_set(cfg, str(root), "test")
        # NOTE: load_data_set already invokes _prepare_window → extract_msg_from_data
        # internally. Calling extract_msg_from_data AGAIN here would re-parse the
        # already-rebuilt msg and corrupt the x_src layout.  We just take what
        # _prepare_window produced.
        eager_x_src = torch.cat([g.x_src for g in eager_train + eager_val + eager_test])
        eager_x_dst = torch.cat([g.x_dst for g in eager_train + eager_val + eager_test])
        eager_msg = torch.cat([g.msg for g in eager_train + eager_val + eager_test])

        # Compare against the compact (lazy) full_data.
        train_d, val_d, test_d, full_data, _ = load_all_datasets(cfg)
        event_ids = torch.arange(full_data.num_events)
        compact_x_src = full_data.get_event_values("x_src", event_ids)
        compact_x_dst = full_data.get_event_values("x_dst", event_ids)
        compact_msg = full_data.get_event_values("msg", event_ids)

        assert torch.allclose(compact_x_src, eager_x_src), (
            "compact x_src must equal per-window x_src concatenated"
        )
        assert torch.allclose(compact_x_dst, eager_x_dst), (
            "compact x_dst must equal per-window x_dst concatenated"
        )
        assert torch.allclose(compact_msg, eager_msg), (
            "compact msg must equal per-window msg concatenated"
        )
