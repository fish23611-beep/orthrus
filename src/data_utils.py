from __future__ import annotations

import bisect
import hashlib
import json
import os
import resource
import sys
import time as time_module
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pickle
import warnings
import torch
from torch_geometric.data import Data, TemporalData
from torch_geometric.loader import TemporalDataLoader

from encoders import OrthrusEncoder
from serialization_compat import load_trusted_torch_artifact


# --------------------------------------------------------------------------- #
# Split name <-> index mapping (centralised — defined in one place only)
# --------------------------------------------------------------------------- #
SPLIT_NAME_TO_INDEX = {"train": 0, "val": 1, "test": 2}
INDEX_TO_SPLIT_NAME = {v: k for k, v in SPLIT_NAME_TO_INDEX.items()}



def _current_rss_mb() -> float | None:
    """Return current resident memory without adding a psutil dependency."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        try:
            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
        except Exception:
            return None


class _LoaderTelemetry:
    """Small phase recorder used by the loader and persisted in runtime.json."""

    def __init__(self) -> None:
        self.phases: dict[str, float | None] = {}
        self.peak_rss_mb: float | None = None

    def observe(self, phase: str, *, emit: bool = True) -> None:
        rss = _current_rss_mb()
        self.phases[phase] = rss
        if rss is not None:
            self.peak_rss_mb = rss if self.peak_rss_mb is None else max(self.peak_rss_mb, rss)
        if emit:
            suffix = "unavailable" if rss is None else f"{rss:.2f} MB"
            print(f"[Dataset loader RSS] {phase}: {suffix}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "path_backed_bounded_memory",
            "rss_mb_by_phase": dict(self.phases),
            "dataset_loader_peak_rss_mb": self.peak_rss_mb,
        }


# --------------------------------------------------------------------------- #
# Compact History Index: eliminates per-batch full artifact reloads
# --------------------------------------------------------------------------- #

class _HistoryAccessTelemetry:
    """Application-level telemetry for history random access.

    Distinguishes:
      - source_artifact_scan_count / source_artifact_scan_bytes_estimate:
        artifacts touched during compact-index construction (one-time per run).
      - fallback_full_window_load_count / fallback_full_window_load_bytes_estimate:
        full TemporalData loads triggered by training/testing lookups AFTER
        index build. This is the only counter that should grow during a run.
      - compact_lookup_count / node_lookup_count / msg_lookup_count:
        random-access lookups served entirely from the in-memory index.
    """

    def __init__(self) -> None:
        # Lookup counters
        self.history_lookup_calls: int = 0
        self.history_lookup_events: int = 0
        self.field_lookup_counts: dict[str, int] = {}
        # Compact tensor lookups (event-level fields served from global tensors)
        self.compact_lookup_count: int = 0
        # Node-level lookups (x_src/x_dst served from per-node table)
        self.node_lookup_count: int = 0
        # msg lookups served by derivation from node table + compact edge_type
        self.msg_lookup_count: int = 0
        # Fallback loads: full TemporalData torch.load after index build
        self.fallback_full_window_load_count: int = 0
        self.fallback_full_window_load_bytes_estimate: int = 0
        # Source-artifact scan counters (one-time index build)
        self.source_artifact_scan_count: int = 0
        self.source_artifact_scan_bytes_estimate: int = 0
        # Index-build metadata
        self.index_build_count: int = 0
        self.index_build_seconds: float = 0.0
        self.index_rss_mb: float | None = None
        self.compact_bytes: int = 0
        self.node_table_bytes: int = 0
        # Persistent sidecar cache (the on-disk fast path that lets us skip
        # the streaming source-artifact scan when the source artifacts and
        # build config have not changed).
        self.persistent_cache_hit: bool = False
        self.persistent_cache_hit_count: int = 0
        # Backwards-compat alias kept for the v1 telemetry contract.
        self.full_artifact_load_count: int = 0
        self.full_artifact_load_bytes_estimate: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": "compact_history_random_access_v2",
            "history_lookup_calls": self.history_lookup_calls,
            "history_lookup_events": self.history_lookup_events,
            "field_lookup_counts": dict(self.field_lookup_counts),
            "compact_lookup_count": self.compact_lookup_count,
            "node_lookup_count": self.node_lookup_count,
            "msg_lookup_count": self.msg_lookup_count,
            "compact_lookup_events": self.compact_lookup_count + self.node_lookup_count + self.msg_lookup_count,
            "fallback_full_window_load_count": self.fallback_full_window_load_count,
            "fallback_full_window_load_bytes_estimate": self.fallback_full_window_load_bytes_estimate,
            "source_artifact_scan_count": self.source_artifact_scan_count,
            "source_artifact_scan_bytes_estimate": self.source_artifact_scan_bytes_estimate,
            "index_build_count": self.index_build_count,
            "index_build_seconds": round(self.index_build_seconds, 3),
            "index_rss_mb": self.index_rss_mb,
            "compact_bytes": self.compact_bytes,
            "node_table_bytes": self.node_table_bytes,
            "persistent_cache_hit": self.persistent_cache_hit,
            "persistent_cache_hit_count": self.persistent_cache_hit_count,
            # Backwards-compat keys
            "full_artifact_load_count": self.fallback_full_window_load_count,
            "full_artifact_load_bytes_estimate": self.fallback_full_window_load_bytes_estimate,
        }


class _CompactIndex:
    """
    Bounded-memory compact index for efficient per-event field lookups.

    During BoundedFullData init, extracts frequently-accessed small fields into
    compact global tensors. Per-node semantic features (x_src/x_dst) are stored
    in a node-level table once each (with strict consistency validation).
    msg is derived on demand from x_src / x_dst / edge_type without loading
    the source TemporalData.

    This eliminates the O(unique_windows_per_batch × full_artifact_size)
    I/O amplification that occurred with cache_windows=1 and random
    history e_id access across many windows.

    Memory budget (THEIA_E3 ~1M nodes, ~430 windows):
        - src/dst (int64):   2 × 8 × 10M ≈ 160 MB
        - t (int64):         8 × 10M ≈ 80 MB
        - edge_type_index (int32): 4 × 10M ≈ 40 MB
        - src_type/dst_type: 2 × 4 × 10M ≈ 80 MB
        - split (int8):      10M ≈ 10 MB
        - node table (float32, 131 dim): 1M × 4 × 131 ≈ 524 MB
        - TOTAL compact:     ~900 MB (bounded, not per-window)
    """

    __slots__ = (
        "_specs", "_window_paths", "_total_events",
        "_src", "_dst", "_t",
        "_src_type", "_dst_type",
        "_edge_type_index", "_edge_type_num_types",
        "_split",
        # Node-level semantic feature table
        "_node_table_x_src", "_node_table_x_dst",
        "_node_table_dim", "_node_table_max_node",
        # msg derivation config
        "_msg_predict_edge_type", "_msg_use_edge_type",
        "_msg_dim",
        "_telemetry",
    )

    # Event-level fields served from compact global tensors (O(1) per event)
    COMPACT_FIELDS = frozenset({
        "src", "dst", "t", "src_type", "dst_type",
        "edge_type_index",
    })

    # Fields that map directly to global tensors or node-level tables.
    # Used by get_event_values routing.
    _COMPACT_INDEXED = frozenset({
        "src", "dst", "t", "src_type", "dst_type",
        "edge_type_index", "edge_type",
        "global_event_index", "split",
        "x_src", "x_dst",
        "msg",
    })

    def __init__(
        self,
        specs: Sequence["_WindowSpec"],
        total_events: int,
    ) -> None:
        self._specs = tuple(specs)
        self._window_paths: dict[int, str] = {}
        self._total_events = total_events

        # Compact tensors (all CPU, int/float)
        self._src = torch.empty(total_events, dtype=torch.int64)
        self._dst = torch.empty(total_events, dtype=torch.int64)
        self._t = torch.empty(total_events, dtype=torch.int64)
        self._src_type = torch.empty(total_events, dtype=torch.int32)
        self._dst_type = torch.empty(total_events, dtype=torch.int32)
        self._edge_type_index = torch.empty(total_events, dtype=torch.int32)
        self._edge_type_num_types = 0
        self._split = torch.empty(total_events, dtype=torch.int8)

        # Node-level semantic feature table. Allocated lazily by build().
        self._node_table_x_src: torch.Tensor | None = None
        self._node_table_x_dst: torch.Tensor | None = None
        self._node_table_dim: int = -1
        self._node_table_max_node: int = -1

        # msg reconstruction policy.
        self._msg_predict_edge_type: bool = False
        self._msg_use_edge_type: bool = True
        self._msg_dim: int = -1

        # Telemetry
        self._telemetry = _HistoryAccessTelemetry()

    @property
    def telemetry(self) -> dict[str, Any]:
        return self._telemetry.as_dict()

    @property
    def num_events(self) -> int:
        return self._total_events

    @property
    def node_table_dim(self) -> int:
        return self._node_table_dim

    def has_node_table(self) -> bool:
        return self._node_table_x_src is not None and self._node_table_x_dst is not None

    def has_msg_derivation(self) -> bool:
        return (
            self.has_node_table()
            and self._edge_type_num_types > 0
            and self._msg_dim > 0
        )

    # ------------------------------------------------------------------
    # Persistent sidecar (optional, safe by default)
    # ------------------------------------------------------------------
    # Sidecar layout (next to ``edge_embeds_dir``):
    #   _compact_index_v2/
    #     manifest.json    {schema_version, dataset, view_mode, fingerprint,
    #                       total_events, edge_type_num_types, msg_dim,
    #                       node_table_dim, node_table_max_node,
    #                       msg_predict_edge_type, completed=true}
    #     compact.pt       {src, dst, t, src_type, dst_type,
    #                       edge_type_index, split}
    #     nodes.pt         {x_src, x_dst}
    #
    # Fingerprint = sha256 of a sorted list of (relpath, size, mtime_ns)
    # tuples from every source window referenced by the specs.
    # Stale/invalid sidecars are silently rebuilt.
    # ------------------------------------------------------------------
    PERSISTENT_SCHEMA_VERSION = 2
    PERSISTENT_DIR_NAME = "_compact_index_v2"

    def _persistent_root(self, cfg: Any) -> str | None:
        """Return the sidecar directory for ``cfg``, or None to disable."""
        try:
            base = cfg.edge_featurization.embed_edges._edge_embeds_dir
        except AttributeError:
            return None
        if not base:
            return None
        return os.path.join(str(base), self.PERSISTENT_DIR_NAME)

    def _manifest_path(self, persistent_root: str, dataset_name: str,
                       view_mode: str | None) -> str:
        safe_dataset = (dataset_name or "default").replace(os.sep, "_")
        safe_view = (view_mode or "default").replace(os.sep, "_")
        return os.path.join(
            persistent_root, f"manifest__{safe_dataset}__{safe_view}.json"
        )

    def _compute_fingerprint(self) -> dict[str, Any]:
        """Return a JSON-serialisable fingerprint for current specs."""
        items: list[tuple[str, int, int]] = []
        for spec in self._specs:
            path = spec.path
            try:
                st = os.stat(path)
                items.append((os.path.basename(path), int(st.st_size),
                              int(st.st_mtime_ns)))
            except OSError:
                # Missing files make the fingerprint empty; any pre-existing
                # sidecar will be invalidated.
                return {"items": [], "sha256": "missing-source"}
        items.sort()
        h = hashlib.sha256()
        for name, size, mtime in items:
            h.update(name.encode())
            h.update(str(size).encode())
            h.update(str(mtime).encode())
            h.update(b"|")
        return {"items": items, "sha256": h.hexdigest()}

    def _try_load_persistent(self, cfg: Any, view_mode: str | None) -> bool:
        """
        Attempt to restore the compact index from the persistent sidecar.

        Returns True if the sidecar was successfully loaded (matches schema,
        dataset name, view_mode and fingerprint). On miss/mismatch the index
        is left untouched and ``build()`` will perform a full rebuild.
        """
        root = self._persistent_root(cfg)
        if root is None:
            return False
        try:
            dataset_name = getattr(getattr(cfg, "dataset", None), "name", "")
        except AttributeError:
            dataset_name = ""
        manifest_path = self._manifest_path(root, dataset_name, view_mode)
        if not os.path.isfile(manifest_path):
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            blob_path = os.path.join(root, "compact.pt")
            nodes_path = os.path.join(root, "nodes.pt")
            if not (os.path.isfile(blob_path) and os.path.isfile(nodes_path)):
                return False
            # Validate schema & dataset identity.
            if (manifest.get("schema_version") != self.PERSISTENT_SCHEMA_VERSION
                    or manifest.get("dataset") != dataset_name
                    or manifest.get("view_mode") != (view_mode or "")):
                return False
            current_fp = self._compute_fingerprint()
            if manifest.get("fingerprint_sha256") != current_fp["sha256"]:
                return False
            if not manifest.get("completed", False):
                return False
            # Validate stored dimensions match the current total_events.
            if int(manifest.get("total_events", -1)) != int(self._total_events):
                return False
            # Load the index tensors. torch.load keeps things simple; sources
            # are trusted because we own the sidecar directory.
            blob = torch.load(blob_path, map_location="cpu", weights_only=True)
            nodes = torch.load(nodes_path, map_location="cpu", weights_only=True)
            self._src = blob["src"]
            self._dst = blob["dst"]
            self._t = blob["t"]
            self._src_type = blob["src_type"]
            self._dst_type = blob["dst_type"]
            self._edge_type_index = blob["edge_type_index"]
            self._split = blob["split"]
            self._edge_type_num_types = int(manifest.get("edge_type_num_types", 0))
            self._node_table_x_src = nodes["x_src"]
            self._node_table_x_dst = nodes["x_dst"]
            self._node_table_dim = int(manifest.get("node_table_dim", -1))
            self._node_table_max_node = int(manifest.get("node_table_max_node", -1))
            self._msg_dim = int(manifest.get("msg_dim", -1))
            self._msg_predict_edge_type = bool(
                manifest.get("msg_predict_edge_type", False)
            )
            self._msg_use_edge_type = not self._msg_predict_edge_type
            self._telemetry.index_build_count += 1
            self._telemetry.source_artifact_scan_count = 0
            self._telemetry.source_artifact_scan_bytes_estimate = 0
            self._telemetry.persistent_cache_hit_count = (
                self._telemetry.persistent_cache_hit_count + 1
            )
            self._telemetry.persistent_cache_hit = True
            return True
        except (OSError, ValueError, KeyError, RuntimeError):
            # Any failure means "sidecar unusable"; rebuild from source.
            return False

    def _write_persistent(self, cfg: Any, view_mode: str | None) -> None:
        """
        Atomically write the freshly-built compact index to disk.

        The build is staged in ``<root>/incomplete-<uuid>`` and renamed into
        place via ``os.replace``, which is atomic on POSIX. An interrupted
        build leaves only ``incomplete-*`` artifacts that are ignored on the
        next startup.
        """
        root = self._persistent_root(cfg)
        if root is None:
            return
        try:
            dataset_name = getattr(getattr(cfg, "dataset", None), "name", "")
        except AttributeError:
            dataset_name = ""
        os.makedirs(root, exist_ok=True)
        # Stage the write into a per-process temp dir so a crashed build does
        # not corrupt the canonical manifest/compact.pt/nodes.pt.
        stage = f".build-{os.getpid()}-{int(time_module.time()*1000)}"
        stage_dir = os.path.join(root, stage)
        manifest_path = self._manifest_path(root, dataset_name, view_mode)
        # If a previous canonical manifest exists, move it aside so the new
        # build cannot atomically clobber a working sidecar until the temp
        # write succeeds.
        backup_manifest = manifest_path + ".prev"
        if os.path.isfile(manifest_path):
            try:
                os.replace(manifest_path, backup_manifest)
            except OSError:
                backup_manifest = None  # type: ignore[assignment]
        try:
            os.makedirs(stage_dir, exist_ok=False)
            blob = {
                "src": self._src.cpu(),
                "dst": self._dst.cpu(),
                "t": self._t.cpu(),
                "src_type": self._src_type.cpu(),
                "dst_type": self._dst_type.cpu(),
                "edge_type_index": self._edge_type_index.cpu(),
                "split": self._split.cpu(),
            }
            nodes = {
                "x_src": self._node_table_x_src.cpu()
                if self._node_table_x_src is not None else torch.empty(0, 0),
                "x_dst": self._node_table_x_dst.cpu()
                if self._node_table_x_dst is not None else torch.empty(0, 0),
            }
            torch.save(blob, os.path.join(stage_dir, "compact.pt"))
            torch.save(nodes, os.path.join(stage_dir, "nodes.pt"))
            fp = self._compute_fingerprint()
            manifest = {
                "schema_version": self.PERSISTENT_SCHEMA_VERSION,
                "dataset": dataset_name,
                "view_mode": view_mode or "",
                "fingerprint_sha256": fp["sha256"],
                "total_events": int(self._total_events),
                "edge_type_num_types": int(self._edge_type_num_types),
                "node_table_dim": int(self._node_table_dim),
                "node_table_max_node": int(self._node_table_max_node),
                "msg_dim": int(self._msg_dim),
                "msg_predict_edge_type": bool(self._msg_predict_edge_type),
                "completed": True,
            }
            manifest_filename = os.path.basename(manifest_path)
            with open(os.path.join(stage_dir, manifest_filename), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, sort_keys=True)
            # Atomic rename of staged files into the canonical names.
            os.replace(
                os.path.join(stage_dir, manifest_filename),
                manifest_path,
            )
            os.replace(
                os.path.join(stage_dir, "compact.pt"),
                os.path.join(root, "compact.pt"),
            )
            os.replace(
                os.path.join(stage_dir, "nodes.pt"),
                os.path.join(root, "nodes.pt"),
            )
        except OSError:
            # Disk-full / permission-denied: silently skip the sidecar.
            pass
        finally:
            # Clean up stage + backup.
            try:
                import shutil
                if os.path.isdir(stage_dir):
                    shutil.rmtree(stage_dir, ignore_errors=True)
            except Exception:
                pass
            if backup_manifest and os.path.isfile(backup_manifest):
                # If the new write succeeded, the previous manifest is no
                # longer authoritative; remove it.
                try:
                    os.remove(backup_manifest)
                except OSError:
                    pass

    def build(self, cfg: Any, view_mode: str | None) -> None:
        """
        Populate compact tensors AND the node-level x_src/x_dst table by scanning
        each window exactly once.

        Uses the same _prepare_window path as the main loader to ensure
        identical field semantics (same featurization, same view_mode).

        Consistency validation: every appearance of a given node_id must yield
        an identical x_src / x_dst. On the first mismatch the index is rejected
        and the build aborts with a descriptive error.

        If the persistent sidecar matches the current ``schema_version``,
        dataset name, view_mode and source-artifact fingerprint, the build
        is restored from disk without re-scanning the source artifacts.
        Otherwise the index is rebuilt from source and the new index is
        persisted atomically.
        """
        t0 = time_module.time()
        if self._try_load_persistent(cfg, view_mode):
            self._telemetry.index_build_seconds += time_module.time() - t0
            self._telemetry.index_rss_mb = _current_rss_mb()
            self._telemetry.compact_bytes = (
                self._src.element_size() * self._src.numel() +
                self._dst.element_size() * self._dst.numel() +
                self._t.element_size() * self._t.numel() +
                self._src_type.element_size() * self._src_type.numel() +
                self._dst_type.element_size() * self._dst_type.numel() +
                self._edge_type_index.element_size() * self._edge_type_index.numel() +
                self._split.element_size() * self._split.numel()
            )
            if self._node_table_x_src is not None:
                self._telemetry.node_table_bytes = (
                    self._node_table_x_src.element_size() * self._node_table_x_src.numel() +
                    self._node_table_x_dst.element_size() * self._node_table_x_dst.numel()
                )
            return

        # ---- Pass 1: gather per-window metadata, collect node ids, max node ----
        # Single streaming scan: track per-node x_src/x_dst consistency without
        # keeping window-level TemporalData around.
        max_node = -1
        node_x_src: dict[int, torch.Tensor] = {}
        node_x_dst: dict[int, torch.Tensor] = {}
        feature_dim = -1
        inconsistent_nodes: list[tuple[int, str]] = []

        global_offset = 0
        num_edge_types = getattr(getattr(cfg, "dataset", None), "num_edge_types", 10)
        self._edge_type_num_types = num_edge_types

        # Resolve msg-derivation policy from cfg.
        decoder_methods_raw = getattr(
            getattr(cfg.detection.gnn_training, "decoder", None), "used_methods", ""
        )
        decoder_methods = {m.strip() for m in str(decoder_methods_raw).split(",")}
        self._msg_predict_edge_type = "predict_edge_type" in decoder_methods
        self._msg_use_edge_type = not self._msg_predict_edge_type

        # We also need x_src dim and (optionally) src_type to know msg_dim.
        # msg_dim = in_dim + in_dim + (edge_type_dim if not predict_edge_type else 0)
        use_node_type_in_node_feats = bool(
            getattr(cfg.detection.gnn_training.encoder, "use_node_type_in_node_feats", True)
        )

        for idx, spec in enumerate(self._specs):
            data = _prepare_window(cfg, spec.path, view_mode)
            n_events = int(data.src.numel())
            scan_size = 0
            try:
                scan_size = os.path.getsize(spec.path)
            except OSError:
                pass
            self._telemetry.source_artifact_scan_count += 1
            self._telemetry.source_artifact_scan_bytes_estimate += int(scan_size)

            if n_events > 0:
                end = global_offset + n_events
                self._src[global_offset:end] = data.src.long()
                self._dst[global_offset:end] = data.dst.long()
                self._t[global_offset:end] = data.t.long()
                self._src_type[global_offset:end] = data.src_type.int()
                self._dst_type[global_offset:end] = data.dst_type.int()
                self._edge_type_index[global_offset:end] = data.edge_type_index.int()
                self._split[global_offset:end] = spec.split_index

                # Populate the node-level feature table for x_src/x_dst.
                # x_src / x_dst must be invariant per node across windows; we
                # enforce that with first-write-wins consistency check.
                x_src = data.x_src
                x_dst = data.x_dst
                if x_src is None or x_dst is None:
                    raise RuntimeError(
                        f"Window {spec.path} lacks x_src/x_dst; cannot build "
                        "node-level feature table"
                    )
                if feature_dim < 0:
                    feature_dim = int(x_src.shape[1])
                    if int(x_dst.shape[1]) != feature_dim:
                        raise RuntimeError(
                            f"Window {spec.path} has mismatched x_src/x_dst "
                            f"dims: {x_src.shape[1]} vs {x_dst.shape[1]}"
                        )
                elif int(x_src.shape[1]) != feature_dim or int(x_dst.shape[1]) != feature_dim:
                    raise RuntimeError(
                        f"Window {spec.path} has inconsistent feature dim "
                        f"(expected {feature_dim}, got "
                        f"x_src={x_src.shape[1]}, x_dst={x_dst.shape[1]})"
                    )

                src_cpu = data.src.long().cpu().tolist()
                # Vectorised uniqueness: keep order of first appearance.
                seen_in_window: set[int] = set()
                for i, node_id in enumerate(src_cpu):
                    if node_id in seen_in_window:
                        continue
                    seen_in_window.add(node_id)
                    feat = x_src[i].detach().cpu()
                    if node_id in node_x_src:
                        if not torch.equal(node_x_src[node_id], feat):
                            inconsistent_nodes.append((node_id, "x_src"))
                            if len(inconsistent_nodes) > 8:
                                break
                    else:
                        node_x_src[node_id] = feat
                    if node_id > max_node:
                        max_node = node_id

                seen_in_window.clear()
                dst_cpu = data.dst.long().cpu().tolist()
                for i, node_id in enumerate(dst_cpu):
                    if node_id in seen_in_window:
                        continue
                    seen_in_window.add(node_id)
                    feat = x_dst[i].detach().cpu()
                    if node_id in node_x_dst:
                        if not torch.equal(node_x_dst[node_id], feat):
                            inconsistent_nodes.append((node_id, "x_dst"))
                            if len(inconsistent_nodes) > 8:
                                break
                    else:
                        node_x_dst[node_id] = feat
                    if node_id > max_node:
                        max_node = node_id

                global_offset = end
                if len(inconsistent_nodes) > 0 and len(inconsistent_nodes) > 8:
                    break
            self._window_paths[idx] = spec.path
            del data

        if inconsistent_nodes:
            sample = ", ".join(
                f"node {n} ({f})" for n, f in inconsistent_nodes[:5]
            )
            raise RuntimeError(
                "x_src/x_dst are not invariant per node across windows; the "
                "node-level compact table is rejected. Offending samples: "
                f"{sample}"
            )

        # ---- Build the dense node table ----
        if feature_dim < 0 or max_node < 0:
            # No events at all — provide a minimal empty table so consumers
            # don't need special-casing.
            self._node_table_dim = 0
            self._node_table_max_node = -1
            self._node_table_x_src = torch.empty(0, 0, dtype=torch.float32)
            self._node_table_x_dst = torch.empty(0, 0, dtype=torch.float32)
        else:
            n_nodes = max_node + 1
            x_src_table = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)
            x_dst_table = torch.zeros((n_nodes, feature_dim), dtype=torch.float32)
            for node_id, feat in node_x_src.items():
                x_src_table[node_id] = feat
            for node_id, feat in node_x_dst.items():
                x_dst_table[node_id] = feat
            self._node_table_x_src = x_src_table
            self._node_table_x_dst = x_dst_table
            self._node_table_dim = feature_dim
            self._node_table_max_node = max_node

        # ---- msg derivation dimension ----
        # msg = cat([x_src[src], x_dst[dst], edge_type]) in non-predict_edge_type
        # msg = cat([x_src[src], x_dst[dst]]) in predict_edge_type
        self._msg_dim = (
            2 * feature_dim + (num_edge_types if self._msg_use_edge_type else 0)
            if feature_dim > 0
            else 0
        )

        # Note: use_node_type_in_node_feats already encoded in x_src/x_dst
        # by extract_msg_from_data / extract_msg_node_type_only, so msg_dim
        # is fully determined by the loaded features.

        self._telemetry.index_build_count += 1
        self._telemetry.index_build_seconds += time_module.time() - t0
        self._telemetry.index_rss_mb = _current_rss_mb()
        self._telemetry.compact_bytes = (
            self._src.element_size() * self._src.numel() +
            self._dst.element_size() * self._dst.numel() +
            self._t.element_size() * self._t.numel() +
            self._src_type.element_size() * self._src_type.numel() +
            self._dst_type.element_size() * self._dst_type.numel() +
            self._edge_type_index.element_size() * self._edge_type_index.numel() +
            self._split.element_size() * self._split.numel()
        )
        if self._node_table_x_src is not None:
            self._telemetry.node_table_bytes = (
                self._node_table_x_src.element_size() * self._node_table_x_src.numel() +
                self._node_table_x_dst.element_size() * self._node_table_x_dst.numel()
            )

        # Persist the freshly-built index for next run.
        self._write_persistent(cfg, view_mode)

    def _build_edge_type_onehot(self, indices: torch.Tensor) -> torch.Tensor:
        """Return float32 one-hot [N, num_edge_types]."""
        N = indices.numel()
        result = torch.zeros(N, self._edge_type_num_types, dtype=torch.float32)
        result.scatter_(1, indices.long().unsqueeze(1), 1.0)
        return result

    def compact_get(self, field: str, flat_ids: torch.Tensor) -> torch.Tensor:
        """Get field directly from global tensors or node table. O(1) per event.

        Returns values with the same shape as the original window-level tensor
        (1D for scalar fields, 2D for edge_type/x_src/x_dst/msg).
        """
        N = flat_ids.numel()

        if field == "src":
            self._telemetry.compact_lookup_count += N
            return self._src[flat_ids].long()
        if field == "dst":
            self._telemetry.compact_lookup_count += N
            return self._dst[flat_ids].long()
        if field == "t":
            self._telemetry.compact_lookup_count += N
            return self._t[flat_ids].long()
        if field == "src_type":
            self._telemetry.compact_lookup_count += N
            return self._src_type[flat_ids].long()
        if field == "dst_type":
            self._telemetry.compact_lookup_count += N
            return self._dst_type[flat_ids].long()
        if field == "edge_type_index":
            self._telemetry.compact_lookup_count += N
            return self._edge_type_index[flat_ids].long()
        if field == "split":
            self._telemetry.compact_lookup_count += N
            return self._split[flat_ids].long()
        if field == "global_event_index":
            self._telemetry.compact_lookup_count += N
            return flat_ids.clone()
        if field == "edge_type":
            self._telemetry.compact_lookup_count += N
            indices = self._edge_type_index[flat_ids].long()
            return self._build_edge_type_onehot(indices)

        if field == "x_src":
            return self._node_field_lookup("x_src", flat_ids, N)
        if field == "x_dst":
            return self._node_field_lookup("x_dst", flat_ids, N)
        if field == "msg":
            return self._msg_lookup(flat_ids, N)

        raise AttributeError(f"Unknown compact field {field!r}")

    def _node_field_lookup(
        self, kind: str, flat_ids: torch.Tensor, N: int
    ) -> torch.Tensor:
        """Serve x_src or x_dst from the per-node table."""
        if self._node_table_x_src is None:
            raise AttributeError(
                "x_src/x_dst are not served by the compact index for this "
                "dataset; the consumer must load them from a window."
            )
        self._telemetry.node_lookup_count += N
        # Resolve per-event source node ids for the requested kind.
        node_ids = self._src if kind == "x_src" else self._dst
        # flat_ids are global event ids; gather the corresponding node ids.
        per_event_node = node_ids[flat_ids].long().clamp_min(0)
        if per_event_node.numel() == 0:
            return torch.zeros(0, self._node_table_dim, dtype=torch.float32)
        max_node_id = int(per_event_node.max().item())
        if max_node_id >= self._node_table_x_src.shape[0]:
            raise IndexError(
                f"node id {max_node_id} exceeds node table size "
                f"{self._node_table_x_src.shape[0]}; data is inconsistent"
            )
        table = self._node_table_x_src if kind == "x_src" else self._node_table_x_dst
        return table[per_event_node]

    def _msg_lookup(self, flat_ids: torch.Tensor, N: int) -> torch.Tensor:
        """Derive msg from x_src[node] || x_dst[node] || (edge_type)."""
        if self._msg_dim <= 0 or self._node_table_x_src is None:
            raise AttributeError(
                "msg cannot be derived from compact index for this dataset"
            )
        self._telemetry.msg_lookup_count += N
        src_nodes = self._src[flat_ids].long().clamp_min(0)
        dst_nodes = self._dst[flat_ids].long().clamp_min(0)
        x_src_per_event = self._node_table_x_src[src_nodes]
        x_dst_per_event = self._node_table_x_dst[dst_nodes]
        if self._msg_use_edge_type:
            indices = self._edge_type_index[flat_ids].long()
            edge_type_oh = self._build_edge_type_onehot(indices)
            return torch.cat([x_src_per_event, x_dst_per_event, edge_type_oh], dim=-1)
        return torch.cat([x_src_per_event, x_dst_per_event], dim=-1)

    def record_lookup(self, field: str, count: int) -> None:
        """Record a history lookup for telemetry."""
        self._telemetry.history_lookup_calls += 1
        self._telemetry.history_lookup_events += count
        self._telemetry.field_lookup_counts[field] = (
            self._telemetry.field_lookup_counts.get(field, 0) + count
        )


@dataclass(frozen=True)
class _WindowSpec:
    path: str
    split_name: str
    split_index: int
    split_window_index: int
    global_window_id: int = -1
    global_offset: int = -1
    num_events: int = -1


def _prepare_window(cfg, filepath: str, view_mode: str | None = None) -> TemporalData:
    """Load and transform exactly one trusted edge-embedding artifact."""
    data = load_trusted_torch_artifact(filepath, expected_type=TemporalData).to("cpu")
    if cfg.edge_featurization.embed_nodes.used_method.strip() == "only_type":
        data = extract_msg_node_type_only([data], cfg)[0]
    else:
        data = extract_msg_from_data([data], cfg)[0]
    _inject_event_indices(data, cfg)
    if view_mode is not None:
        from mstc.dataset_views import apply_dataset_view
        data = apply_dataset_view(data, view_mode)
    return data


class LazyTemporalWindowCollection(Sequence):
    """Chronological path-backed window sequence with no permanent payload cache."""

    def __init__(self, cfg, specs: Sequence[_WindowSpec], *, view_mode: str | None = None):
        self._cfg = cfg
        self._specs = tuple(specs)
        self._view_mode = view_mode

    @property
    def window_paths(self) -> tuple[str, ...]:
        return tuple(spec.path for spec in self._specs)

    def with_view(self, view_mode: str) -> "LazyTemporalWindowCollection":
        return type(self)(self._cfg, self._specs, view_mode=view_mode)

    def with_specs(self, specs: Sequence[_WindowSpec]) -> "LazyTemporalWindowCollection":
        return type(self)(self._cfg, specs, view_mode=self._view_mode)

    def _load(self, spec: _WindowSpec) -> TemporalData:
        data = _prepare_window(self._cfg, spec.path, self._view_mode)
        if spec.global_offset >= 0:
            actual_events = int(data.src.numel())
            if actual_events != spec.num_events:
                raise RuntimeError(
                    f"Window event count changed for {spec.path}: "
                    f"manifest={spec.num_events}, loaded={actual_events}"
                )
            data.global_event_index = torch.arange(
                spec.global_offset, spec.global_offset + spec.num_events, dtype=torch.long,
            )
            data.event_index = data.global_event_index
            data.split = spec.split_index
            data.split_name = spec.split_name
            data.window_id = spec.global_window_id
        return data

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._load(spec) for spec in self._specs[index]]
        return self._load(self._specs[index])

    def __iter__(self) -> Iterator[TemporalData]:
        for spec in self._specs:
            yield self._load(spec)


class _LazyEventField:
    """Tensor-like compatibility facade backed by BoundedFullData lookups."""

    def __init__(self, owner: "BoundedFullData", field: str):
        self._owner = owner
        self._field = field

    def cpu(self):
        return self

    def __len__(self) -> int:
        return self._owner.num_events

    @property
    def shape(self) -> tuple[int, ...]:
        trailing_shape, _ = self._owner.field_metadata(self._field)
        return (self._owner.num_events, *trailing_shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._owner.num_events)
            index = torch.arange(start, stop, step, dtype=torch.long)
        elif not isinstance(index, torch.Tensor):
            index = torch.tensor([index], dtype=torch.long)
            return self._owner.get_event_values(self._field, index)[0]
        return self._owner.get_event_values(self._field, index)


class BoundedFullData:
    """
    Global event view with bounded memory.

    The compact index stores small fields (src, dst, t, src_type, dst_type,
    edge_type_index, split) as global tensors during initialization. This
    eliminates per-batch full artifact reloads when looking up these fields.

    Large fields (msg, x_src, x_dst) remain lazy-loaded from windows.
    """

    _ALIASES = {"event_index": "global_event_index", "event_split": "split"}
    _DERIVED_FIELDS = {"global_event_index", "split"}
    _WINDOW_FIELDS = {
        "msg", "t", "edge_type", "src", "dst", "src_type", "dst_type",
        "edge_type_index", "x_src", "x_dst",
    }
    # Fields that the compact index can serve when its node table is built.
    # x_src/x_dst/msg are served from the node table without loading the source
    # TemporalData. Without the node table, only the event-level fields are
    # available through the compact path.
    _COMPACT_INDEXED = frozenset({
        "src", "dst", "t", "src_type", "dst_type",
        "edge_type_index", "edge_type",
        "global_event_index", "split",
        "x_src", "x_dst",
        "msg",
    })

    def __init__(self, cfg, specs: Sequence[_WindowSpec], *, view_mode: str, max_node: int,
                 telemetry: _LoaderTelemetry,
                 field_metadata: dict[str, tuple[tuple[int, ...], torch.dtype]],
                 cache_windows: int = 1,
                 compact_index: "_CompactIndex | None" = None):
        if cache_windows <= 0:
            raise ValueError("cache_windows must be positive")
        self._cfg = cfg
        self._specs = tuple(specs)
        self._starts = tuple(spec.global_offset for spec in self._specs)
        self._view_mode = view_mode
        self._cache_windows = int(cache_windows)
        self._cache: OrderedDict[int, TemporalData] = OrderedDict()
        self.max_node = int(max_node)
        self.num_events = sum(spec.num_events for spec in self._specs)
        self._field_metadata = dict(field_metadata)
        self.loader_telemetry = telemetry.as_dict()

        # Compact history index (built in load_all_datasets)
        self._compact_index = compact_index

        for field in self._WINDOW_FIELDS | self._DERIVED_FIELDS | set(self._ALIASES):
            setattr(self, field, _LazyEventField(self, field))

    @property
    def cached_window_count(self) -> int:
        return len(self._cache)

    def release_cache(self) -> None:
        self._cache.clear()

    def field_metadata(self, field: str) -> tuple[tuple[int, ...], torch.dtype]:
        field = self._ALIASES.get(field, field)
        if field in self._DERIVED_FIELDS:
            return (), torch.long
        if self._compact_index is not None and field in self._COMPACT_INDEXED:
            if field == "global_event_index":
                return (), torch.int64
            if field == "split":
                return (), torch.int8
            if field == "edge_type":
                # edge_type is reconstructed from edge_type_index
                return self._field_metadata.get("edge_type", ((0,), torch.float32))
            if field in ("x_src", "x_dst"):
                if not self._compact_index.has_node_table():
                    raise AttributeError(
                        f"Field {field!r} requires the compact node table, "
                        "which is unavailable for this dataset."
                    )
                dim = self._compact_index.node_table_dim
                return ((dim,), torch.float32)
            if field == "msg":
                if not self._compact_index.has_msg_derivation():
                    raise AttributeError(
                        f"Field {field!r} cannot be derived from the compact "
                        "index for this dataset."
                    )
                return ((self._compact_index._msg_dim,), torch.float32)
            # src, dst, t, src_type, dst_type, edge_type_index
            return self._field_metadata.get(field, self._field_metadata.get(
                field, ((), torch.int64 if field in ("t",) else torch.int32)
            ))
        try:
            return self._field_metadata[field]
        except KeyError as exc:
            raise AttributeError(f"Unknown full_data field {field!r}") from exc

    def _window_for_event(self, event_id: int) -> int:
        if event_id < 0 or event_id >= self.num_events:
            raise IndexError(f"global event index {event_id} outside [0, {self.num_events})")
        window_index = bisect.bisect_right(self._starts, event_id) - 1
        spec = self._specs[window_index]
        if event_id >= spec.global_offset + spec.num_events:
            raise IndexError(f"global event index {event_id} falls in no window")
        return window_index

    def _load_window(self, window_index: int) -> TemporalData:
        cached = self._cache.pop(window_index, None)
        if cached is not None:
            self._cache[window_index] = cached
            return cached
        spec = self._specs[window_index]
        data = _prepare_window(self._cfg, spec.path, self._view_mode)
        if int(data.src.numel()) != spec.num_events:
            raise RuntimeError(f"Window event count changed for {spec.path}")
        self._cache[window_index] = data
        while len(self._cache) > self._cache_windows:
            self._cache.popitem(last=False)
        return data

    def get_event_values(self, field: str, event_ids: torch.Tensor) -> torch.Tensor:
        field = self._ALIASES.get(field, field)
        ids = torch.as_tensor(event_ids, dtype=torch.long).cpu()
        original_shape = tuple(ids.shape)
        flat_ids = ids.reshape(-1)

        if flat_ids.numel() == 0:
            trailing_shape, dtype = self.field_metadata(field)
            return torch.empty((*original_shape, *trailing_shape), dtype=dtype)

        # Serve compact fields from the compact index (fast path).
        # x_src / x_dst / msg require the node table; if unavailable the fast
        # path is skipped and the legacy window lookup is used (which counts
        # as a fallback_full_window_load).
        if (
            self._compact_index is not None
            and field in self._COMPACT_INDEXED
        ):
            can_serve = True
            if field in ("x_src", "x_dst") and not self._compact_index.has_node_table():
                can_serve = False
            elif field == "msg" and not self._compact_index.has_msg_derivation():
                can_serve = False

            if can_serve:
                self._compact_index.record_lookup(field, flat_ids.numel())
                values = self._compact_index.compact_get(field, flat_ids)
                if values.dim() == 1:
                    return values.reshape(original_shape)
                return values.reshape((*original_shape, *values.shape[1:]))

        # Slow path: window-based lookup for lazy fields
        if field == "global_event_index":
            return flat_ids.clone().reshape(original_shape)

        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        for output_position, event_id in enumerate(flat_ids.tolist()):
            window_index = self._window_for_event(int(event_id))
            local_index = int(event_id) - self._specs[window_index].global_offset
            grouped.setdefault(window_index, []).append((output_position, local_index))

        if field == "split":
            result = torch.empty(flat_ids.numel(), dtype=torch.long)
            for window_index, positions in grouped.items():
                result[[position for position, _ in positions]] = self._specs[window_index].split_index
            return result.reshape(original_shape)

        if field not in self._WINDOW_FIELDS:
            raise AttributeError(f"Unknown full_data field {field!r}")

        result = None
        for window_index, positions in grouped.items():
            data = self._load_window(window_index)
            # Charge the fallback counter for any window loaded outside the
            # initial index build.
            if self._compact_index is not None:
                spec_path = self._specs[window_index].path
                size = 0
                try:
                    size = os.path.getsize(spec_path)
                except OSError:
                    pass
                self._compact_index._telemetry.fallback_full_window_load_count += 1
                self._compact_index._telemetry.fallback_full_window_load_bytes_estimate += int(size)
            if not hasattr(data, field):
                raise ValueError(f"Window {self._specs[window_index].path} lacks field {field!r}")
            local_indices = torch.tensor([local for _, local in positions], dtype=torch.long)
            values = getattr(data, field).cpu().index_select(0, local_indices)
            if result is None:
                result = torch.empty((flat_ids.numel(), *values.shape[1:]), dtype=values.dtype)
            result.index_copy_(
                0,
                torch.tensor([position for position, _ in positions], dtype=torch.long),
                values,
            )
        assert result is not None
        return result.reshape((*original_shape, *result.shape[1:]))

def _validate_event_fields(g: TemporalData, cfg) -> None:
    """
    Validates the structural consistency of a single temporal graph window.

    Raises ValueError or AssertionError on any mismatch.
    """
    num_events = g.src.shape[0]
    assert len(g.src) == num_events, (
        f"src length {len(g.src)} != num_events {num_events}"
    )
    assert len(g.dst) == num_events, (
        f"dst length {len(g.dst)} != num_events {num_events}"
    )
    assert len(g.t) == num_events, (
        f"t length {len(g.t)} != num_events {num_events}"
    )

    assert g.t.dtype == torch.long or g.t.dtype == torch.int64, (
        f"t dtype must be torch.long, got {g.t.dtype}"
    )

    assert (g.t[1:] >= g.t[:-1]).all(), (
        f"timestamps must be non-decreasing; found decreasing order"
    )

    num_node_types = cfg.dataset.num_node_types
    assert (g.src_type >= 0).all() and (g.src_type < num_node_types).all(), (
        f"src_type values must be in [0, {num_node_types}); "
        f"got min={g.src_type.min().item()}, max={g.src_type.max().item()}"
    )
    assert (g.dst_type >= 0).all() and (g.dst_type < num_node_types).all(), (
        f"dst_type values must be in [0, {num_node_types}); "
        f"got min={g.dst_type.min().item()}, max={g.dst_type.max().item()}"
    )

    num_edge_types = cfg.dataset.num_edge_types
    assert (g.edge_type_index >= 0).all() and (g.edge_type_index < num_edge_types).all(), (
        f"edge_type_index must be in [0, {num_edge_types}); "
        f"got min={g.edge_type_index.min().item()}, max={g.edge_type_index.max().item()}"
    )

    expected_edge_type_index = g.edge_type.argmax(dim=-1)
    assert torch.equal(g.edge_type_index, expected_edge_type_index), (
        "edge_type_index does not match edge_type.argmax(dim=-1)"
    )

    expected_local = torch.arange(num_events, dtype=torch.long)
    assert torch.equal(g.local_event_index, expected_local), (
        f"local_event_index must be 0..{num_events - 1}; "
        f"got {g.local_event_index[:5].tolist()}..."
    )


def _inject_event_indices(g: TemporalData, cfg) -> None:
    """
    Computes and attaches per-window event fields and validates them.

    Sets on ``g``:
        src_type      : torch.long, shape [E]
        dst_type      : torch.long, shape [E]
        edge_type_index: torch.long, shape [E]
        local_event_index: torch.long, shape [E]

    Node types are always extracted from the one-hot tensors that were
    stored during message parsing (``g.src_type_onehot`` /
    ``g.dst_type_onehot``).  This guarantees correct semantics regardless
    of the ``use_node_type_in_node_feats`` setting and regardless of
    embedding values.

    For legacy artifacts that lack the one-hot attributes, a fallback
    attempts to extract types from the known message layout.
    """
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types

    msg_dim = g.msg.shape[1]
    short_msg_dim = node_type_dim * 2 + edge_type_dim

    # ------------------------------------------------------------------
    # Node type extraction — authoritative source is the stored one-hot
    # ------------------------------------------------------------------
    if hasattr(g, "src_type_onehot") and hasattr(g, "dst_type_onehot"):
        if g.src_type_onehot.shape != (g.src.shape[0], node_type_dim):
            raise ValueError(
                f"src_type_onehot has wrong shape {g.src_type_onehot.shape}; "
                f"expected ({g.src.shape[0]}, {node_type_dim})"
            )
        if g.dst_type_onehot.shape != (g.src.shape[0], node_type_dim):
            raise ValueError(
                f"dst_type_onehot has wrong shape {g.dst_type_onehot.shape}; "
                f"expected ({g.src.shape[0]}, {node_type_dim})"
            )
        g.src_type = g.src_type_onehot.argmax(dim=-1).long()
        g.dst_type = g.dst_type_onehot.argmax(dim=-1).long()
    elif msg_dim == short_msg_dim:
        # only_type legacy artifact: msg = [src_type | edge_type | dst_type]
        g.src_type = g.msg[:, :node_type_dim].argmax(dim=-1).long()
        g.dst_type = g.msg[:, node_type_dim + edge_type_dim:].argmax(dim=-1).long()
    else:
        # Full-embedding legacy artifact with no one-hot attributes.
        # Attempt to recover types from the known message layout:
        # msg = [src_type_raw | src_emb | edge_type | dst_type_raw | dst_emb]
        if msg_dim == node_type_dim + edge_type_dim + node_type_dim:
            # No embeddings, just types (edge case)
            g.src_type = g.msg[:, :node_type_dim].argmax(dim=-1).long()
            g.dst_type = g.msg[:, node_type_dim + edge_type_dim:].argmax(dim=-1).long()
        elif hasattr(g, "x_src") and g.x_src.shape[1] >= node_type_dim:
            # Heuristic fallback — warn but do not silently use wrong source
            raise ValueError(
                f"Cannot reliably extract node types from x_src/x_dst when "
                f"node type features are not stored. Set "
                f"use_node_type_in_node_feats=True or re-process the source "
                f"artifacts to store src_type_onehot/dst_type_onehot."
            )
        else:
            raise ValueError(
                f"msg has unexpected dimension {msg_dim}; expected either "
                f"{short_msg_dim} (only_type) or one-hot types must be "
                f"stored as src_type_onehot/dst_type_onehot on the graph."
            )

    # ------------------------------------------------------------------
    # Edge type and local index
    # ------------------------------------------------------------------
    g.edge_type_index = g.edge_type.argmax(dim=-1).long()
    g.local_event_index = torch.arange(g.src.shape[0], dtype=torch.long)

    _validate_event_fields(g, cfg)


def _inject_full_data_event_fields(
    train_data: list,
    val_data: list,
    test_data: list,
    full_data: Data,
) -> Data:
    """
    Computes and attaches cumulative global event indices and split labels
    onto ``full_data`` and onto each individual window in the three lists.

    Sets on ``full_data``:
        global_event_index : torch.arange(total_events)
        event_index        : alias of global_event_index
        src_type           : concatenated per-window src_type
        dst_type           : concatenated per-window dst_type
        edge_type_index    : concatenated per-window edge_type_index
        src                : concatenated per-window src
        dst                : concatenated per-window dst
        split              : torch.long, shape [total_events], per-event split index
        event_split        : alias of split
        split_name         : torch.long, shape [total_events], per-event split name (string tensor)

    Sets on each window ``g``:
        global_event_index : global positions [start_offset, ...)
        event_index        : alias of global_event_index
        split              : integer index, one of {0, 1, 2}
        split_name         : original string name ("train" | "val" | "test")
        window_id          : monotonic integer across all splits
    """
    all_windows = [(g, "train") for g in train_data] + \
                  [(g, "val")   for g in val_data]   + \
                  [(g, "test")  for g in test_data]

    cumulative_offset = 0
    window_id_counter = 0

    all_global_event_index: list[torch.Tensor] = []
    all_src_type: list[torch.Tensor] = []
    all_dst_type: list[torch.Tensor] = []
    all_edge_type_index: list[torch.Tensor] = []
    all_src: list[torch.Tensor] = []
    all_dst: list[torch.Tensor] = []
    all_split: list[torch.Tensor] = []

    for g, split_name in all_windows:
        num_events = g.src.shape[0]
        g.global_event_index = torch.arange(
            cumulative_offset, cumulative_offset + num_events, dtype=torch.long
        )
        g.event_index = g.global_event_index
        g.split = SPLIT_NAME_TO_INDEX[split_name]  # integer index
        g.split_name = split_name                   # original string
        g.window_id = window_id_counter
        window_id_counter += 1
        cumulative_offset += num_events

        all_global_event_index.append(g.global_event_index)
        all_src_type.append(g.src_type)
        all_dst_type.append(g.dst_type)
        all_edge_type_index.append(g.edge_type_index)
        all_src.append(g.src)
        all_dst.append(g.dst)
        all_split.append(torch.full((num_events,), SPLIT_NAME_TO_INDEX[split_name], dtype=torch.long))

    full_data.global_event_index = torch.cat(all_global_event_index)
    full_data.event_index = full_data.global_event_index
    full_data.src_type = torch.cat(all_src_type)
    full_data.dst_type = torch.cat(all_dst_type)
    full_data.edge_type_index = torch.cat(all_edge_type_index)
    full_data.src = torch.cat(all_src)
    full_data.dst = torch.cat(all_dst)
    full_data.split = torch.cat(all_split)
    full_data.event_split = full_data.split  # alias

    return full_data


def _eager_load_all_datasets(cfg, train_data, val_data, test_data):
    """Legacy in-memory reference retained for injected/synthetic callers."""
    from mstc.dataset_views import apply_dataset_view

    view_mode = cfg.dataset_view.mode
    train_data = [apply_dataset_view(data, view_mode) for data in train_data]
    val_data = [apply_dataset_view(data, view_mode) for data in val_data]
    test_data = [apply_dataset_view(data, view_mode) for data in test_data]

    all_msg, all_t, all_edge_types, all_src, all_dst = [], [], [], [], []
    max_node = -1
    for dataset in (train_data, val_data, test_data):
        for data in dataset:
            all_msg.append(data.msg)
            all_t.append(data.t)
            all_edge_types.append(data.edge_type)
            all_src.append(data.src)
            all_dst.append(data.dst)
            if data.src.numel() > 0:
                max_node = max(max_node, int(data.src.max()), int(data.dst.max()))

    full_data = Data(
        msg=torch.cat(all_msg),
        t=torch.cat(all_t),
        edge_type=torch.cat(all_edge_types),
        src=torch.cat(all_src),
        dst=torch.cat(all_dst),
    )
    max_node = max_node + 1 if max_node >= 0 else 0
    print(f"Max node in {cfg.dataset.name}: {max_node}")
    return (
        train_data,
        val_data,
        test_data,
        _inject_full_data_event_fields(train_data, val_data, test_data, full_data),
        max_node,
    )


def _scan_lazy_collections(cfg, collections, telemetry):
    """Build exact global offsets while retaining at most one scanned window."""
    scanned_by_split = []
    all_specs = []
    field_metadata = {}
    global_offset = 0
    global_window_id = 0
    max_node = -1

    for collection in collections:
        split_specs = []
        for base_spec in collection._specs:
            data = collection._load(base_spec)
            num_events = int(data.src.numel())
            if num_events:
                max_node = max(max_node, int(data.src.max()), int(data.dst.max()))
            for field in BoundedFullData._WINDOW_FIELDS:
                value = getattr(data, field, None)
                if not isinstance(value, torch.Tensor):
                    continue
                metadata = (tuple(value.shape[1:]), value.dtype)
                previous = field_metadata.setdefault(field, metadata)
                if previous != metadata:
                    raise ValueError(
                        f"Inconsistent {field} metadata: {previous} != {metadata} "
                        f"in {base_spec.path}"
                    )
            spec = _WindowSpec(
                path=base_spec.path,
                split_name=base_spec.split_name,
                split_index=base_spec.split_index,
                split_window_index=base_spec.split_window_index,
                global_window_id=global_window_id,
                global_offset=global_offset,
                num_events=num_events,
            )
            split_specs.append(spec)
            all_specs.append(spec)
            global_offset += num_events
            global_window_id += 1
            telemetry.observe("window scan peak", emit=False)
            del data
        scanned_by_split.append(collection.with_specs(split_specs))
    return (
        *scanned_by_split,
        tuple(all_specs),
        max_node + 1 if max_node >= 0 else 0,
        field_metadata,
    )


def load_all_datasets(cfg):
    """Return full chronological data through a bounded-memory production view.

    Uses CompactHistoryIndex for efficient per-event field lookups without
    per-batch full artifact reloads.
    """
    telemetry = _LoaderTelemetry()
    telemetry.observe("before dataset loading")
    path = cfg.edge_featurization.embed_edges._edge_embeds_dir

    # The lazy keyword is deliberately optional at the call boundary so tests
    # and downstream integrations that inject the historical 3-argument loader
    # continue to use the eager semantic reference.
    try:
        train_data = load_data_set(cfg, path=path, split="train", lazy=True)
        telemetry.observe("after train index/path resolution")
        val_data = load_data_set(cfg, path=path, split="val", lazy=True)
        telemetry.observe("after val index/path resolution")
        test_data = load_data_set(cfg, path=path, split="test", lazy=True)
        telemetry.observe("after test index/path resolution")
    except TypeError as exc:
        if "lazy" not in str(exc):
            raise
        train_data = load_data_set(cfg, path=path, split="train")
        val_data = load_data_set(cfg, path=path, split="val")
        test_data = load_data_set(cfg, path=path, split="test")

    if not all(isinstance(data, LazyTemporalWindowCollection)
               for data in (train_data, val_data, test_data)):
        return _eager_load_all_datasets(cfg, train_data, val_data, test_data)

    view_mode = cfg.dataset_view.mode
    collections = tuple(data.with_view(view_mode) for data in (train_data, val_data, test_data))
    (train_data, val_data, test_data, all_specs, max_node,
     field_metadata) = _scan_lazy_collections(cfg, collections, telemetry)

    # Build compact history index for efficient random access
    total_events = sum(spec.num_events for spec in all_specs)
    compact_index = _CompactIndex(specs=all_specs, total_events=total_events)
    telemetry.observe("before compact index build")
    compact_index.build(cfg, view_mode)
    telemetry.observe("after compact index build")

    full_data = BoundedFullData(
        cfg,
        all_specs,
        view_mode=view_mode,
        max_node=max_node,
        telemetry=telemetry,
        field_metadata=field_metadata,
        cache_windows=1,
        compact_index=compact_index,
    )
    telemetry.observe("after global metadata construction")
    telemetry.observe("before model construction")
    full_data.loader_telemetry = telemetry.as_dict()

    # Add history access telemetry
    if hasattr(full_data, "loader_telemetry") and full_data.loader_telemetry:
        full_data.loader_telemetry["history_access"] = compact_index.telemetry

    print(f"Max node in {cfg.dataset.name}: {max_node}")
    return train_data, val_data, test_data, full_data, max_node


def _resolve_window_specs(cfg, path: str, split: str) -> list[_WindowSpec]:
    requested_split = split
    if cfg._test_mode:
        split = "train"

    split_dir = os.path.join(path, split)
    all_files = sorted(os.listdir(split_dir))
    available = len(all_files)
    limit = getattr(cfg, "_max_windows_per_split", None)
    if limit is not None:
        if limit <= 0:
            raise ValueError(
                f"max_windows_per_split must be a positive integer; got {limit}"
            )
        selected = all_files[:limit]
        print(f"[Bounded smoke] split={split}: selected {len(selected)} / {available} windows")
    else:
        selected = all_files

    # Preserve the historical test-mode artifact source while keeping the
    # requested logical split identity and its global split label.
    return [
        _WindowSpec(
            path=os.path.join(split_dir, filename),
            split_name=requested_split,
            split_index=SPLIT_NAME_TO_INDEX[requested_split],
            split_window_index=index,
        )
        for index, filename in enumerate(selected)
    ]


def load_data_set(cfg, path: str, split: str, *, lazy: bool = False):
    """Resolve a deterministic split, optionally loading each selected window."""
    specs = _resolve_window_specs(cfg, path, split)
    if lazy:
        return LazyTemporalWindowCollection(cfg, specs)

    # Public compatibility mode remains eager.  Formal load_all_datasets uses
    # the path-backed mode above, including when no smoke limit is configured.
    return [_prepare_window(cfg, spec.path) for spec in specs]

def extract_msg_node_type_only(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (node_type_dim * 2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")

    field_to_size = [
        ("src_type_raw", node_type_dim),
        ("edge_type", edge_type_dim),
        ("dst_type_raw", node_type_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size

        # Preserve the raw one-hot type tensors so _inject_event_indices can
        # derive the correct type index directly from the authoritative source.
        g.src_type_onehot = fields["src_type_raw"].clone()
        g.dst_type_onehot = fields["dst_type_raw"].clone()

        x_src = fields["src_type_raw"]
        x_dst = fields["dst_type_raw"]

        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)

        edge_feats = build_edge_feats(fields, msg, cfg)

        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])

    return data_set

def extract_msg_from_data(data_set: list[TemporalData], cfg) -> list[TemporalData]:
    """
    Initializes the attributes of a `Data` object based on the `msg`
    computed in previous tasks.
    """
    emb_dim = cfg.edge_featurization.embed_nodes.emb_dim
    node_type_dim = cfg.dataset.num_node_types
    edge_type_dim = cfg.dataset.num_edge_types

    msg_len = data_set[0].msg.shape[1]
    expected_msg_len = (emb_dim*2) + (node_type_dim*2) + edge_type_dim
    if msg_len != expected_msg_len:
        raise ValueError(f"The msg has an invalid shape, found {msg_len} instead of {expected_msg_len}")

    field_to_size = [
        ("src_type_raw", node_type_dim),
        ("src_emb", emb_dim),
        ("edge_type", edge_type_dim),
        ("dst_type_raw", node_type_dim),
        ("dst_emb", emb_dim),
    ]
    for g in data_set:
        fields = {}
        idx = 0
        for field, size in field_to_size:
            fields[field] = g.msg[:, idx: idx + size]
            idx += size

        # Preserve the raw one-hot type tensors so _inject_event_indices can
        # derive the correct type index without ever consulting the embedding.
        # This is the authoritative source of truth regardless of the
        # use_node_type_in_node_feats setting.
        g.src_type_onehot = fields["src_type_raw"].clone()
        g.dst_type_onehot = fields["dst_type_raw"].clone()

        x_src = fields["src_emb"]
        x_dst = fields["dst_emb"]

        if cfg.detection.gnn_training.encoder.use_node_type_in_node_feats:
            x_src = torch.cat([x_src, fields["src_type_raw"]], dim=-1)
            x_dst = torch.cat([x_dst, fields["dst_type_raw"]], dim=-1)

        if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
            msg = torch.cat([x_src, x_dst], dim=-1)
        else:
            msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)

        edge_feats = build_edge_feats(fields, msg, cfg)

        g.x_src = x_src
        g.x_dst = x_dst
        g.msg = msg
        g.edge_type = fields["edge_type"]
        g.edge_feats = edge_feats
        g.edge_index = torch.stack([g.src, g.dst])

    return data_set

def build_edge_feats(fields, msg, cfg):
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    edge_feats = []
    if "edge_type" in edge_features:
        edge_feats.append(fields["edge_type"])
    if "msg" in edge_features:
        edge_feats.append(msg)
    edge_feats = torch.cat(edge_feats, dim=-1) if len(edge_feats) > 0 else None
    return edge_feats

def _is_event_level_attribute(value, num_events):
    """
    Returns True if ``value`` is a Tensor whose first dimension equals ``num_events``,
    indicating it should be sliced per-batch by PyG's TemporalDataLoader.

    Returns False for window-level scalars (int, str) or other non-sliceable types
    that would cause ``AttributeError: 'int' object has no attribute 'size'`` in
    PyG's TemporalData.index_select.

    This is the complement of the PyG condition::

        if value.size(0) == self.num_events:
            data[key] = value[idx]

    The ``dim() >= 1`` guard is critical: it prevents 0-d scalars (Python int/str)
    from reaching ``size(0)`` and causing AttributeError.
    """
    return (
        isinstance(value, torch.Tensor)
        and value.dim() >= 1
        and value.size(0) == num_events
    )


def custom_temporal_data_loader(data: TemporalData, batch_size: int, *args, **kwargs):
    """
    A `TemporalDataLoader` wrapper that safely handles TemporalData windows
    containing mixed event-level Tensor fields and window-level non-Tensor
    metadata (e.g. ``split=int``, ``split_name=str``, ``window_id=int``).

    Background
    ----------
    PyG's ``TemporalDataLoader`` calls ``TemporalData.index_select`` internally.
    ``index_select`` unconditionally calls ``value.size(0)`` for every store
    attribute.  Window-level metadata (Python ``int`` / ``str``) has no
    ``.size()`` method, causing::

        AttributeError: 'int' object has no attribute 'size'

    This wrapper separates the two classes of attributes before batching and
    re-attaches window-level metadata to each emitted batch.

    Event-level Tensor attributes (``src``, ``dst``, ``t``, ``msg``, etc.) are
    sliced normally by PyG.  The existing contract is preserved:

        - ``g.split`` remains a Python ``int`` on the window
        - ``g.split_name`` remains a Python ``str`` on the window
        - ``g.window_id`` remains a Python ``int`` on the window
        - ``full_data.split`` (``torch.long[E]``) is unaffected (it is the
          event-level tensor, not a window scalar)

    Parameters
    ----------
    data:
        A TemporalData window. May contain both event-level Tensor fields
        and window-level scalar metadata.
    batch_size:
        Number of events per emitted batch.

    Yields
    ------
    TemporalData
        A batch with all event-level fields correctly sliced and
        window-level metadata re-attached.
    """
    num_events = data.num_events

    # ------------------------------------------------------------------
    # 1. Separate event-level tensors from window-level metadata.
    # ------------------------------------------------------------------
    loader_safe_data_dict = {}
    window_metadata = {}

    for key, value in data._store.items():
        if _is_event_level_attribute(value, num_events):
            loader_safe_data_dict[key] = value
        else:
            window_metadata[key] = value

    # ------------------------------------------------------------------
    # 2. Build a loader-safe TemporalData containing only event-level
    #    tensors.  Copy to avoid mutating the caller's original object.
    # ------------------------------------------------------------------
    if loader_safe_data_dict:
        loader_safe_data = TemporalData(**loader_safe_data_dict)
    else:
        # Degenerate: zero-event window — still construct a valid TemporalData
        loader_safe_data = TemporalData()

    # ------------------------------------------------------------------
    # 3. Delegate batching to PyG's TemporalDataLoader.
    # ------------------------------------------------------------------
    loader = TemporalDataLoader(loader_safe_data, batch_size=batch_size, *args, **kwargs)

    for batch in loader:
        # ------------------------------------------------------------------
        # 4. Restore window-level metadata onto the batch unchanged.
        # ------------------------------------------------------------------
        for key, value in window_metadata.items():
            batch[key] = value

        # ------------------------------------------------------------------
        # 5. Reconstruct edge_index from batched src/dst (PyG convention).
        # ------------------------------------------------------------------
        batch.edge_index = torch.stack([batch.src, batch.dst])

        yield batch

def temporal_data_to_data(data: TemporalData) -> Data:
    """
    NeighborLoader requires a `Data` object.
    We need to convert `TemporalData` to `Data` before using it.
    """
    return Data(num_nodes=data.x_src.shape[0], **{k: v for k, v in data._store.items()})

class GraphReindexer:
    """
    Simply transforms an edge_index and its src/dst node features of shape (E, d)
    to a reindexed edge_index with node IDs starting from 0 and src/dst node features of shape
    (max_num_node + 1, d).
    This reindexing is essential for the graph to be computed by a standard GNN model with PyG.
    """
    def __init__(self, num_nodes, device):
        self.num_nodes = num_nodes
        self.device = device

        self.assoc = None
        self.x_src_cache = None
        self.x_dst_cache = None

    def node_features_reshape(self, edge_index, x_src, x_dst, max_num_node=None):
        """
        Converts node features in shape (E, d) to a shape (N, d).
        Returns x as a tuple (x_src, x_dst).
        """
        if self.x_src_cache is None:
            self.x_src_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)
            self.x_dst_cache = torch.zeros((self.num_nodes, x_src.shape[1]), device=self.device)

        max_num_node = max_num_node + 1 if max_num_node else edge_index.max() + 1

        self.x_src_cache = self.x_src_cache.detach()
        self.x_dst_cache = self.x_dst_cache.detach()

        self.x_src_cache[edge_index[0, :]] = x_src
        self.x_dst_cache[edge_index[1, :]] = x_dst
        x = (self.x_src_cache[:max_num_node, :], self.x_dst_cache[:max_num_node, :])

        return x

    def reindex_graph(self, data):
        """
        Reindexes edge_index from 0 + reshapes node features.
        The old edge_index is stored in `data.original_edge_index`
        """
        data = data.clone()
        data.original_edge_index = data.edge_index
        (data.x_src, data.x_dst), data.edge_index = self._reindex_graph(data.edge_index, data.x_src, data.x_dst)
        return data

    def _reindex_graph(self, edge_index, x_src, x_dst):
        """
        Reindexes edge_index with indices starting from 0.
        Also reshapes the node features.
        """
        if self.assoc is None:
            self.assoc = torch.empty((self.num_nodes, ), dtype=torch.long, device=self.device)

        n_id = edge_index.unique()
        self.assoc[n_id] = torch.arange(n_id.size(0), device=edge_index.device)
        edge_index = self.assoc[edge_index]

        x = self.node_features_reshape(edge_index, x_src, x_dst)

        return x, edge_index

def save_training_checkpoint(model, optimizer, epoch: int, path: str, *, cfg=None, scheduler=None):
    """Persist all state needed for deterministic training resume.

    Temporal history is intentionally excluded: C1/C5 replay reconstructs it
    from chronological training data before validation/testing.
    """
    from mstc.experiment_utils import capture_rng_state, stable_config_hash

    if optimizer is None:
        raise ValueError("optimizer is required for a complete training checkpoint")
    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "config_hash": stable_config_hash(cfg) if cfg is not None else None,
        **capture_rng_state(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(path, "checkpoint.pt"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return checkpoint


def load_training_checkpoint(model, path: str, *, optimizer=None, scheduler=None, cfg=None,
                             map_location=None, restore_rng: bool=True):
    """Load a structured checkpoint, with clear legacy model-only fallback."""
    from mstc.experiment_utils import restore_rng_state, stable_config_hash

    checkpoint_path = path if os.path.isfile(path) else os.path.join(path, "checkpoint.pt")
    if not os.path.isfile(checkpoint_path):
        legacy_path = path if os.path.isfile(path) else os.path.join(path, "state_dict.pkl")
        legacy_state = torch.load(legacy_path, map_location=map_location, weights_only=False)
        model.load_state_dict(legacy_state)
        warnings.warn(
            "Loaded legacy model-state-only checkpoint; it supports inference but not training resume.",
            UserWarning,
        )
        return {"epoch": None, "complete": False, "config_hash": None}

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)
        warnings.warn(
            "Loaded legacy model-state-only checkpoint; it supports inference but not training resume.",
            UserWarning,
        )
        return {"epoch": None, "complete": False, "config_hash": None}

    model.load_state_dict(checkpoint["model_state_dict"])
    required = {
        "optimizer_state_dict", "epoch", "python_random_state", "numpy_random_state",
        "torch_cpu_rng_state", "config_hash",
    }
    missing = required.difference(checkpoint)
    if optimizer is not None:
        if missing:
            raise ValueError("checkpoint is not complete enough to resume training; missing " + ", ".join(sorted(missing)))
        if cfg is not None and checkpoint["config_hash"] != stable_config_hash(cfg):
            raise ValueError("checkpoint config_hash does not match the current resolved configuration")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            if "scheduler_state_dict" not in checkpoint:
                raise ValueError("checkpoint has no scheduler state for the supplied scheduler")
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if restore_rng:
            restore_rng_state(checkpoint)
    return {"epoch": checkpoint.get("epoch"), "complete": not missing, "config_hash": checkpoint.get("config_hash")}


def save_model(model, path: str, neigh_loader: bool=True, *, optimizer=None, epoch=None, cfg=None, scheduler=None):
    """Save legacy model weights and, when supplied, a complete training checkpoint."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "state_dict.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    if optimizer is not None and epoch is not None:
        save_training_checkpoint(model, optimizer, epoch, path, cfg=cfg, scheduler=scheduler)
    if neigh_loader and isinstance(model.encoder, OrthrusEncoder):
        torch.save(model.encoder.neighbor_loader, os.path.join(path, "neighbor_loader.pkl"), pickle_protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model, path: str, neigh_loader: bool=True):
    """Load weights for inference from structured or legacy checkpoints."""
    if os.path.isfile(path):
        load_training_checkpoint(model, path, restore_rng=False)
        return model
    structured_path = os.path.join(path, "checkpoint.pt")
    if os.path.isfile(structured_path):
        load_training_checkpoint(model, path, restore_rng=False)
    else:
        model.load_state_dict(torch.load(os.path.join(path, "state_dict.pkl"), weights_only=False))
    neighbor_loader_path = os.path.join(path, "neighbor_loader.pkl")
    if neigh_loader and isinstance(model.encoder, OrthrusEncoder) and os.path.isfile(neighbor_loader_path):
        model.encoder.neighbor_loader = torch.load(neighbor_loader_path, weights_only=False)
    return model
