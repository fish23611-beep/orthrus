from __future__ import annotations

import bisect
import os
import resource
import sys
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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
    def shape(self) -> tuple[int]:
        return (self._owner.num_events,)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._owner.num_events)
            index = torch.arange(start, stop, step, dtype=torch.long)
        elif not isinstance(index, torch.Tensor):
            index = torch.tensor([index], dtype=torch.long)
            return self._owner.get_event_values(self._field, index)[0]
        return self._owner.get_event_values(self._field, index)


class BoundedFullData:
    """Global event view whose large fields remain in per-window artifacts."""

    _ALIASES = {"event_index": "global_event_index", "event_split": "split"}
    _DERIVED_FIELDS = {"global_event_index", "split"}
    _WINDOW_FIELDS = {
        "msg", "t", "edge_type", "src", "dst", "src_type", "dst_type",
        "edge_type_index", "x_src", "x_dst",
    }

    def __init__(self, cfg, specs: Sequence[_WindowSpec], *, view_mode: str, max_node: int,
                 telemetry: _LoaderTelemetry, cache_windows: int = 1):
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
        self.loader_telemetry = telemetry.as_dict()
        for field in self._WINDOW_FIELDS | self._DERIVED_FIELDS | set(self._ALIASES):
            setattr(self, field, _LazyEventField(self, field))

    @property
    def cached_window_count(self) -> int:
        return len(self._cache)

    def release_cache(self) -> None:
        self._cache.clear()

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
            return torch.empty(original_shape, dtype=torch.long)
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
    return (*scanned_by_split, tuple(all_specs), max_node + 1 if max_node >= 0 else 0)


def load_all_datasets(cfg):
    """Return full chronological data through a bounded-memory production view."""
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
    train_data, val_data, test_data, all_specs, max_node = _scan_lazy_collections(
        cfg, collections, telemetry,
    )
    full_data = BoundedFullData(
        cfg,
        all_specs,
        view_mode=view_mode,
        max_node=max_node,
        telemetry=telemetry,
        cache_windows=1,
    )
    telemetry.observe("after global metadata construction")
    telemetry.observe("before model construction")
    full_data.loader_telemetry = telemetry.as_dict()
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
