"""Runtime DatasetViews for C6 host/network input ablations.

The module operates on one temporal window at a time and deliberately has no
``detection`` or PyG dependency: it only requires a data object with tensor
attributes such as ``src``, ``dst``, ``x_src`` and ``x_dst``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


HOST_ONLY = "host_only"
HOST_NETWORK_STRUCTURE = "host_network_structure"
HOST_NETWORK_FULL = "host_network_full"
VALID_DATASET_VIEWS = frozenset(
    {HOST_ONLY, HOST_NETWORK_STRUCTURE, HOST_NETWORK_FULL}
)

# Single, documented THEIA mapping.  Its authoritative project definition is
# config.ntype2id: subject=1, file=2, netflow=3.  C3 persists explicit
# src_type/dst_type fields, so views never infer types from feature columns.
NETFLOW_TYPE_INDEX = 3


def _clone_data(data: Any) -> Any:
    """Return an independent copy without mutating the caller's window."""
    if hasattr(data, "clone"):
        return data.clone()
    return copy.deepcopy(data)


def _event_count(data: Any) -> int:
    if not hasattr(data, "src"):
        raise ValueError("DatasetView input must expose an event-aligned 'src' field")
    return int(data.src.shape[0])


def _filter_event_aligned_fields(data: Any, keep_mask: torch.Tensor) -> None:
    """Apply one event mask to every recognised event-aligned field.

    Tensor fields whose leading dimension equals the original event count are
    event-aligned. ``edge_index`` is the one standard column-aligned exception
    and is filtered on its second dimension. Lists and tuples of event length
    are handled as well, so newly-added C4/C5 event metadata is not silently
    left misaligned.
    """
    event_count = int(keep_mask.numel())
    indices = keep_mask.nonzero(as_tuple=False).flatten()
    store = getattr(data, "_store", None)
    items = list(store.items()) if store is not None else list(vars(data).items())
    for name, value in items:
        if name == "edge_index" and isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] == event_count:
            setattr(data, name, value.index_select(1, indices.to(value.device)))
        elif isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == event_count:
            setattr(data, name, value.index_select(0, indices.to(value.device)))
        elif isinstance(value, list) and len(value) == event_count:
            setattr(data, name, [value[i] for i in indices.tolist()])
        elif isinstance(value, tuple) and len(value) == event_count:
            setattr(data, name, tuple(value[i] for i in indices.tolist()))


def _semantic_width(features: torch.Tensor, type_onehot: Any) -> int:
    """Find the semantic prefix without guessing a trailing type width."""
    if not isinstance(type_onehot, torch.Tensor) or type_onehot.ndim != 2:
        return int(features.shape[1])
    type_dim = int(type_onehot.shape[1])
    if features.ndim != 2 or features.shape[1] < type_dim:
        return int(features.shape[1])
    if torch.equal(features[:, -type_dim:], type_onehot.to(features.device, features.dtype)):
        return int(features.shape[1] - type_dim)
    return int(features.shape[1])


def _rebuild_derived_event_features(data: Any) -> None:
    """Keep msg/edge_feats consistent after endpoint feature masking."""
    if not (hasattr(data, "x_src") and hasattr(data, "x_dst")):
        return
    x_src, x_dst = data.x_src, data.x_dst
    edge_type = getattr(data, "edge_type", None)
    if hasattr(data, "msg") and isinstance(data.msg, torch.Tensor):
        base = torch.cat([x_src, x_dst], dim=-1)
        if isinstance(edge_type, torch.Tensor) and data.msg.shape[1] == base.shape[1] + edge_type.shape[1]:
            data.msg = torch.cat([base, edge_type], dim=-1)
        elif data.msg.shape[1] == base.shape[1]:
            data.msg = base
    if hasattr(data, "edge_feats") and isinstance(data.edge_feats, torch.Tensor) and isinstance(edge_type, torch.Tensor):
        if data.edge_feats.shape[1] == data.msg.shape[1]:
            data.edge_feats = data.msg.clone()
        elif data.edge_feats.shape[1] == edge_type.shape[1] + data.msg.shape[1]:
            data.edge_feats = torch.cat([edge_type, data.msg], dim=-1)


def _zero_netflow_semantics(data: Any) -> None:
    if not (hasattr(data, "src_type") and hasattr(data, "dst_type")):
        raise ValueError("host_network_structure requires explicit src_type and dst_type")
    for feature_name, type_name in (("x_src", "src_type_onehot"), ("x_dst", "dst_type_onehot")):
        if not hasattr(data, feature_name):
            continue
        features = getattr(data, feature_name)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            continue
        types = getattr(data, type_name, None)
        semantic_width = _semantic_width(features, types)
        if semantic_width == 0:
            continue
        mask = getattr(data, "src_type" if feature_name == "x_src" else "dst_type") == NETFLOW_TYPE_INDEX
        if mask.any():
            features[mask, :semantic_width] = 0
    _rebuild_derived_event_features(data)


def apply_dataset_view(data: Any, mode: str) -> Any:
    """Apply one canonical DatasetView and return a non-mutating copy.

    ``host_only`` removes every event with a netflow endpoint and rebuilds its
    local event indices. ``host_network_structure`` retains topology and type
    one-hot features while zeroing only the netflow semantic prefix. The full
    mode is an identity copy for baseline compatibility.
    """
    if not isinstance(mode, str) or mode not in VALID_DATASET_VIEWS:
        raise ValueError(
            f"Invalid dataset view {mode!r}. Allowed values: {sorted(VALID_DATASET_VIEWS)}"
        )
    result = _clone_data(data)
    event_count = _event_count(result)
    if mode == HOST_NETWORK_FULL:
        return result
    if mode == HOST_NETWORK_STRUCTURE:
        _zero_netflow_semantics(result)
        return result

    if not (hasattr(result, "src_type") and hasattr(result, "dst_type")):
        raise ValueError("host_only requires explicit src_type and dst_type")
    keep_mask = (result.src_type != NETFLOW_TYPE_INDEX) & (result.dst_type != NETFLOW_TYPE_INDEX)
    _filter_event_aligned_fields(result, keep_mask)
    result.local_event_index = torch.arange(
        int(keep_mask.sum().item()), dtype=torch.long, device=result.src.device
    )
    return result
