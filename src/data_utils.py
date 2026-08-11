import os

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


def load_all_datasets(cfg):
    train_data = load_data_set(cfg, path=cfg.edge_featurization.embed_edges._edge_embeds_dir, split="train")
    val_data = load_data_set(cfg, path=cfg.edge_featurization.embed_edges._edge_embeds_dir, split="val")
    test_data = load_data_set(cfg, path=cfg.edge_featurization.embed_edges._edge_embeds_dir, split="test")
    # C6-B8: transform each window before any full-data/history indices exist.
    from mstc.dataset_views import apply_dataset_view
    view_mode = cfg.dataset_view.mode
    train_data = [apply_dataset_view(data, view_mode) for data in train_data]
    val_data = [apply_dataset_view(data, view_mode) for data in val_data]
    test_data = [apply_dataset_view(data, view_mode) for data in test_data]

    all_msg, all_t, all_edge_types, all_src, all_dst = [], [], [], [], []
    max_node = -1
    for dataset in [train_data, val_data, test_data]:
        for data in dataset:
            all_msg.append(data.msg)
            all_t.append(data.t)
            all_edge_types.append(data.edge_type)
            all_src.append(data.src)
            all_dst.append(data.dst)
            if data.src.numel() > 0:
                max_node = max(max_node, torch.cat([data.src, data.dst]).max().item())

    all_msg = torch.cat(all_msg)
    all_t = torch.cat(all_t)
    all_edge_types = torch.cat(all_edge_types)
    all_src = torch.cat(all_src)
    all_dst = torch.cat(all_dst)
    full_data = Data(
        msg=all_msg,
        t=all_t,
        edge_type=all_edge_types,
        src=all_src,
        dst=all_dst,
    )
    max_node = max_node + 1 if max_node >= 0 else 0
    print(f"Max node in {cfg.dataset.name}: {max_node}")

    full_data = _inject_full_data_event_fields(train_data, val_data, test_data, full_data)

    return train_data, val_data, test_data, full_data, max_node

def load_data_set(cfg, path: str, split: str) -> list[TemporalData]:
    """
    Returns a list of time window graphs for a given `split` (train/val/test set).
    """
    if cfg._test_mode:
        split = "train"

    data_list = []
    for f in sorted(os.listdir(os.path.join(path, split))):
        filepath = os.path.join(path, split, f)
        # ORTHRUS-generated TemporalData artifacts require full pickle deserialization
        # (weights_only=False). Using trusted loader ensures PyTorch 2.6 compatibility
        # while maintaining trust boundary and type verification.
        data = load_trusted_torch_artifact(filepath, expected_type=TemporalData).to("cpu")
        data_list.append(data)

    if cfg.edge_featurization.embed_nodes.used_method.strip() == "only_type":
        data_list = extract_msg_node_type_only(data_list, cfg)
    else:
        data_list = extract_msg_from_data(data_list, cfg)

    for g in data_list:
        _inject_event_indices(g, cfg)

    return data_list

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

def custom_temporal_data_loader(data: TemporalData, batch_size: int, *args, **kwargs):
    """
    A simple `TemporalDataLoader` which also update the edge_index with the
    sampled edges of size `batch_size`. By default, only attributes of shape (E, d)
    are updated, `edge_index` is thus not updated automatically.
    """
    loader = TemporalDataLoader(data, batch_size=batch_size, *args, **kwargs)
    for batch in loader:
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
