import torch
import torch.nn as nn
import torch.nn.functional as F

from mstc.time_gap import TimeGapStatistics
from mstc.history_store import HistoryStore
from mstc.multiscale_sampler import MultiScaleNeighborLoader
from mstc.multiscale_encoder import MultiScaleOrthrusEncoder

from provnet_utils import *
from config import *
from model import *
from encoders import *
from decoders import *
from data_utils import *
from temporal import LastAggregator, LastNeighborLoader


def build_model(data_sample, device, cfg, max_node_num, time_gap_statistics=None):
    """Build the selected baseline or C4 MSTC model."""
    msg_dim, edge_dim, in_dim = get_dimensions_from_data_sample(data_sample)

    graph_reindexer = GraphReindexer(num_nodes=max_node_num, device=device)
    encoder = encoder_factory(
        cfg, msg_dim=msg_dim, in_dim=in_dim, edge_dim=edge_dim,
        graph_reindexer=graph_reindexer, device=device, max_node_num=max_node_num,
        time_gap_statistics=time_gap_statistics,
    )
    decoders = decoder_factory(cfg, in_dim=in_dim, device=device, max_node_num=max_node_num)
    return model_factory(
        encoder, decoders, cfg, in_dim=in_dim, graph_reindexer=graph_reindexer,
        device=device, max_node_num=max_node_num, time_gap_statistics=time_gap_statistics,
    )

def model_factory(encoder, decoders, cfg, in_dim, graph_reindexer, device, max_node_num, time_gap_statistics=None):
    variant = getattr(getattr(cfg, "model", None), "variant", "orthrus_baseline")
    if variant == "orthrus_baseline":
        return Orthrus(
            encoder=encoder,
            decoders=decoders,
            num_nodes=max_node_num,
            device=device,
            in_dim=in_dim,
            out_dim=cfg.detection.gnn_training.node_out_dim,
            use_contrastive_learning="predict_edge_contrastive" in cfg.detection.gnn_training.decoder.used_methods,
            graph_reindexer=graph_reindexer,
        ).to(device)
    if variant == "mstc":
        return MSTCOrthrus(
            encoder=encoder,
            edge_decoder=decoders[0] if decoders else None,
            time_gap_decoder=time_gap_decoder_factory(cfg),
            time_gap_statistics=time_gap_statistics,
            lambda_time=cfg.detection.gnn_training.decoder.time_gap.lambda_time,
            graph_reindexer=graph_reindexer,
        ).to(device)
    raise ValueError(f"Unknown model.variant: {variant}")

def _build_graph_encoder(cfg, in_dim, edge_dim, temporal_dim, node_hid_dim, node_out_dim, num_heads, activation_fn, dropout):
    """Build the shared GraphTransformer used by both recent and multiscale encoders."""
    return GraphTransformer(
        in_dim=in_dim,
        hid_dim=node_hid_dim,
        out_dim=node_out_dim,
        edge_dim=edge_dim or None,
        activation=activation_fn,
        dropout=dropout,
        num_heads=num_heads,
    )


def _build_graphsage_encoder(cfg, in_dim, edge_dim, temporal_dim, node_hid_dim, node_out_dim, num_heads, activation_fn, dropout):
    """Build the shared GraphSAGEBackbone used by both recent and multiscale encoders.

    GraphSAGE does not consume edge features, so the ``edge_dim`` argument is
    accepted but ignored. ``num_heads`` is also unused because GraphSAGE has no
    multi-head attention; it is kept in the signature for parity with
    :func:`_build_graph_encoder`.
    """
    del edge_dim, num_heads
    return GraphSAGEBackbone(
        in_dim=in_dim,
        hid_dim=node_hid_dim,
        out_dim=node_out_dim,
        activation=activation_fn,
        dropout=dropout,
    )



def _build_semantic_mlp_encoder(in_dim, node_hid_dim, node_out_dim, activation_fn, dropout):
    """Build the stateless current-event Semantic MLP encoder."""
    return SemanticMLPEncoder(
        SemanticMLPBackbone(
            in_dim=in_dim,
            hid_dim=node_hid_dim,
            out_dim=node_out_dim,
            activation=activation_fn,
            dropout=dropout,
        )
    )


def _resolve_backbone(cfg):
    """Resolve the encoder backbone name from cfg, defaulting to 'graph_transformer'."""
    encoder_cfg = getattr(getattr(cfg, "detection", None), "gnn_training", None)
    encoder_cfg = getattr(encoder_cfg, "encoder", None)
    backbone = getattr(encoder_cfg, "backbone", None)
    if backbone is None:
        return "graph_transformer"
    backbone = str(backbone).strip().lower()
    if backbone in ("graph_transformer", "graphsage", "semantic_mlp"):
        return backbone
    raise ValueError(
        f"Unknown encoder.backbone={backbone!r}. "
        "Expected 'graph_transformer', 'graphsage', or 'semantic_mlp'."
    )


def encoder_factory(cfg, msg_dim, in_dim, edge_dim, graph_reindexer, device, max_node_num, time_gap_statistics=None):
    node_hid_dim = cfg.detection.gnn_training.node_hid_dim
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    temporal_dim = cfg.detection.gnn_training.encoder.temporal_dim
    dropout = cfg.detection.gnn_training.encoder.graph_attention.dropout
    activation_str = cfg.detection.gnn_training.encoder.graph_attention.activation
    num_heads = cfg.detection.gnn_training.encoder.graph_attention.num_heads
    use_node_feats_in_gnn = cfg.detection.gnn_training.encoder.use_node_feats_in_gnn

    # Edge dimension
    edge_dim_agg = 0
    edge_features = list(map(lambda x: x.strip(), cfg.detection.gnn_training.encoder.edge_features.split(",")))
    for edge_feat in edge_features:
        if edge_feat == "edge_type":
            edge_dim_agg += cfg.dataset.num_edge_types
        elif edge_feat == "msg":
            edge_dim_agg += msg_dim
        elif edge_feat == "none":
            pass
        else:
            raise ValueError(f"Invalid edge feature {edge_feat}")

    original_in_dim = in_dim
    in_dim = temporal_dim
    activation_fn = activation_fn_factory(activation_str)

    context_mode = getattr(cfg.detection.gnn_training.encoder, "context", None)
    mode = getattr(context_mode, "mode", "recent") if context_mode else "recent"
    multiscale_enabled = getattr(context_mode, "multiscale", None) and getattr(context_mode.multiscale, "enabled", False)
    backbone = _resolve_backbone(cfg)


    if backbone == "semantic_mlp":
        if mode != "none":
            raise ValueError("backbone='semantic_mlp' requires context.mode='none'")
        return _build_semantic_mlp_encoder(
            original_in_dim, node_hid_dim, node_out_dim, activation_fn, dropout
        )

    if backbone == "graph_transformer":
        graph_encoder = _build_graph_encoder(
            cfg, in_dim, edge_dim_agg, temporal_dim, node_hid_dim, node_out_dim, num_heads, activation_fn, dropout
        )
    elif backbone == "graphsage":
        graph_encoder = _build_graphsage_encoder(
            cfg, in_dim, edge_dim_agg, temporal_dim, node_hid_dim, node_out_dim, num_heads, activation_fn, dropout
        )
        # GraphSAGE does not consume edge features; force-disable at the encoder
        # level so we do not pay the cost of gathering/building them upstream.
        edge_features = ["none"]
        edge_dim_agg = 0
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")

    if mode == "recent":
        neighbor_loader = LastNeighborLoader(max_node_num, size=cfg.detection.gnn_training.encoder.neighbor_size, device=device)
        return OrthrusEncoder(
            encoder=graph_encoder,
            neighbor_loader=neighbor_loader,
            in_dim=original_in_dim,
            temporal_dim=temporal_dim,
            use_node_feats_in_gnn=use_node_feats_in_gnn,
            graph_reindexer=graph_reindexer,
            edge_features=edge_features,
            device=device,
            num_nodes=max_node_num,
            edge_dim=edge_dim_agg,
        )

    elif mode == "multiscale":
        if not multiscale_enabled:
            raise ValueError("context.mode=multiscale requires multiscale.enabled=True")
        ms_cfg = context_mode.multiscale

        # Check for scale_boundaries_seconds (new schema) or scale_boundaries (legacy)
        scale_bounds = getattr(time_gap_statistics, "scale_boundaries_seconds", None)
        if scale_bounds is None:
            scale_bounds = getattr(time_gap_statistics, "scale_boundaries", None)
        if not scale_bounds:
            raise ValueError(
                "context.mode=multiscale requires fitted TimeGapStatistics "
                "from the training split"
            )
        # scale_boundaries_seconds are already in seconds; convert to ns
        tau_short_ns = round(scale_bounds[0] * 1_000_000_000)
        tau_medium_ns = round(scale_bounds[1] * 1_000_000_000)
        # tau_extreme_ns (Q99) is stored but NOT used as hard cutoff
        # Long scale: delta > tau_medium_ns (no upper bound)
        tau_extreme_ns = round(scale_bounds[2] * 1_000_000_000) if len(scale_bounds) > 2 else None

        history_store = HistoryStore(
            num_nodes=max_node_num,
            candidate_capacity=int(ms_cfg.candidate_capacity),
            device=str(ms_cfg.history_device),
        )
        neighbor_loader = MultiScaleNeighborLoader(
            history_store=history_store,
            tau_short_ns=tau_short_ns,
            tau_medium_ns=tau_medium_ns,
            short_budget=int(ms_cfg.neighbor_budgets[0]),
            medium_budget=int(ms_cfg.neighbor_budgets[1]),
            long_budget=int(ms_cfg.neighbor_budgets[2]),
        )
        return MultiScaleOrthrusEncoder(
            shared_graph_encoder=graph_encoder,
            neighbor_loader=neighbor_loader,
            in_dim=original_in_dim,
            temporal_dim=temporal_dim,
            node_out_dim=node_out_dim,
            use_node_feats_in_gnn=use_node_feats_in_gnn,
            edge_features=edge_features,
            device=device,
            gate_hidden_dim=int(ms_cfg.gate_hidden_dim),
            use_scale_embedding=bool(ms_cfg.use_scale_embedding),
            fusion=str(ms_cfg.fusion).strip().lower() if hasattr(ms_cfg, "fusion") else "gated",
        )

    else:
        raise ValueError(f"Unknown context.mode: {mode!r}. Expected 'recent' or 'multiscale'.")

def decoder_factory(cfg, in_dim, device, max_node_num):
    node_out_dim = cfg.detection.gnn_training.node_out_dim
    if not cfg.detection.gnn_training.decoder.predict_edge_type.enabled:
        return []

    decoders = []
    for method in map(lambda x: x.strip(), cfg.detection.gnn_training.decoder.used_methods.split(",")):
        if method == "predict_edge_type":
            def cross_entropy(x, y, inference=False, **kwargs):
                reduction = "none" if inference else "mean"
                return F.cross_entropy(x, y, reduction=reduction)

            loss_fn = cross_entropy
            activation = activation_fn_factory(cfg.detection.gnn_training.decoder.predict_edge_type.custom.activation)
            
            decoder = EdgeTypeDecoder(
                in_dim=node_out_dim,
                num_edge_types=cfg.dataset.num_edge_types,
                loss_fn=loss_fn,
                dropout=cfg.detection.gnn_training.decoder.predict_edge_type.custom.dropout,
                num_layers=cfg.detection.gnn_training.decoder.predict_edge_type.custom.num_layers,
                activation=activation,
            )
            decoders.append(decoder)
        
        else:
            raise ValueError(f"Invalid decoder {method}")
        
    return decoders

def time_gap_decoder_factory(cfg):
    """Build the optional C4 decoder without affecting baseline decoders."""
    time_cfg = cfg.detection.gnn_training.decoder.time_gap
    if not time_cfg.enabled:
        return None
    return TimeGapDecoder(
        in_dim=cfg.detection.gnn_training.node_out_dim,
        hidden_dim=time_cfg.hidden_dim,
        num_classes=time_cfg.num_classes,
    )


def requires_time_gap_statistics(cfg) -> bool:
    """Return whether the configured MSTC model consumes training time statistics."""
    if getattr(getattr(cfg, "model", None), "variant", None) != "mstc":
        return False

    training_cfg = getattr(getattr(cfg, "detection", None), "gnn_training", None)
    decoder_cfg = getattr(training_cfg, "decoder", None)
    time_gap_cfg = getattr(decoder_cfg, "time_gap", None)
    time_gap_enabled = bool(getattr(time_gap_cfg, "enabled", False))

    encoder_cfg = getattr(training_cfg, "encoder", None)
    context_cfg = getattr(encoder_cfg, "context", None)
    multiscale_enabled = (
        getattr(context_cfg, "mode", None) == "multiscale"
        and bool(getattr(getattr(context_cfg, "multiscale", None), "enabled", False))
    )
    return time_gap_enabled or multiscale_enabled


def fit_time_gap_statistics(train_data, scale_quantiles=None, time_bucket_quantiles=None):
    """Fit C4 boundaries once from chronological training data only.

    Parameters
    ----------
    train_data
        Training data list.
    scale_quantiles
        Quantile probabilities for scale boundaries (e.g., [0.5, 0.9, 0.99]).
        If None, uses default [0.5, 0.9, 0.99].
    time_bucket_quantiles
        Quantile probabilities for time bucket boundaries (e.g., [0.2, 0.4, 0.6, 0.8]).
        If None, uses default [0.2, 0.4, 0.6, 0.8].
    """
    return TimeGapStatistics(
        scale_quantiles=scale_quantiles,
        time_bucket_quantiles=time_bucket_quantiles,
    ).fit(train_data)

def batch_loader_factory(cfg, data, graph_reindexer):
    return custom_temporal_data_loader(data, batch_size=cfg.detection.gnn_training.encoder.batch_size)
    
    data = graph_reindexer.reindex_graph(data)
    return [data]

def activation_fn_factory(activation: str):
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Invalid activation function {activation}")

def optimizer_factory(cfg, parameters):
    lr = cfg.detection.gnn_training.lr
    weight_decay = cfg.detection.gnn_training.weight_decay

    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay) # TODO: parametrize

def get_dimensions_from_data_sample(data):
    msg_dim = data.msg.shape[1]
    edge_dim = data.edge_feats.shape[1] if hasattr(data, "edge_feats") else None
    in_dim = data.x_src.shape[1]
    
    return msg_dim, edge_dim, in_dim
