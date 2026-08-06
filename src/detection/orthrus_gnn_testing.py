from tqdm import tqdm

from encoders import OrthrusEncoder
from model import MSTCOrthrus
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch


@torch.no_grad()
def test(
        data,
        full_data,
        model,
        nodeid2msg,
        split,
        model_epoch_file,
        cfg,
        device,
):
    model.eval()

    time_with_loss = {}  # key: time锛? value锛?the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()

    # NOTE: warning, this may reindexes the graph
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        outputs = model(batch, full_data, inference=True)
        is_mstc = isinstance(outputs, dict)
        each_edge_loss = outputs["score_raw"] if is_mstc else outputs
        tot_loss += each_edge_loss.sum().item()

        # If the graph has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index

        num_events = each_edge_loss.shape[0]
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        edge_type_indices = getattr(batch, "edge_type_index", edge_types - 1)
        for i in range(num_events):
            srcnode = int(edge_index[0, i])
            dstnode = int(edge_index[1, i])

            # Get node messages based on include_node_messages config
            include_msgs = getattr(getattr(cfg, "testing", None), "include_node_messages", True)
            if include_msgs and nodeid2msg:
                srcmsg = nodeid2msg.get(srcnode, f"node:{srcnode}")
                dstmsg = nodeid2msg.get(dstnode, f"node:{dstnode}")
            else:
                srcmsg = f"node:{srcnode}"
                dstmsg = f"node:{dstnode}"

            t_var = int(batch.t[i])
            edge_type_idx = edge_types[i].item()
            edge_type = rel2id[edge_type_idx]
            score_raw = each_edge_loss[i]
            event_index = int(batch.global_event_index[i]) if hasattr(batch, "global_event_index") else event_count + i
            src_type = int(batch.src_type[i]) if hasattr(batch, "src_type") else None
            dst_type = int(batch.dst_type[i]) if hasattr(batch, "dst_type") else None

            if is_mstc:
                loss_type = float(outputs["loss_type"][i])
                loss_time_src = float(outputs["loss_time_src"][i])
                loss_time_dst = float(outputs["loss_time_dst"][i])
                loss_time = float(outputs["loss_time"][i])
                src_time_target = int(outputs["src_time_target"][i]) if outputs["src_time_target"].numel() else None
                dst_time_target = int(outputs["dst_time_target"][i]) if outputs["dst_time_target"].numel() else None
                src_time_prediction = int(outputs["src_time_logits"][i].argmax()) if outputs["src_time_logits"].numel() else None
                dst_time_prediction = int(outputs["dst_time_logits"][i].argmax()) if outputs["dst_time_logits"].numel() else None
            else:
                loss_type = float(score_raw)
                loss_time_src = loss_time_dst = loss_time = 0.0
                src_time_target = dst_time_target = None
                src_time_prediction = dst_time_prediction = None

            temp_dic = {
                'event_index': event_index,
                'time': t_var,
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'src_type': src_type,
                'dst_type': dst_type,
                'edge_type': edge_type,
                'edge_type_index': int(edge_type_indices[i]),
                'loss': float(score_raw),
                'loss_type': loss_type,
                'loss_time_src': loss_time_src,
                'loss_time_dst': loss_time_dst,
                'loss_time': loss_time,
                'score_raw': float(score_raw),
                'src_time_target': src_time_target,
                'dst_time_target': dst_time_target,
                'src_time_prediction': src_time_prediction,
                'dst_time_prediction': dst_time_prediction,
            }
            edge_list.append(temp_dic)

        event_count += num_events
    tot_loss /= event_count

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list[-1]["time"])

    end = time.perf_counter()
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(edge_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

    # log(
    #     f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')


@torch.no_grad()
def _replay_train_history(model, train_data, full_data, cfg, device):
    """
    Replay the training data through the model to build temporal history.

    Runs the model on each training graph in eval/no_grad mode to populate
    the neighbor loader history WITHOUT computing or storing gradients.
    After this, the model can directly process val/test 鈥?they will see
    the full training-set history.

    This replaces the old approach of loading a serialized neighbor_loader from
    a checkpoint, which saved the LAST-EPOCH history only and required
    O(num_nodes 脳 neighbor_size) disk space.
    """
    model.eval()
    for g in train_data:
        g.to(device=device)
        try:
            batch_loader = batch_loader_factory(cfg, g, model.graph_reindexer)
            for batch in batch_loader:
                # Forward pass only 鈥?builds neighbor-loader history via insert(src, dst)
                # Signature matches model(batch, full_data, inference=True) used in test()
                model(batch, full_data, inference=True)
        finally:
            g.to("cpu")


def _is_detection_only_mode(cfg) -> bool:
    """Check if pipeline mode is detection_only."""
    return (
        hasattr(cfg, "pipeline")
        and hasattr(cfg.pipeline, "mode")
        and cfg.pipeline.mode == "detection_only"
    )


def _get_metadata_cache(cfg):
    """Get or create MetadataCache instance from cfg."""
    if not hasattr(cfg, "_metadata_dir") or not cfg._metadata_dir:
        return None
    from mstc.metadata_cache import MetadataCache
    return MetadataCache(cfg._metadata_dir)


def _load_nodeid2msg(cfg) -> dict:
    """
    Load node ID to message mapping with cache-first strategy.

    Priority:
    1. nodeid2msg.pkl cache
    2. node_metadata.pkl (derive display field)
    3. full_pipeline: DB fallback
    4. detection_only: raise error
    """
    cache = _get_metadata_cache(cfg)
    include_msgs = getattr(getattr(cfg, "testing", None), "include_node_messages", True)

    # If include_node_messages is False, skip loading and return empty
    if not include_msgs:
        return {}

    # Try nodeid2msg cache first
    if cache is not None and cache.has_nodeid2msg():
        try:
            return cache.load_nodeid2msg()
        except Exception:
            pass  # Fall through to next option

    # Try node_metadata cache
    if cache is not None and cache.has_node_metadata():
        try:
            return cache.derive_nodeid2msg_from_metadata()
        except Exception:
            pass  # Fall through to next option

    # Cache miss - check mode
    if _is_detection_only_mode(cfg):
        if cache is not None:
            missing = cache.validate_required(["nodeid2msg"])
            raise FileNotFoundError(
                f"detection_only mode: required caches missing: {missing}. "
                f"Set testing.include_node_messages=false to skip node messages."
            )
        else:
            raise FileNotFoundError(
                "detection_only mode: no metadata cache configured and include_node_messages is true. "
                "Set testing.include_node_messages=false to skip node messages."
            )

    # full_pipeline: use DB fallback
    cur, _ = init_database_connection(cfg)
    nodeid2msg = gen_nodeid2msg(cur=cur)
    return {k: str(v) for k, v in nodeid2msg.items()}


def main(cfg):
    # Load node messages with cache-first strategy
    nodeid2msg = _load_nodeid2msg(cfg)
    nodeid2msg = {k: str(v) for k, v in nodeid2msg.items()}  # pre-compute because it's too slow in main loop

    train_data, val_data, test_data, full_data, max_node_num = load_all_datasets(cfg)

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = ["model_epoch_1"] if cfg._from_weights else listdir_sorted(gnn_models_dir)

    device = get_device(cfg)

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        torch.cuda.empty_cache()
        time_gap_statistics = None
        model_variant = getattr(getattr(cfg, "model", None), "variant", None)
        time_gap_cfg = getattr(
            getattr(getattr(cfg, "detection", None), "gnn_training", None),
            "decoder",
            None,
        )
        time_gap_enabled = getattr(getattr(time_gap_cfg, "time_gap", None), "enabled", False) if time_gap_cfg is not None else False
        if model_variant == "mstc" and time_gap_enabled:
            time_gap_statistics = fit_time_gap_statistics(train_data)
        model = build_model(
            data_sample=test_data[0], device=device, cfg=cfg, max_node_num=max_node_num,
            time_gap_statistics=time_gap_statistics,
        )
        model = load_model(model, os.path.join(gnn_models_dir, trained_model))

        if cfg._from_weights:
            model.load_state_dict(torch.load(os.path.join(cfg._from_weights_path, f"{cfg.dataset.name}.pkl")))

        # ------------------------------------------------------------------ #
        # Correct replay protocol (per checkpoint, before val+test):
        #   1. reset_state()  鈥?clears any stale history
        #   2. replay train   鈥?rebuilds history from scratch
        #   3. val (no reset) 鈥?sees full train history
        #   4. test (no reset) 鈥?sees train + val history
        # Val and test must NOT reset/replay between them.
        # ------------------------------------------------------------------ #
        if isinstance(model, MSTCOrthrus):
            model.reset_state()
            log(f"    [replay] reset_state() called for {trained_model}")
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'reset_state'):
            model.encoder.reset_state()
            log(f"    [replay] reset_state() called for {trained_model}")

        _replay_train_history(model, train_data, full_data, cfg, device)
        log(f"    [replay] train history rebuilt for {trained_model}")

        # Process val then test 鈥?NO reset/replay between them
        for graphs, split in [
            (val_data, "val"),
            (test_data, "test"),
        ]:
            log(f"    Testing {split} set...")
            for g in tqdm(graphs, desc=f"{split} set with {trained_model}"):
                g.to(device=device)
                test(
                    data=g,
                    full_data=full_data,
                    model=model,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    cfg=cfg,
                    device=device,
                )
                g.to("cpu")

        del model


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
