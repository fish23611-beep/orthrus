import gc
import os
from tqdm import tqdm

from encoders import OrthrusEncoder
from model import MSTCOrthrus
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch

from mstc.experiment_utils import dump_environment, events_per_second, peak_cpu_memory_mb, update_runtime, update_runtime_nested
from run_metadata import _is_valid_path


def _cleanup_checkpoint_cuda(model, device):
    """
    Explicitly break CUDA-holding references and reclaim GPU memory at a
    checkpoint boundary.

    The model hierarchy (Orthrus → encoder → neighbor_loader/graph_reindexer, or
    MSTCOrthrus → encoder → neighbor_loader/graph_reindexer) contains CUDA
    tensors that Python's reference counting alone cannot free because of
    internal reference cycles.  After ``del model`` the object remains alive
    until the next garbage-collection pass, causing allocated CUDA memory to
    grow monotonically across checkpoint iterations.

    This function explicitly nulls every known CUDA-carrying attribute so that
    ``del model`` + ``gc.collect()`` + ``torch.cuda.synchronize()`` +
    ``torch.cuda.empty_cache()`` can reclaim the memory deterministically.

    No-op on CPU-only devices.
    """
    if device is None or device.type == "cpu":
        return

    # Null CUDA tensors inside the encoder (shared across Orthrus / MSTCOrthrus)
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        # LastNeighborLoader / MultiScaleNeighborLoader: CUDA tensors in
        # self.neighbors, self.e_id, self._assoc
        nl = getattr(encoder, "neighbor_loader", None)
        if nl is not None:
            for attr in ("neighbors", "e_id", "_assoc"):
                t = getattr(nl, attr, None)
                if t is not None and t.is_cuda:
                    setattr(nl, attr, torch.empty(0, device=t.device))

        # GraphReindexer may hold per-batch CUDA reindexing tensors
        reindexer = getattr(encoder, "graph_reindexer", None)
        if reindexer is not None:
            for attr_name in dir(reindexer):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(reindexer, attr_name)
                except AttributeError:
                    continue
                if isinstance(attr, torch.Tensor) and attr.is_cuda:
                    setattr(reindexer, attr_name, torch.empty(0, device=attr.device))

    # Orthrus: last_h_storage and last_h_non_empty_nodes are CUDA tensors
    for attr in ("last_h_storage", "last_h_non_empty_nodes"):
        t = getattr(model, attr, None)
        if t is not None and isinstance(t, torch.Tensor) and t.is_cuda:
            setattr(model, attr, torch.empty(0, device=t.device))

    # Force a synchronous GC pass so cyclic Python objects are deallocated
    # before we call empty_cache.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()


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
    return {"processed_event_count": event_count, "test_seconds": end - start}


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


def _run_scoped_metadata_dir(cfg):
    """Resolve the run-scoped metadata directory for time statistics.

    Mirrors the logic in orthrus_gnn_training._run_scoped_metadata_dir.
    Priority:
    1. cfg._run_dir/metadata/  (canonical: run-isolated)
    2. cfg._metadata_dir       (legacy fallback: shared preprocessing path)
    """
    run_dir = getattr(cfg, "_run_dir", None)
    if _is_valid_path(run_dir):
        return os.path.join(os.fspath(run_dir), "metadata")
    metadata_dir = getattr(cfg, "_metadata_dir", None)
    if _is_valid_path(metadata_dir):
        return metadata_dir
    return None


def _load_or_fit_time_gap_statistics(cfg, train_data):
    """Load persisted statistics, or fit train-only in full-pipeline fallback."""
    metadata_dir = _run_scoped_metadata_dir(cfg)
    time_stats_path = (
        os.path.join(metadata_dir, "time_statistics.json")
        if metadata_dir
        else None
    )

    if time_stats_path and os.path.exists(time_stats_path):
        statistics = TimeGapStatistics.load(time_stats_path)
        log(f"Loaded time_statistics.json from {time_stats_path}")
        return statistics

    if _is_detection_only_mode(cfg):
        raise FileNotFoundError(
            "detection_only mode: time_statistics.json not found at "
            f"{time_stats_path}. Training must save the artifact first."
        )

    log(
        f"time_statistics.json not found at {time_stats_path}; fitting from "
        "train_data only (full_pipeline backward-compatible fallback)"
    )
    return fit_time_gap_statistics(train_data)


def _get_metadata_cache(cfg):
    """Get or create MetadataCache instance from cfg."""
    metadata_dir = getattr(cfg, "_metadata_dir", None)
    if not _is_valid_path(metadata_dir):
        return None
    from mstc.metadata_cache import MetadataCache
    return MetadataCache(metadata_dir)


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

    time_gap_statistics = None
    if requires_time_gap_statistics(cfg):
        time_gap_statistics = _load_or_fit_time_gap_statistics(cfg, train_data)

    # For each model trained at a given epoch, we test. C8-B may inject one
    # explicit inference checkpoint (a checkpoint directory or checkpoint file).
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    inference_checkpoint = getattr(cfg, "_inference_checkpoint", None)
    if isinstance(inference_checkpoint, str) and inference_checkpoint:
        checkpoint_path = os.fspath(inference_checkpoint)
        all_trained_models = [(os.path.basename(checkpoint_path.rstrip(os.sep)) or "checkpoint", checkpoint_path)]
    elif cfg._from_weights:
        all_trained_models = [("model_epoch_1", os.path.join(gnn_models_dir, "model_epoch_1"))]
    else:
        all_trained_models = [(name, os.path.join(gnn_models_dir, name)) for name in listdir_sorted(gnn_models_dir)]
    runtime_dir = getattr(cfg, "_run_dir", None)
    if not _is_valid_path(runtime_dir):
        runtime_dir = os.path.dirname(gnn_models_dir)
    dump_environment(cfg, runtime_dir)
    if hasattr(full_data, "loader_telemetry"):
        update_runtime_nested(runtime_dir, "dataset_loader", "testing", full_data.loader_telemetry)

    device = get_device(cfg)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)
    testing_started = time.perf_counter()
    processed_events = 0

    for trained_model, model_path in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        model = build_model(
            data_sample=test_data[0], device=device, cfg=cfg, max_node_num=max_node_num,
            time_gap_statistics=time_gap_statistics,
        )
        model = load_model(model, model_path)

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
                result = test(
                    data=g,
                    full_data=full_data,
                    model=model,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    cfg=cfg,
                    device=device,
                )
                if isinstance(result, dict):
                    processed_events += int(result.get("processed_event_count", 0))
                g.to("cpu")

        # ------------------------------------------------------------------ #
        # Checkpoint-boundary CUDA lifecycle cleanup:
        #
        #   Problem: del model is insufficient because Python reference cycles
        #   (e.g. model ↔ encoder ↔ neighbor_loader ↔ graph_reindexer) keep
        #   the old checkpoint's CUDA tensors alive until the next gc pass.
        #   Over 6 checkpoints this accumulates several hundred MB of unreleased
        #   CUDA memory, eventually triggering OOM on model_epoch_2 test.
        #
        #   Fix: explicitly break all known CUDA-holding references, invoke the
        #   garbage collector to reclaim cyclic Python objects, and synchronise
        #   the CUDA allocator so the freed memory is returned to the pool.
        #
        #   This does NOT change the replay protocol (reset → replay → val →
        #   test), does NOT modify model parameters or hyperparameters, and does
        #   NOT affect the val/test order or the no-reset-between-val-and-test
        #   invariant.
        # ------------------------------------------------------------------ #
        _cleanup_checkpoint_cuda(model, device)

        del model

    test_seconds = time.perf_counter() - testing_started
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    update_runtime(runtime_dir, "testing", {
        "test_seconds": test_seconds,
        "events_per_second": events_per_second(processed_events, test_seconds),
        "processed_event_count": processed_events,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "peak_cpu_memory_mb": peak_cpu_memory_mb(),
    })


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
