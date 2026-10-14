import logging
from time import perf_counter as timer

import torch.nn as nn

from encoders import OrthrusEncoder
from model import MSTCOrthrus
from config import *
from data_utils import *
from factory import *
from mstc.experiment_utils import dump_environment, events_per_second, peak_cpu_memory_mb, update_runtime, update_runtime_nested
from wandb_control import wandb_log
from runtime.runtime_profiler import get_profiler


def train(data,
          full_data,
          model,
          optimizer,
          cfg
          ):
    model.train()

    losses = []
    profiler = get_profiler()
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch, batch_token in profiler.profile_batches(batch_loader):
        profiler.add_counter("training.batches")
        batch_events = int(batch.edge_index.size(1))
        profiler.add_counter("training.events", batch_events)
        profiler.record_basic_batch_stat("batch_events", batch_events)
        profiler.record_batch_stat("batch_events", batch_events)
        with profiler.timer("training.T_zero_grad", detail=True,
                            parent="training.T_total_batch"):
            optimizer.zero_grad()

        with profiler.timer("training.T_forward", detail=True, parent="training.T_total_batch"), \
                profiler.cuda_timer("training.T_forward_cuda_interval", detail=True, parent="training.T_total_batch"):
            outputs = model(batch, full_data)
        with profiler.timer("training.T_loss_extract", detail=True,
                            parent="training.T_total_batch"):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            sync_started = timer() if profiler.detail_active else None
            log(
                "MSTC batch loss={:.4f}, type={:.4f}, time={:.4f}, src={:.4f}, dst={:.4f}".format(
                    outputs["loss"].item(),
                    outputs["loss_type"].item(),
                    outputs["loss_time"].item(),
                    outputs["loss_time_src"].item(),
                    outputs["loss_time_dst"].item(),
                )
            )
            if sync_started is not None:
                profiler.record(
                    "training.T_logging_items", timer() - sync_started,
                    parent="training.T_total_batch",
                )

        with profiler.timer("training.T_backward", detail=True, parent="training.T_total_batch"), \
                profiler.cuda_timer("training.T_backward_cuda_interval", detail=True, parent="training.T_total_batch"):
            loss.backward()
        with profiler.timer("training.T_optimizer", detail=True, parent="training.T_total_batch"), \
                profiler.cuda_timer("training.T_optimizer_cuda_interval", detail=True, parent="training.T_total_batch"):
            optimizer.step()
        with profiler.timer("training.T_loss_item", detail=True,
                            parent="training.T_total_batch"):
            losses.append(loss.item())
        profiler.end_batch(batch_token, "training.T_total_batch")
    return np.mean(losses)


def _runtime_dir(cfg, fallback_dir):
    run_dir = getattr(cfg, "_run_dir", None)
    return run_dir if isinstance(run_dir, (str, os.PathLike)) and os.fspath(run_dir) else os.path.dirname(fallback_dir)


def _event_count(data):
    for field in ("t", "src", "dst"):
        value = getattr(data, field, None)
        if value is not None:
            return int(value.numel())
    return 0


def _receipt_scalar(value):
    """Return JSON-safe window metadata without inspecting tensor contents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _run_scoped_metadata_dir(cfg):
    """Resolve the run-scoped metadata directory for time statistics.

    Priority:
    1. cfg._run_dir/metadata/  (canonical: run-isolated)
    2. cfg._metadata_dir       (legacy fallback: shared preprocessing path)

    The run-scoped path is preferred because TimeGapStatistics are fitted from
    the current training split (which varies with dataset_view mode), so different
    experiments with different train_data views must not share one artifact.
    """
    run_dir = getattr(cfg, "_run_dir", None)
    if isinstance(run_dir, (str, os.PathLike)) and os.fspath(run_dir):
        return os.path.join(os.fspath(run_dir), "metadata")
    # Legacy fallback: use shared preprocessing metadata dir
    metadata_dir = getattr(cfg, "_metadata_dir", None)
    if metadata_dir:
        return metadata_dir
    return None


def _fit_and_save_time_gap_statistics(train_data, cfg):
    """Fit from train data only and persist to the run-scoped metadata directory."""
    metadata_dir = _run_scoped_metadata_dir(cfg)
    if not metadata_dir:
        raise ValueError(
            "TimeGapStatistics requires either cfg._run_dir or cfg._metadata_dir "
            "so training can save metadata/time_statistics.json."
        )

    # Extract scale_quantiles from config if available
    scale_quantiles = getattr(
        getattr(getattr(cfg, "detection", None), "gnn_training", None),
        "encoder", None
    )
    scale_quantiles = getattr(
        getattr(scale_quantiles, "context", None), "multiscale", None
    )
    scale_quantiles = getattr(scale_quantiles, "scale_quantiles", None)

    statistics = fit_time_gap_statistics(
        train_data,
        scale_quantiles=scale_quantiles,
    )
    time_stats_path = os.path.join(metadata_dir, "time_statistics.json")
    os.makedirs(metadata_dir, exist_ok=True)
    statistics.save(time_stats_path)
    log(f"Saved time_statistics.json to {time_stats_path}")
    return statistics


def main(cfg):
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)
    runtime_dir = _runtime_dir(cfg, gnn_models_dir)
    dump_environment(cfg, runtime_dir)
    device = get_device(cfg)
    profiler = get_profiler()

    # CUDA APIs are guarded so CPU-only training remains supported.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    training_started = timer()

    train_data, _, _, full_data, max_node_num = load_all_datasets(cfg)
    if hasattr(full_data, "loader_telemetry"):
        update_runtime_nested(runtime_dir, "dataset_loader", "training", full_data.loader_telemetry)

    time_gap_statistics = None
    if requires_time_gap_statistics(cfg):
        time_gap_statistics = _fit_and_save_time_gap_statistics(train_data, cfg)

    with profiler.timer("loading.T_model_construction"):
        model = build_model(
            data_sample=train_data[0], device=device, cfg=cfg, max_node_num=max_node_num,
            time_gap_statistics=time_gap_statistics,
        )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

    num_epochs = 1 if cfg._from_weights else cfg.detection.gnn_training.num_epochs
    start_epoch = 1
    resume_path = getattr(cfg.detection.gnn_training, "resume_checkpoint", None)
    if resume_path:
        resume_state = load_training_checkpoint(
            model, resume_path, optimizer=optimizer, cfg=cfg, map_location=device,
        )
        if not resume_state["complete"] or resume_state["epoch"] is None:
            raise ValueError("legacy model-only checkpoints cannot be used for training resume")
        start_epoch = int(resume_state["epoch"]) + 1
        log(f"Resuming training from epoch {start_epoch}")

    tot_loss = 0.0
    epoch_times = []
    processed_events = 0
    completed_windows = []
    training_batch_size = int(cfg.detection.gnn_training.encoder.batch_size)

    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Training"):
        start = timer()

        # Before each epoch, reset the coupled encoder/time state for MSTC.
        if isinstance(model, MSTCOrthrus):
            model.reset_state()
        elif isinstance(model.encoder, OrthrusEncoder):
            model.encoder.reset_state()

        tot_loss = 0
        for window_index, g in enumerate(train_data):
            window_event_count = _event_count(g)
            window_id = _receipt_scalar(getattr(g, "window_id", window_index))
            profiler.set_context(
                epoch=epoch,
                window_index=window_index,
                window_id=getattr(g, "window_id", window_index),
            )
            profiler.add_counter("training.windows")
            with profiler.timer("training.T_h2d"), \
                    profiler.cuda_timer("training.T_h2d_cuda_interval", detail=False,
                                        parent="training.T_h2d"):
                g.to(device=device)
            loss = train(
                data=g,  # avoids alteration of the graph across epochs
                full_data=full_data,  # full list of edge messages (do not store on CPU)
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
            tot_loss += loss
            processed_events += window_event_count
            # log(f"Loss {loss:4f}")
            g.to("cpu")
            # Appended only after the complete window, including its CPU return,
            # succeeds. This is a lightweight receipt, not a timing record.
            completed_windows.append({
                "epoch": epoch,
                "window_index": window_index,
                "window_id": window_id,
                "event_count": window_event_count,
                "expected_batch_count": (
                    (window_event_count + training_batch_size - 1) // training_batch_size
                ),
            })


        tot_loss /= len(train_data)
        log(f'GNN training loss Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')
        
        epoch_times.append(timer() - start)
        
        peak_memory_mb = None
        if device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            log(f'Peak CUDA memory usage Epoch {epoch}: {peak_memory_mb:.2f} MB')

        wandb_log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "peak_cuda_memory_GB": round(peak_memory_mb / 1024, 2) if peak_memory_mb is not None else None,
        })

        # Keep legacy state_dict.pkl for inference, plus complete resume state.
        if cfg._test_mode or epoch % 1 == 0:
            model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            with profiler.timer("training.T_checkpoint"):
                save_model(
                    model, model_path, neigh_loader=False, optimizer=optimizer,
                    epoch=epoch, cfg=cfg,
                )

    total_train_seconds = timer() - training_started
    runtime = {
        "train_seconds_per_epoch": epoch_times,
        "mean_train_seconds_per_epoch": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        "total_train_seconds": total_train_seconds,
        "events_per_second": events_per_second(processed_events, total_train_seconds),
        "processed_event_count": processed_events,
        "peak_gpu_memory_mb": peak_memory_mb if epoch_times else None,
        "peak_cpu_memory_mb": peak_cpu_memory_mb(),
        "execution_receipt": {
            "schema_version": 1,
            "status": "completed",
            "start_epoch": start_epoch,
            "end_epoch": num_epochs,
            "batch_size": training_batch_size,
            "windows": completed_windows,
        },
    }
    update_runtime(runtime_dir, "training", runtime)
    update_runtime(runtime_dir, "model", {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    wandb_log({"train_epoch_time": round(np.mean(epoch_times), 2) if epoch_times else float("nan")})


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
