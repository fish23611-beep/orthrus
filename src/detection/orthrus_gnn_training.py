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


def train(data,
          full_data,
          model,
          optimizer,
          cfg
          ):
    model.train()

    losses = []
    batch_loader = batch_loader_factory(cfg, data, model.graph_reindexer)

    for batch in batch_loader:
        optimizer.zero_grad()

        outputs = model(batch, full_data)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            log(
                "MSTC batch loss={:.4f}, type={:.4f}, time={:.4f}, src={:.4f}, dst={:.4f}".format(
                    outputs["loss"].item(),
                    outputs["loss_type"].item(),
                    outputs["loss_time"].item(),
                    outputs["loss_time_src"].item(),
                    outputs["loss_time_dst"].item(),
                )
            )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
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


def main(cfg):
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)
    runtime_dir = _runtime_dir(cfg, gnn_models_dir)
    dump_environment(cfg, runtime_dir)
    device = get_device(cfg)

    # CUDA APIs are guarded so CPU-only training remains supported.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    training_started = timer()

    train_data, _, _, full_data, max_node_num = load_all_datasets(cfg)
    if hasattr(full_data, "loader_telemetry"):
        update_runtime_nested(runtime_dir, "dataset_loader", "training", full_data.loader_telemetry)

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
    for epoch in tqdm(range(start_epoch, num_epochs + 1), desc="Training"):
        start = timer()

        # Before each epoch, reset the coupled encoder/time state for MSTC.
        if isinstance(model, MSTCOrthrus):
            model.reset_state()
        elif isinstance(model.encoder, OrthrusEncoder):
            model.encoder.reset_state()

        tot_loss = 0
        for g in train_data:
            g.to(device=device)
            loss = train(
                data=g,  # avoids alteration of the graph across epochs
                full_data=full_data,  # full list of edge messages (do not store on CPU)
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
            tot_loss += loss
            processed_events += _event_count(g)
            # log(f"Loss {loss:4f}")
            g.to("cpu")

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
            save_model(model, model_path, neigh_loader=False, optimizer=optimizer, epoch=epoch, cfg=cfg)

    total_train_seconds = timer() - training_started
    runtime = {
        "train_seconds_per_epoch": epoch_times,
        "mean_train_seconds_per_epoch": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        "total_train_seconds": total_train_seconds,
        "events_per_second": events_per_second(processed_events, total_train_seconds),
        "processed_event_count": processed_events,
        "peak_gpu_memory_mb": peak_memory_mb if epoch_times else None,
        "peak_cpu_memory_mb": peak_cpu_memory_mb(),
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
