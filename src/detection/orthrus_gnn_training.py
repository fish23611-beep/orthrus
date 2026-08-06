import logging
from time import perf_counter as timer

import torch.nn as nn
import wandb

from encoders import OrthrusEncoder
from model import MSTCOrthrus
from config import *
from data_utils import *
from factory import *


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


def main(cfg):
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    os.makedirs(gnn_models_dir, exist_ok=True)
    device = get_device(cfg)
    
    # Reset the peak memory usage counter
    if device == torch.device("cuda"):
        torch.cuda.reset_peak_memory_stats(device=device)

    train_data, _, _, full_data, max_node_num = load_all_datasets(cfg)

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
    tot_loss = 0.0
    epoch_times = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
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
            # log(f"Loss {loss:4f}")
            g.to("cpu")

        tot_loss /= len(train_data)
        log(f'GNN training loss Epoch: {epoch:02d}, Loss: {tot_loss:.4f}')
        
        epoch_times.append(timer() - start)
        
        # Log peak CUDA memory usage
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)  # Convert to GB
        log(f'Peak CUDA memory usage Epoch {epoch}: {peak_memory:.2f} GB')
        
        wandb.log({
            "train_epoch": epoch,
            "train_loss": round(tot_loss, 4),
            "peak_cuda_memory_GB": round(peak_memory, 2),
        })

        # Check points
        if cfg._test_mode or epoch % 1 == 0:
            model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
            save_model(model, model_path)
            
    wandb.log({
        "train_epoch_time": round(np.mean(epoch_times), 2),
    })


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
