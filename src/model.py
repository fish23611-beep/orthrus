from provnet_utils import *
from config import *
from typing import Optional

import torch
import torch.nn as nn

from mstc.time_gap import TimeGapStatistics


class Orthrus(nn.Module):
    def __init__(self,
            encoder: nn.Module,
            decoders: list[nn.Module],
            num_nodes: int,
            in_dim: int,
            out_dim: int,
            use_contrastive_learning: bool,
            device,
            graph_reindexer,
        ):
        super(Orthrus, self).__init__()

        self.encoder = encoder
        self.decoders = nn.ModuleList(decoders)
        self.use_contrastive_learning = use_contrastive_learning
        self.graph_reindexer = graph_reindexer
        
        self.last_h_storage, self.last_h_non_empty_nodes = None, None
        if self.use_contrastive_learning:
            self.last_h_storage = torch.empty((num_nodes, out_dim), device=device)
            self.last_h_non_empty_nodes = torch.tensor([], dtype=torch.long, device=device)
        
    def forward(self, batch, full_data, inference=False):
        train_mode = not inference
        x = (batch.x_src, batch.x_dst)
        edge_index = batch.edge_index

        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data,
                inference=inference,

                edge_types= batch.edge_type
            )

            h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) \
                if isinstance(h, torch.Tensor) \
                else h
        
            if x[0].shape[0] != edge_index.shape[1]:
                x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])
            
            if self.use_contrastive_learning:
                involved_nodes = edge_index.flatten()
                self.last_h_storage[involved_nodes] = torch.cat([h_src, h_dst]).detach()
                self.last_h_non_empty_nodes = torch.cat([involved_nodes, self.last_h_non_empty_nodes]).unique()
            
            # Train mode: loss | Inference mode: edge scores
            loss_or_scores = (torch.zeros(1) if train_mode else \
                torch.zeros(edge_index.shape[1], dtype=torch.float)).to(h_src.device)
            
            for decoder in self.decoders:
                loss = decoder(
                    h_src=h_src,
                    h_dst=h_dst,
                    x=x,
                    edge_index=edge_index,
                    edge_type=batch.edge_type,
                    inference=inference,
                    last_h_storage=self.last_h_storage,
                    last_h_non_empty_nodes=self.last_h_non_empty_nodes,
                )
                if loss.numel() != loss_or_scores.numel():
                    raise TypeError(f"Shapes of loss/score do not match ({loss.numel()} vs {loss_or_scores.numel()})")
                loss_or_scores = loss_or_scores + loss

            return loss_or_scores


class MSTCOrthrus(nn.Module):
    """C4 multi-task wrapper that keeps causal time state beside the encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        edge_decoder: Optional[nn.Module],
        time_gap_decoder: Optional[nn.Module],
        time_gap_statistics: Optional[TimeGapStatistics],
        lambda_time: float = 0.3,
        graph_reindexer=None,
    ) -> None:
        super().__init__()
        if time_gap_decoder is not None and time_gap_statistics is None:
            raise ValueError("A fitted TimeGapStatistics is required when time task is enabled.")
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.time_gap_decoder = time_gap_decoder
        self.time_gap_statistics = time_gap_statistics
        self.lambda_time = lambda_time
        self.graph_reindexer = graph_reindexer
        self.last_seen_per_node: dict[int, int] = {}

    @property
    def time_task_enabled(self) -> bool:
        return self.time_gap_decoder is not None

    def reset_state(self) -> None:
        """Reset both C1 encoder history and C4 time history together."""
        if hasattr(self.encoder, "reset_state"):
            self.encoder.reset_state()
        self.last_seen_per_node.clear()

    def forward(self, batch, full_data, inference: bool = False) -> dict[str, torch.Tensor]:
        train_mode = not inference
        edge_index = batch.edge_index
        x = (batch.x_src, batch.x_dst)
        batch_size = edge_index.size(1)

        # Targets are derived from the state snapshot before this batch.  The
        # returned state is committed only after both task outputs are computed.
        updated_last_seen = None
        src_target = dst_target = None
        if self.time_task_enabled:
            src_target, dst_target, updated_last_seen = self.time_gap_statistics.transform_batch(
                batch, self.last_seen_per_node
            )

        with torch.set_grad_enabled(train_mode):
            h = self.encoder(
                edge_index=edge_index,
                t=batch.t,
                x=x,
                msg=batch.msg,
                edge_feats=batch.edge_feats if hasattr(batch, "edge_feats") else None,
                full_data=full_data,
                inference=inference,
                edge_types=batch.edge_type,
            )
            h_src, h_dst = (h[edge_index[0]], h[edge_index[1]]) if isinstance(h, torch.Tensor) else h
            if x[0].shape[0] != batch_size:
                x = (batch.x_src[edge_index[0]], batch.x_dst[edge_index[1]])

            zero_scalar = h_src.sum() * 0.0
            zero_per_event = zero_scalar.expand(batch_size)
            if self.edge_decoder is None:
                edge_logits = h_src.new_empty((batch_size, 0))
                loss_type = zero_per_event if inference else zero_scalar
            else:
                edge_logits = self.edge_decoder.logits(h_src, h_dst)
                loss_type = self.edge_decoder.loss(
                    h_src, h_dst, batch.edge_type, inference=inference
                )

            if not self.time_task_enabled:
                src_time_logits = h_src.new_empty((batch_size, 0))
                dst_time_logits = h_src.new_empty((batch_size, 0))
                loss_time = zero_per_event if inference else zero_scalar
                loss_time_src = zero_per_event if inference else zero_scalar
                loss_time_dst = zero_per_event if inference else zero_scalar
            else:
                src_target = src_target.to(h_src.device)
                dst_target = dst_target.to(h_src.device)
                src_time_logits, dst_time_logits = self.time_gap_decoder(h_src, h_dst)
                reduction = "none" if inference else "mean"
                loss_time, loss_time_src, loss_time_dst = self.time_gap_decoder.loss(
                    src_time_logits, dst_time_logits, src_target, dst_target, reduction=reduction
                )

            score_raw = loss_type + self.lambda_time * loss_time

        if updated_last_seen is not None:
            self.last_seen_per_node = updated_last_seen

        if train_mode:
            return {
                "loss": score_raw,
                "loss_type": loss_type,
                "loss_time": loss_time,
                "loss_time_src": loss_time_src,
                "loss_time_dst": loss_time_dst,
            }
        return {
            "score_raw": score_raw,
            "loss_type": loss_type,
            "loss_time": loss_time,
            "loss_time_src": loss_time_src,
            "loss_time_dst": loss_time_dst,
            "edge_logits": edge_logits,
            "src_time_logits": src_time_logits,
            "dst_time_logits": dst_time_logits,
            "src_time_target": src_target if src_target is not None else torch.empty(0, dtype=torch.long, device=h_src.device),
            "dst_time_target": dst_target if dst_target is not None else torch.empty(0, dtype=torch.long, device=h_src.device),
        }
