import torch
from typing import Literal, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F

class EdgeTypeDecoder(nn.Module):
    def __init__(self, in_dim, num_edge_types, loss_fn, dropout, num_layers, activation):
        super(EdgeTypeDecoder, self).__init__()
        self.lin_src = nn.Linear(in_dim, in_dim*2)
        self.lin_dst = nn.Linear(in_dim, in_dim*2)
        
        if num_layers == 2:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim * 2),
                nn.Dropout(dropout),
                activation,
                
                nn.Linear(in_dim * 2, num_edge_types),
            )
        elif num_layers == 3:
            self.lin_seq = nn.Sequential(
                nn.Linear(in_dim * 4, in_dim * 4),
                nn.Dropout(dropout),
                activation,
            
                nn.Linear(in_dim * 4, in_dim * 2),
                nn.Dropout(dropout),
                activation,
            
                nn.Linear(in_dim * 2, num_edge_types),
            )
        else:
            raise ValueError(f"Invalid number of layers, found {num_layers}")
        
        self.loss_fn = loss_fn
        
    def logits(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        """Return edge-type logits without changing the legacy forward contract."""
        h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
        return self.lin_seq(h)

    def loss(self, h_src, h_dst, edge_type, inference=False) -> torch.Tensor:
        logits = self.logits(h_src, h_dst)
        edge_type_classes = edge_type.argmax(dim=1)
        return self.loss_fn(logits, edge_type_classes, inference=inference)

    def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
        return self.loss(h_src, h_dst, edge_type, inference=inference)


class TimeGapDecoder(nn.Module):
    """Predict source and destination time-gap buckets from endpoint states."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int = 6,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_classes != 6:
            raise ValueError("TimeGapDecoder requires the six C3 time-gap buckets.")
        self.num_classes = num_classes
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            activation if activation is not None else nn.ReLU(),
        )
        self.src_head = nn.Linear(hidden_dim, num_classes)
        self.dst_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, h_src: torch.Tensor, h_dst: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared_mlp(torch.cat([h_src, h_dst], dim=-1))
        return self.src_head(hidden), self.dst_head(hidden)

    def loss(
        self,
        src_logits: torch.Tensor,
        dst_logits: torch.Tensor,
        src_target: torch.Tensor,
        dst_target: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return joint, source, and destination cross-entropy losses."""
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        src_loss = F.cross_entropy(src_logits, src_target, reduction=reduction)
        dst_loss = F.cross_entropy(dst_logits, dst_target, reduction=reduction)
        return 0.5 * (src_loss + dst_loss), src_loss, dst_loss
