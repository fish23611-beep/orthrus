"""Focused unit tests for the C4 TimeGapDecoder."""

import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from decoders import TimeGapDecoder


def test_forward_uses_shared_hidden_representation_then_two_heads():
    decoder = TimeGapDecoder(in_dim=4, hidden_dim=7)
    h_src = torch.randn(3, 4)
    h_dst = torch.randn(3, 4)

    src_logits, dst_logits = decoder(h_src, h_dst)

    assert src_logits.shape == (3, 6)
    assert dst_logits.shape == (3, 6)
    assert decoder.shared_mlp[0].out_features == 7
    assert decoder.src_head.in_features == 7
    assert decoder.dst_head.in_features == 7


def test_loss_reductions_and_joint_definition_are_finite_and_differentiable():
    torch.manual_seed(0)
    decoder = TimeGapDecoder(in_dim=5, hidden_dim=9)
    h_src = torch.randn(4, 5, requires_grad=True)
    h_dst = torch.randn(4, 5, requires_grad=True)
    src_target = torch.tensor([0, 1, 2, 5], dtype=torch.long)
    dst_target = torch.tensor([5, 2, 1, 0], dtype=torch.long)

    src_logits, dst_logits = decoder(h_src, h_dst)
    loss_none, src_none, dst_none = decoder.loss(
        src_logits, dst_logits, src_target, dst_target, reduction="none"
    )
    loss_mean, src_mean, dst_mean = decoder.loss(
        src_logits, dst_logits, src_target, dst_target, reduction="mean"
    )

    assert loss_none.shape == (4,)
    assert src_none.shape == (4,)
    assert dst_none.shape == (4,)
    assert loss_mean.ndim == src_mean.ndim == dst_mean.ndim == 0
    assert torch.allclose(loss_none, 0.5 * (src_none + dst_none))
    assert torch.allclose(loss_mean, 0.5 * (src_mean + dst_mean))
    assert torch.isfinite(loss_none).all()
    assert torch.isfinite(loss_mean)

    loss_mean.backward()
    assert h_src.grad is not None and torch.isfinite(h_src.grad).all()
    assert h_dst.grad is not None and torch.isfinite(h_dst.grad).all()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in decoder.parameters())