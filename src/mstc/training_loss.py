"""Train-only, fixed full-split endpoint weighting; independent of model state."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

MODES = frozenset({'event_mean', 'node_contribution_balanced', 'node_contribution_shrunk'})


def weighting_mode(cfg):
    mode = getattr(cfg.detection.gnn_training, 'train_loss_weighting', 'event_mean')
    if mode not in MODES:
        raise ValueError(f'Unknown train_loss_weighting: {mode!r}')
    if mode != 'event_mean' and getattr(getattr(cfg, 'model', None), 'variant', None) != 'mstc':
        raise ValueError('Node contribution weighting requires model.variant=mstc')
    return mode


def endpoint_arrays(src, dst):
    arrays = []
    for value in (src, dst):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        a = np.asarray(value)
        if a.ndim != 1 or a.dtype.kind not in 'iu' or (a.size and (a.min() < 0 or a.max() > np.iinfo(np.int64).max)):
            raise ValueError('Node IDs must be one-dimensional nonnegative int64-compatible integers')
        arrays.append(a.astype(np.int64, copy=False))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('Endpoint lengths differ')
    return arrays


class NodeContributionWeights:
    """For M target events and V observed nodes: W_e=(M/V)*(1/c_s+1/c_d).

    sum_e W_e=M, and mean_e W_e*loss_e is exactly the mean of per-node
    mean endpoint losses. Self loops increment c twice and contribute 2/c.
    No validation/test frequencies, labels, clipping or batch normalization.
    """
    def __init__(self, counts, num_events):
        counts = np.asarray(counts)
        if counts.ndim != 1 or counts.dtype.kind not in 'iu' or not len(counts) or (counts < 0).any():
            raise ValueError('Counts must be a nonnegative integer vector')
        if isinstance(num_events, bool) or not isinstance(num_events, (int, np.integer)) or num_events <= 0:
            raise ValueError('num_events must be a positive integer')
        if counts.max() > np.iinfo(np.int64).max or int(counts.sum(dtype=object)) != 2 * int(num_events):
            raise ValueError('Endpoint counts must sum to twice the target events')
        self.counts = counts.astype(np.int64, copy=True)
        self.counts.flags.writeable = False
        self.num_events = int(num_events)
        self.num_nodes = int(np.count_nonzero(counts))
        self.normalization = self.num_events / self.num_nodes
        self._device_inverse = {}

    @classmethod
    def fit(cls, train_windows, *, num_nodes):
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, (int, np.integer)) or num_nodes <= 0:
            raise ValueError('num_nodes must be a positive capacity')
        counts = np.zeros(num_nodes, dtype=np.int64); events = 0
        for window in train_windows:
            split = getattr(window, 'split_name', 'train')
            if split != 'train':
                raise ValueError('Only training target windows may fit weights')
            src, dst = endpoint_arrays(window.src, window.dst)
            for ids in (src, dst):
                if ids.size and ids.max() >= num_nodes:
                    raise ValueError('Training node ID exceeds capacity')
                np.add.at(counts, ids, 1)
            events += len(src)
        return cls(counts, events)

    def _counts_for_endpoints(self, src, dst):
        src, dst = endpoint_arrays(src, dst)
        if len(src) and max(src.max(), dst.max()) >= len(self.counts):
            raise ValueError('Node ID exceeds fitted training universe')
        cs, cd = self.counts[src], self.counts[dst]
        if (cs <= 0).any() or (cd <= 0).any():
            raise ValueError('Zero count: event endpoint was absent from training fit')
        return cs, cd

    def _endpoint_inverse(self):
        return np.divide(1., self.counts, out=np.zeros(len(self.counts)), where=self.counts > 0)

    def numpy_weights(self, src, dst):
        cs, cd = self._counts_for_endpoints(src, dst)
        weights = self.normalization * (1. / cs + 1. / cd)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Invalid event weights; no implicit clipping is permitted')
        return weights

    def event_weights(self, src, dst, *, dtype=torch.float64):
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            raise ValueError('Batch endpoints must be tensors')
        if src.dtype not in (torch.int32, torch.int64) or dst.dtype not in (torch.int32, torch.int64):
            raise ValueError('Batch IDs must be integer tensors')
        if src.ndim != 1 or src.shape != dst.shape or src.device != dst.device or not src.numel():
            raise ValueError('Batch endpoints must be nonempty aligned vectors on the same device')
        if bool((src < 0).any() | (dst < 0).any() | (src >= len(self.counts)).any() | (dst >= len(self.counts)).any()):
            raise ValueError('Batch node ID outside fitted training universe')
        key = str(src.device)
        if key not in self._device_inverse:
            inverse = self._endpoint_inverse()
            self._device_inverse[key] = torch.tensor(inverse, dtype=torch.float64, device=src.device)
        inverse = self._device_inverse[key]
        left, right = inverse[src.long()], inverse[dst.long()]
        if bool((left <= 0).any() | (right <= 0).any()):
            raise ValueError('Zero count at batch endpoint')
        weights = ((left + right) * self.normalization).to(dtype=dtype)
        if not bool(torch.isfinite(weights).all() & (weights > 0).all()):
            raise ValueError('Nonfinite or nonpositive event weights')
        return weights

    def reduce(self, event_losses, src, dst):
        if not isinstance(event_losses, torch.Tensor) or event_losses.ndim != 1 or not event_losses.is_floating_point():
            raise ValueError('Expected differentiable floating per-event losses')
        if event_losses.shape != src.shape or event_losses.device != src.device:
            raise ValueError('Per-event loss/endpoints shape or device mismatch')
        if not bool(torch.isfinite(event_losses).all()):
            raise ValueError('Nonfinite event loss')
        weights = self.event_weights(src, dst, dtype=event_losses.dtype)
        return (event_losses * weights).mean()

    def summary(self):
        values = self.counts[self.counts > 0]
        return dict(mode='node_contribution_balanced',target_events=self.num_events,observed_nodes=self.num_nodes,
                    endpoint_contributions=int(values.sum()),normalization=self.normalization,
                    counts_sha256=hashlib.sha256(self.counts.tobytes()).hexdigest(),
                    contribution_quantiles={str(q):float(x) for q,x in zip([.5,.9,.99,.999],np.quantile(values,[.5,.9,.99,.999]))},
                    counts_source='training target endpoints only',batch_renormalization=False)


class ShrunkNodeContributionWeights(NodeContributionWeights):
    """Fixed half-event / half-smoothed-node loss, with W in (0.5, 2.5].

    tau is train-only: at least mean endpoint count, enlarged until the
    conservative smoothed-component bound is <=4. No batch normalization.
    """
    def __init__(self, counts, num_events):
        super().__init__(counts, num_events)
        active = self.counts[self.counts > 0].astype(np.float64)
        minimum = float(active.min())
        self.tau = 2. * self.num_events / self.num_nodes

        def total(tau):
            return float(np.sum(active / (active + tau), dtype=np.float64))

        def upper(tau):
            return (2. * self.num_events / total(tau)) / (minimum + tau)

        if upper(self.tau) > 4.:
            low, high = self.tau, self.tau
            while upper(high) > 4.:
                high *= 2.
                if not np.isfinite(high):
                    raise ValueError('Cannot fit finite training shrinkage')
            for _ in range(64):
                middle = low + (high - low) / 2.
                if upper(middle) > 4.:
                    low = middle
                else:
                    high = middle
            self.tau = high
        self.normalization = self.num_events / total(self.tau)
        self.weight_upper_bound = .5 + .5 * upper(self.tau)

    def _endpoint_inverse(self):
        return np.divide(1., self.counts + self.tau,
                         out=np.zeros(len(self.counts)), where=self.counts > 0)

    def numpy_weights(self, src, dst):
        cs, cd = self._counts_for_endpoints(src, dst)
        return .5 + .5 * self.normalization * (1. / (cs + self.tau) + 1. / (cd + self.tau))

    def event_weights(self, src, dst, *, dtype=torch.float64):
        return .5 + .5 * super().event_weights(src, dst, dtype=dtype)

    def summary(self):
        result = super().summary()
        result.update(mode='node_contribution_shrunk', formula_version=1,
                      alpha=.5, smooth_component_cap=4., tau=self.tau,
                      weight_lower_bound=.5, weight_upper_bound=self.weight_upper_bound,
                      tau_rule='at_least_mean_endpoint_count_then_bound_bisection_64')
        return result


def fit_training_weights(cfg, train_windows, *, num_nodes):
    """Only called by training; validation/test scoring never uses weights."""
    mode = weighting_mode(cfg)
    if mode == 'event_mean':
        return None
    cls = ShrunkNodeContributionWeights if mode == 'node_contribution_shrunk' else NodeContributionWeights
    return cls.fit(train_windows, num_nodes=num_nodes)
