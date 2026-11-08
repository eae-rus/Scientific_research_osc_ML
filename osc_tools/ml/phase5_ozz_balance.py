"""Window-balanced supervision and threshold-independent OZZ diagnostics."""
import numpy as np
import torch
from torch.nn import functional as F


class NoUsableEvent(ValueError):
    """Expected data condition; labels cannot anchor a usable interval."""


def event_start(labels, length, spp, group, rng, continuous=False):
    """Anchor the source arc type in a usable (post Fourier warm-up) interval."""
    blocks = int(round(length/(10*spp)))
    positions = np.floor(np.arange(1, blocks*80+1)*spp/8+.5).astype(int)-1
    left = np.floor((np.arange(blocks*80)//80)*10*spp+.5).astype(int)
    eligible = (positions if continuous else positions-left) >= int(np.floor(spp+.5))-1
    token = np.searchsorted(positions, np.arange(length))
    usable = eligible[np.minimum(token, len(eligible)-1)] & (token < len(eligible))
    first = int(np.flatnonzero(usable)[0])
    anchors = np.flatnonzero(labels[:, group] > .5)
    anchors = anchors[anchors >= first]
    if not len(anchors):
        raise NoUsableEvent(f'RTDS group {group}: no usable positive labels')
    for anchor in rng.permutation(anchors):
        low = max(0, int(anchor)-length+1)
        high = min(len(labels)-length, int(anchor)-first)
        if high < low:
            continue
        for _ in range(32):
            start = int(rng.integers(low, high+1))
            if usable[anchor-start]:
                return start
        # Exact fallback even for isolated one-sample events.
        offsets = np.flatnonzero(usable[int(anchor)-high:int(anchor)-low+1])
        if len(offsets):
            return high-int(rng.choice(offsets))
    raise NoUsableEvent('Event falls entirely inside Fourier warm-up intervals')


def estimate_balance(dataset, samples=400, cap=100.):
    """Match window-mean loss; use training data only, never validation labels."""
    fractions = []
    for i in range(min(samples, len(dataset))):
        batch = dataset[i]
        mask = batch['valid'] & batch['supervised']
        if mask.any():
            fractions.append(batch['target'][mask].float().mean(0))
    if not fractions:
        raise ValueError('No supervised training intervals for class balance')
    p = torch.stack(fractions).mean(0)
    if (p == 0).any():
        raise ValueError('Class absent from balance probe; increase balance_samples')
    weight = ((1-p)/p).clamp(1., cap)
    return dict(samples=len(fractions), positive_fraction=p.tolist(),
                positive_weight=weight.tolist(), normalizer=(1-p+p*weight).tolist())


def transition_weights(target, valid, boundary_weight=.5):
    """Keep every label, downweight the two bins adjacent to an observed transition.

    Last dimension is class; penultimate dimension is time. Invalid intervals
    cannot create a boundary. One-bin events retain a nonzero positive loss.
    """
    if not 0 < boundary_weight <= 1:
        raise ValueError('boundary_weight must be in (0, 1]')
    edge = (target[..., 1:, :] != target[..., :-1, :])
    edge = edge & (valid[..., 1:] & valid[..., :-1]).unsqueeze(-1)
    boundary = torch.zeros_like(target, dtype=torch.bool)
    boundary[..., 1:, :] |= edge
    boundary[..., :-1, :] |= edge
    return torch.where(boundary, boundary_weight, 1.)


def balanced_window_loss(logits, hard_target, smoothing, balance, focal_gamma=0., interval_weight=None):
    """Weight actual events, not the artificial positive mass from smoothing."""
    if not logits.numel():
        return logits.sum() * 0.
    target = smoothing + (1-2*smoothing)*hard_target
    weight = logits.new_tensor(balance['positive_weight'])
    norm = logits.new_tensor(balance['normalizer'])
    element_weight = torch.where(hard_target > .5, weight, 1.) / norm
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if focal_gamma:
        pt = torch.where(hard_target > .5, logits.sigmoid(), (-logits).sigmoid())
        loss = loss * (1-pt).pow(focal_gamma)
    if interval_weight is not None:
        loss = loss * interval_weight
    return (loss * element_weight).mean()


def average_precision_metrics(logits, target):
    """Stepwise PR integral (average precision), grouping tied scores exactly."""
    scores, truth = np.asarray(logits), np.asarray(target) > .5
    values, prevalence = [], []
    for c in range(truth.shape[1]):
        y = truth[:, c]
        prevalence.append(float(y.mean()) if len(y) else None)
        if not y.any():
            values.append(None)
            continue
        order = np.argsort(-scores[:, c], kind='stable')
        s, y = scores[order, c], y[order]
        ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]
        tp = np.cumsum(y)[ends]
        recall = tp / y.sum()
        precision = tp / (ends+1)
        values.append(float(np.sum(np.diff(np.r_[0., recall])*precision)))
    present = [v for v in values if v is not None]
    return dict(average_precision_per_class=values, macro_ap=float(np.mean(present)) if present else None,
                positive_fraction_per_class=prevalence)
