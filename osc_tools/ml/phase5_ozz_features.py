"""Continuous causal Fourier history for the global OZZ model (same B layout)."""
import math
import numpy as np
from osc_tools.ml.phase5_hierarchical_data import ExactFrequencyBuilder, token_positions
from osc_tools.ml.phase5_fourier import _PreparedBuilder


def continuous_features(segment, spp, blocks, basis, provenance, history=None):
    history = np.empty((0, 8), np.float32) if history is None else history
    raw = np.concatenate((history, segment))
    positions = token_positions(0, blocks*80, spp)
    absolute = positions+len(history)
    rounded = int(math.floor(spp+.5))
    specs = ((1, 9), (2, 1), (4, 1), (6, 1), (10, 1))
    phasors = {(rounded*p, h): np.full((len(positions), 8, h), np.nan+1j*np.nan, np.complex64)
               for p, h in specs}
    builder = ExactFrequencyBuilder(spp)
    # Bound temporary arrays, while retaining ten periods across every CPU chunk.
    for first in range(0, len(segment), 65536):
        selected = np.flatnonzero((positions >= first) & (positions < first+65536))
        if not len(selected):
            continue
        ends = absolute[selected]
        left = max(0, int(ends[0])-int(math.floor(10*spp+.5))+1)
        right = int(ends[-1])+1
        for periods, harmonics in specs:
            phasors[(rounded*periods, harmonics)][selected] = builder._phasors_prefix(
                raw[left:right], ends-left, rounded*periods, harmonics)
    features, missing, meta = _PreparedBuilder(phasors).build(
        segment, rounded, positions, voltage_basis=basis,
        channel_provenance=provenance, phasor_backend='prefix')
    return features, missing, meta, positions


def expand_labels(labels, samples_per_period):
    """Union of [positive sample, positive sample + one period], all classes.

    O(N*C), independent of event lengths/counts and without a sliding-window tensor.
    Compute on the whole record once, before selecting training windows.
    """
    if not np.isfinite(samples_per_period) or samples_per_period <= 0:
        raise ValueError('Invalid period')
    positive = np.asarray(labels) > .5
    index = np.arange(len(positive), dtype=np.int64)[:, None]
    last = np.maximum.accumulate(np.where(positive, index, -len(positive)-1), axis=0)
    return (last >= 0) & (index-last <= samples_per_period)


def cached_expanded_labels(labels, samples_per_period, cache_root=None):
    """Small derived label-only cache, shared across windows, workers and restarts."""
    from pathlib import Path
    import hashlib
    import os
    positive = np.asarray(labels) > .5
    if not positive.any():
        return positive
    digest = hashlib.sha256(b'forward_inclusive_one_period_v1')
    digest.update(str((positive.shape, float(samples_per_period))).encode())
    digest.update(positive.tobytes())
    root = (Path(cache_root) if cache_root is not None else
            Path(__file__).resolve().parents[2]/'reports/phase5/ozz_label_cache')
    path = root/(digest.hexdigest()+'.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved['target']
    expanded = expand_labels(positive, samples_per_period)
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, target=expanded)
    os.replace(temporary, path)
    return expanded
