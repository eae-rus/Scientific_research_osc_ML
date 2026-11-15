"""Current-only corruption for supervised OZZ adaptation."""
import numpy as np


def augment_impulse(raw, rng, probability, history_count=0):
    """Corrupt one measured channel or all measured channels for 1–3 samples."""
    if probability == 0 or rng.random() >= probability:
        return
    signal = raw[history_count:]
    # Respect missing channels and channels already zeroed by dropout.
    available = np.flatnonzero(np.any(np.isfinite(signal) & (signal != 0), axis=0))
    if not len(available):
        return
    channels = available if rng.random() < .5 else [int(rng.choice(available))]
    width = int(rng.integers(1, min(3, len(signal)) + 1))
    start = int(rng.integers(len(signal) - width + 1))
    zero = rng.random() < .5
    factor = 0. if zero else rng.choice([-1., 1.]) * rng.uniform(100., 1000.)
    for channel in channels:
        column = signal[:, channel]
        values = column[np.isfinite(column) & (column != 0)]
        amplitude = np.quantile(np.abs(values), .95) if not zero else 0.
        burst = column[start:start + width]
        burst[np.isfinite(burst)] = factor * amplitude


def augment_current_channels(raw, provenance, rng, all_probability=0., phase_probability=0., zero_probability=.5):
    """Label-independent current corruption; voltages and true labels stay intact.

    Missing means NaN/state=0; zero corruption keeps a channel's original presence.
    Never manufacture a measured zero where the source was already missing.
    Disabled augmentation consumes no RNG, preserving old recipes exactly.
    """
    if not (0 <= all_probability <= 1 and 0 <= phase_probability <= 1
            and all_probability+phase_probability <= 1 and 0 <= zero_probability <= 1):
        raise ValueError('Invalid current dropout probabilities')
    if all_probability+phase_probability == 0:
        return raw, provenance, 'full'
    draw = rng.random()
    if draw < all_probability:
        columns, name = [0, 1, 2, 3], 'all_currents'
    elif draw < all_probability+phase_probability:
        columns = [int(rng.choice(3, p=[.2, .6, .2]))]
        name = 'phase_current'
    else:
        return raw, provenance, 'full'
    result = raw.copy()
    state = None if provenance is None else provenance.copy()
    if rng.random() < zero_probability:
        result[:, columns] = np.where(np.isfinite(result[:, columns]), 0., np.nan)
        return result, state, name+'_zero'
    result[:, columns] = np.nan
    if state is not None:
        state[columns] = 0
    return result, state, name+'_missing'

