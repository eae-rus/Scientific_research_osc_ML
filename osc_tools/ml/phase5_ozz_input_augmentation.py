"""Current-only corruption for supervised OZZ adaptation."""
import numpy as np


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

