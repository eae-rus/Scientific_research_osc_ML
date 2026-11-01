"""Batched local Fourier features with independent ten-period histories."""
from functools import lru_cache
import math
import numpy as np
from osc_tools.ml.spectral_features import SpectralFeatureBuilder, SpectralFeatureConfig


@lru_cache(maxsize=128)
def _rotations(length, spp, periods, harmonic, starts):
    omega = 2*np.pi*harmonic/(periods*spp)
    carrier = np.exp(-1j*omega*np.arange(length, dtype=np.float64))
    correction = np.exp(1j*omega*np.asarray(starts, dtype=np.int64))
    carrier.flags.writeable = correction.flags.writeable = False
    return carrier, correction


class _PreparedBuilder(SpectralFeatureBuilder):
    def __init__(self, phasors):
        super().__init__(SpectralFeatureConfig('B'))
        self.phasors = phasors

    def _phasors_prefix(self, raw, positions, window, count):
        return self.phasors[(window, count)]


def build_batched_features(segment, spp, blocks, voltage_basis, provenance):
    """Same accumulation order/phase origin as ExactFrequencyBuilder.

    Groups equal sample grids, chunks at most 16 blocks, and computes the
    symmetry/polar conversion once for the full token sequence. No history
    crosses a ten-period boundary. Only signal-independent rotations are cached.
    """
    if spp < 8:
        raise ValueError('At least eight raw samples per period required')
    rounded = int(math.floor(spp+.5))
    positions = np.floor(np.arange(1, 80*blocks+1)*spp/8+.5).astype(np.int64)-1
    specs = [(1, 9), (2, 1), (4, 1), (6, 1), (10, 1)]
    values = {(rounded*p, h): np.full((blocks, 80, 8, h), np.nan+1j*np.nan, np.complex64)
              for p, h in specs}
    groups = {}
    for b in range(blocks):
        left, right = int(math.floor(b*10*spp+.5)), int(math.floor((b+1)*10*spp+.5))
        local = positions[b*80:(b+1)*80]-left
        groups.setdefault((right-left, tuple(local)), []).append((b, left, right))
    for (length, grid), entries in groups.items():
        local = np.asarray(grid)
        chunk_size = max(1, min(16, 4096//length))
        for offset in range(0, len(entries), chunk_size):
            chunk = entries[offset:offset+chunk_size]
            indices = [entry[0] for entry in chunk]
            raw = np.stack([np.asarray(segment[a:b], dtype=np.float32) for _, a, b in chunk])
            finite = np.isfinite(raw)
            counts = np.concatenate((np.zeros((len(chunk), 1, 8), np.int64), np.cumsum(finite, axis=1)), axis=1)
            safe = np.where(finite, raw, 0).astype(np.float64)
            for periods, harmonics in specs:
                window = int(math.floor(periods*spp+.5))
                eligible = (local >= window-1) & (local < length)
                ends = local[eligible]; starts = ends-window+1
                if not len(ends):
                    continue
                valid = counts[:, ends+1]-counts[:, starts] == window
                for harmonic in range(1, harmonics+1):
                    if harmonic >= periods*spp/2:
                        continue
                    carrier, correction = _rotations(length, spp, periods, harmonic, tuple(starts))
                    weighted = safe*carrier[None, :, None]
                    prefix = np.concatenate((np.zeros((len(chunk), 1, 8), np.complex128),
                                             np.cumsum(weighted, axis=1)), axis=1)
                    value = (prefix[:, ends+1]-prefix[:, starts])*correction[None, :, None]*2/window
                    destination = values[(rounded*periods, harmonics)]
                    destination[np.asarray(indices)[:, None], np.flatnonzero(eligible)[None, :], :, harmonic-1] = np.where(valid, value, np.nan+1j*np.nan)
    prepared = {key: value.reshape(blocks*80, 8, key[1]) for key, value in values.items()}
    features, missing, meta = _PreparedBuilder(prepared).build(
        segment, rounded, positions, voltage_basis=voltage_basis,
        channel_provenance=provenance, phasor_backend='prefix')
    # Preserve the existing metadata contract (last block's local positions).
    last_left = int(math.floor((blocks-1)*10*spp+.5))
    meta['positions'] = (positions[-80:]-last_left).tolist()
    return features, missing, meta, positions
