"""Shared descriptive statistics for record-level overvoltage maxima.

These functions consume already selected scalar values; they do not label
waveforms, select fault zones or define voltage normalization.
"""
import numpy as np


def _values(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Expected at least two finite record-level maxima")
    return array


def describe_overvoltages(values):
    array = _values(values)
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "median": float(np.median(array)), "sample_sd": float(array.std(ddof=1)),
        "sample_variance": float(array.var(ddof=1)), "min": float(array.min()),
        "q1": float(np.quantile(array, .25, method="linear")),
        "q3": float(np.quantile(array, .75, method="linear")),
        "max": float(array.max()),
    }


def interval_fraction(values, lower=None, upper=None):
    """Count a closed interval; either absent bound means an open half-line."""
    array = _values(values)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Lower bound exceeds upper bound")
    mask = np.ones(array.size, dtype=bool)
    if lower is not None:
        mask &= array >= lower
    if upper is not None:
        mask &= array <= upper
    count = int(mask.sum())
    return {"count": count, "denominator": int(array.size),
            "percent": count / array.size * 100}


def empirical_cdf(values):
    """Return unique maxima and right-continuous cumulative record fractions."""
    array = _values(values)
    unique, counts = np.unique(array, return_counts=True)
    return unique, np.cumsum(counts) / array.size
