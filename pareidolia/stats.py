#!/usr/bin/env python3

"""
Functions used to evaluate statistical significance of changes.
cmdoret, 20200403
"""
import numpy as np


def vals_to_percentiles(vals: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Return the percentiles corresponding to values in a distribution."""
    # Get indices corresponding to vals in the distribution
    sorted_dist = np.sort(dist)
    percentiles = np.searchsorted(sorted_dist, vals) / len(sorted_dist)
    return percentiles
