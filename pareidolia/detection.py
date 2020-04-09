#!/usr/bin/env python3

"""
Functions used for change detection.
cmdoret, 20200403
"""
from typing import Iterable
import scipy.sparse as sp
import numpy as np


def median_bg(mats: Iterable[sp.spmatrix]) -> sp.spmatrix:
    """
    Given multiple sparse matrices with the same shape, format
    and nonzero coordinates, generate a background matrix made
    up of the median signal
    """
    if np.all([sp.issparse(m) for m in mats]):
        if not np.all([m.format == mats[0].format for m in mats]):
            raise ValueError("All input matrices must be in the same format.")
    else:
        raise ValueError("Input must be a scipy sparse matrix.")
    bg = mats[0].copy()
    bg.data = np.median([m.data for m in mats], axis=0)
    return bg


def reps_bg_diff(mats: Iterable[sp.spmatrix]) -> np.ndarray:
    """
    Given multiple sample matrices, return a 1D array of all differences
    between each sample matrix's pixels and their corresponding median
    background value. All input matrices must have the same shape, nonzero
    coordinates and format.
    """
    bg = median_bg(mats)
    diffs = [m.data - bg.data for m in mats]
    diffs = np.hstack(diffs)
    return diffs


def get_sse_mat(mats: Iterable[sp.spmatrix]) -> sp.spmatrix:
    """
    Given multiple matrices, return the matrix of position-wise sum of
    squared errors to median. All input matrices must have the same shape,
    nonzero coordinates and format. Output matrix will have the same format
    as inputs.
    """
    bg = median_bg(*mats)
    sse = bg.copy()
    se = np.array([(m.data - bg.data) ** 2 for m in mats])
    sse.data = np.sum(se, axis=0)
    return sse
