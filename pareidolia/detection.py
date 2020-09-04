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

    Parameters
    ----------
    mats : Iterable of sp.spmatrix:
        A list of data matrices from different samples.

    Returns
    -------
    sp.spmatrix :
        The median background matrix, where each pixel is the median of
        the corresponding pixel from all samples.
    """
    # Check that all matrices are sparse, have the same format and sparsity
    if not np.all([sp.issparse(m) for m in mats]):
        raise ValueError("Input must be a scipy sparse matrix.")
    if not np.all([m.format == mats[0].format for m in mats]):
        raise ValueError("All input matrices must be in the same format.")
    if not np.all([m.getnnz() == mats[0].getnnz() for m in mats]):
        raise ValueError("All input matrices must have the same sparsity.")
    bg = mats[0].copy()
    bg.data = np.median([m.data for m in mats], axis=0)
    return bg


def reps_bg_diff(mats: Iterable[sp.spmatrix]) -> np.ndarray:
    """
    Given multiple sample matrices, return a 1D array of all differences
    between each sample matrix's pixels and their corresponding median
    background value. All input matrices must have the same shape, nonzero
    coordinates and format.

    Parameters
    ----------
    mats : Iterable of sp.spmatrix
        The list of data matrices from different samples.

    Returns
    -------
    np.ndarray of floats :
        The distribution of pixel differences to the median background. If
        there are S matrices of P nonzero pixels, this 1D array will contain
        P*S elements.
    """
    med_bg = median_bg(mats)
    diffs = [m.data - med_bg.data for m in mats]
    diffs = np.hstack(diffs)
    return diffs


def get_sse_mat(mats: Iterable[sp.spmatrix]) -> sp.spmatrix:
    """
    Given multiple matrices, return the matrix of position-wise sum of
    squared errors to median. All input matrices must have the same shape,
    nonzero coordinates and format. Output matrix will have the same format
    as inputs.

    Parameters
    ----------
    mats : Iterable of sp.spmatrix
        The list of data matrices from different samples.

    Returns
    -------
    sp.spmatrix :
        A sparse matrix where each nonzero pixel is the sum of squared
        difference of all samples to their median background, at the
        corresponding nonzero pixel.

    """
    bg = median_bg(mats)
    sse = bg.copy()
    se = np.array([(m.data - bg.data) ** 2 for m in mats])
    sse.data = np.sum(se, axis=0)
    return sse
