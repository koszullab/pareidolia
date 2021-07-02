#!/usr/bin/env python3

"""
Functions used for change detection.
cmdoret, 20200403
"""
from typing import Iterable
import scipy.sparse as sp
import numpy as np
import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup


def median_bg(mats: Iterable[sp.spmatrix]) -> sp.spmatrix:
    """
    Given multiple sparse matrices with the same shape, format
    and nonzero coordinates, generate a background matrix made
    up of the median signal

    Parameters
    ----------
    mats : Iterable of scipy.sparse.spmatrix:
        A list of data matrices from different samples.

    Returns
    -------
    scipy.sparse.spmatrix :
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
    mats : Iterable of scipy.sparse.spmatrix
        The list of data matrices from different samples.

    Returns
    -------
    numpy.ndarray of floats :
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
    mats : Iterable of scipy.sparse.spmatrix
        The list of data matrices from different samples.

    Returns
    -------
    scipy.sparse.spmatrix :
        A sparse matrix where each nonzero pixel is the sum of squared
        difference of all samples to their median background, at the
        corresponding nonzero pixel.

    """
    bg = median_bg(mats)
    sse = bg.copy()
    se = np.array([(m.data - bg.data) ** 2 for m in mats])
    sse.data = np.sum(se, axis=0)
    return sse


def get_win_density(
    mat: sp.csr_matrix, win_size: int = 3, sym_upper: bool = False
) -> sp.csr_matrix:
    """Compute local pixel density in sparse matrices using convolution.
    The convolution is performed in 'full' mode: Computations are performed
    all the way to the edges by trimming the kernel.

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        The input sparse matrix to convolve.
    win_size : int
        The size of the area in which to compute the proportion of nonzero
        pixels. This will be the kernel size used in convolution.
    sym_upper : bool
        Whether the input matrix is symmetric upper, in which case only the
        upper triangle is returned.


    Returns
    -------
    scipy.sparse.csr_matrix:
        The result of the convolution of the uniform kernel
        (win_size x win_size) on the binarized input matrix.
        Each value represents the proportion of nonzero pixels
        in the neighbourhood.
    """
    ms, ns = mat.shape
    # Generate a binary matrix (pixels are either empty or full)
    bin_mat = mat.copy()
    if sym_upper:
        bin_mat = sp.triu(bin_mat)
    bin_mat.data = bin_mat.data.astype(bool)
    # Adding a frame of zeros around the signal
    tmp = sp.csr_matrix((win_size - 1, ns), dtype=bool)
    bin_mat = sp.vstack([tmp, bin_mat, tmp], format=mat.format)
    tmp = sp.csr_matrix((ms + 2 * (win_size - 1), win_size - 1), dtype=bool)
    bin_mat = sp.hstack([tmp, bin_mat, tmp], format=mat.format)
    # Convolve the uniform kernel with this matrix to get the proportion of nonzero
    # pixels in each neighbourhood
    kernel = np.ones((win_size, win_size))
    win_area = win_size ** 2
    density = cud.xcorr2(bin_mat, kernel / win_area)

    # Compute convolution of uniform kernel with a frame of ones to get number
    # of missing pixels in each window.
    frame = cup.frame_missing_mask(
        sp.csr_matrix(mat.shape, dtype=bool), kernel.shape, sym_upper=sym_upper
    )
    frame = cud.xcorr2(frame, kernel).tocoo()
    # From now on, frame.data contains the number of 'present' samples.
    # (where there is at least one missing pixel)
    frame.data = win_area - frame.data

    # Adjust the proportion for values close to the border (lower denominator)
    density[frame.row, frame.col] = (
        density[frame.row, frame.col].A1 * win_area / frame.data
    )
    # Trim the frame out from the signal
    density = density[
        win_size - 1 : -win_size + 1, win_size - 1 : -win_size + 1
    ]
    if sym_upper:
        density = sp.triu(density)
    return density
