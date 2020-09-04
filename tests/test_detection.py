#!/usr/bin/env python3

"""
Unit tests for functions in detection.py
cmdoret, 20200904
"""
import numpy as np
import scipy.sparse as sp
import pytest
import pareidolia.detection as pad

# Run test on square and rectangle matrices with even and odd dimensions,
MAP_PARAMS = ("shape", ((2, 2), (3, 3), (3, 5), (5, 3), (2, 5), (5, 2)))


def gen_mats(shape, n_mats=4, density=0.8):
    """
    Helper function generating a list of random sparse matrices with identical
    sparsity structure.
    """
    print(shape)
    # Get total number of pixels in the matrix
    n_coords = shape[0] * shape[1]
    # Pick a number of (1D) pixel indices to switch on
    picked_coords = np.random.choice(
        n_coords, round(density * n_coords), replace=False
    )
    # Convert 1D coords to 2D
    picked_rows = picked_coords // shape[1]
    picked_cols = picked_coords % shape[1]
    # Generate empty sparse matrices and fill the right pixels
    mats = [None] * n_mats
    for i in range(n_mats):
        mat = sp.csr_matrix(shape)
        mat[picked_rows, picked_cols] = np.random.random()
        mats[i] = mat

    return mats


@pytest.mark.parametrize(*MAP_PARAMS)
def test_median_bg(shape):
    """Test computation of median background from multiple sparse matrices."""
    # Generate multiple matrices of identical shape and sparsity
    mats = gen_mats(shape)
    # Compute median background
    obs_bg = pad.median_bg(mats)
    # Use numpy's utilities to compute the median background in dense format
    dense_mats = [m.toarray() for m in mats]
    exp_bg = np.apply_along_axis(np.median, 2, np.dstack(dense_mats))
    # The median bg should be the same as numpy's result on dense matrices
    assert np.all(obs_bg == exp_bg)
    bad_mats = mats.copy()
    # A sparse matrix with different format should yield ValueError
    with pytest.raises(ValueError):
        bad_mats[0] = mats[0].tocoo()
        pad.median_bg(bad_mats)
    with pytest.raises(ValueError):  # Same for sparsity structure
        bad_mats[0] = mats[0]
        bad_mats[0].data[0] = 0
        bad_mats[0].eliminate_zeros()
        pad.median_bg(bad_mats)
    with pytest.raises(ValueError):  # Same for dense matrix
        bad_mats[0] = mats[0].toarray()
        pad.median_bg(bad_mats)


def test_reps_bg_diff():
    ...


def test_get_sse_mat():
    ...
