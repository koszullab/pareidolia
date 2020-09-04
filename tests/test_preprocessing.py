#!/usr/bin/env python3

"""
Unit tests for functions in preprocess.py
cmdoret, 20200514
"""
import copy
import itertools as it
import numpy as np
import scipy.sparse as sp
import pareidolia.preprocess as pap
import pytest

# Run test on square and rectangle matrices with even and odd dimensions,
# each with several seeds
MAP_PARAMS = (
    "seed,shape",
    list(it.product([1, 41, 115], [(3, 3), (7, 8), (100, 100), (120, 130)])),
)


def gen_mats(seed, shape, n=4, fmt="csr", density=0.1):
    """Helper function generating a list of random sparse matrices"""

    np.random.seed(seed)
    dens = density * np.random.random()
    mats = [sp.random(*shape, density=dens, format=fmt) for i in range(n)]
    return mats


@pytest.mark.parametrize(*MAP_PARAMS)
def test_get_nnz_union(seed, shape):
    """Test generation of nnz union from multiple sparse matrices."""
    # Generate 4 matrices of 100x100 px with random density between 0 and 10%
    mats = gen_mats(seed, shape)
    nnz_to_set = lambda nnz: set([coord for coord in zip(nnz[0], nnz[1])])
    # Use pareidolia's fast get_nnz function to get union (observed)
    union_obs = pap.get_nnz_union(mats)
    union_obs = {coord for coord in zip(union_obs[:, 0], union_obs[:, 1])}
    # Build expected union set naively (slow)
    union_exp = set()
    for mat in mats:
        nnz_mat = nnz_to_set(mat.nonzero())
        union_exp = union_exp.union(nnz_mat)
        # Check correctness of set: union is a superset of current matrix.
        assert union_obs.issuperset(nnz_mat)
    # Check completeness of set: ensuring XOR is empty between naive set union
    # and actual implementation.
    assert not len(union_exp ^ union_obs)


def test_get_nnz_union_errors():
    """Test proper error handling in get_nnz_union"""
    mats_wrong_type = [[]] * 3
    mats_wrong_format = [sp.coo_matrix((10, 10))] * 3
    with pytest.raises(ValueError):
        assert pap.get_nnz_union(mats_wrong_format)
    with pytest.raises(TypeError):
        assert pap.get_nnz_union(mats_wrong_type)


@pytest.mark.parametrize(*MAP_PARAMS)
def test_fill_nnz(seed, shape):
    """Test if a nnz superset can be used to fill multiple matrices correctly."""
    # Generate 4 matrices of 100x100 pixls with random density between 0 and 10%
    mats = gen_mats(seed, shape)
    nnz_union = pap.get_nnz_union(mats)
    # Check if all matrices have been filled to the sparsity of nnz_union
    for mat in mats:
        filled = pap.fill_nnz(mat, nnz_union)
        assert filled.nnz == nnz_union.shape[0]


@pytest.mark.parametrize(*MAP_PARAMS)
def test_get_common_valid_bins(seed, shape):
    """Test if common valid bins indices are correct"""

    mats = gen_mats(seed, shape)
    np.random.seed(seed)
    max_idx = min(shape)
    # We chose a random "bad" bin to test
    bad_bin_exp = np.random.choice(max_idx, size=1)
    for mat in mats:
        # Set all pixels to 1 and turn the bin off
        mat[:, :] = 1.0
        mat[bad_bin_exp, :] = 0.0
        mat[:, bad_bin_exp] = 0.0
        mat.eliminate_zeros()
    # Check for correct exception on rectangle matrices
    if shape[0] != shape[1]:
        with pytest.raises(NotImplementedError):
            assert pap.get_common_valid_bins(mats)
    else:
        valid_obs = pap.get_common_valid_bins(mats, n_mads=1)
        bad_bin_obs = [b for b in range(shape[0]) if b not in valid_obs]
        assert len(bad_bin_obs) == 1
        assert bad_bin_obs[0] == bad_bin_exp

