#!/usr/bin/env python3

"""
Unit tests for functions in preprocess.py
cmdoret, 20200514
"""
import itertools as it
import numpy as np
import scipy.sparse as sp
import pareidolia.preprocess as pap
import pytest

MAP_PARAMS = ('seed,shape', list(it.product([1, 41, 115],[(3, 3), (7, 8), (100, 100), (120, 130)])))
@pytest.mark.parametrize(*MAP_PARAMS)
def test_get_nnz(seed, shape):
    """Test generation of nnz union from multiple sparse matrices."""
    # Generate 4 matrices of 100x100 pixls with random density between 0 and 10%
    np.random.seed(1)
    mats = [sp.random(100, 100, density=np.random.random() / 10, format='csr') for i in range(4)]
    nnz_union = pap.get_nnz_union(mats)
    # Check set naively (slow)
    nnz_to_set = lambda nnz: set([coord for coord in zip(nnz[0], nnz[1])])
    nnz_union_set = {coord for coord in zip(nnz_union[:, 0], nnz_union[:, 1])}
    for mat in mats:
        nnz_m_set = nnz_to_set(mat.nonzero())
        for coord in nnz_m_set:
            # Checks correctness of set
            try:
                nnz_union_set.pop(coord)
            except KeyError:
                raise KeyError("nnz set misses some values present in input matrices")
    # Check completeness of set
    assert not len(nnz_m_set)

@pytest.mark.parametrize(*MAP_PARAMS)
def test_fill_nnz(seed, shape):
    """Test if a nnz superset can be used to fill multiple matrices correctly."""
    # Generate 4 matrices of 100x100 pixls with random density between 0 and 10%
    np.random.seed(1)
    mats = [sp.random(100, 100, density=np.random.random() / 10, format='csr') for i in range(4)]
    nnz_union = pap.get_nnz_union(mats)
    # Check if all matrices have been filled to the sparsity of nnz_union
    for mat in mats:
        filled = pap.fill_nnz(mat)
        assert filled.nnz == nnz_union.shape[0]
