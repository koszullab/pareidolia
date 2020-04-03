#!/usr/bin/env python3

"""
Functions used to clean and prepare matrices for detection.
cmdoret, 20200403
"""
from typing import Iterable, Optional, Tuple, Iterator, Set
import scipy.sparse as  sp
import numpy as np
import pandas as pd
import cooler
import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup


def get_common_valid_bins(
        *mats: sp.csr_matrix,
        max_dist: Optional[int] =50
    ) -> np.ndarray:
    """
    Generates an array of valid bins indices, using the intersection
    of valid bins from all input sparse matrices. All input matrices must
    have the same shape.
    """
    common_valid = None
    for m in mats:
        valid = cup.get_detectable_bins(m, n_mads=5)
        if common_valid is None:
            common_valid = set(valid[0])
        else:
            common_valid = common_valid.intersection(set(valid[0]))
    return np.array(list(common_valid))


def get_min_contacts(*cools: cooler.Cooler, region: Optional[str]=None) -> int:
    """
    Given several cooler objects, returns the number of contacts in the least
    covered matrix. Optionally, a region can be given in UCSC format, in which
    case coverage will be computed only in the matrix from that region.
    """
    contacts = np.zeros(len(cools))
    for i, c in enumerate(cools):
        if region is None:
            contacts[i] = c.info['sum']
        else:
            contacts[i] = c.matrix(balance=False, sparse=True).fetch(region).sum()
    return int(min(contacts))


def preprocess_hic(
        c: cooler.Cooler,
        min_contacts: Optional[int]=None,
        region: Optional[str]=None
    ) -> sp.csr_matrix:
    """
    Given an input cooler object, returns the preprocessed Hi-C matrix.
    Preprocessing involves (in that order): subsetting region, subsampling
    contacts, normalisation, detrending (obs / exp). Balancing weights must
    be pre-computer in the referenced cool file. Region must be in UCSC format.
    """
    mat = c.matrix(sparse=True, balance=False)
    if region is None:
        mat = mat[:]
    else:
        mat = mat.fetch(region)
    if min_contacts is not None:
        # Preprocess matrices (get to same coverage, balance and detrend)
        mat = cup.subsample_contacts(mat, min_contacts).tocoo()
    valid = cup.get_detectable_bins(mat, n_mads=5)
    # balance region with weights precomputed on the whole matrix
    biases = c.bins().fetch(region)['weight'].values
    mat.data = mat.data * biases[mat.row] * biases[mat.col]
    mat = cup.detrend(mat.tocsr(), smooth=True, detectable_bins=valid[0])
    # Replace NaNs by 0s
    mat.data = np.nan_to_num(mat.data)
    mat.eliminate_zeros()
    return mat


def get_correlation(mat: sp.csr_matrix, kernel: np.ndarray) -> sp.csr_matrix:
    """Get the pearson coefficient map between input matrix and kernel."""
    return cud.normxcorr2(mat, kernel, full=False, missing_mask=None, sym_upper=True, max_dist=50)


def yield_nnz(mat: sp.spmatrix) -> Iterator[Tuple[int]]:
    """
    Helper function to extract nonzero values from a scipy.sparse matrix and
    returns an iterator of nonzero coordinates, in the form of (row, col) tuples.
    """
    nnr, nnc = mat.nonzero()
    nnr = nnr.tolist()
    nnc = nnc.tolist()
    return zip(nnr, nnc)


def get_nnz_set(*mats: sp.csr_matrix, mode: str='intersection') -> Set[Tuple[int]]:
    """
    Given an arbitrary number of sparse matrices, build a set
    containing the intersection or union of nonzero coordinates
    from all matrices. Each coordinate is stored in the form of
    (row, col) tuples.
    """
    # Check for input type
    try:
        if np.all([m.format == 'csr' for m in mats]):
            for i, mat in enumerate(mats):
                # Use first matrix to initialize set
                if i == 0:
                    nnz_set = set(yield_nnz(mat))
                # Iteratively reduce set by keeping only elements present in each matrix
                else:
                    if mode == 'union':
                        nnz_set = nnz_set.union(set(yield_nnz(mat)))
                    elif mode == 'intersection':
                        nnz_set = nnz_set.intersection(set(yield_nnz(mat)))
                    else:
                        raise ValueError("mode must be either 'union' or 'intersection'")
        else:
            raise ValueError("input sparse matrices must be in csr format")
    except AttributeError:
        raise TypeError("Input type must be scipy.sparse.csr_matrix")
    
    return nnz_set


def filter_nnz(mat: sp.csr_matrix, nnz_set: Set[Tuple[int]]) -> sp.csr_matrix:
    """
    Given an input sparse matrix and a set of nonzero coordinates,
    filter the matrix to keep only positions present in the set.
    """
    out = mat.copy()
    drop_coords_mask = np.array([n not in nnz_set for n in yield_nnz(mat)], dtype=bool)
    out.data[drop_coords_mask] = 0.0
    out.eliminate_zeros()
    return out


def fill_nnz(mat: sp.csr_matrix, nnz_set:Set[Tuple[int]], fill_value: float=0.0) -> sp.csr_matrix:
    """
    Given an input sparse matrix and a set of nonzero coordinates,
    fill the matrix to ensure all values in the set are stored explicitely.
    """
    # Get the set of nonzero coordinate in the input matrix
    nnz_mat = set(yield_nnz(mat))
    nnz_list = list(nnz_set)
    out = mat.copy()
    add_rows = [c[0] for c in nnz_list if c not in nnz_mat]
    add_cols = [c[1] for c in nnz_list if c not in nnz_mat]
    # Store fill_value in all nnz_set coordinates that are zero in the matrix
    out[add_rows, add_cols] = fill_value
    return out
