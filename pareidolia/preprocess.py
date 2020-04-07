#!/usr/bin/env python3

"""
Functions used to clean and prepare sparse matrices for detection.
cmdoret, 20200403
"""
from typing import Iterable, Tuple, Iterator, Set
import numpy as np
import scipy.sparse as sp
import chromosight.utils.detection as cud
import chromosight.utils.preprocessing as cup


def get_common_valid_bins(
    mats: Iterable[sp.csr_matrix], n_mads: float = 5,
) -> np.ndarray:
    """
    Generates an array of valid bins indices, using the intersection
    of valid bins from all input sparse matrices. All input matrices must
    be square and have the same shape.
    """
    common_valid = None
    for mat in mats:
        if mat.shape[0] != mat.shape[1]:
            NotImplementedError("Only square matrices are valid input.")
        # Get the list of valid bins in the current matrix
        valid = cup.get_detectable_bins(mat, n_mads=n_mads)
        # Initialize set of common bins with the first matrix
        if common_valid is None:
            common_valid = set(valid[0])
        # Remove elements absent from current matrix from the common set
        else:
            common_valid = common_valid.intersection(set(valid[0]))
    return np.array(list(common_valid))


def get_correlation(mat: sp.csr_matrix, kernel: np.ndarray) -> sp.csr_matrix:
    """Get the pearson coefficient map between input matrix and kernel."""
    return cud.normxcorr2(
        mat, kernel, full=False, missing_mask=None, sym_upper=True, max_dist=50
    )


def yield_nnz(mat: sp.spmatrix) -> Iterator[Tuple[int]]:
    """
    Helper function to extract nonzero values from a scipy.sparse matrix and
    returns an iterator of nonzero coordinates, in the form of (row, col)
    tuples.
    """
    # Split nonzero coordinates into rows and columns
    nnr, nnc = mat.nonzero()
    nnr = nnr.tolist()
    nnc = nnc.tolist()
    return zip(nnr, nnc)


def get_nnz_set(
    mats: Iterable[sp.csr_matrix], mode: str = "intersection"
) -> Set[Tuple[int]]:
    """
    Given an arbitrary number of sparse matrices, build a set containing the
    intersection or union of nonzero coordinates from all matrices. Each
    coordinate is stored in the form of (row, col) tuples.
    """
    # Check for input type
    try:
        if np.all([m.format == "csr" for m in mats]):
            for i, mat in enumerate(mats):
                # Use first matrix to initialize set
                if i == 0:
                    nnz_set = set(yield_nnz(mat))
                # Iteratively reduce set by keeping only elements present in each matrix
                else:
                    if mode == "union":
                        nnz_set = nnz_set.union(set(yield_nnz(mat)))
                    elif mode == "intersection":
                        nnz_set = nnz_set.intersection(set(yield_nnz(mat)))
                    else:
                        raise ValueError(
                            "mode must be either 'union' or 'intersection'"
                        )
        else:
            raise ValueError("input sparse matrices must be in csr format")
    except AttributeError:
        raise TypeError("Input type must be scipy.sparse.csr_matrix")

    return nnz_set


def filter_nnz(mat: sp.csr_matrix, nnz_set: Set[Tuple[int]]) -> sp.csr_matrix:
    """
    Given an input sparse matrix and a set of nonzero coordinates, filter the
    matrix to keep only positions present in the set.
    """
    out = mat.copy()
    drop_coords_mask = np.array(
        [n not in nnz_set for n in yield_nnz(mat)], dtype=bool
    )
    out.data[drop_coords_mask] = 0.0
    out.eliminate_zeros()
    return out


def fill_nnz(
    mat: sp.csr_matrix, nnz_set: Set[Tuple[int]], fill_value: float = 0.0
) -> sp.csr_matrix:
    """
    Given an input sparse matrix and a set of nonzero coordinates, fill the
    matrix to ensure all values in the set are stored explicitely.
    """
    # Get the set of nonzero coordinate in the input matrix
    nnz_mat = set(yield_nnz(mat))
    nnz_list = list(nnz_set)
    out = mat.copy()
    # get all all nnz_set coordinates that are zero in the matrix
    add_rows = [c[0] for c in nnz_list if c not in nnz_mat]
    add_cols = [c[1] for c in nnz_list if c not in nnz_mat]
    # Replace implicit zeros by fill_value at these coordinates
    out[add_rows, add_cols] = fill_value
    return out
