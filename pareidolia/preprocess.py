#!/usr/bin/env python3

"""
Functions used to clean and prepare sparse matrices for detection.
cmdoret, 20200403
"""
from typing import Iterable, Tuple, Iterator, Set
import numpy as np
import scipy.sparse as sp
import chromosight.utils.preprocessing as cup


def get_common_valid_bins(
    mats: Iterable['sp.csr_matrix[float]'], n_mads: float = 5,
) -> 'np.ndarray[int]':
    """
    Generates an array of valid bins indices, using the intersection
    of valid bins from all input sparse matrices. All input matrices must
    be square and have the same shape. Valid bins are defined based on their
    proportion of nonzero pixels.
    """
    common_valid = None
    for mat in mats:
        if mat.shape[0] != mat.shape[1]:
            raise NotImplementedError("Only square matrices are valid input.")
        # Get the list of valid bins in the current matrix
        valid = cup.get_detectable_bins(mat, n_mads=n_mads)
        # Initialize set of common bins with the first matrix
        if common_valid is None:
            common_valid = set(valid[0])
        # Remove elements absent from current matrix from the common set
        else:
            common_valid = common_valid.intersection(set(valid[0]))
    return np.array(list(common_valid))


def get_nnz_union(mats: Iterable["sp.csr_matrix[float]"]) -> "np.ndarray[int]":
    """
    Given a list of sparse matrices, return the union of their nonzero
    coordinates, in the form of a 2D numpy array with 1 coordinate per
    row, with 2 columns representing coordinates rows and columns.
    """
    try:
        # Check for input type
        if np.all([m.format == "csr" for m in mats]):
            for i, mat in enumerate(mats):
                # Use first matrix to initialize set
                if i == 0:
                    union_mat = mat.copy()
                # Iteratively sum matrices
                else:
                    union_mat += mat
                union_mat.eliminate_zeros()
            # Retrieve positions of nonzero entries into an array
            all_nnz = np.ascontiguousarray(np.vstack(union_mat.nonzero()).T)
        else:
            raise ValueError("input sparse matrices must be in csr format")
    except AttributeError:
        raise TypeError("Input type must be scipy.sparse.csr_matrix")

    return all_nnz


def fill_nnz(
    mat: "sp.csr_matrix", all_nnz: "np.ndarray[int]", fill_value: float = 1e-9
) -> sp.csr_matrix:
    """
    Given an input sparse matrix and a superset of nonzero coordinates, fill the
    matrix to ensure all values in the set are stored explicitely with the
    value of fill_value.
    """
    # Get the set of nonzero coordinate in the input matrix
    mat_nnz = np.ascontiguousarray(np.vstack(mat.nonzero()).T)
    out = mat.copy()
    ncols = all_nnz.shape[1]
    # Tricking numpy into treating rows as single values using a custom dtype
    # based on: https://stackoverflow.com/a/8317403/8440675
    dtype = {
        "names": [f"f{i}" for i in range(ncols)],
        "formats": ncols * [all_nnz.dtype],
    }
    # get all all_nnz coordinates that are zero in the matrix
    add_mask = np.in1d(all_nnz.view(dtype), mat_nnz.view(dtype), invert=True)
    # Replace implicit zeros by fill_value at these coordinates
    out[all_nnz[add_mask, 0], all_nnz[add_mask, 1]] = fill_value
    return out
