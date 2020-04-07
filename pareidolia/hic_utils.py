#!/usr/bin/env python3

"""
Functions used to apply Hi-C specific transformations and interact with cooler
objects.
cmdoret, 20200404
"""

from typing import Iterable, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cooler
import chromosight.utils.preprocessing as cup
import pareidolia.io as pio


def get_min_contacts(
    coolers: Iterable[cooler.Cooler], region: Optional[str] = None
) -> int:
    """
    Given several cooler objects, returns the number of contacts in the least
    covered matrix. Optionally, a region can be given in UCSC format, in which
    case coverage will be computed only in the matrix from that region.
    """
    contacts = np.zeros(len(coolers))
    # Get coverage for each cool file
    for i, clr in enumerate(coolers):
        if region is None:
            contacts[i] = clr.info["sum"]
        else:
            contacts[i] = (
                clr.matrix(balance=False, sparse=True).fetch(region).sum()
            )
    # Return the minimum coverage value (in number of contacts)
    return int(min(contacts))


def preprocess_hic(
    clr: cooler.Cooler,
    min_contacts: Optional[int] = None,
    region: Optional[str] = None,
) -> sp.csr_matrix:
    """
    Given an input cooler object, returns the preprocessed Hi-C matrix.
    Preprocessing involves (in that order): subsetting region, subsampling
    contacts, normalisation, detrending (obs / exp). Balancing weights must
    be pre-computer in the referenced cool file. Region must be in UCSC format.
    """
    # Load raw matrix and subset region if requested
    mat = clr.matrix(sparse=True, balance=False)
    if region is None:
        mat = mat[:]
    else:
        mat = mat.fetch(region)
    # get to same coverage
    if min_contacts is not None:
        mat = cup.subsample_contacts(mat, min_contacts).tocoo()
    valid = cup.get_detectable_bins(mat, n_mads=5)

    # balance region with weights precomputed on the whole matrix
    biases = clr.bins().fetch(region)["weight"].values
    mat.data = mat.data * biases[mat.row] * biases[mat.col]
    # Detrend for P(s)
    mat = cup.detrend(mat.tocsr(), smooth=True, detectable_bins=valid[0])
    # Replace NaNs by 0s
    mat.data = np.nan_to_num(mat.data)
    mat.eliminate_zeros()
    return mat


def change_detection_pipeline(
    cool_files: Iterable[str],
    conditions: Iterable[str],
    bed2d_file: Optional[str],
    region: str = None,
) -> pd.DataFrame:
    """
    Run end to end change detection pipeline on input cool files. A list of
    conditions of the same lengths as the sample list must be provided.
    Positions with significant changes will be reported in a pandas
    dataframe. Optionally, a 2D bed file with positions of interest can be
    specified, in which case
    """
    if len(cool_files) != len(conditions):
        raise ValueError(
            "The lists of cool files and conditions must have the same length"
        )
    samples = pd.DataFrame(
        {"cond": conditions, "cool": pio.get_cools(cool_files)}
    )
