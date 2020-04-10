#!/usr/bin/env python3

"""
Functions used to apply Hi-C specific transformations and interact with cooler
objects.
cmdoret, 20200404
"""

import sys
from typing import Iterable, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cooler
import chromosight.utils.preprocessing as cup
import chromosight.utils.detection as cud
import chromosight.utils.io as cio
import chromosight.kernels as ck
import pareidolia.io as pai
import pareidolia.preprocess as pap
import pareidolia.detection as pad


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
    bins = clr.bins()
    if region is None:
        mat = mat[:]
        bins = bins[:]
    else:
        mat = mat.fetch(region)
        bins = bins.fetch(region)
    try:
        biases = bins["weight"].values
    except KeyError as err:
        sys.stderr.write("Error: Input cooler must be balanced.\n")
        raise err
    # get to same coverage
    if min_contacts is not None:
        mat = cup.subsample_contacts(mat, min_contacts).tocoo()
    valid = cup.get_detectable_bins(mat, n_mads=5)

    # balance region with weights precomputed on the whole matrix
    mat.data = mat.data * biases[mat.row] * biases[mat.col]
    # Detrend for P(s)
    mat = cup.detrend(mat.tocsr(), smooth=True, detectable_bins=valid[0])
    # Replace NaNs by 0s
    mat.data = np.nan_to_num(mat.data)
    mat.eliminate_zeros()
    return mat


def coords_to_bins(clr: cooler.Cooler, coords: pd.DataFrame) -> np.ndarray:
    """
        Converts genomic coordinates to a list of bin ids based on the whole genome
        contact map.

        Parameters
        ----------
        coords : pandas.DataFrame
            Table of genomic coordinates, with columns chrom, pos.

        Returns
        -------
        numpy.array of ints :
            Indices in the whole genome matrix contact map.

        """
    coords.pos = (coords.pos // clr.binsize) * clr.binsize
    # Coordinates are merged with bins, both indices are kept in memory so that
    # the indices of matching bins can be returned in the order of the input
    # coordinates
    idx = (
        clr.bins()[:]
        .reset_index()
        .rename(columns={"index": "bin_idx"})
        .merge(
            coords.reset_index().rename(columns={"index": "coord_idx"}),
            left_on=["chrom", "start"],
            right_on=["chrom", "pos"],
            how="right",
        )
        .set_index("bin_idx")
        .sort_values("coord_idx")
        .index.values
    )
    return idx


def change_detection_pipeline(
    cool_files: Iterable[str],
    conditions: Iterable[str],
    kernel: Union[np.ndarray, str] = "loops",
    bed2d_file: Optional[str] = None,
    region: Optional[str] = None,
    max_dist: Optional[int] = None,
    subsample: bool = True,
    percentile_thresh: float = 95,
) -> pd.DataFrame:
    """
    Run end to end pattern change detection pipeline on input cool files. A
    list of conditions of the same lengths as the sample list must be provided.

    Changes for a specific pattern are computed. A valid chromosight pattern
    name can be supplied (e.g. loops, borders, hairpins, ...) or a kernel matrix
    can be supplied directly instead.

    Positions with significant changes will be reported in a pandas
    dataframe. Optionally, a 2D bed file with positions of interest can be
    specified, in which case change value at these positions will be reported
    instead.
    """
    # Make sure each sample has an associated condition
    if len(cool_files) != len(conditions):
        raise ValueError(
            "The lists of cool files and conditions must have the same length"
        )

    # If a pattern name was provided, load corresponding chromosight kernel
    if isinstance(kernel, str):
        kernel = getattr(ck, kernel)["kernels"][0]
    # Associate samples with their conditions
    samples = pd.DataFrame(
        {"cond": conditions, "cool": pai.get_coolers(cool_files)}
    )
    # Remember condition values in a fixed order
    conditions = np.unique(conditions)
    # Define range of interest from region
    clr = samples.cool.values[0]
    if region is None:
        s, e = 0, clr.shape[0]
        bins = clr.bins()[:]
    else:
        s, e = clr.extent(region)
        bins = clr.bins().fetch(region).reset_index(drop=True)
    # Compute number of contacts in the matrix with the lowest coverage
    if subsample:
        min_contacts = get_min_contacts(samples.cool, region=region)
    else:
        min_contacts = None
    # Preprocess all matrices (subsample, balance, detrend)
    samples["mat"] = samples.cool.apply(
        lambda clr: preprocess_hic(
            clr, min_contacts=min_contacts, region=region
        )
    )
    # Retrieve the indices of bins which are valid in all samples (not missing
    # because of repeated sequences or low coverage)
    common_bins = pap.get_common_valid_bins(samples.mat)
    # Generate a missing mask from these bins
    missing_mask = cup.make_missing_mask(
        samples.mat[0].shape,
        common_bins,
        common_bins,
        max_dist=max_dist,
        sym_upper=True,
    )
    # Remove all missing values form each sample's matrix
    samples.mat = samples.mat.apply(
        lambda mat: cup.erase_missing(sp.triu(mat), common_bins, common_bins)
    )
    # Generate correlation maps for all samples using chromosight's algorithm
    samples["corr"] = samples.mat.apply(
        lambda mat: cud.normxcorr2(
            mat, kernel, full=True, missing_mask=missing_mask, sym_upper=True,
        )[0]
    )
    # Get the union of nonzero coordinates across all samples
    total_nnz_set = pap.get_nnz_set(samples["corr"], mode="union")
    # Store explicit zeros at these coordinates
    samples["corr"] = samples["corr"].apply(
        lambda cor: pap.fill_nnz(cor, total_nnz_set)
    )
    # Compute background for each condition
    backgrounds = samples.groupby("cond")["corr"].apply(
        lambda g: pad.median_bg(g.reset_index(drop=True))
    )
    # Get the distribution of difference to background within each condition
    within_diffs = np.hstack(
        samples.groupby("cond")["corr"].apply(
            lambda g: pad.reps_bg_diff(g.reset_index(drop=True))
        )
    )
    # Use average difference to first background as change metric
    diff = sp.csr_matrix(backgrounds[0].shape)
    for c in conditions[1:]:
        diff += backgrounds[c] - backgrounds[conditions[0]]
    diff.data /= len(backgrounds) - 1
    # Apply threshold to differences based on within-condition variations
    thresh = np.percentile(
        abs(within_diffs[within_diffs != 0]), percentile_thresh
    )
    diff.data[np.abs(diff.data) < thresh] = 0.0
    # If positions were provided, return the change value for each of them
    if bed2d_file:
        positions = cio.load_bed2d(bed2d_file)
        # Convert both coordinates from genomic coords to bins
        for i in [1, 2]:
            positions["pos"] = (
                positions[f"start{i}"] + positions[f"end{i}"]
            ) // 2
            positions.chrom = positions[f"chrom{i}"]
            positions[f"bin{i}"] = coords_to_bins(clr, positions)
        positions = positions.drop(columns=["pos", "chrom"])
        # Subset positions to region of interest
    # Otherwise report individual spots of change using chromosight
    else:
        # Pick "foci" of changed pixels and their local maxima
        positions, _ = cud.picker(abs(diff), thresh)
        # Get genomic positions from matrix coordinates
        positions = pd.DataFrame(positions, columns=["bin1", "bin2"])
        for i in [1, 2]:
            coords = (
                bins.loc[positions[f"bin{i}"], ["chrom", "start", "end"]]
                .reset_index(drop=True)
                .rename(
                    columns={
                        "chrom": f"chrom{i}",
                        "start": f"start{i}",
                        "end": f"end{i}",
                    }
                )
            )
            positions = pd.concat([coords, positions], axis=1)
    # Retrieve diff values for each coordinate
    positions["diff"] = positions.apply(lambda p: diff[p.bin1, p.bin2], axis=1)
    positions = positions.loc[
        :,
        [
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            "bin1",
            "bin2",
            "diff",
        ],
    ]
    return positions
