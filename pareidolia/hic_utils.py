#!/usr/bin/env python3

"""
Functions used to apply Hi-C specific transformations and interact with cooler
objects.
cmdoret, 20200404
"""

import sys
import itertools as it
from typing import Iterable, Optional, Union, Tuple
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
import multiprocessing as mp


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
    # get to same coverage if requested and matrix is not empty
    if mat.sum() and (min_contacts is not None):
        mat = cup.subsample_contacts(mat, min_contacts).tocoo()
    valid = cup.get_detectable_bins(mat, n_mads=5)

    # balance region with weights precomputed on the whole matrix
    mat.data = mat.data * biases[mat.row] * biases[mat.col]
    # Detrend for P(s)
    mat = cup.detrend(mat.tocsr(), smooth=False, detectable_bins=valid[0])
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


def detection_matrix(
    samples: pd.DataFrame,
    kernel: np.ndarray,
    region: Optional[str] = None,
    subsample: Optional[int] = None,
    max_dist: Optional[int] = None,
    percentile_thresh: Optional[float] = 95.0,
    n_cpus: int = 4,
) -> Tuple[Optional[sp.csr_matrix], Optional[float]]:
    """
    Run the detection process for a single matrix. This is abstracted from all
    notions of chromosomes and genomic coordinates.
    """
    # We consider the matrix is symmetric upper (i.e. intrachromosomal)
    sym_upper = True
    # Compute number of contacts in the matrix with the lowest coverage
    if subsample:
        min_contacts = get_min_contacts(samples.cool, region=region)
    else:
        min_contacts = None
    # Define the condition of the first sample as the baseline condition
    control = samples.cond.values[0]
    # Preprocess all matrices (subsample, balance, detrend)
    # Samples pocessed in parallel if requested
    if n_cpus > 1:
        pool = mp.Pool(n_cpus)
        map_fun = pool.starmap
    else:
        map_fun = lambda x, y: [x(*args) for args in y]

    samples["mat"] = map_fun(
        preprocess_hic,
        zip(samples.cool, it.repeat(min_contacts), it.repeat(region)),
    )
    print(f"{region} preprocessed", file=sys.stderr)

    # Return nothing if the matrix is smaller than kernel
    if np.any(np.array(samples["mat"][0].shape) <= np.array(kernel.shape)):
        return None, None
    # Retrieve the indices of bins which are valid in all samples (not missing
    # because of repeated sequences or low coverage)
    common_bins = pap.get_common_valid_bins(samples["mat"])
    # Trim diagonals beyond max_dist to spare resources
    samples["mat"] = map_fun(
        cup.diag_trim, zip(samples["mat"], it.repeat(max_dist))
    )
    # Generate a missing mask from these bins
    missing_mask = cup.make_missing_mask(
        samples["mat"][0].shape,
        common_bins,
        common_bins,
        max_dist=max_dist,
        sym_upper=sym_upper,
    )
    # Remove all missing values form each sample's matrix
    samples["mat"] = map_fun(
        cup.erase_missing,
        zip(
            map(sp.triu, samples["mat"]),
            it.repeat(common_bins),
            it.repeat(common_bins),
            it.repeat(sym_upper),
        ),
    )
    print(f"{region} missing bins erased", file=sys.stderr)
    # Generate correlation maps for all samples using chromosight's algorithm
    corrs = map_fun(
        cud.normxcorr2,
        zip(
            samples.mat.values,
            it.repeat(kernel),
            it.repeat(max_dist),
            it.repeat(True),
            it.repeat(True),
            it.repeat(missing_mask),
            it.repeat(0.75),
            it.repeat(None),
            it.repeat(False),
        ),
    )
    samples["mat"] = [tup[0] for tup in corrs]
    del corrs
    print(f"{region} correlation matrices computed", file=sys.stderr)
    # Get the union of nonzero coordinates across all samples
    total_nnz_set = pap.get_nnz_union(samples["mat"])
    # Fill zeros at these coordinates
    samples["mat"] = samples["mat"].apply(
        lambda cor: pap.fill_nnz(cor, total_nnz_set)
    )
    if n_cpus > 1:
        pool.close()
    # Compute background for each condition
    backgrounds = samples.groupby("cond")["mat"].apply(
        lambda g: pad.median_bg(g.reset_index(drop=True))
    )
    # Get the distribution of difference to background within each condition
    within_diffs = np.hstack(
        samples.groupby("cond")["mat"].apply(
            lambda g: pad.reps_bg_diff(g.reset_index(drop=True))
        )
    )
    # Use average difference to first background as change metric
    diff = sp.csr_matrix(backgrounds[0].shape)
    conditions = np.unique(samples.cond)
    conditions = conditions[conditions != control]
    for c in conditions:
        diff += backgrounds[c] - backgrounds[control]
    diff.data /= len(backgrounds) - 1
    # Apply threshold to differences based on within-condition variations
    try:
        if percentile_thresh is None:
            thresh = None
        else:
            thresh = np.percentile(
                abs(within_diffs[within_diffs != 0]), percentile_thresh
            )
            diff.data[np.abs(diff.data) < thresh] = 0.0
    # If there is no nonzero value (e.g. very small matrices), return nothing
    except IndexError:
        diff = None
        thresh = None

    return diff, thresh


def change_detection_pipeline(
    cool_files: Iterable[str],
    conditions: Iterable[str],
    kernel: Union[np.ndarray, str] = "loops",
    bed2d_file: Optional[str] = None,
    region: Optional[Union[Iterable[str], str]] = None,
    max_dist: Optional[int] = None,
    subsample: bool = True,
    percentile_thresh: Optional[float] = 95.0,
    n_cpus: int = 4,
) -> pd.DataFrame:
    """
    Run end to end pattern change detection pipeline on input cool files. A
    list of conditions of the same lengths as the sample list must be provided.
    The first condition in the list is used as the reference (control) state.

    Changes for a specific pattern are computed. A valid chromosight pattern
    name can be supplied (e.g. loops, borders, hairpins, ...) or a kernel
    matrix can be supplied directly instead. maximum scanning distance can be
    specified directly (in basepairs) to override the kernel default value.

    Positions with significant changes will be reported in a pandas
    dataframe. Significance is determined based on the percentile threshold,
    between 1 and 100. Optionally, a 2D bed file with positions of interest can
    be specified, in which case change value at these positions will be
    reported instead. When using a bed2d file, the threshold is optional (one
    can report either scores at all positions, or only where they are
    significant).

    Positive diff_scores mean the pattern intensity was increased relative to
    control (first condition).
    """
    # Make sure each sample has an associated condition
    if len(cool_files) != len(conditions):
        raise ValueError(
            "The lists of cool files and conditions must have the same length"
        )

    # If a pattern name was provided, load corresponding chromosight kernel
    if isinstance(kernel, str):
        kernel_name = kernel
        try:
            kernel = getattr(ck, kernel_name)["kernels"][0]
            if max_dist is None:
                max_dist = getattr(ck, kernel_name)["max_dist"]
        except AttributeError:
            raise AttributeError(f"{kernel_name} is not a valid pattern name")
    elif isinstance(kernel, np.ndarray):
        kernel_name = "custom kernel"
    else:
        raise ValueError(
            "Kernel must either be a valid chromosight pattern name, or a 2D numpy.ndarray of floats"
        )
    # Associate samples with their conditions
    samples = pd.DataFrame(
        {"cond": conditions, "cool": pai.get_coolers(cool_files)}
    )
    print(
        f"Changes will be computed relative to condition: {samples.cond.values[0]}"
    )
    # Define each chromosome as a region, if None specified
    clr = samples.cool.values[0]
    if max_dist is not None:
        max_dist = max_dist // clr.binsize
    if region is None:
        regions = clr.chroms()[:]["name"].tolist()
    elif isinstance(region, str):
        region = [region]
    pos_cols = [
        "chrom1",
        "start1",
        "end1",
        "chrom2",
        "start2",
        "end2",
        "bin1",
        "bin2",
        "diff_score",
    ]
    if bed2d_file:
        positions = cio.load_bed2d(bed2d_file)
        for col in ["diff_score", "bin1", "bin2"]:
            positions[col] = np.nan
    else:
        positions = pd.DataFrame(columns=pos_cols)
    for reg in regions:
        # Subset bins to the range of interest
        bins = clr.bins().fetch(reg).reset_index(drop=True)
        diff, thresh = detection_matrix(
            samples,
            kernel,
            region=reg,
            subsample=subsample,
            max_dist=max_dist,
            percentile_thresh=percentile_thresh,
            n_cpus=n_cpus,
        )
        # If the matrix was too small, skip it
        if thresh is None and percentile_thresh is not None:
            continue
        # If positions were provided, return the change value for each of them
        if bed2d_file:
            tmp_chr = reg.split(":")[0]
            tmp_rows = (positions.chrom1 == tmp_chr) & (
                positions.chrom2 == tmp_chr
            )
            # If there are no positions of interest on this chromosome, just
            # skip it
            if not np.any(tmp_rows):
                continue
            tmp_pos = positions.loc[tmp_rows, :]
            # Convert both coordinates from genomic coords to bins
            for i in [1, 2]:
                tmp_pos["chrom"] = tmp_pos[f"chrom{i}"]
                tmp_pos["pos"] = (
                    tmp_pos[f"start{i}"] + tmp_pos[f"end{i}"]
                ) // 2
                tmp_pos[f"bin{i}"] = coords_to_bins(clr, tmp_pos).astype(int)
                # Save bin coordinates from current chromosome to the full table
                positions.loc[tmp_rows, f"bin{i}"] = tmp_pos[f"bin{i}"]
            tmp_pos = tmp_pos.drop(columns=["pos", "chrom"])
            # Retrieve diff values for each coordinate
            positions.loc[tmp_rows, "diff_score"] = diff[
                tmp_pos.start1 // clr.binsize, tmp_pos.start2 // clr.binsize
            ].A1
        # Otherwise report individual spots of change using chromosight
        else:
            # Pick "foci" of changed pixels and their local maxima
            tmp_pos, _ = cud.picker(abs(diff), thresh)
            # Get genomic positions from matrix coordinates
            tmp_pos = pd.DataFrame(tmp_pos, columns=["bin1", "bin2"])
            for i in [1, 2]:
                coords = (
                    bins.loc[tmp_pos[f"bin{i}"], ["chrom", "start", "end"]]
                    .reset_index(drop=True)
                    .rename(
                        columns={
                            "chrom": f"chrom{i}",
                            "start": f"start{i}",
                            "end": f"end{i}",
                        }
                    )
                )
                # Add axis' columns to  dataframe
                tmp_pos = pd.concat([coords, tmp_pos], axis=1)
            # Retrieve diff values for each coordinate
            tmp_pos["diff_score"] = diff[tmp_pos.bin1, tmp_pos.bin2].A1
            # Append new chromosome's rows
            positions = pd.concat([positions, tmp_pos], axis=0)
    positions = positions.loc[:, pos_cols]
    return positions
