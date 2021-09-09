#!/usr/bin/env python3

"""
Functions used to apply Hi-C specific transformations and interact with cooler
objects.
cmdoret, 20200404
"""

import sys
import itertools as it
from typing import Iterable, Optional, Union, Tuple
from functools import reduce
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cooler
import multiprocessing as mp
import chromosight.utils.preprocessing as cup
import chromosight.utils.detection as cud
import chromosight.utils.io as cio
import chromosight.kernels as ck
import pareidolia.io as pai
import pareidolia.preprocess as pap
import pareidolia.detection as pad
import pareidolia.stats as pas


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


def make_density_filter(
    mats: Iterable[sp.csr_matrix],
    density_thresh: float = 0.10,
    win_size=3,
    sym_upper=False,
) -> sp.csr_matrix:
    """Given a list of sparse matrices, generate a "density filter". This
    new sparse matrix is a boolean mask where values indicate whether the
    proportion of nonzero pixels in the neighbourhood of diameter win_size
    is above the input threshold in all input matrices.

    Parameters
    ----------
    mats : Iterable of scipy.sparse.csr_matrix
        The matrices to be combined into a filter.
    density_thresh : float
        The required proportion of nonzero pixels in the neighbourhood pass
        the filter.
    win_size : int
        The diameter of the neighbourhood in which to compute the proportion
        of nonzero pixels.
    sym_upper : bool
        Whether the matrix is symmetric upper. In this case, computations are
        performed in the upper triangle.

    Returns
    -------
    scipy.sparse.csr_matrix of bools:
        The sparse boolean mask representing the density filter. Values are
        True where all input matrices passed the threshold.
    """

    # Filter out regions where contacts were too sparse in all contact matrices
    # Pixel density is first converted to a boolean matrix (pass / fail)
    filters = map(
        lambda m: pad.get_win_density(
            m, win_size=win_size, sym_upper=sym_upper
        )
        > density_thresh,
        mats,
    )
    # matrices are reduced through an element-wise multiplication acting as an AND
    # filter: AND(AND(AND(m1, m2), m3), m4)
    inter_filter = reduce(lambda x, y: x.multiply(y), filters)
    return inter_filter


def _ttest_matrix(
    samples: pd.DataFrame, control: str
) -> Tuple[sp.csr_matrix, float]:
    """
    Performs pixel-wise t-test comparisons between conditions to detect differential
    interactions.
    """
    # Compute background for each condition
    arr_control = np.dstack(
        [m.data for m in samples.mat[samples.cond == control]]
    )[0, :, :]
    arr_alt = np.dstack(
        [m.data for m in samples.mat[samples.cond != control]]
    )[0, :, :]
    diff = samples["mat"][0]
    diff.data = pas.vec_ttest(arr_control, arr_alt)
    # The threshold is the t-value corresponding to p=0.05
    # thresh = pas.pval_to_tval(
    #    1 - percentile_thresh / 100,
    #    arr_control.shape[1] + arr_alt.shape[1],
    # )
    return diff


def _median_bg_subtraction(
    samples: pd.DataFrame,
    control: str,
    cnr_thresh: Optional[float] = 0.3,
    cnr_max: float = 10.0,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Performs the median background subtraction to extract differential signal
    from multiple Hi-C matrix. Returns the filtered differential matrix and
    the contrast-to-noise-ratio matrix used for the filtering.
    """
    # Compute background for each condition
    backgrounds = samples.groupby("cond")["mat"].apply(
        lambda g: pad.median_bg(g.reset_index(drop=True))
    )
    # Compute sse for each condition
    sse = samples.groupby("cond")["mat"].apply(
        lambda g: pad.get_sse_mat(g.reset_index(drop=True))
    )
    conditions = np.unique(samples.cond)
    # Compute difference between conditions and contrast
    # to noise ratio
    cnr = sse[0].copy()
    cnr.data = np.zeros(sse[0].data.shape)
    diff = sp.csr_matrix(sse[0].shape)
    for c in conditions:
        if c != control:
            # Break ties to preserve sparsity (do not introduce 0s)
            ties = backgrounds[c].data == backgrounds[control].data
            backgrounds[c].data[ties] += 1e-08
            curr_diff = backgrounds[c] - backgrounds[control]
            diff += curr_diff
            cnr.data += np.abs(curr_diff.data) / np.sqrt(sse[c].data)
    cnr.data /= len(conditions)
    # Erase spurious or extreme values
    cnr.data[cnr.data < 0.0] = 0.0
    cnr.data[cnr.data > cnr_max] = cnr_max
    # Use average difference to first background as change metric
    diff.data /= len(conditions) - 1
    # Threshold data on background / sse value
    if cnr_thresh is not None:
        diff.data[cnr.data < cnr_thresh] = 0.0
    return diff, cnr


def detection_matrix(
    samples: pd.DataFrame,
    kernel: np.ndarray,
    region: Optional[str] = None,
    subsample: Optional[int] = None,
    max_dist: Optional[int] = None,
    pearson_thresh: Optional[float] = None,
    density_thresh: Optional[float] = None,
    cnr_thresh: Optional[float] = 0.3,
    n_cpus: int = 4,
) -> Tuple[Optional[sp.csr_matrix], Optional[sp.csr_matrix]]:
    """
    Run the detection process for a single chromosome or region. This is abstracted from all
    notions of chromosomes and genomic coordinates.
    """
    # We consider the matrix is symmetric upper (i.e. intrachromosomal)
    sym_upper = True
    # Diagonals will be trimmed at max_dist with a margin for convolution
    if max_dist is None:
        trim_dist = None
    else:
        mat_size = samples.cool[0].matrix(sparse=True).fetch(region).shape[0]
        trim_dist = min(mat_size, max_dist + max(kernel.shape))
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

    # Hi-C specific preprocessing individual matrices (subsample, balance, detrend)
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
    # Trim diagonals beyond max_dist (with kernel margin for the convolution)
    # to spare resources
    if trim_dist is not None:
        samples["mat"] = map_fun(
            cup.diag_trim, zip(samples["mat"], it.repeat(trim_dist))
        )
    # Generate a missing mask from these bins
    missing_mask = cup.make_missing_mask(
        samples["mat"][0].shape,
        common_bins,
        common_bins,
        max_dist=trim_dist,
        sym_upper=sym_upper,
    )
    # Remove all missing values from each sample's matrix
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

    # Compute a density filter: regions with sufficient proportion of nonzero
    # pixels in kernel windows, in all samples. We will use it for downstream
    # which filter
    if (density_thresh is not None) and (density_thresh > 0):
        density_filter = make_density_filter(
            samples["mat"],
            density_thresh=density_thresh,
            win_size=kernel.shape[0],
            sym_upper=sym_upper,
        )
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
    # Erase pixels where all samples are below pearson threshold
    if pearson_thresh is not None:
        pearson_fail = [
            (m.data < pearson_thresh).astype(bool) for m in samples["mat"]
        ]
        pearson_fail = np.bitwise_and.reduce(pearson_fail)
        # Threshold maps using pearson correlations to reduce noisy detections
        for i, m in enumerate(samples["mat"]):
            m.data[pearson_fail] = 0.0
            samples["mat"][i] = m

    if n_cpus > 1:
        pool.close()

    # Use median background
    diff, cnr = _median_bg_subtraction(samples, control, cnr_thresh)

    # Erase pixels which do not pass the density filter in all samples
    if (density_thresh is not None) and (density_thresh > 0):
        diff = diff.multiply(density_filter)
    # Remove all values beyond user-specified max_dist
    if max_dist is not None:
        diff = cup.diag_trim(diff, max_dist + 2)
    return diff, cnr


def change_detection_pipeline(
    cool_files: Iterable[str],
    conditions: Iterable[str],
    kernel: Union[np.ndarray, str] = "loops",
    bed2d_file: Optional[str] = None,
    region: Optional[Union[Iterable[str], str]] = None,
    max_dist: Optional[int] = None,
    min_dist: Optional[int] = None,
    subsample: bool = True,
    pearson_thresh: Optional[float] = None,
    density_thresh: Optional[float] = 0.10,
    cnr_thresh: Optional[float] = 1.0,
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

    Positive diff_scores mean the pattern intensity was increased relative to
    control (first condition).

    Positions with significant changes will be reported in a pandas
    dataframe. In addition to the score, a contrast-to-noise ratio between 0 and
    10 is given to give an estimation of the signal quality. Optionally, a 2D
    bed file with positions of interest can be specified, in which case change
    value at these positions will be reported instead. When using a bed2d file.

    Parameters
    ----------
    cool_files :
        The list of paths to cool files for the input samples.
    conditions :
        The list of conditions matching the samples.
    kernel :
        Either the kernel to use as pattern as a numpy array, or the name of a
        valid chromosight pattern.
    bed2d_file :
        Path to a bed2D file containing a list of 2D positions. If this is
        provided, pattern changes at these coordinates will be quantified.
        Otherwise, they will be detected based on a threshold.
    region :
        Either a single UCSC format region string, or a list of multiple
        regions. The analysis will be restricted to those regions.
    max_dist :
        Maximum interaction distance (in basepairs) to consider in the analysis.
        If this is not specified and a chromosight kernel was specified, the
        default max_dist for that kernel is used. If the case of a custom kernel,
        the whole matrix will be scanned if no max_dist is specified.
    subsample :
        Whether all input matrices should be subsampled to the same number of
        contacts as the least covered sample.
    pearson_thresh :
        The pearson correlation threshold to use when detecting patterns. If None,
        the default value for the kernel is used.
    density_thresh :
        The pixel density threshold to require. Low coverage windows with a
        proportion of nonzero pixels below this value are discarded.
    n_cpus :
        Number of CPU cores to allocate for parallel operations.

    Returns
    -------
    pd.DataFrame :
        The list of reported 2D coordinates and their change intensities.
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
            if min_dist is None:
                min_dist = getattr(ck, kernel_name)["min_dist"]
            if pearson_thresh is None:
                pearson_thresh = getattr(ck, kernel_name)["pearson"]
        except AttributeError:
            raise AttributeError(f"{kernel_name} is not a valid pattern name")
        print(f"Loading default parameter for kernel '{kernel_name}'...")
        print(f"pearson_thresh: {pearson_thresh}")
        print(f"min_dist: {min_dist}")
        print(f"max_dist: {max_dist}")
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
    if min_dist is None:
        min_dist = 0
    else:
        min_dist = min_dist // clr.binsize
    if region is None:
        regions = clr.chroms()[:]["name"].tolist()
    elif isinstance(region, str):
        regions = [region]
    else:
        regions = region
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
        "cnr",
    ]
    if bed2d_file:
        positions = cio.load_bed2d(bed2d_file)
        for col in ["diff_score", "cnr", "bin1", "bin2"]:
            positions[col] = np.nan
    else:
        positions = pd.DataFrame(columns=pos_cols)
    for reg in regions:
        # Subset bins to the range of interest
        bins = clr.bins().fetch(reg).reset_index(drop=True)
        diff, cnr = detection_matrix(
            samples,
            kernel,
            region=reg,
            subsample=subsample,
            max_dist=max_dist,
            pearson_thresh=pearson_thresh,
            density_thresh=density_thresh,
            n_cpus=n_cpus,
            cnr_thresh=cnr_thresh,
        )

        # If the matrix was too small or no difference was found, skip it
        if diff is None or diff.nnz == 0:
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
            positions.loc[tmp_rows, "cnr"] = cnr[
                tmp_pos.start1 // clr.binsize, tmp_pos.start2 // clr.binsize
            ].A1
        # Otherwise report individual spots of change using chromosight
        else:
            # Pick "foci" of changed pixels and their local maxima
            tmp_pos, _ = cud.pick_foci(abs(diff), 0.01, min_size=3)
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
            try:
                tmp_pos["diff_score"] = diff[tmp_pos.bin1, tmp_pos.bin2].A1
            # No position found, go to next region
            except AttributeError:
                continue
            tmp_pos["cnr"] = cnr[tmp_pos.bin1, tmp_pos.bin2].A1
            # Append new chromosome's rows
            positions = pd.concat([positions, tmp_pos], axis=0)
            # For 1D patterns (e.g. borders) set diagonal positions.
            if max_dist == 0:
                positions[["bin1", "chrom1", "start1", "end1"]] = positions[
                    ["bin2", "chrom2", "start2", "end2"]
                ]
    positions = positions.loc[:, pos_cols]
    positions = positions.loc[
        abs(positions.bin2 - positions.bin1) >= min_dist, :
    ].reset_index(drop=True)
    print(positions)
    return positions
