#!/usr/bin/env python3

"""
Unit tests for functions in hic_utils.py
cmdoret, 20200407
"""
import itertools as it
import pathlib
import pytest
import numpy as np
import pandas as pd
import cooler
import pareidolia.hic_utils as pah
import pareidolia.io as pai

DATA = pathlib.Path("data_test")
# Synthetic matrices and their known loop coordinates
COOLS = [str(c) for c in DATA.glob("A_[1-6]*.cool")]
LOOPS = np.loadtxt(DATA / "A_loops.txt")
# Matrices with a diagonal gradient
COOLS_COMP = [str(c) for c in DATA.glob("smooth_[1-6]*.cool")]
COOL_IN = ("cool", COOLS)
REGION = "chr0:100000-120000"


def test_get_min_contacts():
    """Test if lowest contact value is found correctly"""
    min_exp = 44035
    min_obs = pah.get_min_contacts(pai.get_coolers(COOLS))
    assert min_obs == min_exp


def test_get_min_contacts_region():
    """Test if lowest contact value is found correctly"""
    min_exp = 376
    min_obs = pah.get_min_contacts(pai.get_coolers(COOLS), REGION)
    assert min_obs == min_exp


@pytest.mark.parametrize(*COOL_IN)
def test_preprocess_hic(cool):
    """Test if preprocessing pipeline works without errors"""
    clr = cooler.Cooler(cool)
    pah.preprocess_hic(clr)
    pah.preprocess_hic(clr, region=REGION)
    pah.preprocess_hic(clr, min_contacts=44035)
    pah.preprocess_hic(clr, min_contacts=376, region=REGION)


def test_coords_to_bins():
    """Test if coordinates are converted correctly"""
    clr = cooler.Cooler(COOLS[0])
    # Note: last input coords is our of bounds, np.nan expected
    coords = pd.DataFrame(
        {
            "chrom": ["chr0", "chr0", "chr0", "chr0"],
            "pos": [0, 100004, 1444999, 1445000],
        }
    )
    exp_idx = np.array([0, 20, 288, np.nan])
    obs_idx = pah.coords_to_bins(clr, coords)
    np.testing.assert_equal(obs_idx, exp_idx)


def test_change_detection():
    """Test if change detection pipeline finds some relevant positions"""
    # Run loop change detection between matrices with and without loops
    cools = COOLS + COOLS_COMP
    conds = ["A"] * len(COOLS) + ["B"] * len(COOLS_COMP)
    obs_pos = pah.change_detection_pipeline(
        cools, conds, subsample=False, percentile_thresh=95, mode="median"
    )
    # Build a set of fuzzy (+/3 pixels around) positions found
    valid_pos = set()
    for pos in obs_pos.loc[:, ["bin1", "bin2"]].values:
        for shift in it.combinations(range(-3, 4), 2):
            valid_pos.add((pos[0] + shift[0], pos[1] + shift[1]))
    # Count the number of real loop positions that were found
    found = 0
    for target in LOOPS:
        if tuple(target.astype(int)) in valid_pos:
            found += 1
    assert found / LOOPS.shape[0] >= 0.3


def test_change_quantification():
    """Test if change detection pipeline change at input positions"""
    # Run loop change detection between matrices with and without loops
    cools = COOLS + COOLS_COMP
    conds = ["A"] * len(COOLS) + ["B"] * len(COOLS_COMP)
    obs_pos = pah.change_detection_pipeline(
        cools,
        conds,
        bed2d_file=str(DATA / "A_loops.bed2d"),
        subsample=False,
        percentile_thresh=95,
    )
    diff = obs_pos.diff_score
    # Check if change was detected in at least 30% of the positions
    assert len(diff[diff < 0]) >= len(diff) * 0.3


def test_change_no_threshold():
    """Test if change detection pipeline without threshold reports all input positions"""
    # Run loop change detection between matrices with and without loops
    cools = COOLS + COOLS_COMP
    conds = ["A"] * len(COOLS) + ["B"] * len(COOLS_COMP)
    bed2d = pd.read_csv(str(DATA / "A_loops.bed2d"), sep="\t")
    obs_pos = pah.change_detection_pipeline(
        cools,
        conds,
        bed2d_file=str(DATA / "A_loops.bed2d"),
        subsample=False,
        percentile_thresh=None,
    )
    diff = obs_pos.diff_score
    # Check if diff scores are returned for all positions
    assert len(diff[~np.isnan(diff)]) == bed2d.shape[0]
