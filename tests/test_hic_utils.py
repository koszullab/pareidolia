
#!/usr/bin/env python3

"""
Unit tests for functions in hic_utils.py
cmdoret, 20200407
"""
import pathlib
import pytest
import numpy as np
import pandas as pd
import cooler
import pareidolia.hic_utils as pah
import pareidolia.io as pai

DATA = pathlib.Path("data_test")
COOLS = [str(c) for c in DATA.glob("A_[1-6]*.cool")]
COOLS_COMP = [str(c) for c in DATA.glob("B_[1-6]*.cool")]
COOL_IN = ('cool', COOLS)
REGION = 'chr0:100000-120000'

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
    coords = pd.DataFrame({
        'chrom': ['chr0', 'chr0', 'chr0', 'chr0'],
        'pos': [0, 100004, 1444999, 1445000]
    })
    exp_idx = np.array([0, 20, 288, np.nan])
    obs_idx = pah.coords_to_bins(clr, coords)
    np.testing.assert_equal(obs_idx, exp_idx)

def test_change_detection_pipeline():
    """Test if change detection pipeline works without errors"""
    cools = COOLS + COOLS_COMP
    conds = ["A"] * len(COOLS) + ["B"] * len(COOLS_COMP)
    pah.change_detection_pipeline(cools, conds)
    