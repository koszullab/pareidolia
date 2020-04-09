#!/usr/bin/env python3

"""
Unit tests for functions in hic_utils.py
cmdoret, 20200407
"""
import pathlib
import pytest
import pareidolia.io as pai
import cooler

DATA = pathlib.Path("data_test")
COOLS = [str(c) for c in DATA.glob("A_[1-6]*.cool")]


def test_get_coolers():
    """Test loading of cool files from multiple samples"""
    # Load multiple files
    assert len(pai.get_coolers(COOLS)) == len(COOLS)
    # Load single file correctly
    assert isinstance(pai.get_coolers([COOLS[0]])[0], cooler.Cooler)
    # Load single file wrong, should crash
    with pytest.raises(TypeError):
        assert pai.get_coolers(COOLS[0])
    # Dimensions unmatched, should give an explicit error
    with pytest.raises(ValueError) as err:
        assert pai.get_coolers(COOLS + [str(DATA / "natural.cool")])
    assert str(err.value) == "Shapes are inconsistent."
    # Resolutions unmatched, should give an explicit error
    with pytest.raises(ValueError) as err:
        assert pai.get_coolers(COOLS + [str(DATA / "A_rebin.cool")])
    assert str(err.value) == "Resolutions are inconsistent."
