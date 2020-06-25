#!/usr/bin/env python3

"""
Unit tests for command line interface
cmdoret, 20200407
"""
import os
import pathlib
import tempfile
import pytest
from click.testing import CliRunner
import numpy as np
import pareidolia.cli as pac
DATA = pathlib.Path("data_test")
# Synthetic matrices and their known loop coordinates
COOLS = [str(c) for c in DATA.glob("A_[1-6]*.cool")]
LOOPS = np.loadtxt(DATA / 'A_loops.txt')
# Matrices with a diagonal gradient
COOLS_COMP = [str(c) for c in DATA.glob("smooth_[1-6]*.cool")]
COOL_IN = ('cool', COOLS)
REGION = 'chr0:100000-120000'


def test_parse_cli_list():
    """Check if comma-separated lists are split correctly"""
    commas = 'x1,x2,x3'
    exp_list = ['x1', 'x2', 'x3']
    obs_list = pac._parse_cli_list(commas)
    assert len(exp_list) == len(obs_list)
    for obs, exp in zip(obs_list, exp_list):
        assert obs == exp


def test_change_detection():
    """Test if change detection pipeline can run from CLI"""
    # Run loop change detection between matrices with and without loops
    cools = ','.join(COOLS + COOLS_COMP)
    conds = ','.join(["A"] * len(COOLS) + ["B"] * len(COOLS_COMP))
    out_f = tempfile.NamedTemporaryFile(delete=False)
    runner = CliRunner()
    # Default parameters
    default_result = runner.invoke(
        pac.pareidolia_cmd, [cools, conds, out_f.name]
    )
    assert default_result.exit_code == 0
    # With a bed2d file
    bed2d_result = runner.invoke(
        pac.pareidolia_cmd,
        [cools, conds, '--bed2d-file', str(DATA / 'A_loops.bed2d'), out_f.name]
    )
    assert bed2d_result.exit_code == 0
    # Should crash with a ValueError if given a nonexistent kernel
    bad_kernel_result = runner.invoke(
        pac.pareidolia_cmd,
        [cools, conds, '--kernel', 'invalid_kernel', out_f.name]
    )
    assert isinstance(bad_kernel_result.exception, ValueError)
    os.unlink(out_f.name)
