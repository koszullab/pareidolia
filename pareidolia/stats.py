#!/usr/bin/env python3

"""
Functions used to evaluate statistical significance of changes.
cmdoret, 20200403
"""
import numpy as np
import scipy.stats as ss


def vals_to_percentiles(vals: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Return the percentiles corresponding to values in a distribution.

    Parameters
    ----------
    vals : np.ndarray of floats
        The values for which percentiles should be returned.
    dist : np.ndarray of floats
        The distribution to use when computing the percentiles.

    Returns
    -------
    np.ndarray of floats :
        The array of percentiles corresponding to input values
    """
    # Get indices corresponding to vals in the distribution
    sorted_dist = np.sort(dist)
    percentiles = np.searchsorted(sorted_dist, vals) / len(sorted_dist)
    return percentiles


def vec_ttest(arr1, arr2):
    """
    Given 2 numpy arrays containing data for 2 different conditions, where
    each array has shape shape rxd where r is the number of replicates and N
    is the number of independent tests to perform. N must be equal in both array,
    however the number of replicates can differ.
    """
    if not (len(arr1.shape) == len(arr2.shape) == 2):
        raise ValueError("Both arrays must be 2-dimensional.")
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError(
            "Both arrays must have the same number of test datasets"
        )
    # Number of sample in each test (both conditions)
    n_samples = arr1.shape[1] + arr2.shape[1]
    # Compute mean and variance for each test independently
    means_1, means_2 = arr1.mean(axis=1), arr2.mean(axis=1)
    var1, var2 = arr1.var(axis=1, ddof=1), arr2.var(axis=1, ddof=1)
    stds = np.sqrt((var1 + var2) / 2)
    stds[stds == 0] = np.nan
    # alt - control
    t_stats = (means_2 - means_1) / (stds * np.sqrt(1 / n_samples))
    t_stats[np.isnan(t_stats)] = 0

    return t_stats


# TODO: Write a vectorized linear model function for cases with more than
# 2 (quantitative) conditions
def vec_lm():
    ...


def tvals_to_pvals(tvals, n_samples):
    """Compute p-values from t-values
    Given an array of test statistics from multiple t-tests and the (fixed)
    number of samples used for each test, compute p-values for all tests
    """
    # get p-values from test statistics using the cumulative distribution
    # function of the student distribution
    return 1 - ss.t.cdf(np.abs(tvals), df=n_samples - 1)


def pval_to_tval(pval, n_samples):
    """Conversion of a single p-value to corresponding t
    Given a desired p-value and sample size, return the corresponding t-value.
    The absolute value for the 2 sided hypothesis is returned.
    """
    return abs(ss.t.ppf(pval / 2, df=n_samples - 1))
