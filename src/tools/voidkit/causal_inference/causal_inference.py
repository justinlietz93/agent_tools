"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Tuple, Dict, Any

def granger_causality(
    data: np.ndarray, 
    max_lag: int, 
    test: str = 'ssr_chi2test'
) -> Dict[str, Any]:
    """
    Performs a Granger causality test on a multivariate time series.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array of shape (n_obs, 2) where each column is a time series.
    max_lag : int
        The maximum number of lags to test for.
    test : str, optional
        The test to perform, by default 'ssr_chi2test'.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the test results.
    """
    return grangercausalitytests(data, maxlag=max_lag, verbose=False)

def calculate_transfer_entropy(
    x: np.ndarray, 
    y: np.ndarray, 
    lag: int = 1, 
    n_bins: int = 10
) -> float:
    """
    Calculates the transfer entropy from time series x to time series y.

    Parameters
    ----------
    x : np.ndarray
        The source time series.
    y : np.ndarray
        The target time series.
    lag : int, optional
        The time lag, by default 1.
    n_bins : int, optional
        The number of bins to use for discretizing the data, by default 10.

    Returns
    -------
    float
        The transfer entropy from x to y.
    """
    if len(x) != len(y):
        raise ValueError("Time series must have the same length.")

    # Discretize the data
    x_binned = np.digitize(x, bins=np.linspace(x.min(), x.max(), n_bins + 1)) - 1
    y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins + 1)) - 1

    # Create lagged versions of the series
    y_t = y_binned[lag:]
    y_t_minus_1 = y_binned[:-lag]
    x_t_minus_1 = x_binned[:-lag]

    # Calculate probabilities
    p_y_t_y_t_minus_1_x_t_minus_1 = np.zeros((n_bins, n_bins, n_bins))
    p_y_t_minus_1_x_t_minus_1 = np.zeros((n_bins, n_bins))
    p_y_t_y_t_minus_1 = np.zeros((n_bins, n_bins))
    p_y_t_minus_1 = np.zeros(n_bins)

    for i in range(len(y_t)):
        p_y_t_y_t_minus_1_x_t_minus_1[y_t[i], y_t_minus_1[i], x_t_minus_1[i]] += 1
        p_y_t_minus_1_x_t_minus_1[y_t_minus_1[i], x_t_minus_1[i]] += 1
        p_y_t_y_t_minus_1[y_t[i], y_t_minus_1[i]] += 1
        p_y_t_minus_1[y_t_minus_1[i]] += 1

    # Normalize to get probabilities
    p_y_t_y_t_minus_1_x_t_minus_1 /= len(y_t)
    p_y_t_minus_1_x_t_minus_1 /= len(y_t)
    p_y_t_y_t_minus_1 /= len(y_t)
    p_y_t_minus_1 /= len(y_t)

    # Calculate transfer entropy
    te = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if p_y_t_y_t_minus_1_x_t_minus_1[i, j, k] > 0:
                    p_cond1 = p_y_t_y_t_minus_1_x_t_minus_1[i, j, k] / p_y_t_minus_1_x_t_minus_1[j, k]
                    p_cond2 = p_y_t_y_t_minus_1[i, j] / p_y_t_minus_1[j]
                    te += p_y_t_y_t_minus_1_x_t_minus_1[i, j, k] * np.log2(p_cond1 / p_cond2)
    
    return te
