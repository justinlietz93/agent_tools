"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Union

def calculate_wasserstein_distance(
    u_values: np.ndarray,
    v_values: np.ndarray,
    u_weights: np.ndarray = None,
    v_weights: np.ndarray = None
) -> float:
    """
    Calculates the 1-D Wasserstein distance between two distributions.

    Parameters
    ----------
    u_values : np.ndarray
        A 1D array of values for the first distribution.
    v_values : np.ndarray
        A 1D array of values for the second distribution.
    u_weights : np.ndarray, optional
        Weights for the first distribution, by default None.
    v_weights : np.ndarray, optional
        Weights for the second distribution, by default None.

    Returns
    -------
    float
        The 1-D Wasserstein distance.
    """
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)
