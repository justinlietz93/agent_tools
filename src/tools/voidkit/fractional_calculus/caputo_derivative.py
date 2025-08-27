"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.special import gamma

def caputo_derivative(
    f: np.ndarray, 
    alpha: float, 
    dt: float = 1.0
) -> np.ndarray:
    """
    Calculates the Caputo fractional derivative of a time series.

    This implementation uses the Grunwald-Letnikov formula.

    Parameters
    ----------
    f : np.ndarray
        A 1D numpy array representing the time series.
    alpha : float
        The order of the fractional derivative (0 < alpha < 1).
    dt : float, optional
        The time step between samples, by default 1.0.

    Returns
    -------
    np.ndarray
        The Caputo fractional derivative of the time series.
    """
    n = len(f)
    result = np.zeros(n)
    
    for i in range(n):
        summation = 0
        for k in range(i + 1):
            # Grunwald-Letnikov coefficients
            coeff = (gamma(k - alpha) / (gamma(-alpha) * gamma(k + 1)))
            summation += coeff * f[i - k]
        result[i] = summation / (dt**alpha)
        
    return result
