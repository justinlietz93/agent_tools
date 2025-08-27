"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Tuple

def fit_power_law(data: np.ndarray) -> Tuple[float, float]:
    """
    Fits a power-law distribution to data using a linear fit on a log-log plot.

    Parameters
    ----------
    data : np.ndarray
        A 1D numpy array of the data to be fitted (e.g., avalanche sizes).

    Returns
    -------
    Tuple[float, float]
        A tuple containing the exponent of the power law and the R-squared value
        of the fit.
    """
    # Create a histogram of the data
    counts, bin_edges = np.histogram(data, bins=np.logspace(np.log10(min(data)), np.log10(max(data)), len(data)//10))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter out zero counts to avoid log(0)
    non_zero = counts > 0
    log_x = np.log10(bin_centers[non_zero])
    log_y = np.log10(counts[non_zero])

    # Fit a line to the log-log data
    coeffs, residuals, _, _, _ = np.polyfit(log_x, log_y, 1, full=True)
    
    exponent = coeffs[0]
    
    # Calculate R-squared
    ss_res = residuals[0]
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return exponent, r_squared
