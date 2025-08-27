"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Callable, Any

def calculate_jacobian(
    func: Callable[[np.ndarray, Any], np.ndarray],
    point: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Calculates the Jacobian matrix of a dynamical system at a given point.

    Parameters
    ----------
    func : Callable[[np.ndarray, Any], np.ndarray]
        A function representing the dynamical system.
    point : np.ndarray
        The point at which to calculate the Jacobian.
    epsilon : float, optional
        The step size for the finite difference approximation, by default 1e-6.

    Returns
    -------
    np.ndarray
        The Jacobian matrix at the given point.
    """
    n = len(point)
    jacobian = np.zeros((n, n))
    f0 = func(point)
    for i in range(n):
        p_plus = point.copy()
        p_plus[i] += epsilon
        f_plus = func(p_plus)
        jacobian[:, i] = (f_plus - f0) / epsilon
    return jacobian
