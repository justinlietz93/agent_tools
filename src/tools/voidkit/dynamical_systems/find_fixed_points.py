"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Callable, List, Any

def find_fixed_points(
    func: Callable[[np.ndarray, Any], np.ndarray],
    initial_guesses: List[np.ndarray]
) -> np.ndarray:
    """
    Finds the fixed points (equilibria) of a dynamical system.

    Parameters
    ----------
    func : Callable[[np.ndarray, Any], np.ndarray]
        A function representing the dynamical system, where func(y, *args) = dy/dt.
    initial_guesses : List[np.ndarray]
        A list of initial guesses for the fixed points.

    Returns
    -------
    np.ndarray
        An array of the found fixed points.
    """
    fixed_points = []
    for guess in initial_guesses:
        fixed_point, _, _, _ = fsolve(func, guess, full_output=True)
        fixed_points.append(fixed_point)
    return np.array(fixed_points)
