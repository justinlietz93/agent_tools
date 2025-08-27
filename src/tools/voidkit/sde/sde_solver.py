"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Callable, Tuple

def sde_solver(
    drift_func: Callable[[np.ndarray], np.ndarray],
    diffusion_func: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a system of stochastic differential equations (SDEs) using the
    Euler-Maruyama method.

    dx/dt = drift_func(x) + diffusion_func(x) * dW_t

    Parameters
    ----------
    drift_func : Callable[[np.ndarray], np.ndarray]
        The drift function of the SDE.
    diffusion_func : Callable[[np.ndarray], np.ndarray]
        The diffusion function of the SDE.
    initial_state : np.ndarray
        The initial state of the system.
    t_span : Tuple[float, float]
        The time span of the simulation (t_start, t_end).
    dt : float
        The time step of the simulation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The time points of the simulation.
        - The state of the system at each time point.
    """
    n_steps = int((t_span[1] - t_span[0]) / dt)
    times = np.linspace(t_span[0], t_span[1], n_steps + 1)
    states = np.zeros((n_steps + 1, len(initial_state)))
    states[0] = initial_state

    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), len(initial_state))
        drift = drift_func(states[i])
        diffusion = diffusion_func(states[i])
        states[i+1] = states[i] + drift * dt + diffusion * dW

    return times, states
