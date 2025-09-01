"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def simulate_sparsity_evolution(
    initial_sparsity: float,
    kappa: float,
    mu: float,
    spike_events: np.ndarray,
    dt: float,
    t_max: float
) -> np.ndarray:
    """
    Simulates the evolution of graph sparsity over time.

    ds/dt = -κ * s * (1 - s) + μ * Σspike_t

    Parameters
    ----------
    initial_sparsity : float
        The initial sparsity of the graph.
    kappa : float
        The decay parameter.
    mu : float
        The growth parameter.
    spike_events : np.ndarray
        A 1D array of the number of spike events at each time step.
    dt : float
        The time step.
    t_max : float
        The maximum simulation time.

    Returns
    -------
    np.ndarray
        An array of the sparsity at each time step.
    """
    n_steps = int(t_max / dt)
    sparsity = np.zeros(n_steps + 1)
    sparsity[0] = initial_sparsity

    for i in range(n_steps):
        ds_dt = -kappa * sparsity[i] * (1 - sparsity[i]) + mu * spike_events[i]
        sparsity[i+1] = sparsity[i] + ds_dt * dt
        
    return sparsity

def calculate_path_score(
    weights: np.ndarray,
    spike_times: np.ndarray,
    distances: np.ndarray,
    lambda_reg: float
) -> float:
    """
    Calculates a score for a path in the graph.

    path_score = Σw_ij * spike_t * e^(-d_ij/λ)

    Parameters
    ----------
    weights : np.ndarray
        The weights of the edges in the path.
    spike_times : np.ndarray
        The spike times at each node in the path.
    distances : np.ndarray
        The distances between nodes in the path.
    lambda_reg : float
        The distance decay parameter.

    Returns
    -------
    float
        The calculated path score.
    """
    path_score = 0.0
    for i in range(len(weights)):
        path_score += weights[i] * spike_times[i] * np.exp(-distances[i] / lambda_reg)
    return path_score
