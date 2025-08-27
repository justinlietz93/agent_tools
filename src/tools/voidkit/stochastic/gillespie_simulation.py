"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Callable, List, Tuple

def gillespie_simulation(
    initial_state: np.ndarray,
    propensity_func: Callable[[np.ndarray], np.ndarray],
    stoichiometry: np.ndarray,
    t_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a Gillespie simulation (Stochastic Simulation Algorithm).

    Parameters
    ----------
    initial_state : np.ndarray
        A 1D numpy array of the initial counts of each species.
    propensity_func : Callable[[np.ndarray], np.ndarray]
        A function that takes the current state and returns the propensities
        (reaction rates) for each reaction.
    stoichiometry : np.ndarray
        A 2D numpy array of shape (n_reactions, n_species) that defines the
        change in species counts for each reaction.
    t_max : float
        The maximum simulation time.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The time points of the simulation.
        - The state of the system at each time point.
    """
    times = [0.0]
    states = [initial_state.copy()]
    
    t = 0.0
    current_state = initial_state.copy()

    while t < t_max:
        propensities = propensity_func(current_state)
        total_propensity = np.sum(propensities)

        if total_propensity == 0:
            break

        # Time to next reaction
        dt = -np.log(np.random.rand()) / total_propensity
        
        # Which reaction occurs?
        reaction_probs = propensities / total_propensity
        reaction_index = np.random.choice(len(propensities), p=reaction_probs)

        # Update state
        current_state += stoichiometry[reaction_index]
        t += dt

        times.append(t)
        states.append(current_state.copy())

    return np.array(times), np.array(states)
