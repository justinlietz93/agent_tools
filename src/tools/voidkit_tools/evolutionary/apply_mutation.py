"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def apply_mutation(
    weights: np.ndarray,
    mutation_rate: float = 0.01,
    mutation_scale: float = 0.1
) -> np.ndarray:
    """
    Applies random mutations to a set of weights.

    Parameters
    ----------
    weights : np.ndarray
        The weights to mutate.
    mutation_rate : float, optional
        The probability of each weight being mutated, by default 0.01.
    mutation_scale : float, optional
        The standard deviation of the Gaussian noise to add to the weights,
        by default 0.1.

    Returns
    -------
    np.ndarray
        The mutated weights.
    """
    mutation_mask = np.random.rand(*weights.shape) < mutation_rate
    mutations = np.random.normal(0, mutation_scale, weights.shape)
    
    mutated_weights = weights.copy()
    mutated_weights[mutation_mask] += mutations[mutation_mask]
    
    return mutated_weights
