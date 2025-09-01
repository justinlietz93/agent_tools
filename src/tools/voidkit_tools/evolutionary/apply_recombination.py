"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def apply_recombination(
    weights1: np.ndarray,
    weights2: np.ndarray,
    recombination_prob: float = 0.5
) -> np.ndarray:
    """
    Performs crossover/recombination between two sets of weights.

    Parameters
    ----------
    weights1 : np.ndarray
        The first set of weights.
    weights2 : np.ndarray
        The second set of weights.
    recombination_prob : float, optional
        The probability of choosing a weight from the first set, by default 0.5.

    Returns
    -------
    np.ndarray
        The new set of weights after recombination.
    """
    if weights1.shape != weights2.shape:
        raise ValueError("Weight arrays must have the same shape.")
        
    recombination_mask = np.random.rand(*weights1.shape) < recombination_prob
    
    new_weights = weights1.copy()
    new_weights[~recombination_mask] = weights2[~recombination_mask]
    
    return new_weights
