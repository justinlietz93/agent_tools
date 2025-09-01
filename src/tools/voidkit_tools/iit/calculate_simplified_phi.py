"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Dict

def calculate_simplified_phi(
    weights: np.ndarray,
    spike_rates: np.ndarray,
    spike_probabilities: np.ndarray
) -> float:
    """
    Calculates a simplified version of the integrated information (Φ) metric.

    Φ = Σ_i log(1 + I(i,j) / H(j))
    where:
    I(i,j) = w_ij * spike_rate_j
    H(j) = -Σ p(spike_j) log p(spike_j)

    Parameters
    ----------
    weights : np.ndarray
        A 2D numpy array of synaptic weights (w_ij).
    spike_rates : np.ndarray
        A 1D numpy array of spike rates for each neuron.
    spike_probabilities : np.ndarray
        A 1D numpy array of spike probabilities for each neuron.

    Returns
    -------
    float
        The calculated simplified Φ value.
    """
    n_neurons = weights.shape[0]
    phi = 0.0

    for j in range(n_neurons):
        # Calculate entropy H(j)
        p_j = spike_probabilities[j]
        if p_j > 0 and p_j < 1:
            h_j = -(p_j * np.log2(p_j) + (1 - p_j) * np.log2(1 - p_j))
        else:
            h_j = 0

        if h_j > 0:
            for i in range(n_neurons):
                # Calculate information I(i,j)
                i_ij = weights[i, j] * spike_rates[j]
                
                # Add to total Phi
                phi += np.log2(1 + np.abs(i_ij) / h_j)
                
    return phi
