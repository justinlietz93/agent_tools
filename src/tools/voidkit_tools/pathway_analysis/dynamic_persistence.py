"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def calculate_dynamic_persistence_threshold(
    base_threshold: float = 0.9,
    input_diversity: float = 0.0,
    alpha: float = 0.05
) -> float:
    """
    Calculates a dynamic persistence threshold based on input diversity.

    thresh_p(t) = base_threshold - alpha * input_diversity

    Parameters
    ----------
    base_threshold : float, optional
        The base persistence threshold, by default 0.9.
    input_diversity : float, optional
        A measure of the diversity of recent inputs, by default 0.0.
    alpha : float, optional
        The scaling factor for input diversity, by default 0.05.

    Returns
    -------
    float
        The calculated dynamic persistence threshold.
    """
    return base_threshold - alpha * input_diversity

def calculate_interference_score(
    spike_rates_persistent: np.ndarray,
    output_diversity: float
) -> float:
    """
    Calculates a score to predict potential interference with persistent pathways.

    I_score = torch.mean(spike_rates[persistent_paths] * (1 - output_diversity))

    Parameters
    ----------
    spike_rates_persistent : np.ndarray
        The spike rates of the neurons in the persistent pathways.
    output_diversity : float
        A measure of the diversity of the network's output.

    Returns
    -------
    float
        The calculated interference score.
    """
    return np.mean(spike_rates_persistent * (1 - output_diversity))
