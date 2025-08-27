"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Dict, List

def calculate_tda_metrics(
    persistence_diagrams: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calculates TDA metrics from persistence diagrams.

    Parameters
    ----------
    persistence_diagrams : List[np.ndarray]
        A list of persistence diagrams for each dimension, as returned by ripser.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the calculated TDA metrics, including:
        - 'total_b1_persistence': The sum of persistence of 1-dimensional features.
        - 'component_count': The number of connected components (0-dimensional features).
    """
    if not isinstance(persistence_diagrams, list):
        raise TypeError("persistence_diagrams must be a list of numpy arrays.")

    metrics = {}

    # H0: Connected components
    h0 = persistence_diagrams[0]
    # The number of components is the number of points with infinite persistence.
    # In ripser, infinite persistence is represented by np.inf.
    metrics['component_count'] = np.sum(np.isinf(h0[:, 1]))

    # H1: Loops/cycles
    if len(persistence_diagrams) > 1:
        h1 = persistence_diagrams[1]
        persistence = h1[:, 1] - h1[:, 0]
        metrics['total_b1_persistence'] = np.sum(persistence)
    else:
        metrics['total_b1_persistence'] = 0.0
        
    return metrics
