"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# src/tda/compute_persistent_homology.py

import numpy as np
from typing import Dict
from ripser import ripser

def compute_persistent_homology(
    data: np.ndarray, 
    max_dim: int = 1,
    is_distance_matrix: bool = False
) -> Dict[str, np.ndarray]:
    """
    Computes the persistent homology of a point cloud OR a distance matrix.
    VERSION 3: This version is now fully robust.

    Args:
        data (np.ndarray): A 2D NumPy array representing either:
                             - A point cloud (n_points, n_features)
                             - A square distance matrix (n_points, n_points)
        max_dim (int, optional): The maximum dimension of homology to compute.
        is_distance_matrix (bool): Flag indicating if the input data is a
                                     pre-computed distance matrix.

    Returns:
        Dict[str, np.ndarray]: A dictionary from ripser containing the results.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise TypeError("Input data must be a 2D NumPy array.")
    if is_distance_matrix and data.shape[0] != data.shape[1]:
        raise ValueError("A distance matrix must be square.")

    if is_distance_matrix:
        # Data is a pre-computed square distance matrix
        result = ripser(data, maxdim=max_dim, distance_matrix=True)
    else:
        # Data is a point cloud
        result = ripser(data, maxdim=max_dim, distance_matrix=False)
    
    return result