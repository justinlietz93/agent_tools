"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import List, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

def construct_vietoris_rips(
    points: np.ndarray, 
    max_edge_length: float,
    max_dim: int = 2
) -> List[Tuple[Union[int, Tuple[int, ...]], float]]:
    """
    Constructs a Vietoris-Rips complex from a set of points up to a given dimension.

    Parameters
    ----------
    points : np.ndarray
        A numpy array of shape (n_points, n_features) representing the data.
    max_edge_length : float
        The maximum edge length to consider for the complex.
    max_dim : int, optional
        The maximum dimension of the simplices to include, by default 2.

    Returns
    -------
    List[Tuple[Union[int, Tuple[int, ...]], float]]
        A list of simplices, where each simplex is represented as a tuple
        containing the vertices and the filtration value at which it appears.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise TypeError("points must be a 2D numpy array.")
    if not isinstance(max_edge_length, (int, float)) or max_edge_length < 0:
        raise ValueError("max_edge_length must be a non-negative number.")
    if not isinstance(max_dim, int) or max_dim < 0:
        raise ValueError("max_dim must be a non-negative integer.")

    n_points = points.shape[0]
    dist_matrix = squareform(pdist(points))

    rips_complex = []
    
    # 0-simplices (vertices)
    for i in range(n_points):
        rips_complex.append(([i], 0.0))

    # 1-simplices (edges)
    edges = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if dist_matrix[i, j] <= max_edge_length:
                rips_complex.append(([i, j], dist_matrix[i, j]))
                edges.append(tuple(sorted((i, j))))
    
    # Higher-dimensional simplices
    if max_dim > 1:
        current_simplices = edges
        for dim in range(2, max_dim + 1):
            next_simplices = []
            for simplex in combinations(range(n_points), dim + 1):
                is_valid_simplex = True
                max_dist = 0
                for edge in combinations(simplex, 2):
                    if dist_matrix[edge[0], edge[1]] > max_edge_length:
                        is_valid_simplex = False
                        break
                    max_dist = max(max_dist, dist_matrix[edge[0], edge[1]])
                
                if is_valid_simplex:
                    rips_complex.append((list(simplex), max_dist))
            
    return rips_complex
