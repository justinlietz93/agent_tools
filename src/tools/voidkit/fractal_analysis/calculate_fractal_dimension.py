"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def calculate_fractal_dimension(points: np.ndarray, threshold: float = 0.9) -> float:
    """
    Calculates the fractal dimension of a point set using the box-counting algorithm.

    Parameters
    ----------
    points : np.ndarray
        A numpy array of shape (n_points, n_features) representing the data.
    threshold : float, optional
        The threshold for the number of points in a box to be considered "filled",
        by default 0.9.

    Returns
    -------
    float
        The estimated fractal dimension.
    """
    # Find the bounding box of the point set
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # A list of scales to use
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    
    counts = []
    for scale in scales:
        box_size = (max_coords - min_coords) / scale
        
        # Create a grid of boxes
        grid = {}
        for point in points:
            box_index = tuple(np.floor((point - min_coords) / box_size).astype(int))
            if box_index not in grid:
                grid[box_index] = 0
            grid[box_index] += 1
            
        # Count the number of non-empty boxes
        counts.append(len(grid))
        
    # Fit a line to the log-log plot of counts vs. scales
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    
    return -coeffs[0]
