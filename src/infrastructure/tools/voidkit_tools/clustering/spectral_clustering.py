"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from typing import Tuple

def spectral_clustering_with_temporal_kernel(
    spike_rates: np.ndarray,
    spike_times: np.ndarray,
    sigma: float = 1.0,
    tau: float = 1.0,
    max_clusters: int = 10
) -> Tuple[int, np.ndarray]:
    """
    Performs spectral clustering with a temporal kernel to find the optimal
    number of clusters.

    The affinity matrix W is defined as:
    W_ij = exp(-||rate_i - rate_j||² / σ² - |Δt_ij| / τ)

    Parameters
    ----------
    spike_rates : np.ndarray
        A 1D numpy array of spike rates for each neuron.
    spike_times : np.ndarray
        A 1D numpy array of the last spike time for each neuron.
    sigma : float, optional
        The width of the Gaussian kernel for the rates, by default 1.0.
    tau : float, optional
        The time constant for the temporal kernel, by default 1.0.
    max_clusters : int, optional
        The maximum number of clusters to test for, by default 10.

    Returns
    -------
    Tuple[int, np.ndarray]
        A tuple containing:
        - The optimal number of clusters found.
        - The labels for each neuron.
    """
    n_neurons = len(spike_rates)
    
    # Calculate rate differences
    rate_diff = spike_rates[:, np.newaxis] - spike_rates[np.newaxis, :]
    
    # Calculate time differences
    time_diff = spike_times[:, np.newaxis] - spike_times[np.newaxis, :]
    
    # Calculate affinity matrix
    affinity_matrix = np.exp(-rate_diff**2 / sigma**2 - np.abs(time_diff) / tau)
    
    # Find optimal k using the eigengap heuristic
    eigenvalues, _ = np.linalg.eigh(affinity_matrix)
    eigengaps = np.diff(eigenvalues)
    optimal_k = np.argmax(eigengaps) + 1
    
    if optimal_k > max_clusters:
        optimal_k = max_clusters
        
    # Perform spectral clustering with the optimal k
    sc = SpectralClustering(n_clusters=optimal_k, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(affinity_matrix)
    
    return optimal_k, labels
