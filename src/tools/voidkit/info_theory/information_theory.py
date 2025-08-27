"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy

def calculate_entropy(pk: np.ndarray, base: int = 2) -> float:
    """
    Calculates the Shannon entropy of a discrete probability distribution.

    Parameters
    ----------
    pk : np.ndarray
        A 1D numpy array representing the probability distribution.
        Must sum to 1.
    base : int, optional
        The logarithmic base to use, by default 2.

    Returns
    -------
    float
        The Shannon entropy of the distribution.
    """
    if not np.isclose(np.sum(pk), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    return scipy_entropy(pk, base=base)

def calculate_mutual_information(
    p_xy: np.ndarray, base: int = 2
) -> float:
    """
    Calculates the mutual information between two random variables.

    Parameters
    ----------
    p_xy : np.ndarray
        A 2D numpy array representing the joint probability distribution
        P(X, Y).
    base : int, optional
        The logarithmic base to use, by default 2.

    Returns
    -------
    float
        The mutual information I(X; Y).
    """
    if not np.isclose(np.sum(p_xy), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    p_x_p_y = np.outer(p_x, p_y)
    
    # We need to avoid log(0) for cases where p_xy is 0.
    # The contribution to the sum is 0 in these cases.
    non_zero = p_xy > 0
    
    return np.sum(p_xy[non_zero] * np.log(p_xy[non_zero] / p_x_p_y[non_zero])) / np.log(base)

def calculate_kl_divergence(
    p: np.ndarray, q: np.ndarray, base: int = 2
) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two
    discrete probability distributions.

    Parameters
    ----------
    p : np.ndarray
        A 1D numpy array representing the true probability distribution.
    q : np.ndarray
        A 1D numpy array representing the approximate probability distribution.
    base : int, optional
        The logarithmic base to use, by default 2.

    Returns
    -------
    float
        The KL divergence D_KL(P || Q).
    """
    if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length.")

    return scipy_entropy(p, q, base=base)
