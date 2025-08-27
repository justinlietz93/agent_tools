"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def calculate_free_energy(
    spike_rates: np.ndarray,
    target_rate: float,
    weights: np.ndarray,
    lambda_reg: float
) -> float:
    """
    Calculates the free energy of the system.

    F = Σ_i (spike_rate_i - target_rate)^2 + λ * Σ w_ij^2

    Parameters
    ----------
    spike_rates : np.ndarray
        A 1D numpy array of spike rates for each neuron.
    target_rate : float
        The target firing rate for the neurons.
    weights : np.ndarray
        A 2D numpy array of synaptic weights.
    lambda_reg : float
        The regularization parameter for the weights.

    Returns
    -------
    float
        The calculated free energy.
    """
    rate_error = np.sum((spike_rates - target_rate)**2)
    weight_regularization = lambda_reg * np.sum(weights**2)
    return rate_error + weight_regularization

def minimize_free_energy_step(
    weights: np.ndarray,
    spike_rates: np.ndarray,
    target_rate: float,
    lambda_reg: float,
    eta: float,
    delta_t: float,
    tau: float
) -> np.ndarray:
    """
    Performs one step of gradient descent to minimize the free energy.

    dw_ij/dt = -η * ∂F/∂w_ij * e^(-Δt/τ)

    Parameters
    ----------
    weights : np.ndarray
        The current synaptic weights.
    spike_rates : np.ndarray
        The current spike rates.
    target_rate : float
        The target firing rate.
    lambda_reg : float
        The weight regularization parameter.
    eta : float
        The learning rate.
    delta_t : float
        The time difference for the STDP-like modulation.
    tau : float
        The time constant for the STDP-like modulation.

    Returns
    -------
    np.ndarray
        The updated weights.
    """
    # The partial derivative of F with respect to w_ij is 2 * lambda * w_ij
    grad_F = 2 * lambda_reg * weights
    
    # STDP-like modulation
    modulation = np.exp(-delta_t / tau)
    
    # Update rule
    delta_w = -eta * grad_F * modulation
    
    return weights + delta_w
