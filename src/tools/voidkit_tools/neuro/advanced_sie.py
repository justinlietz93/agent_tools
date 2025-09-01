"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def calculate_stabilized_reward(
    td_error: float,
    novelty: float,
    habituation: float,
    self_benefit: float,
    external_reward: float,
    w_r_base: float = 0.6,
    w_n: float = 0.3,
    w_s: float = 0.1,
    lambda_reg: float = 0.05
) -> float:
    """
    Calculates a stabilized multi-objective reward function.

    R_tot(t) = w_r * TD_error(t) + w_n * novelty(t) * (1 - tanh(habituation(t))) + w_s * self_benefit(t)
    where w_r = w_r_base * e^(-λ * |external_reward|)

    Parameters
    ----------
    td_error : float
    novelty : float
    habituation : float
    self_benefit : float
    external_reward : float
    w_r_base : float, optional
    w_n : float, optional
    w_s : float, optional
    lambda_reg : float, optional

    Returns
    -------
    float
        The calculated stabilized reward.
    """
    w_r = w_r_base * np.exp(-lambda_reg * np.abs(external_reward))
    
    reward = (w_r * td_error + 
              w_n * novelty * (1 - np.tanh(habituation)) + 
              w_s * self_benefit)
              
    return reward

def apply_quadratic_stdp_modulation(
    eta_base: float = 0.12,
    beta: float = 0.15,
    tau: float = 15.0,
    delta_t: float = 0.0,
    total_reward: float = 0.0
) -> float:
    """
    Calculates the STDP weight change with quadratic reward modulation.

    Δw_ij = η * (1 + β * R_tot^2) * e^(-Δt/τ)

    Parameters
    ----------
    eta_base : float, optional
    beta : float, optional
    tau : float, optional
    delta_t : float, optional
    total_reward : float, optional

    Returns
    -------
    float
        The calculated STDP weight change.
    """
    eta = eta_base * (1 + beta * total_reward**2)
    return eta * np.exp(-delta_t / tau)
