"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# FUM_AdvancedMath.fum.revgsp_formulas
#
# Provides the pure, canonical implementations of the mathematical formulas
# for the RE-VGSP learning rule.

import torch

def calculate_modulated_learning_rate(base_eta: float, total_reward: float) -> float:
    """
    Calculates the effective learning rate, modulated by the global reward signal.
    A positive reward enables learning; a negative reward could enable anti-learning.
    
    Ref: Blueprint Rule 2, `eta_effective(total_reward)`
    Time Complexity: O(1)
    """
    # Simple linear scaling, but could be more complex (e.g., sigmoid)
    return base_eta * total_reward

def calculate_modulated_trace_decay(base_gamma: float, plv: float) -> float:
    """
    Calculates the effective eligibility trace decay factor, modulated by the
    local network resonance (Phase-Locking Value). High resonance (high PLV)
    should lead to more stable traces (higher gamma, closer to 1.0).
    
    Ref: Blueprint Rule 2, `gamma(PLV)`
    Time Complexity: O(1)
    """
    # Simple scaling: if plv is 1.0 (perfect sync), gamma is base_gamma.
    # If plv is 0.0 (no sync), gamma is lower, making traces decay faster.
    return base_gamma * (0.5 + (0.5 * plv))

def calculate_plasticity_impulse(delta_t: torch.Tensor, phase_pre: torch.Tensor, phase_post: torch.Tensor) -> torch.Tensor:
    """
    Calculates the phase-sensitive Plasticity Impulse (PI) for a batch of
    pre-post spike pairs.

    Ref: Blueprint Rule 8.1
    Time Complexity: O(k) where k is the number of spike pairs.
    """
    base_pi = torch.exp(-torch.abs(delta_t) / 10.0)
    phase_difference_cosine = torch.cos(phase_pre - phase_post)
    phase_modulation = (1 + phase_difference_cosine) / 2
    return base_pi * phase_modulation

def update_eligibility_trace(e_ij_prev: torch.Tensor, pi: torch.Tensor, gamma_eff: float) -> torch.Tensor:
    """
    Updates the eligibility traces for a batch of synapses.

    Ref: Blueprint Rule 2 & 2.1
    Time Complexity: O(N) where N is number of synapses.
    """
    return (gamma_eff * e_ij_prev) + pi

def calculate_weight_change(e_ij: torch.Tensor, w_ij: torch.Tensor, eta_eff: float, lambda_decay: float) -> torch.Tensor:
    """
    Calculates the final weight change for a batch of synapses using the
    effective learning rate and eligibility traces.

    Ref: Blueprint Rule 2
    Time Complexity: O(N) where N is number of synapses.
    """
    reinforcement = eta_eff * e_ij
    decay = lambda_decay * w_ij
    return reinforcement - decay