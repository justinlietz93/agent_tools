# FUM_AdvancedMath.fum.sie_formulas
#
# Provides the pure, canonical implementations of the mathematical formulas
# for the Self-Improvement Engine (SIE).
# 
# Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.
#
# This research is protected under a dual-license to foster open academic
# research while ensuring commercial applications are aligned with the project's 
# ethical principles. 
# Commercial use requires written permission from Justin K. Lietz. 
# See LICENSE file for full terms.
# ===========================================================================

import torch
import logging
import time

logger = logging.getLogger(__name__)

def calculate_td_error(V_current: float, R_external: float, V_next: float, gamma: float) -> float:
    """
    Calculates the Temporal Difference (TD) error for a state transition.
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(1)
    """
    return R_external + (gamma * V_next) - V_current

def calculate_novelty_score(N_s: int) -> float:
    """
    Calculates the novelty score for a state based on its visitation count.
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(1)
    """
    # Inverse visitation count, add epsilon for stability
    return 1.0 / (N_s + 1e-6)

def calculate_habituation_score(recent_count: int, history_length: int) -> float:
    """
    Calculates a habituation score based on the frequency of the current
    input in a recent history, as per Learning_and_Guidance.md
    Ref: Blueprint Rule 3 (Component of the SIE)
    Time Complexity: O(1)
    """
    if history_length == 0:
        return 0.0
    
    return min(recent_count / history_length, 1.0)

def calculate_hsi(firing_rates: torch.Tensor, target_var: float) -> float:
    """
    Calculates the Homeostatic Stability Index (HSI).
    Ref: Blueprint Rule 3.1 & FUM Nomenclature
    Time Complexity: O(N) where N is number of neurons.
    """
    current_var = torch.var(firing_rates)
    return 1.0 - (torch.abs(current_var - target_var) / target_var)

def calculate_total_reward(w_td: float, td_error_norm: float,
                           w_nov: float, novelty_norm: float,
                           w_hab: float, habituation_norm: float,
                           w_hsi: float, hsi_norm: float) -> float:
    """
    Calculates the composite total_reward signal from its four weighted,
    normalized components.
    Ref: Blueprint Rule 3
    Time Complexity: O(1)
    """
    reward = (w_td * td_error_norm +
              w_nov * novelty_norm -
              w_hab * habituation_norm +
              w_hsi * hsi_norm)
    
    # 🔍 VALIDATION LOG: Track SIE-REVGSP handshake signal for timing analysis
    logger.debug(f"SIE total_reward: {reward:.6f} [TD:{w_td*td_error_norm:.4f}, NOV:{w_nov*novelty_norm:.4f}, HAB:{-w_hab*habituation_norm:.4f}, HSI:{w_hsi*hsi_norm:.4f}]")
    
    # 🚨 HANDSHAKE VALIDATION: Log extreme values that could disrupt RE-VGSP learning
    if abs(reward) > 10.0:
        logger.warning(f"SIE total_reward extreme value: {reward:.6f} - may cause RE-VGSP eta_eff instability")
    elif abs(reward) < 0.01:
        logger.warning(f"SIE total_reward very small: {reward:.6f} - may cause weak RE-VGSP learning")
    
    return reward