"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Tuple

def apply_stc(
    current_weight: float,
    eligibility_trace: float,
    synaptic_tag: float,
    reward: float,
    tag_decay: float = 0.9,
    consolidation_threshold: float = 0.7,
    consolidation_rate: float = 0.1
) -> Tuple[float, float, float]:
    """
    Applies a Synaptic Tagging and Capture (STC) rule.

    Parameters
    ----------
    current_weight : float
        The current synaptic weight.
    eligibility_trace : float
        The current eligibility trace for the synapse (e.g., from STDP).
    synaptic_tag : float
        The current synaptic tag value.
    reward : float
        The global or local reward signal.
    tag_decay : float, optional
        The decay rate of the synaptic tag, by default 0.9.
    consolidation_threshold : float, optional
        The reward threshold for triggering consolidation, by default 0.7.
    consolidation_rate : float, optional
        The rate at which the weight is consolidated, by default 0.1.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - The updated synaptic weight.
        - The updated eligibility trace (reset after use).
        - The updated synaptic tag.
    """
    # Update the synaptic tag based on the eligibility trace
    new_tag = tag_decay * synaptic_tag + eligibility_trace

    # Check for consolidation
    if reward > consolidation_threshold:
        # Consolidate the weight based on the tag
        new_weight = current_weight + consolidation_rate * new_tag
        # Reset the tag after consolidation
        new_tag = 0.0
    else:
        new_weight = current_weight

    # Reset the eligibility trace
    new_eligibility_trace = 0.0
    
    return new_weight, new_eligibility_trace, new_tag
