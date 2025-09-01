"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Tuple, List, Union, Optional

def apply_stdp(
    spike_times_pre: Union[List[float], np.ndarray],
    spike_times_post: Union[List[float], np.ndarray],
    current_weight: float,
    is_inhibitory: bool = False,
    A_plus_base: float = 0.1,
    A_minus_base: float = 0.12,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    eligibility_trace: float = 0.0,
    gamma: float = 0.9,
    cluster_reward: float = 0.0,
    max_reward: float = 1.0,
    spike_rate_pre: float = 0.0,
    dt: float = 1.0,
    target_rate: float = 0.3,
    eta: float = 1.0,
    A_plus_inh: Optional[float] = None,
    A_minus_inh: Optional[float] = None,
    tau_plus_inh: Optional[float] = None,
    tau_minus_inh: Optional[float] = None,
    weight_bounds: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Applies Spike-Timing-Dependent Plasticity (STDP) rules to update synaptic weights
    based on the relative timing of pre- and post-synaptic spikes.
    
    Parameters
    ----------
    spike_times_pre : array_like
        List or array of spike times for the pre-synaptic neuron (in ms).
    
    spike_times_post : array_like
        List or array of spike times for the post-synaptic neuron (in ms).
    
    current_weight : float
        The current synaptic weight (w_ij) between the neurons.
    
    is_inhibitory : bool, optional
        Flag indicating whether the synapse is inhibitory (True) or excitatory (False).
        Default is False (excitatory).
    
    A_plus_base : float, optional
        Base potentiation strength for excitatory synapses. Default is 0.1.
    
    A_minus_base : float, optional
        Base depression strength for excitatory synapses. Default is 0.12.
    
    tau_plus : float, optional
        Time constant for potentiation (in ms) for excitatory synapses. Default is 20.0.
    
    tau_minus : float, optional
        Time constant for depression (in ms) for excitatory synapses. Default is 20.0.
    
    eligibility_trace : float, optional
        The current eligibility trace for the synapse. Default is 0.0.
    
    gamma : float, optional
        Decay factor for the eligibility trace. Default is 0.9.
    
    cluster_reward : float, optional
        The reward signal specific to the post-synaptic neuron's cluster. Default is 0.0.
    
    max_reward : float, optional
        The maximum possible reward value. Default is 1.0.
    
    spike_rate_pre : float, optional
        The recent firing rate of the pre-synaptic neuron (in Hz). Default is 0.0.
    
    dt : float, optional
        The time step in ms. Default is 1.0.
    
    target_rate : float, optional
        Target firing rate for homeostatic regulation (in Hz). Default is 0.3.
    
    eta : float, optional
        Base learning rate. Default is 1.0.
    
    A_plus_inh : float, optional
        Potentiation strength for inhibitory synapses. If None, uses A_plus_base.
    
    A_minus_inh : float, optional
        Depression strength for inhibitory synapses. If None, uses A_minus_base.
    
    tau_plus_inh : float, optional
        Time constant for potentiation (in ms) for inhibitory synapses. If None, uses tau_plus.
    
    tau_minus_inh : float, optional
        Time constant for depression (in ms) for inhibitory synapses. If None, uses tau_minus.
    
    weight_bounds : tuple of float, optional
        Minimum and maximum allowed values for the synaptic weight.
        If None, uses (0.0, 1.0) for excitatory and (-1.0, 0.0) for inhibitory synapses.
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - The updated synaptic weight (w_ij)
        - The updated eligibility trace (e_ij)
    
    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., negative time constants).
    TypeError
        If input parameters have incorrect types.
    
    Notes
    -----
    This function implements the STDP rules for the FUM (Fully Unified Model) project,
    handling both excitatory and inhibitory synapses, eligibility traces, and parameter
    heterogeneity as described in the FUM documentation.
    
    The STDP rule for excitatory synapses is:
    - Δw_ij = A_+ * exp(-Δt / τ_+) if Δt > 0 (pre precedes post, potentiation)
    - Δw_ij = A_- * exp(Δt / τ_-) if Δt < 0 (post precedes pre, depression)
    
    The STDP rule for inhibitory synapses is the reverse:
    - Δw_ij = A_+_inh * exp(Δt / τ_+_inh) if Δt < 0 (post precedes pre, potentiation)
    - Δw_ij = A_-_inh * exp(-Δt / τ_-_inh) if Δt > 0 (pre precedes post, depression)
    
    The eligibility trace is updated as:
    - e_ij(t+dt) = gamma * e_ij(t) + Δw_ij
    
    Examples
    --------
    >>> # Example for excitatory synapse
    >>> spike_times_pre = [10.0, 20.0, 30.0]
    >>> spike_times_post = [15.0, 25.0, 35.0]
    >>> current_weight = 0.5
    >>> new_weight, new_trace = apply_stdp(spike_times_pre, spike_times_post, current_weight)
    >>> print(f"New weight: {new_weight:.4f}, New trace: {new_trace:.4f}")
    
    >>> # Example for inhibitory synapse
    >>> spike_times_pre = [10.0, 20.0, 30.0]
    >>> spike_times_post = [5.0, 15.0, 25.0]
    >>> current_weight = -0.5
    >>> new_weight, new_trace = apply_stdp(
    ...     spike_times_pre, spike_times_post, current_weight, 
    ...     is_inhibitory=True, cluster_reward=0.5
    ... )
    >>> print(f"New weight: {new_weight:.4f}, New trace: {new_trace:.4f}")
    """
    # Input validation
    # Check types
    if not isinstance(spike_times_pre, (list, np.ndarray)):
        raise TypeError("spike_times_pre must be a list or numpy array")
    if not isinstance(spike_times_post, (list, np.ndarray)):
        raise TypeError("spike_times_post must be a list or numpy array")
    if not isinstance(current_weight, (int, float)):
        raise TypeError("current_weight must be a number")
    if not isinstance(is_inhibitory, bool):
        raise TypeError("is_inhibitory must be a boolean")
    
    # Check values
    if not (isinstance(A_plus_base, (int, float)) and A_plus_base > 0):
        raise ValueError("A_plus_base must be a positive number")
    if not (isinstance(A_minus_base, (int, float)) and A_minus_base > 0):
        raise ValueError("A_minus_base must be a positive number")
    if not (isinstance(tau_plus, (int, float)) and tau_plus > 0):
        raise ValueError("tau_plus must be a positive number")
    if not (isinstance(tau_minus, (int, float)) and tau_minus > 0):
        raise ValueError("tau_minus must be a positive number")
    if not isinstance(eligibility_trace, (int, float)):
        raise TypeError("eligibility_trace must be a number")
    if not (isinstance(gamma, (int, float)) and 0 <= gamma <= 1):
        raise ValueError("gamma must be a number between 0 and 1")
    if not (isinstance(cluster_reward, (int, float)) and cluster_reward >= 0):
        raise ValueError("cluster_reward must be a non-negative number")
    if not (isinstance(max_reward, (int, float)) and max_reward > 0):
        raise ValueError("max_reward must be a positive number")
    if cluster_reward > max_reward:
        raise ValueError("cluster_reward cannot exceed max_reward")
    if not (isinstance(spike_rate_pre, (int, float)) and spike_rate_pre >= 0):
        raise ValueError("spike_rate_pre must be a non-negative number")
    if not (isinstance(dt, (int, float)) and dt > 0):
        raise ValueError("dt must be a positive number")
    if not (isinstance(target_rate, (int, float)) and target_rate > 0):
        raise ValueError("target_rate must be a positive number")
    if not (isinstance(eta, (int, float)) and eta > 0):
        raise ValueError("eta must be a positive number")
    
    # Check optional parameters if provided
    if A_plus_inh is not None and not (isinstance(A_plus_inh, (int, float)) and A_plus_inh > 0):
        raise ValueError("A_plus_inh must be a positive number")
    if A_minus_inh is not None and not (isinstance(A_minus_inh, (int, float)) and A_minus_inh > 0):
        raise ValueError("A_minus_inh must be a positive number")
    if tau_plus_inh is not None and not (isinstance(tau_plus_inh, (int, float)) and tau_plus_inh > 0):
        raise ValueError("tau_plus_inh must be a positive number")
    if tau_minus_inh is not None and not (isinstance(tau_minus_inh, (int, float)) and tau_minus_inh > 0):
        raise ValueError("tau_minus_inh must be a positive number")
    
    # Check weight bounds if provided
    if weight_bounds is not None:
        if not isinstance(weight_bounds, tuple) or len(weight_bounds) != 2:
            raise TypeError("weight_bounds must be a tuple of (min_weight, max_weight)")
        if not (isinstance(weight_bounds[0], (int, float)) and isinstance(weight_bounds[1], (int, float))):
            raise TypeError("weight_bounds values must be numbers")
        if weight_bounds[0] >= weight_bounds[1]:
            raise ValueError("weight_bounds[0] must be less than weight_bounds[1]")
    
    # Check consistency between weight and inhibitory flag
    if is_inhibitory and current_weight > 0:
        raise ValueError("Inhibitory synapses must have negative weights")
    if not is_inhibitory and current_weight < 0:
        raise ValueError("Excitatory synapses must have positive weights")
    
    # Convert inputs to numpy arrays if they aren't already
    spike_times_pre = np.asarray(spike_times_pre)
    spike_times_post = np.asarray(spike_times_post)
    
    # Set default weight bounds if not provided
    if weight_bounds is None:
        if is_inhibitory:
            weight_bounds = (-1.0, 0.0)
        else:
            weight_bounds = (0.0, 1.0)
    
    # Initialize weight change
    delta_w = 0.0
    
    # Implement STDP rules for excitatory synapses
    if not is_inhibitory:
        # Modulate A_plus based on cluster reward and pre-synaptic firing rate
        A_plus = A_plus_base * (cluster_reward / max_reward)
        
        # Homeostatic regulation based on pre-synaptic firing rate
        if spike_rate_pre > 0:
            A_plus *= spike_rate_pre / target_rate
        
        # Vectorized implementation for excitatory synapses
        if len(spike_times_pre) > 0 and len(spike_times_post) > 0:
            # Compute all pairwise time differences (Δt = t_post - t_pre)
            delta_t_matrix = np.subtract.outer(spike_times_post, spike_times_pre)
            
            # Potentiation: when pre-synaptic spike precedes post-synaptic spike (Δt > 0)
            potentiation_mask = delta_t_matrix > 0
            if np.any(potentiation_mask):
                potentiation_values = A_plus * np.exp(-delta_t_matrix[potentiation_mask] / tau_plus)
                delta_w += np.sum(potentiation_values)
            
            # Depression: when post-synaptic spike precedes pre-synaptic spike (Δt < 0)
            depression_mask = delta_t_matrix < 0
            if np.any(depression_mask):
                depression_values = A_minus_base * np.exp(delta_t_matrix[depression_mask] / tau_minus)
                delta_w -= np.sum(depression_values)
    
    # Implement STDP rules for inhibitory synapses
    elif is_inhibitory:
        # Set inhibitory parameters if not provided
        if A_plus_inh is None:
            A_plus_inh = A_plus_base
        if A_minus_inh is None:
            A_minus_inh = A_minus_base
        if tau_plus_inh is None:
            tau_plus_inh = tau_plus
        if tau_minus_inh is None:
            tau_minus_inh = tau_minus
        
        # Vectorized implementation for inhibitory synapses
        if len(spike_times_pre) > 0 and len(spike_times_post) > 0:
            # Compute all pairwise time differences (Δt = t_post - t_pre)
            delta_t_matrix = np.subtract.outer(spike_times_post, spike_times_pre)
            
            # For inhibitory synapses, the timing dependency is reversed:
            # Depression: when pre-synaptic spike precedes post-synaptic spike (Δt > 0)
            depression_mask = delta_t_matrix > 0
            if np.any(depression_mask):
                depression_values = A_minus_inh * np.exp(-delta_t_matrix[depression_mask] / tau_minus_inh)
                delta_w += np.sum(depression_values)
            
            # Potentiation: when post-synaptic spike precedes pre-synaptic spike (Δt < 0)
            potentiation_mask = delta_t_matrix < 0
            if np.any(potentiation_mask):
                potentiation_values = A_plus_inh * np.exp(delta_t_matrix[potentiation_mask] / tau_plus_inh)
                delta_w -= np.sum(potentiation_values)
    
    # Implement eligibility trace integration
    # Update the eligibility trace: e_ij(t+dt) = gamma * e_ij(t) + Δw_ij
    new_eligibility_trace = gamma * eligibility_trace + delta_w
    
    # Apply SIE modulation (learning rate modulation)
    # Δw_ij = eta_effective * Δw_ij
    delta_w = eta * delta_w
    
    # Update the weight based on the eligibility trace and apply bounds
    new_weight = current_weight + delta_w
    
    # Apply weight bounds
    new_weight = np.clip(new_weight, weight_bounds[0], weight_bounds[1])
    
    return new_weight, new_eligibility_trace
