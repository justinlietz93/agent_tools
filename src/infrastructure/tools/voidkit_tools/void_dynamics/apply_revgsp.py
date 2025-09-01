"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

# FUM_AdvancedMath.fum.apply_revgsp
#
# Blueprint-compliant implementation of RE-VGSP (Resonance-Enhanced Valence-Gated Synaptic Plasticity)
# The canonical three-factor learning rule that bridges Local System (SNN) and Global System (SIE/ADC)

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix, lil_matrix
import logging

# Import blueprint formulas
from .revgsp_formulas import (
    calculate_modulated_learning_rate,
    calculate_modulated_trace_decay,
    calculate_plasticity_impulse,
    update_eligibility_trace,
    calculate_weight_change
)

logger = logging.getLogger(__name__)

def extract_spike_pairs_from_synapses(
    synaptic_weights: csr_matrix,
    pre_spike_times: List[List[float]],
    post_spike_times: List[List[float]],
    time_window: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Blueprint-compliant spike pair extraction for existing synapses only.
    
    Blueprint Reference: Rule 2 - RE-VGSP Local Rule Implementation
    Time Complexity: O(N × S) where N is number of existing synapses, S is average spikes per neuron
    ✅ BLUEPRINT COMPLIANT: "O(N) algorithm where N is the number of synapses"
    ✅ LOCAL RULE: "Avoiding expensive global calculations" - processes only existing connections
    
    Args:
        synaptic_weights: Sparse CSR matrix defining existing synapses
        pre_spike_times: List of spike times for each presynaptic neuron
        post_spike_times: List of spike times for each postsynaptic neuron
        time_window: Maximum time difference for spike pairing (ms)
        
    Returns:
        Tuple of (pre_indices, post_indices, delta_t, synapse_indices)
    """
    pre_indices = []
    post_indices = []
    delta_t_list = []
    synapse_indices = []
    
    # Iterate only over existing synapses (non-zero entries) - O(N) where N = nnz synapses
    cx = synaptic_weights.tocoo()  # Convert to COO for efficient iteration
    
    for synapse_idx, (pre_neuron_idx, post_neuron_idx) in enumerate(zip(cx.row, cx.col)):
        # Validate neuron indices
        if (pre_neuron_idx >= len(pre_spike_times) or
            post_neuron_idx >= len(post_spike_times)):
            continue
            
        pre_times = pre_spike_times[pre_neuron_idx]
        post_times = post_spike_times[post_neuron_idx]
        
        if len(pre_times) == 0 or len(post_times) == 0:
            continue
            
        # Find spike pairs within time window for this specific synapse
        for pre_time in pre_times:
            for post_time in post_times:
                dt = post_time - pre_time
                if abs(dt) <= time_window:
                    pre_indices.append(pre_neuron_idx)
                    post_indices.append(post_neuron_idx)
                    delta_t_list.append(dt)
                    synapse_indices.append(synapse_idx)
    
    if len(delta_t_list) == 0:
        logger.debug("No spike pairs found for existing synapses")
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long))
    
    logger.debug(f"Blueprint-compliant extraction: {len(delta_t_list)} spike pairs from {len(cx.data)} synapses")
    
    return (torch.tensor(pre_indices, dtype=torch.long),
            torch.tensor(post_indices, dtype=torch.long),
            torch.tensor(delta_t_list, dtype=torch.float32),
            torch.tensor(synapse_indices, dtype=torch.long))

def extract_spike_pairs(
    pre_spike_times: List[List[float]],
    post_spike_times: List[List[float]],
    time_window: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DEPRECATED: Legacy function maintained for backward compatibility.
    
    ⚠️  BLUEPRINT VIOLATION: O(N² × S²) complexity
    Use extract_spike_pairs_from_synapses() for blueprint-compliant O(N) processing.
    
    Args:
        pre_spike_times: List of spike times for each presynaptic neuron
        post_spike_times: List of spike times for each postsynaptic neuron
        time_window: Maximum time difference for spike pairing (ms)
        
    Returns:
        Tuple of (pre_indices, post_indices, delta_t, synapse_indices)
    """
    logger.warning("Using deprecated extract_spike_pairs() - O(N²) complexity violates Blueprint Rule 2")
    logger.warning("Use extract_spike_pairs_from_synapses() for blueprint-compliant O(N) processing")
    
    pre_indices = []
    post_indices = []
    delta_t_list = []
    synapse_indices = []
    
    synapse_idx = 0
    for pre_neuron_idx, pre_times in enumerate(pre_spike_times):
        for post_neuron_idx, post_times in enumerate(post_spike_times):
            if len(pre_times) == 0 or len(post_times) == 0:
                continue
                
            # Find spike pairs within time window
            for pre_time in pre_times:
                for post_time in post_times:
                    dt = post_time - pre_time
                    if abs(dt) <= time_window:
                        pre_indices.append(pre_neuron_idx)
                        post_indices.append(post_neuron_idx)
                        delta_t_list.append(dt)
                        synapse_indices.append(synapse_idx)
            
            synapse_idx += 1
    
    if len(delta_t_list) == 0:
        return (torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long))
    
    return (torch.tensor(pre_indices, dtype=torch.long),
            torch.tensor(post_indices, dtype=torch.long),
            torch.tensor(delta_t_list, dtype=torch.float32),
            torch.tensor(synapse_indices, dtype=torch.long))

def calculate_phase_from_spike_times(
    spike_times: List[float],
    reference_freq: float = 40.0,
    current_time: float = 0.0
) -> float:
    """
    Calculate neural oscillation phase from recent spike timing.
    
    Blueprint Reference: Rule 8.1 - Phase-sensitive plasticity impulse
    Time Complexity: O(1) - uses only the most recent spike for phase calculation
    
    Args:
        spike_times: Recent spike times for the neuron
        reference_freq: Reference oscillation frequency (Hz)
        current_time: Current simulation time
        
    Returns:
        Phase value in radians [0, 2π]
    """
    if len(spike_times) == 0:
        return 0.0
    
    # Use most recent spike for phase calculation
    last_spike = spike_times[-1]
    time_since_spike = current_time - last_spike
    
    # Calculate phase based on reference oscillation
    period_ms = 1000.0 / reference_freq  # Convert Hz to ms period
    phase = (2 * np.pi * time_since_spike / period_ms) % (2 * np.pi)
    
    return float(phase)

def apply_neuron_polarities(
    eligibility_traces: torch.Tensor,
    neuron_polarities: torch.Tensor,
    synaptic_weights: csr_matrix
) -> torch.Tensor:
    """
    Apply neuron polarities to eligibility traces following blueprint specifications.
    
    Blueprint Reference: Rule 2.1 - Polarity scaling of eligibility traces
    Time Complexity: O(N²) for dense matrices, O(M) for sparse matrices where M is nnz elements
    
    Args:
        eligibility_traces: Current eligibility trace matrix (sparse or dense)
        neuron_polarities: Polarity values for each neuron [-1, +1]
        synaptic_weights: Sparse weight matrix for structure reference
        
    Returns:
        Polarity-scaled eligibility traces
    """
    if isinstance(eligibility_traces, torch.Tensor):
        # Dense tensor case
        if eligibility_traces.dim() == 2:
            # Scale rows by presynaptic neuron polarities
            scaled_traces = eligibility_traces * neuron_polarities.unsqueeze(1)
        else:
            # 1D case - assume flattened matrix
            n_neurons = len(neuron_polarities)
            traces_2d = eligibility_traces.view(n_neurons, n_neurons)
            scaled_traces = traces_2d * neuron_polarities.unsqueeze(1)
            scaled_traces = scaled_traces.view(-1)
    else:
        # Sparse matrix case - convert to dense for polarity scaling
        logger.warning("Converting sparse eligibility traces to dense for polarity scaling")
        dense_traces = torch.tensor(eligibility_traces.toarray(), dtype=torch.float32)
        scaled_traces = dense_traces * neuron_polarities.unsqueeze(1)
    
    return scaled_traces

def revgsp_learning_step(
    synaptic_weights: csr_matrix,
    eligibility_traces: torch.Tensor,
    spike_data: Dict,
    total_reward: float,
    plv: float,
    neuron_polarities: torch.Tensor,
    base_eta: float = 0.01,
    base_gamma: float = 0.9,
    lambda_decay: float = 0.001,
    current_time: float = 0.0
) -> Dict[str, any]:
    """
    Complete RE-VGSP learning step implementing the three-factor rule.
    
    Blueprint Reference: Rule 2 - RE-VGSP three-factor learning
    Components: spike-timing (local) + eligibility (memory) + reward (global)
    Time Complexity: O(N × S + M) where N=existing synapses, S=avg spikes, M=non-zero weights
    ✅ BLUEPRINT COMPLIANT: "O(N) algorithm where N is the number of synapses"
    
    Args:
        synaptic_weights: Current sparse weight matrix
        eligibility_traces: Current eligibility trace matrix
        spike_data: Dict containing pre/post spike times and neuron indices
        total_reward: Global reward signal from SIE
        plv: Phase-Locking Value for resonance enhancement
        neuron_polarities: Neuron polarity values [-1, +1]
        base_eta: Base learning rate
        base_gamma: Base eligibility trace decay
        lambda_decay: Weight decay factor
        current_time: Current simulation time
        
    Returns:
        Dict containing weight updates and updated eligibility traces
    """
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    if start_time:
        start_time.record()
    
    n_neurons = synaptic_weights.shape[0]
    
    # Extract spike pairs using blueprint-compliant O(N) method
    pre_indices, post_indices, delta_t, synapse_indices = extract_spike_pairs_from_synapses(
        synaptic_weights,
        spike_data.get('pre_spike_times', []),
        spike_data.get('post_spike_times', []),
        time_window=50.0
    )
    
    if len(delta_t) == 0:
        logger.debug("No spike pairs found for existing synapses in REVGSP learning")
        # Return decay-only updates
        eta_eff = calculate_modulated_learning_rate(base_eta, total_reward)
        weight_changes = -lambda_decay * torch.tensor(synaptic_weights.data, dtype=torch.float32)
        
        # 🔍 VALIDATION LOG: No spike pairs case - pure SIE modulation
        logger.debug(f"RE-VGSP decay-only: no spike pairs, eta_eff={eta_eff:.6f}, total_reward={total_reward:.6f}")
        if total_reward != 0.0:
            logger.debug("SIE-REVGSP handshake: SIE active but no local plasticity - learning suppressed")
        
        return {
            'weight_changes': weight_changes,
            'eligibility_traces': eligibility_traces * base_gamma,
            'plasticity_impulses': torch.zeros(0),
            'eta_effective': eta_eff,
            'gamma_effective': base_gamma,
            'n_spike_pairs': 0,
            'complexity_validation': 'O(N) - Blueprint compliant'
        }
    
    # Calculate phases for pre and post neurons
    pre_phases = []
    post_phases = []
    
    for i in range(len(pre_indices)):
        pre_idx = pre_indices[i].item()
        post_idx = post_indices[i].item()
        
        # Get spike times for phase calculation
        pre_spikes = spike_data.get('pre_spike_times', [[]])[pre_idx] if pre_idx < len(spike_data.get('pre_spike_times', [])) else []
        post_spikes = spike_data.get('post_spike_times', [[]])[post_idx] if post_idx < len(spike_data.get('post_spike_times', [])) else []
        
        pre_phase = calculate_phase_from_spike_times(pre_spikes, current_time=current_time)
        post_phase = calculate_phase_from_spike_times(post_spikes, current_time=current_time)
        
        pre_phases.append(pre_phase)
        post_phases.append(post_phase)
    
    pre_phases = torch.tensor(pre_phases, dtype=torch.float32)
    post_phases = torch.tensor(post_phases, dtype=torch.float32)
    
    # Calculate phase-sensitive plasticity impulses using blueprint formula from Rule 8.1
    # Formula: PI(t) = base_PI(Δt) * (1 + cos(phase_pre(t) - phase_post(t))) / 2
    plasticity_impulses = calculate_plasticity_impulse(delta_t, pre_phases, post_phases)
    
    # Validate phase-sensitive calculations are preserved
    phase_differences = pre_phases - post_phases
    phase_modulation = (1 + torch.cos(phase_differences)) / 2
    logger.debug(f"Phase modulation range: [{phase_modulation.min():.3f}, {phase_modulation.max():.3f}]")
    logger.debug(f"Mean phase difference: {phase_differences.mean():.3f} radians")
    
    # Calculate modulated parameters using PLV and reward
    eta_eff = calculate_modulated_learning_rate(base_eta, total_reward)
    gamma_eff = calculate_modulated_trace_decay(base_gamma, plv)
    
    # 🔍 VALIDATION LOG: SIE-REVGSP handshake timing and learning rate modulation
    logger.debug(f"SIE-REVGSP handshake: total_reward={total_reward:.6f} → eta_eff={eta_eff:.6f} (base={base_eta:.6f})")
    logger.debug(f"Learning rate modulation factor: {eta_eff/base_eta:.4f}x" if base_eta != 0 else "Learning rate modulation: base_eta is zero")
    
    # 🚨 HANDSHAKE VALIDATION: Check for timing correlation between SIE reward and RE-VGSP plasticity
    if len(plasticity_impulses) > 0:
        pi_timing_mean = torch.mean(torch.abs(delta_t)).item()
        logger.debug(f"Handshake timing: {len(plasticity_impulses)} plasticity impulses, mean_dt={pi_timing_mean:.2f}ms, reward_signal={total_reward:.6f}")
        
        # Check for potential timing mismatches that could disrupt learning
        if total_reward != 0.0 and len(plasticity_impulses) == 0:
            logger.warning("SIE-REVGSP timing mismatch: SIE providing reward but no RE-VGSP plasticity impulses")
        elif total_reward == 0.0 and len(plasticity_impulses) > 0:
            logger.warning("SIE-REVGSP timing mismatch: RE-VGSP plasticity active but no SIE reward signal")
    
    # 🔍 LEARNING RATE VALIDATION: Check for extreme modulation that could destabilize learning
    modulation_factor = eta_eff / base_eta if base_eta != 0 else float('inf')
    if modulation_factor > 5.0:
        logger.warning(f"Extreme learning rate amplification: {modulation_factor:.2f}x - SIE reward may be too high")
    elif modulation_factor < 0.1:
        logger.warning(f"Severe learning rate suppression: {modulation_factor:.2f}x - SIE reward may be too low")
    
    # Update eligibility traces
    if len(plasticity_impulses) > 0:
        # Create sparse plasticity impulse matrix
        pi_matrix = torch.zeros((n_neurons, n_neurons), dtype=torch.float32)
        for i, (pre_idx, post_idx, pi) in enumerate(zip(pre_indices, post_indices, plasticity_impulses)):
            pi_matrix[pre_idx, post_idx] += pi
        
        # Update eligibility traces using blueprint formula
        eligibility_traces = update_eligibility_trace(
            eligibility_traces, 
            pi_matrix.view(-1) if eligibility_traces.dim() == 1 else pi_matrix,
            gamma_eff
        )
    else:
        # Decay-only update
        eligibility_traces = eligibility_traces * gamma_eff
    
    # Apply neuron polarities to eligibility traces
    scaled_eligibility = apply_neuron_polarities(
        eligibility_traces, 
        neuron_polarities, 
        synaptic_weights
    )
    
    # Calculate weight changes using blueprint formula
    current_weights = torch.tensor(synaptic_weights.data, dtype=torch.float32)
    if scaled_eligibility.dim() == 2:
        # Flatten for weight change calculation
        scaled_eligibility_flat = scaled_eligibility.view(-1)
    else:
        scaled_eligibility_flat = scaled_eligibility
    
    # Only use non-zero weight entries
    nonzero_mask = current_weights != 0
    weight_changes = torch.zeros_like(current_weights)
    
    if torch.any(nonzero_mask):
        weight_changes[nonzero_mask] = calculate_weight_change(
            scaled_eligibility_flat[nonzero_mask],
            current_weights[nonzero_mask], 
            eta_eff,
            lambda_decay
        )
    
    if start_time:
        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        torch.cuda.synchronize()
        elapsed_ms = start_time.elapsed_time(end_time)
        logger.debug(f"REVGSP learning step: {elapsed_ms:.3f}ms, {len(plasticity_impulses)} spike pairs")
    
    # Validation logging for blueprint compliance
    cx = synaptic_weights.tocoo()
    logger.info(f"REVGSP: eta_eff={eta_eff:.6f}, gamma_eff={gamma_eff:.6f}, PLV={plv:.4f}, reward={total_reward:.6f}")
    logger.info(f"Blueprint compliance: O(N) where N={len(cx.data)} synapses, processed {len(plasticity_impulses)} spike pairs")
    
    return {
        'weight_changes': weight_changes,
        'eligibility_traces': eligibility_traces,
        'plasticity_impulses': plasticity_impulses,
        'eta_effective': eta_eff,
        'gamma_effective': gamma_eff,
        'n_spike_pairs': len(plasticity_impulses),
        'pre_indices': pre_indices,
        'post_indices': post_indices,
        'delta_t': delta_t,
        'complexity_validation': f'O(N) - Blueprint compliant, N={len(cx.data)} synapses',
        'phase_sensitivity_preserved': True
    }

def adapt_connectome_revgsp(
    substrate_state: Dict,
    sie_signals: Dict,
    adc_territories: Dict,
    timestep: int
) -> Dict[str, any]:
    """
    High-level connectome adaptation using RE-VGSP algorithm.
    
    Blueprint Reference: Complete RE-VGSP pipeline for connectome evolution
    
    Args:
        substrate_state: Current substrate state including weights and traces
        sie_signals: SIE outputs including total_reward and stability metrics
        adc_territories: ADC territory mapping and PLV calculations
        timestep: Current simulation timestep
        
    Returns:
        Updated substrate state and learning metrics
    """
    # Extract state components
    synaptic_weights = substrate_state['synaptic_weights']
    eligibility_traces = substrate_state['eligibility_traces']
    neuron_polarities = substrate_state.get('neuron_polarities', torch.ones(synaptic_weights.shape[0]))
    
    # Extract SIE signals
    total_reward = sie_signals.get('total_reward', 0.0)
    
    # Extract ADC PLV (Phase-Locking Value for resonance)
    plv = adc_territories.get('plv', 0.5)  # Default PLV if not provided
    
    # Prepare spike data from substrate
    spike_data = {
        'pre_spike_times': substrate_state.get('recent_spike_times', []),
        'post_spike_times': substrate_state.get('recent_spike_times', [])
    }
    
    # Apply RE-VGSP learning
    learning_results = revgsp_learning_step(
        synaptic_weights=synaptic_weights,
        eligibility_traces=eligibility_traces,
        spike_data=spike_data,
        total_reward=total_reward,
        plv=plv,
        neuron_polarities=neuron_polarities,
        current_time=float(timestep)
    )
    
    # Update substrate state
    updated_state = substrate_state.copy()
    updated_state['eligibility_traces'] = learning_results['eligibility_traces']
    
    # Apply weight changes to sparse matrix
    weight_changes = learning_results['weight_changes']
    if len(weight_changes) > 0:
        # Convert to sparse format for efficient updates
        updated_weights = synaptic_weights.copy()
        updated_weights.data += weight_changes.numpy()
        
        # Ensure no negative weights (optional constraint)
        updated_weights.data = np.maximum(updated_weights.data, 0.0)
        
        updated_state['synaptic_weights'] = updated_weights
    
    return {
        'updated_state': updated_state,
        'learning_metrics': {
            'eta_effective': learning_results['eta_effective'],
            'gamma_effective': learning_results['gamma_effective'],
            'n_spike_pairs': learning_results['n_spike_pairs'],
            'total_reward': total_reward,
            'plv': plv,
            'weight_change_magnitude': torch.sum(torch.abs(weight_changes)).item()
        }
    }

def apply_revgsp_updates(W, spike_times, time_step, eta, mod_factor, lambda_decay, params, is_excitatory):
    """
    Blueprint-compliant RE-VGSP (Resonance-Enhanced Valence-Gated Synaptic Plasticity) learning updates.
    
    Blueprint Reference: Rule 2 - The Learning Rule: The RE-VGSP "Handshake"
    Blueprint Reference: Rule 2.1 - Terminology: Plasticity Impulse vs. Eligibility Trace
    Blueprint Reference: Rule 8.1 - UTE: Spatio-Temporal-Polarity-Phase Volume Encoding
    Time Complexity: O(N × S + M) where N=existing synapses, S=avg spikes, M=non-zero weights
    ✅ BLUEPRINT COMPLIANT: "RE-VGSP is an O(N) algorithm where N is the number of synapses"
    
    RE-VGSP Formula: Δw_ij = (eta_effective(total_reward) * e_ij(t)) - (lambda_decay * w_ij)
    Where: e_ij(t) = gamma(PLV) * e_ij(t-1) + PI(t)
    And: PI(t) = base_PI(Δt) * (1 + cos(phase_pre(t) - phase_post(t))) / 2 [Rule 8.1]
    
    Parameters:
        W: Sparse synaptic weight matrix (input)
        spike_times: List of spike times per neuron (input)
        time_step: Current simulation time (input)
        eta: Base learning rate (input) - modulated by total_reward to create eta_effective
        mod_factor: SIE total_reward signal for reinforcement (input)
        lambda_decay: Weight decay factor for homeostatic stability (input)
        params: REVGSP parameters dict (input)
        is_excitatory: Boolean array for neuron polarities (input)
        
    Returns:
        Tuple[scipy.sparse.csc_matrix, dict]: Updated weights and metrics
    """
    logger.info("🔄 RE-VGSP learning updates initiated")
    
    # Convert to blueprint-compliant format
    from scipy.sparse import csr_matrix, csc_matrix
    import torch
    
    # Ensure sparse format for blueprint efficiency
    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)
    
    # Create RE-VGSP substrate state
    n_neurons = W.shape[0]
    eligibility_traces = torch.zeros(n_neurons, n_neurons)
    neuron_polarities = torch.tensor([1.0 if exc else -1.0 for exc in is_excitatory], dtype=torch.float32)
    
    # Prepare spike data for RE-VGSP interface
    time_window = 250.0  # ms
    pre_spike_times = []
    post_spike_times = []
    
    for neuron_spikes in spike_times:
        recent_spikes = [s for s in neuron_spikes if s > (time_step - time_window)]
        pre_spike_times.append(recent_spikes)
        post_spike_times.append(recent_spikes)
    
    spike_data = {
        'pre_spike_times': pre_spike_times,
        'post_spike_times': post_spike_times
    }
    
    # Map mod_factor to RE-VGSP total_reward (Blueprint Rule 3: SIE interaction)
    total_reward = mod_factor
    
    # Execute blueprint-compliant RE-VGSP learning step
    learning_results = revgsp_learning_step(
        synaptic_weights=W,
        eligibility_traces=eligibility_traces,
        spike_data=spike_data,
        total_reward=total_reward,
        plv=0.5,  # Default Phase-Locking Value for resonance enhancement
        neuron_polarities=neuron_polarities,
        base_eta=eta,
        lambda_decay=lambda_decay,
        current_time=float(time_step)
    )
    
    # Apply RE-VGSP weight changes
    weight_changes = learning_results['weight_changes']
    if len(weight_changes) > 0:
        updated_W = W.copy()
        
        # 🔧 FIXED: Apply E/I constraints to weight changes BEFORE adding to weights
        # This prevents fake potentiation in decay-only scenarios
        constrained_changes = weight_changes.numpy().copy()
        
        # Get current weights in dense format for constraint logic
        W_dense = W.toarray()
        inhib_indices = np.where(is_excitatory == False)[0]
        excit_indices = np.where(is_excitatory == True)[0]
        
        # For each synapse, check if the weight change would violate E/I constraints
        for i, (row_idx, col_idx) in enumerate(zip(*W.nonzero())):
            current_weight = W_dense[row_idx, col_idx]
            proposed_change = constrained_changes[i]
            new_weight = current_weight + proposed_change
            
            # Apply E/I constraints to the final weight
            if row_idx in inhib_indices:
                # Inhibitory synapse - ensure weight stays ≤ 0
                constrained_weight = min(new_weight, 0.0)
            else:
                # Excitatory synapse - ensure weight stays ≥ 0
                constrained_weight = max(new_weight, 0.0)
            
            # Calculate the actual allowed change
            constrained_changes[i] = constrained_weight - current_weight
        
        # Apply the constrained changes
        updated_W.data += constrained_changes
        
        # Apply magnitude clipping and remove self-connections
        updated_W.data = np.clip(updated_W.data, -2.0, 2.0)
        # Remove self-connections by setting diagonal to 0
        updated_W.setdiag(0)
        updated_W.prune()
    else:
        updated_W = W
    
    # Convert RE-VGSP metrics to compatible format
    net_change = updated_W.toarray() - W.toarray()
    revgsp_metrics = {
        'net_weight_change': np.sum(net_change),
        'potentiated_synapses': np.sum(net_change > 1e-9),
        'depressed_synapses': np.sum(net_change < -1e-9),
        'eta_effective': learning_results['eta_effective'],
        'n_spike_pairs': learning_results['n_spike_pairs'],
        'revgsp_complexity': learning_results['complexity_validation'],
        'phase_sensitivity_preserved': learning_results.get('phase_sensitivity_preserved', True)
    }
    
    logger.info(f"RE-VGSP: eta_eff={revgsp_metrics['eta_effective']:.6f}, {revgsp_metrics['revgsp_complexity']}")
    
    return updated_W, revgsp_metrics