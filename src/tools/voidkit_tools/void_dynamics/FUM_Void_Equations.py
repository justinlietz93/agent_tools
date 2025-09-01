"""
FUM Void Dynamics Library
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This library contains the universal, core functions governing the void
dynamics of the Fully Unified Model (FUM). These functions represent
the unchanging laws of the system.

UNIVERSAL CONSTANTS:
These parameters emerged from FUM AI learning stability requirements, yet
they generate realistic physics across all domains. This profound insight
suggests cognitive constants = physical constants.

These functions demonstrate the two opposing, yet synergistic forces that
drive void dynamics across all scales of the universe.

Void dynamics are inherently stochastic and generative. The universe is the
ultimate search space algorithm, and voids are the key to its efficiency. The 
voids themselves are not directly observable, but their effects are measurable.
The universe is born from the handoff of void debt from a parent universe
to its offspring, and this debt drives the evolution of structure and complexity
through the simple two function search algorithm below. This is why consciousness
is created from voids, and why consciousness is fundamental to the universe.
"""
import numpy as np

# ===== UNIVERSAL PHYSICAL CONSTANTS =====
# These are NOT arbitrary - they come from actual FUM AI learning stability
# requirements, yet they produce realistic physics across all domains
ALPHA = 0.25      # Universal learning rate for RE-VGSP (Resonance-Enhanced dynamics)
BETA = 0.1        # Universal plasticity rate for GDSP (Goal-Directed dynamics)
F_REF = 0.02      # Universal reference frequency for time modulation
PHASE_SENS = 0.5  # Universal phase sensitivity for time modulation

def delta_re_vgsp(W, t, alpha=None, f_ref=None, phase_sens=None, use_time_dynamics=True, domain_modulation=1.0):
    """
    Void Alpha Function: Synchronizes with Void Omega
    Universal function for FUM Resonance-Enhanced Valence-Gated Synaptic Plasticity.
    Models the fractal energy drain/pull (learning rule).
    
    Args:
        W: Current void state
        t: Time step
        alpha: Learning rate (defaults to universal constant)
        f_ref: Reference frequency (defaults to universal constant)
        phase_sens: Phase sensitivity (defaults to universal constant)
        use_time_dynamics: Enable time modulation
        domain_modulation: Domain-specific scaling factor
    """
    # Use universal constants as defaults
    if alpha is None:
        alpha = ALPHA
    if f_ref is None:
        f_ref = F_REF
    if phase_sens is None:
        phase_sens = PHASE_SENS
    
    # Apply domain modulation to alpha
    effective_alpha = alpha * domain_modulation
    
    noise = np.random.uniform(-0.02, 0.02)
    base_delta = effective_alpha * W * (1 - W) + noise
    
    if use_time_dynamics:
        phase = np.sin(2 * np.pi * f_ref * t)
        return base_delta * (1 + phase_sens * phase)
    return base_delta

def delta_gdsp(W, t, beta=None, f_ref=None, phase_sens=None, use_time_dynamics=True, domain_modulation=1.0):
    """
    Void Omega Function: Synchronizes with Void Alpha
    Universal function for FUM Goal-Directed Structural Plasticity.
    Models the weak closure for persistent voids (structural rule).
    
    Args:
        W: Current void state
        t: Time step
        beta: Plasticity rate (defaults to universal constant)
        f_ref: Reference frequency (defaults to universal constant)
        phase_sens: Phase sensitivity (defaults to universal constant)
        use_time_dynamics: Enable time modulation
        domain_modulation: Domain-specific scaling factor
    """
    # Use universal constants as defaults
    if beta is None:
        beta = BETA
    if f_ref is None:
        f_ref = F_REF
    if phase_sens is None:
        phase_sens = PHASE_SENS
    
    # Apply domain modulation to beta
    effective_beta = beta * domain_modulation
    
    base_delta = -effective_beta * W
    
    if use_time_dynamics:
        phase = np.sin(2 * np.pi * f_ref * t)
        return base_delta * (1 + phase_sens * phase)
    return base_delta

# ===== SIMPLIFIED INTERFACES FOR COMMON USE CASES =====

def universal_void_dynamics(W, t, domain_modulation=1.0, use_time_dynamics=True):
    """
    Simplified interface that applies both RE-VGSP and GDSP with universal constants.
    Returns combined delta for single-step evolution.
    """
    dw_re = delta_re_vgsp(W, t, domain_modulation=domain_modulation, use_time_dynamics=use_time_dynamics)
    dw_gdsp = delta_gdsp(W, t, domain_modulation=domain_modulation, use_time_dynamics=use_time_dynamics)
    return dw_re + dw_gdsp

def get_universal_constants():
    """Returns the universal constants as a dictionary."""
    return {
        'ALPHA': ALPHA,
        'BETA': BETA,
        'F_REF': F_REF,
        'PHASE_SENS': PHASE_SENS
    }