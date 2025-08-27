"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def generate_fractal_spike_train(
    fractal_dimension: float,
    k: float,
    tau_f: float,
    duration: float,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generates a spike train based on a fractal dynamics rule.

    spike_rate_i = k * D_f * e^(-t/τ_f)

    Parameters
    ----------
    fractal_dimension : float
        The fractal dimension D_f.
    k : float
        A scaling factor.
    tau_f : float
        The time constant for the exponential decay.
    duration : float
        The duration of the spike train to generate.
    dt : float, optional
        The time step, by default 1.0.

    Returns
    -------
    np.ndarray
        A 1D numpy array of spike times.
    """
    n_steps = int(duration / dt)
    time = np.arange(0, duration, dt)
    
    spike_rate = k * fractal_dimension * np.exp(-time / tau_f)
    
    # Generate spikes using a Poisson process with the given rate
    spike_train = np.random.rand(n_steps) < (spike_rate * dt / 1000.0) # Assuming dt is in ms
    
    return np.where(spike_train)[0] * dt
