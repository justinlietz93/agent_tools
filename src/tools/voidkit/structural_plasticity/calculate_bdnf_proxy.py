"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import List

def calculate_bdnf_proxy(
    spike_times_pre: np.ndarray,
    spike_times_post: np.ndarray,
    rewards: np.ndarray,
    time_window: float = 50.0
) -> float:
    """
    Calculates a proxy for Brain-Derived Neurotrophic Factor (BDNF) levels,
    which can be used to trigger structural plasticity.

    This proxy is based on the principle that correlated, rewarded activity
    promotes structural growth.

    Parameters
    ----------
    spike_times_pre : np.ndarray
        Spike times of the pre-synaptic neuron.
    spike_times_post : np.ndarray
        Spike times of the post-synaptic neuron.
    rewards : np.ndarray
        An array of reward signals, aligned with the spike times.
    time_window : float, optional
        The time window (in ms) to consider for correlated activity, by default 50.0.

    Returns
    -------
    float
        The calculated BDNF proxy value.
    """
    bdnf_proxy = 0.0
    for t_pre in spike_times_pre:
        # Find post-synaptic spikes within the time window
        post_in_window = spike_times_post[
            (spike_times_post > t_pre) & (spike_times_post <= t_pre + time_window)
        ]
        if len(post_in_window) > 0:
            # Find the reward associated with this correlated activity
            # (a simple approach is to take the max reward in the window)
            reward_in_window = rewards[
                (rewards[:, 0] > t_pre) & (rewards[:, 0] <= t_pre + time_window)
            ]
            if len(reward_in_window) > 0:
                max_reward = np.max(reward_in_window[:, 1])
                bdnf_proxy += max_reward * len(post_in_window)
    
    return bdnf_proxy
