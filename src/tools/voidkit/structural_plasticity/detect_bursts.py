"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def detect_bursts(
    spike_times: np.ndarray, 
    max_interspike_interval: float = 10.0,
    min_spikes_in_burst: int = 3
) -> np.ndarray:
    """
    Detects bursts of spikes in a spike train.

    A burst is defined as a sequence of spikes where the inter-spike interval
    is less than or equal to max_interspike_interval.

    Parameters
    ----------
    spike_times : np.ndarray
        A 1D numpy array of spike times.
    max_interspike_interval : float, optional
        The maximum time between spikes to be considered part of a burst, 
        by default 10.0.
    min_spikes_in_burst : int, optional
        The minimum number of spikes required to form a burst, by default 3.

    Returns
    -------
    np.ndarray
        An array of the start and end times of the detected bursts.
    """
    if len(spike_times) < min_spikes_in_burst:
        return np.array([])

    interspike_intervals = np.diff(spike_times)
    is_in_burst = interspike_intervals <= max_interspike_interval

    bursts = []
    current_burst_start = -1

    for i in range(len(is_in_burst)):
        if is_in_burst[i] and current_burst_start == -1:
            current_burst_start = spike_times[i]
        elif not is_in_burst[i] and current_burst_start != -1:
            burst_end = spike_times[i]
            if (i - np.where(spike_times == current_burst_start)[0][0] + 1) >= min_spikes_in_burst:
                bursts.append([current_burst_start, burst_end])
            current_burst_start = -1
            
    if current_burst_start != -1:
        burst_end = spike_times[-1]
        if (len(spike_times) - np.where(spike_times == current_burst_start)[0][0]) >= min_spikes_in_burst:
            bursts.append([current_burst_start, burst_end])

    return np.array(bursts)
