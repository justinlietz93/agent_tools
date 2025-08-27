"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import List, Dict

def detect_neuronal_avalanches(
    spike_times: np.ndarray, 
    bin_width: float = 1.0
) -> Dict[str, List[int]]:
    """
    Detects neuronal avalanches from a spike train.

    An avalanche is a continuous sequence of time bins with at least one spike,
    preceded and succeeded by an empty time bin.

    Parameters
    ----------
    spike_times : np.ndarray
        A 1D numpy array of spike times.
    bin_width : float, optional
        The width of the time bins in the same units as spike_times, by default 1.0.

    Returns
    -------
    Dict[str, List[int]]
        A dictionary containing the sizes and durations of the detected avalanches.
    """
    if len(spike_times) == 0:
        return {'sizes': [], 'durations': []}

    # Bin the spike times
    max_time = np.max(spike_times)
    bins = np.arange(0, max_time + bin_width, bin_width)
    binned_spikes, _ = np.histogram(spike_times, bins=bins)

    avalanches = {'sizes': [], 'durations': []}
    in_avalanche = False
    current_avalanche_size = 0
    current_avalanche_duration = 0

    for n_spikes in binned_spikes:
        if n_spikes > 0:
            if not in_avalanche:
                in_avalanche = True
            current_avalanche_size += n_spikes
            current_avalanche_duration += 1
        else:
            if in_avalanche:
                in_avalanche = False
                avalanches['sizes'].append(current_avalanche_size)
                avalanches['durations'].append(current_avalanche_duration)
                current_avalanche_size = 0
                current_avalanche_duration = 0
    
    if in_avalanche:
        avalanches['sizes'].append(current_avalanche_size)
        avalanches['durations'].append(current_avalanche_duration)

    return avalanches
