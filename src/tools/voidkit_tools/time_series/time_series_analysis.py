"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple

def calculate_fft(
    signal: np.ndarray, 
    dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Fast Fourier Transform (FFT) of a signal.

    Parameters
    ----------
    signal : np.ndarray
        A 1D numpy array representing the time series signal.
    dt : float, optional
        The time step between samples, by default 1.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - The frequencies for the FFT.
        - The complex-valued FFT of the signal.
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, dt)[:n//2]
    return xf, yf[0:n//2]

def calculate_autocorrelation(signal: np.ndarray) -> np.ndarray:
    """
    Calculates the autocorrelation of a signal.

    Parameters
    ----------
    signal : np.ndarray
        A 1D numpy array representing the time series signal.

    Returns
    -------
    np.ndarray
        The autocorrelation of the signal.
    """
    mean_subtracted = signal - np.mean(signal)
    autocorr = np.correlate(mean_subtracted, mean_subtracted, mode='full')
    return autocorr[autocorr.size//2:] / (len(signal) * np.var(signal))

def calculate_cross_correlation(
    signal1: np.ndarray, 
    signal2: np.ndarray
) -> np.ndarray:
    """
    Calculates the cross-correlation between two signals.

    Parameters
    ----------
    signal1 : np.ndarray
        The first 1D numpy array.
    signal2 : np.ndarray
        The second 1D numpy array.

    Returns
    -------
    np.ndarray
        The cross-correlation of the two signals.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length.")
    
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)
    std1 = np.std(signal1)
    std2 = np.std(signal2)

    if std1 == 0 or std2 == 0:
        return np.zeros(len(signal1))

    cross_corr = np.correlate(signal1 - mean1, signal2 - mean2, mode='full')
    return cross_corr[cross_corr.size//2:] / (len(signal1) * std1 * std2)
