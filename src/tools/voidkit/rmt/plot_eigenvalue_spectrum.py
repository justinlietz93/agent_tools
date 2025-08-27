"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union

def plot_eigenvalue_spectrum(
    matrix: np.ndarray,
    bins: Union[int, str] = 'auto',
    save_path: str = None
):
    """
    Calculates the eigenvalues of a matrix and plots their distribution.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D numpy array.
    bins : int or str, optional
        The number of bins for the histogram, by default 'auto'.
    save_path : str, optional
        The path to save the plot image, by default None (plot is shown).
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise TypeError("Input must be a 2D numpy array.")
    
    eigenvalues = np.linalg.eigvals(matrix)
    
    plt.figure(figsize=(10, 6))
    plt.hist(np.real(eigenvalues), bins=bins, alpha=0.7, label='Real Part')
    plt.hist(np.imag(eigenvalues), bins=bins, alpha=0.7, label='Imaginary Part')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.title("Eigenvalue Spectrum")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
