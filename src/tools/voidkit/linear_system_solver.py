"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np

def linear_system_solver(matrix_A, vector_b):
    """
    Solves a system of linear equations in the form Ax = b.
    
    Parameters
    ----------
    matrix_A : array_like
        Coefficient matrix of the linear system. Must be square and non-singular.
        
    vector_b : array_like
        Right-hand side vector of the linear system.
        
    Returns
    -------
    np.ndarray
        Solution vector x that satisfies Ax = b.
        
    Raises
    ------
    TypeError
        If matrix_A is not a 2D array or vector_b is not a 1D array.
        If inputs contain non-numeric data.
    ValueError
        If matrix_A is not square or if the shapes of matrix_A and vector_b are incompatible.
        If inputs contain NaN or infinity values.
    np.linalg.LinAlgError
        If matrix_A is singular or the computation does not converge.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Define a simple 2x2 system: 3x + 2y = 7, x + y = 3
    >>> A = np.array([[3, 2], [1, 1]])
    >>> b = np.array([7, 3])
    >>> x = linear_system_solver(A, b)
    >>> print(f"Solution: x = {x[0]}, y = {x[1]}")
    Solution: x = 1.0, y = 2.0
    
    Notes
    -----
    This function uses numpy.linalg.solve to compute the solution to the linear
    system Ax = b. The solution is computed using LU factorization.
    """
    # Check if inputs are provided
    if matrix_A is None:
        raise ValueError("matrix_A cannot be None")
    if vector_b is None:
        raise ValueError("vector_b cannot be None")
    
    # Convert inputs to numpy arrays
    try:
        matrix_A = np.asarray(matrix_A, dtype=float)
        vector_b = np.asarray(vector_b, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert inputs to numeric arrays: {str(e)}")
    
    # Check for NaN or infinity values
    if np.isnan(matrix_A).any() or np.isinf(matrix_A).any():
        raise ValueError("matrix_A contains NaN or infinity values")
    if np.isnan(vector_b).any() or np.isinf(vector_b).any():
        raise ValueError("vector_b contains NaN or infinity values")
    
    # Input validation for dimensions
    if matrix_A.ndim != 2:
        raise TypeError(f"matrix_A must be a 2D array, got {matrix_A.ndim}D array")
    
    if vector_b.ndim != 1:
        raise TypeError(f"vector_b must be a 1D array, got {vector_b.ndim}D array")
    
    # Input validation for shape
    if matrix_A.shape[0] != matrix_A.shape[1]:
        raise ValueError(f"matrix_A must be square, got shape {matrix_A.shape}")
    
    if matrix_A.shape[0] != vector_b.shape[0]:
        raise ValueError(f"Incompatible shapes: matrix_A has {matrix_A.shape[0]} rows, but vector_b has {vector_b.shape[0]} elements")
    
    # Check if matrix is empty
    if matrix_A.size == 0 or vector_b.size == 0:
        raise ValueError("Empty arrays are not valid inputs")
    
    # Solve the linear system
    try:
        x = np.linalg.solve(matrix_A, vector_b)
        return x
    except np.linalg.LinAlgError as e:
        if "singular" in str(e).lower():
            raise np.linalg.LinAlgError("The coefficient matrix is singular. The system has no unique solution.") from e
        else:
            raise np.linalg.LinAlgError(f"Failed to solve the linear system: {str(e)}") from e
