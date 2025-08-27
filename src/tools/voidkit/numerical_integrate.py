"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy import integrate
from typing import Callable, Tuple, Optional, Any, Union

def numerical_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    args: Tuple = ()
) -> Tuple[float, float]:
    """
    Numerically calculate the definite integral of a given function over a specified interval.
    
    This function uses scipy.integrate.quad to compute the definite integral of func(x)
    from a to b. It provides a robust interface to standard numerical integration libraries.
    
    Parameters
    ----------
    func : callable
        The function to integrate. Must be a Python callable that accepts a float
        as its first argument and returns a float. The function can have additional
        parameters which can be passed through the args parameter.
    
    a : float
        The lower limit of integration.
    
    b : float
        The upper limit of integration.
    
    args : tuple, optional
        Additional arguments to pass to the function. Default is an empty tuple.
    
    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - The computed value of the definite integral
        - The estimated absolute error in the result
    
    Raises
    ------
    TypeError
        If func is not callable, or if a or b are not numbers.
    ValueError
        If a is greater than or equal to b.
    RuntimeError
        If the integration fails to converge or encounters other numerical issues.
    
    Notes
    -----
    This function is a wrapper around scipy.integrate.quad, which uses adaptive
    quadrature methods to compute the integral with error control. The underlying
    algorithm is based on QUADPACK, a Fortran library for numerical integration.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Simple integral of x^2 from 0 to 1 (equals 1/3)
    >>> result, error = numerical_integrate(lambda x: x**2, 0, 1)
    >>> print(f"Result: {result:.6f}, Error: {error:.6e}")
    Result: 0.333333, Error: 3.700743e-15
    
    >>> # Integral with additional parameters
    >>> def integrand(x, a, b):
    ...     return a * np.sin(b * x)
    >>> result, error = numerical_integrate(integrand, 0, np.pi, args=(2.0, 1.0))
    >>> print(f"Result: {result:.6f}, Error: {error:.6e}")
    Result: 4.000000, Error: 4.440892e-14
    
    >>> # Integral of a more complex function
    >>> result, error = numerical_integrate(lambda x: np.exp(-x**2), -np.inf, np.inf)
    >>> print(f"Result: {result:.6f}, Error: {error:.6e}")
    Result: 1.772454, Error: 1.420416e-14
    """
    # Input validation
    if not callable(func):
        raise TypeError("func must be a callable")
    
    if not isinstance(a, (int, float)):
        raise TypeError("Lower limit 'a' must be a number")
    
    if not isinstance(b, (int, float)):
        raise TypeError("Upper limit 'b' must be a number")
    
    if a >= b:
        raise ValueError("Lower limit 'a' must be less than upper limit 'b'")
    
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    
    try:
        # Perform the numerical integration using scipy.integrate.quad
        result, error = integrate.quad(func, a, b, args=args)
        return result, error
    except Exception as e:
        # Handle potential errors from the integration
        if "singular" in str(e).lower():
            raise RuntimeError(f"Integration failed due to singularity in the function: {str(e)}")
        elif "converge" in str(e).lower():
            raise RuntimeError(f"Integration failed to converge: {str(e)}")
        else:
            raise RuntimeError(f"Integration failed: {str(e)}")
