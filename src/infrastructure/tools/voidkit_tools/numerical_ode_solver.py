"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Union, List, Optional, Any

def numerical_ode_solver(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Union[List[float], np.ndarray],
    t_eval: Optional[Union[List[float], np.ndarray]] = None,
    args: Tuple = (),
    method: str = 'RK45',
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None
) -> Any:
    """
    Numerically solve an ordinary differential equation (ODE) initial value problem.
    
    This function solves a system of first-order ODEs:
        dy/dt = fun(t, y, *args)
    with initial conditions:
        y(t0) = y0
    
    Parameters
    ----------
    fun : callable
        Function that defines the ODE system. The calling signature is fun(t, y, *args).
        It must return an array-like with the same shape as y.
    
    t_span : tuple of float
        Interval of integration (t0, tf). The solver starts with t=t0 and integrates
        until it reaches t=tf.
    
    y0 : array_like
        Initial state. Must be a 1-D array or list of floats.
    
    t_eval : array_like or None, optional
        Times at which to store the computed solution. If None (default), the solver
        will choose the time points automatically.
    
    args : tuple, optional
        Extra arguments to pass to the function `fun`. Default is an empty tuple.
    
    method : str, optional
        Integration method to use. Options include:
        - 'RK45': Explicit Runge-Kutta method of order 5(4) (default)
        - 'RK23': Explicit Runge-Kutta method of order 3(2)
        - 'DOP853': Explicit Runge-Kutta method of order 8
        - 'Radau': Implicit Runge-Kutta method of the Radau IIA family of order 5
        - 'BDF': Implicit multi-step variable-order (1 to 5) method
        - 'LSODA': Adams/BDF method with automatic stiffness detection
    
    rtol : float, optional
        Relative tolerance for the solver. Default is 1e-3.
    
    atol : float, optional
        Absolute tolerance for the solver. Default is 1e-6.
    
    max_step : float or None, optional
        Maximum allowed step size. If None (default), the solver will determine it automatically.
    
    Returns
    -------
    sol : OdeResult
        Object with the following attributes:
        - t: array, times at which the solution was computed
        - y: array, values of the solution at corresponding times in t
        - sol: callable, interpolated solution
        - t_events, y_events: arrays (only if events were detected)
        - nfev, njev, nlu: number of evaluations of the right-hand side, Jacobian, LU decompositions
        - status: int, reason for algorithm termination
        - message: str, human-readable description of the termination reason
        - success: bool, whether the solver succeeded
    
    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., t_span is not a 2-element tuple,
        y0 is not array-like, or method is not recognized).
    TypeError
        If input parameters have incorrect types.
    RuntimeError
        If the solver fails to converge or encounters other runtime issues.
    
    Examples
    --------
    Example 1: Simple exponential decay
    
    >>> def exponential_decay(t, y, rate_constant):
    ...     return -rate_constant * y
    ...
    >>> t_span = (0, 10)
    >>> y0 = [1.0]
    >>> rate_constant = 0.1
    >>> sol = numerical_ode_solver(exponential_decay, t_span, y0, args=(rate_constant,))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(sol.t, sol.y[0])
    >>> plt.xlabel('Time')
    >>> plt.ylabel('y(t)')
    >>> plt.title('Exponential Decay')
    
    Example 2: Lotka-Volterra predator-prey model
    
    >>> def lotka_volterra(t, z, a, b, c, d):
    ...     x, y = z
    ...     dx_dt = a * x - b * x * y
    ...     dy_dt = -c * y + d * x * y
    ...     return [dx_dt, dy_dt]
    ...
    >>> t_span = (0, 15)
    >>> y0 = [10, 5]  # Initial populations
    >>> params = (1.5, 1, 3, 1)  # a, b, c, d
    >>> t_eval = np.linspace(0, 15, 1000)  # Specific evaluation points
    >>> sol = numerical_ode_solver(lotka_volterra, t_span, y0, t_eval, args=params)
    >>> plt.plot(sol.t, sol.y[0], label='Prey')
    >>> plt.plot(sol.t, sol.y[1], label='Predator')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Population')
    >>> plt.legend()
    >>> plt.title('Lotka-Volterra Model')
    """
    # Input validation
    if not callable(fun):
        raise TypeError("The 'fun' parameter must be a callable function.")
    
    if not isinstance(t_span, tuple) or len(t_span) != 2:
        raise ValueError("The 't_span' parameter must be a tuple of two floats: (t0, tf).")
    
    if t_span[0] >= t_span[1]:
        raise ValueError("The 't_span' parameter must have t0 < tf.")
    
    try:
        y0_array = np.asarray(y0, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("The 'y0' parameter must be array-like with numeric values.")
    
    if y0_array.ndim != 1:
        raise ValueError("The 'y0' parameter must be a 1-D array or list.")
    
    if t_eval is not None:
        try:
            t_eval_array = np.asarray(t_eval, dtype=float)
        except (TypeError, ValueError):
            raise TypeError("The 't_eval' parameter must be array-like with numeric values.")
    
    if not isinstance(args, tuple):
        raise TypeError("The 'args' parameter must be a tuple.")
    
    # Validate method parameter
    valid_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Valid methods are: {', '.join(valid_methods)}")
    
    # Validate tolerance parameters
    if not isinstance(rtol, (int, float)) or rtol <= 0:
        raise ValueError("The 'rtol' parameter must be a positive number.")
    
    if not isinstance(atol, (int, float)) or atol <= 0:
        raise ValueError("The 'atol' parameter must be a positive number.")
    
    if max_step is not None:
        if not isinstance(max_step, (int, float)):
            raise TypeError("The 'max_step' parameter must be a number or None.")
        if max_step <= 0:
            raise ValueError("The 'max_step' parameter must be positive when specified.")
    
    # Solve the ODE system
    try:
        # First, verify that the function works with the provided arguments
        try:
            test_result = fun(t_span[0], y0_array, *args)
            test_result_array = np.asarray(test_result, dtype=float)
            if test_result_array.shape != y0_array.shape:
                raise ValueError(
                    f"The function 'fun' returned an array of shape {test_result_array.shape}, "
                    f"but expected shape {y0_array.shape} matching the initial state 'y0'."
                )
        except Exception as e:
            if isinstance(e, ValueError) and "shape" in str(e):
                raise
            else:
                raise ValueError(
                    f"Error when testing the ODE function with initial conditions: {str(e)}. "
                    f"Make sure 'fun' accepts arguments (t, y, *args) and returns an array of the same shape as y."
                )
        
        # Now solve the ODE
        # Only include max_step in kwargs if it's not None to avoid SciPy's internal validation issues
        kwargs = {
            'fun': fun,
            't_span': t_span,
            'y0': y0_array,
            't_eval': t_eval,
            'args': args,
            'method': method,
            'rtol': rtol,
            'atol': atol
        }
        
        if max_step is not None:
            kwargs['max_step'] = max_step
            
        sol = solve_ivp(**kwargs)
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        return sol
        
    except Exception as e:
        # Handle specific exceptions from solve_ivp with more informative messages
        if "The solver successfully" in str(e):
            # This is actually a success message, not an error
            return sol
        elif "The solver could not" in str(e) or "Maximum number of steps" in str(e):
            raise RuntimeError(
                f"ODE solver failed to converge: {str(e)}. Try adjusting rtol, atol, or max_step parameters."
            )
        elif "Invalid initial condition" in str(e):
            raise ValueError(f"Invalid initial condition: {str(e)}")
        elif isinstance(e, RuntimeError) and "ODE solver failed" in str(e):
            raise
        elif isinstance(e, ValueError) and ("shape" in str(e) or "testing the ODE function" in str(e)):
            raise
        else:
            raise RuntimeError(f"Error in numerical_ode_solver: {str(e)}")
