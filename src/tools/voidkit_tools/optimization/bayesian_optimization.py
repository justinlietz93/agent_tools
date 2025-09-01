"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

def bayesian_optimization(
    objective_func: Callable[[List[Any]], float],
    param_space: List[Dict[str, Any]],
    n_calls: int = 50,
    n_initial_points: int = 10,
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Performs Bayesian optimization on a given objective function.

    Parameters
    ----------
    objective_func : Callable[[List[Any]], float]
        The objective function to minimize. It must take a list of parameters
        and return a single float value.
    param_space : List[Dict[str, Any]]
        A list of dictionaries defining the search space for each parameter.
        Each dictionary should have 'type', 'name', and 'range' keys.
        - 'type': 'real', 'integer', or 'categorical'.
        - 'name': The name of the parameter.
        - 'range': A tuple for 'real' and 'integer' (e.g., (1e-6, 1e-1)),
                   or a list of categories for 'categorical'.
    n_calls : int, optional
        The number of calls to the objective function, by default 50.
    n_initial_points : int, optional
        The number of random points to sample before starting the optimization,
        by default 10.
    random_state : int, optional
        The random state for reproducibility, by default 0.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results of the optimization, including:
        - 'best_params': A dictionary of the best parameters found.
        - 'best_value': The minimum value of the objective function found.
        - 'result_object': The full result object from gp_minimize.
    """
    space = []
    param_names = []
    for p in param_space:
        param_names.append(p['name'])
        if p['type'] == 'real':
            space.append(Real(p['range'][0], p['range'][1], name=p['name']))
        elif p['type'] == 'integer':
            space.append(Integer(p['range'][0], p['range'][1], name=p['name']))
        elif p['type'] == 'categorical':
            space.append(Categorical(p['range'], name=p['name']))
        else:
            raise ValueError(f"Unsupported parameter type: {p['type']}")

    # Wrapper for the objective function to match skopt's expected input
    def objective_wrapper(params):
        return objective_func(params)

    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=random_state
    )

    best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}

    return {
        'best_params': best_params,
        'best_value': result.fun,
        'result_object': result
    }
