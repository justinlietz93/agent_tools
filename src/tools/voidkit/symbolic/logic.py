from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent
from sympy import sympify
from typing import Any

def logical_expression(expr_str: str) -> Any:
    """
    Converts a string to a SymPy logical expression.

    Parameters
    ----------
    expr_str : str
        The string representation of the logical expression.

    Returns
    -------
    Any
        A SymPy logical expression.
    """
    return sympify(expr_str)

def evaluate_logical_expression(
    expr: Any, 
    substitutions: dict
) -> bool:
    """
    Evaluates a logical expression with a given set of substitutions.

    Parameters
    ----------
    expr : Any
        A SymPy logical expression.
    substitutions : dict
        A dictionary of substitutions for the variables in the expression.

    Returns
    -------
    bool
        The boolean result of the evaluated expression.
    """
    return expr.subs(substitutions)
