from sympy import diff, integrate
from typing import Any

def differentiate(
    expr: Any, 
    variable: Any, 
    order: int = 1
) -> Any:
    """
    Symbolically differentiates an expression with respect to a variable.

    Parameters
    ----------
    expr : Any
        A SymPy expression.
    variable : Any
        The SymPy symbol to differentiate with respect to.
    order : int, optional
        The order of the derivative, by default 1.

    Returns
    -------
    Any
        The derivative of the expression.
    """
    return diff(expr, variable, order)

def integrate_expression(
    expr: Any, 
    variable: Any
) -> Any:
    """
    Symbolically integrates an expression with respect to a variable.

    Parameters
    ----------
    expr : Any
        A SymPy expression.
    variable : Any
        The SymPy symbol to integrate with respect to.

    Returns
    -------
    Any
        The indefinite integral of the expression.
    """
    return integrate(expr, variable)
