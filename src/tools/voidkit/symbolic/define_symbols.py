

from sympy import symbols, Function, sympify
from typing import Union, List, Dict, Any

def define_symbols(names: Union[str, List[str]]) -> Union[Any, List[Any]]:
    """
    Defines one or more symbolic variables.

    Parameters
    ----------
    names : Union[str, List[str]]
        A string or list of strings with the names of the symbols.

    Returns
    -------
    Union[Any, List[Any]]
        A single symbol or a list of symbols.
    """
    return symbols(names)

def define_function(name: str) -> Function:
    """
    Defines a symbolic function.

    Parameters
    ----------
    name : str
        The name of the function.

    Returns
    -------
    Function
        A symbolic function.
    """
    return Function(name)

def string_to_expression(expr_str: str, local_dict: Dict[str, Any] = None) -> Any:
    """
    Converts a string to a SymPy expression.

    Parameters
    ----------
    expr_str : str
        The string representation of the expression.
    local_dict : Dict[str, Any], optional
        A dictionary of local symbols to use when parsing the string, by default None.

    Returns
    -------
    Any
        A SymPy expression.
    """
    return sympify(expr_str, locals=local_dict)
