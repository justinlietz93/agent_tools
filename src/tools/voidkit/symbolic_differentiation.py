import sympy
from typing import Union, List, Optional, Tuple

def symbolic_differentiate(expression: Union[str, sympy.Expr], 
                          variable: Union[str, sympy.Symbol, List[Union[str, sympy.Symbol]]], 
                          order: int = 1) -> sympy.Expr:
    """
    Performs symbolic differentiation on a mathematical expression.
    
    This function uses SymPy to compute the derivative of a symbolic expression
    with respect to one or more variables. It can handle both string inputs and
    SymPy expression objects.
    
    Parameters:
    -----------
    expression : Union[str, sympy.Expr]
        The mathematical expression to differentiate. Can be provided as a string
        or as a SymPy expression object.
    
    variable : Union[str, sympy.Symbol, List[Union[str, sympy.Symbol]]]
        The variable(s) with respect to which the differentiation should be performed.
        Can be provided as a string, a SymPy Symbol, or a list of strings/SymPy Symbols
        for partial derivatives.
    
    order : int, optional
        The order of the derivative to compute. Default is 1 (first derivative).
        Must be a positive integer.
    
    Returns:
    --------
    sympy.Expr
        The resulting derivative as a SymPy expression.
    
    Raises:
    -------
    TypeError
        If the input types are not as expected.
    ValueError
        If the order is not a positive integer, if the variable is not present in the expression,
        or if other validation fails.
    sympy.SympifyError
        If the expression string cannot be parsed into a valid SymPy expression.
    
    Examples:
    ---------
    >>> symbolic_differentiate("x**2 + 2*x + 1", "x")
    2*x + 2
    
    >>> from sympy import symbols, sin
    >>> x, y = symbols('x y')
    >>> expr = x**2 * sin(y)
    >>> symbolic_differentiate(expr, "x")
    2*x*sin(y)
    
    >>> symbolic_differentiate("x**2 * sin(y)", ["x", "y"])
    2*x*cos(y)
    """
    # Validate order parameter
    if not isinstance(order, int) or order < 1:
        raise ValueError(f"Order must be a positive integer, got {order}")
    
    # Convert expression to SymPy expression if it's a string
    if isinstance(expression, str):
        # In SymPy, ^ is not used for exponentiation (** is used instead)
        # This is a common error, so we'll check for it specifically
        if '^' in expression:
            raise sympy.SympifyError("Invalid syntax: '^' is not a valid operator in SymPy expressions. Use '**' for exponentiation.")
        
        try:
            expr = sympy.sympify(expression)
        except sympy.SympifyError as e:
            raise sympy.SympifyError(f"Failed to parse expression string: {e}")
    elif isinstance(expression, sympy.Expr):
        expr = expression
    else:
        raise TypeError(f"Expression must be a string or a SymPy expression, got {type(expression).__name__}")
    
    # Process the variable(s)
    if isinstance(variable, list):
        # Handle list of variables for partial derivatives
        if not variable:
            raise ValueError("Variable list cannot be empty")
        
        vars_list = []
        for var in variable:
            if isinstance(var, str):
                vars_list.append(sympy.Symbol(var))
            elif isinstance(var, sympy.Symbol):
                vars_list.append(var)
            else:
                raise TypeError(f"Variables in list must be strings or SymPy Symbols, got {type(var).__name__}")
        
        # Check if variables are in the expression
        free_symbols = expr.free_symbols
        for var in vars_list:
            if var not in free_symbols:
                # This is a special case - differentiating with respect to a variable not in the expression
                # returns zero according to calculus rules, so we'll handle it gracefully
                if len(vars_list) == 1:
                    return sympy.Integer(0)
                else:
                    # For multiple variables, we should warn the user
                    raise ValueError(f"Variable '{var}' is not present in the expression")
        
        # Apply differentiation for each variable in sequence
        result = expr
        for var in vars_list:
            result = sympy.diff(result, var, order)
        return result
    
    elif isinstance(variable, str):
        # Convert string to SymPy Symbol
        var_sym = sympy.Symbol(variable)
        
        # Check if variable is in the expression
        if var_sym not in expr.free_symbols:
            # Differentiating with respect to a variable not in the expression returns zero
            return sympy.Integer(0)
            
        return sympy.diff(expr, var_sym, order)
    
    elif isinstance(variable, sympy.Symbol):
        # Check if variable is in the expression
        if variable not in expr.free_symbols:
            # Differentiating with respect to a variable not in the expression returns zero
            return sympy.Integer(0)
            
        # Use the SymPy Symbol directly
        return sympy.diff(expr, variable, order)
    
    else:
        raise TypeError(f"Variable must be a string, a SymPy Symbol, or a list of these, got {type(variable).__name__}")
