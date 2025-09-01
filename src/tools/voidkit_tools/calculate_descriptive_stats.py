import numpy as np
from scipy import stats

def calculate_descriptive_stats(data, nan_policy='propagate'):
    """
    Calculate descriptive statistics for a given dataset.
    
    Parameters
    ----------
    data : array_like
        Input data array. Must be convertible to a 1D NumPy array of numbers.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaN values:
        - 'propagate': returns nan for statistics when NaN values are present (default)
        - 'omit': ignores NaN values when computing statistics
        - 'raise': raises an error if NaN values are present
    
    Returns
    -------
    dict
        Dictionary containing the following descriptive statistics:
        - 'count': Number of observations
        - 'mean': Arithmetic mean
        - 'stdev': Standard deviation (sample)
        - 'variance': Variance (sample)
        - 'min': Minimum value
        - 'max': Maximum value
        - 'skewness': Skewness (third standardized moment)
        - 'kurtosis': Kurtosis (fourth standardized moment, Fisher's definition: normal = 0.0)
    
    Raises
    ------
    TypeError
        If input data cannot be converted to a numeric array.
    ValueError
        If nan_policy is not one of {'propagate', 'omit', 'raise'}.
        If input data is empty.
        If NaN values are present and nan_policy is 'raise'.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = [1, 2, 3, 4, 5]
    >>> stats = calculate_descriptive_stats(data)
    >>> print(stats['mean'])
    3.0
    >>> print(stats['stdev'])
    1.5811388300841898
    """
    # Validate nan_policy parameter
    valid_policies = ['propagate', 'omit', 'raise']
    if nan_policy not in valid_policies:
        raise ValueError(f"nan_policy must be one of {valid_policies}, got '{nan_policy}'")
    
    # Check if data is None
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Convert input to numpy array
    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert input data to numeric array: {str(e)}")
    
    # Ensure data is 1D
    if data_array.ndim > 1:
        raise ValueError(f"Input data must be 1-dimensional, got {data_array.ndim}-dimensional data")
    
    # Check if data is empty
    if data_array.size == 0:
        raise ValueError("Input data cannot be empty")
    
    # Check for NaN values if policy is 'raise'
    if nan_policy == 'raise' and np.isnan(data_array).any():
        raise ValueError("Input data contains NaN values")
    
    # Special case for single value
    if data_array.size == 1:
        value = data_array[0]
        return {
            'count': 1,
            'mean': value,
            'variance': 0.0,
            'stdev': 0.0,
            'min': value,
            'max': value,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    # Special case for constant values (zero variance)
    if np.all(data_array == data_array[0]):
        value = data_array[0]
        return {
            'count': data_array.size,
            'mean': value,
            'variance': 0.0,
            'stdev': 0.0,
            'min': value,
            'max': value,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    # Handle NaN values
    has_nan = np.isnan(data_array).any()
    
    # Use scipy.stats.describe to efficiently calculate multiple statistics
    try:
        # For propagate policy with NaNs, we need special handling for min/max
        if nan_policy == 'propagate' and has_nan:
            # Get non-NaN values for min/max calculation
            non_nan_data = data_array[~np.isnan(data_array)]
            
            result = stats.describe(data_array, nan_policy=nan_policy)
            
            # Process the DescribeResult object into a dictionary with standardized keys
            stats_dict = {
                'count': np.sum(~np.isnan(data_array)),  # Count of non-NaN values
                'mean': result.mean,
                'variance': result.variance,
                'stdev': np.sqrt(result.variance) if not np.isnan(result.variance) else np.nan,
                'min': np.min(non_nan_data) if non_nan_data.size > 0 else np.nan,
                'max': np.max(non_nan_data) if non_nan_data.size > 0 else np.nan,
                'skewness': result.skewness,
                'kurtosis': result.kurtosis
            }
        else:
            result = stats.describe(data_array, nan_policy=nan_policy)
            
            # Process the DescribeResult object into a dictionary with standardized keys
            stats_dict = {
                'count': result.nobs,
                'mean': result.mean,
                'variance': result.variance,
                'stdev': np.sqrt(result.variance),
                'min': result.minmax[0],
                'max': result.minmax[1],
                'skewness': result.skewness,
                'kurtosis': result.kurtosis
            }
        
        return stats_dict
        
    except Exception as e:
        # Handle specific known errors
        if "zero variance" in str(e).lower():
            # Special handling for zero variance case
            # Calculate what we can and set others to appropriate values
            stats_dict = {
                'count': len(data_array),
                'mean': np.mean(data_array),
                'variance': 0.0,
                'stdev': 0.0,
                'min': np.min(data_array),
                'max': np.max(data_array),
                'skewness': 0.0,  # Undefined for constant data, set to 0
                'kurtosis': 0.0   # Undefined for constant data, set to 0
            }
            return stats_dict
        else:
            # Re-raise with a more informative error message
            raise type(e)(f"Error calculating descriptive statistics: {str(e)}")
