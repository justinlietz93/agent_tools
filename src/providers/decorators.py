"""
Decorator utilities for AI providers.

This module provides useful decorators to simplify error handling, retry logic,
and other cross-cutting concerns for AI provider implementations.
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, TypeVar, Dict, List, Union, Tuple

# Import exceptions
from .exceptions import (
    ApiCallError, 
    ApiResponseError, 
    MaxRetriesExceededError,
    JsonParsingError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Return type for generic functions

def with_retry(max_attempts: int = 3, delay_base: float = 2.0):
    """
    Decorator that adds retry logic to API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_base: Base for exponential backoff delay calculation
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ApiCallError, ApiResponseError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = delay_base ** attempt
                        logger.warning(f"API call failed, retrying in {delay}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"API call failed after {max_attempts} retries: {e}")
            
            # If we get here, all attempts have failed
            if last_exception:
                raise MaxRetriesExceededError(f"Maximum retries exceeded: {last_exception}")
            else:
                raise ApiCallError("All API call attempts failed")
                
        return wrapper
    return decorator


def with_error_handling(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that handles common API errors and provides consistent logging.
    
    Args:
        func: The function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ApiCallError as e:
            logger.error(f"API call error in {func.__name__}: {e}")
            raise
        except ApiResponseError as e:
            logger.error(f"API response error in {func.__name__}: {e}")
            raise
        except JsonParsingError as e:
            logger.error(f"JSON parsing error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise ApiCallError(f"Unexpected error in {func.__name__}: {e}") from e
            
    return wrapper


def with_fallback(fallback_provider: str = 'openai'):
    """
    Decorator that provides a fallback to another provider if the primary one fails.
    
    Args:
        fallback_provider: Name of fallback provider to use
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary provider {func.__module__} failed: {e}, falling back to {fallback_provider}")
                
                # Import the fallback provider and call it with the same arguments
                # We import inside the function to avoid circular imports
                if fallback_provider == 'openai':
                    from . import openai_client
                    return openai_client.run_openai_client(*args, **kwargs)
                elif fallback_provider == 'anthropic':
                    from . import anthropic_client
                    return anthropic_client.run_anthropic_client(*args, **kwargs)
                elif fallback_provider == 'deepseek':
                    from . import deepseek_client
                    return deepseek_client.run_deepseek_client(*args, **kwargs)
                elif fallback_provider == 'gemini':
                    from . import gemini_client
                    return gemini_client.run_gemini_client(*args, **kwargs)
                else:
                    logger.error(f"Unknown fallback provider: {fallback_provider}")
                    raise
                
        return wrapper
    return decorator


def cache_result(maxsize: int = 128, ttl: int = 3600):
    """
    Decorator that caches function results with a time-to-live.
    
    Args:
        maxsize: Maximum size of the cache
        ttl: Time-to-live in seconds
        
    Returns:
        Decorator function
    """
    cache = {}
    timestamps = {}
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create a hashable key from the arguments
            key = str((args, frozenset(sorted(kwargs.items()))))
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if key in cache and current_time - timestamps[key] < ttl:
                logger.debug(f"Cache hit for {func.__name__}")
                return cache[key]
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            
            # Manage cache size with simple LRU approach
            if len(cache) >= maxsize:
                oldest_key = min(timestamps.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del timestamps[oldest_key]
            
            cache[key] = result
            timestamps[key] = current_time
            return result
            
        return wrapper
    return decorator
