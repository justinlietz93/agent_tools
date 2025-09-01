"""
Custom exceptions for LLM provider interactions.
"""

class ProviderError(Exception):
    """Base class for provider-related errors."""
    pass

class ApiKeyError(ProviderError):
    """Error related to API key configuration or validity."""
    pass

class ApiCallError(ProviderError):
    """Error during the API call itself (e.g., network issue, server error)."""
    pass

class ApiResponseError(ProviderError):
    """Error related to the structure or content of the API response."""
    pass

class ApiBlockedError(ApiResponseError):
    """Error indicating the request was blocked, often due to safety filters."""
    def __init__(self, message, reason=None, ratings=None):
        super().__init__(message)
        self.reason = reason
        self.ratings = ratings

    def __str__(self):
        details = f"Reason: {self.reason}" if self.reason else "No reason provided"
        if self.ratings:
            details += f", Safety Ratings: {self.ratings}"
        return f"{super().__str__()} ({details})"


class JsonParsingError(ApiResponseError):
    """Error when failing to parse JSON from the API response."""
    pass

class JsonProcessingError(ApiResponseError):
    """Error during processing of the parsed JSON response."""
    pass

class ModelCallError(ProviderError):
    """Error during a model API call."""
    pass

class MaxRetriesExceededError(ProviderError):
    """Error when maximum number of retries is exceeded."""
    pass
