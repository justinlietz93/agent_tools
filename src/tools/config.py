"""
Configuration module for loading environment variables and settings.

This module handles:
1. Loading API keys from .env file
2. Setting default configurations
3. Validating required settings (lax; does not force Deepseek key for local/Ollama)
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for tool settings and API keys."""
    
    # OpenAI-compatible endpoints (support DeepSeek, Ollama, custom)
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_BASE_URL: str = os.getenv('OPENAI_BASE_URL', '')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', '')
    
    # DeepSeek (OpenAI-compatible)
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_BASE_URL: str = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    DEEPSEEK_MODEL: str = os.getenv('DEEPSEEK_MODEL', 'deepseek-reasoner')
    
    # Ollama (OpenAI-compatible shim at /v1)
    OLLAMA_API_KEY: str = os.getenv('OLLAMA_API_KEY', 'ollama')  # placeholder, often unused
    OLLAMA_BASE_URL: str = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    OLLAMA_MODEL: str = os.getenv('OLLAMA_MODEL', 'llama3.1')
    
    # Optional API Keys
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    GOOGLE_SEARCH_API_KEY: str = os.getenv('GOOGLE_SEARCH_API_KEY', '')
    GOOGLE_SEARCH_ENGINE_ID: str = os.getenv('GOOGLE_SEARCH_ENGINE_ID', '')
    BING_SEARCH_API_KEY: str = os.getenv('BING_SEARCH_API_KEY', '')
    SERP_API_KEY: str = os.getenv('SERP_API_KEY', '')
    
    # Optional configurations with defaults
    WEB_BROWSER_TIMEOUT: int = int(os.getenv('WEB_BROWSER_TIMEOUT', '30'))
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv('DEFAULT_SEARCH_RESULTS', '10'))
    
    # Package manager settings
    PACKAGE_MANAGER_CONFIG = {
        "use_module_pip": True,  # Default to python -m pip for better venv support
        "pip_command": None,     # Custom pip command if needed
    }
    
    # Validation flags
    REQUIRE_DEEPSEEK_KEY: bool = os.getenv('REQUIRE_DEEPSEEK_KEY', 'false').lower() in ('1', 'true', 'yes')

    def __init__(self):
        """Initialize configuration without forcing Deepseek API key."""
        # No hard validation here to allow local/Ollama-first usage
        pass
    
    @classmethod
    def validate_api_keys(cls) -> None:
        """
        Optionally validate that required API keys are set.
        By default, we do NOT require Deepseek key to allow local/Ollama usage.
        """
        if cls.REQUIRE_DEEPSEEK_KEY and not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "Missing required API key: DEEPSEEK_API_KEY. "
                "Set REQUIRE_DEEPSEEK_KEY=true to enforce; otherwise it's optional."
            )
    
    @classmethod
    def get_api_key(cls, key_name: str) -> Optional[str]:
        """
        Get an API key by name.
        
        Args:
            key_name: Name of the API key to retrieve
            
        Returns:
            The API key value or None if not found
        """
        return getattr(cls, key_name, None)

# Lax validation on import (no exception by default)
Config.validate_api_keys()