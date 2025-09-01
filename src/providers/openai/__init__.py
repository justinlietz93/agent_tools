"""
OpenAI provider package.

Exports:
- OpenAIProvider: Adapter implementing LLMProvider for OpenAI

See:
- [client.py](Cogito/src/providers/openai/client.py)
"""

from .client import OpenAIProvider

__all__ = ["OpenAIProvider"]