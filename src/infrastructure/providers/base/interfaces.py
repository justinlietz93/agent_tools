"""
Provider-agnostic interfaces (ABCs/Protocols) for the providers layer.

This module defines the boundary contracts that upstream code should depend on:
- LLMProvider: minimal chat interface taking a normalized ChatRequest and returning ChatResponse
- Capability mixins (Protocols) to advertise optional features without tight coupling
- ModelListingProvider: unified way to (re)load provider model registries

Adapters for each concrete provider (OpenAI, Anthropic, etc.) should implement LLMProvider
and optionally capability mixins. Upstream does not import SDKs directly.

Related DTOs are defined in: src/providers/base/models.py
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Optional
from .models import ChatRequest, ChatResponse, ModelRegistrySnapshot


@runtime_checkable
class LLMProvider(Protocol):
    """
    Minimal interface for Large Language Model providers.

    Implementations should map ChatRequest fields to their SDK-specific parameters,
    normalize responses to ChatResponse, and never leak SDK objects upstream.
    """

    @property
    def provider_name(self) -> str:
        """Canonical provider identifier, e.g., 'openai', 'anthropic'."""
        ...

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a single chat completion request.

        - Must not raise for common provider errors; instead encode details in ChatResponse.meta.extra
          and return a best-effort text/parts. Reserve exceptions for programmer errors.
        """
        ...


@runtime_checkable
class SupportsJSONOutput(Protocol):
    """
    Capability marker for providers that can request JSON-native responses.

    Providers implementing this may honor ChatRequest.response_format == 'json_object'
    or an equivalent representation, and normalize JSON content into a text string
    (serialized) or parts as appropriate.
    """
    def supports_json_output(self) -> bool:
        return True


@runtime_checkable
class SupportsResponsesAPI(Protocol):
    """
    Capability marker for providers that expose a 'responses' style API
    (e.g., OpenAI o1/o3-mini) separate from classic chat completions.
    """
    def uses_responses_api(self, model: str) -> bool:
        """
        Return True if the given model should be executed via the Responses API path.
        """
        ...


@runtime_checkable
class ModelListingProvider(Protocol):
    """
    Interface for providers that can materialize a model registry snapshot
    (e.g., via a 'get models' endpoint or local tooling like 'ollama list').
    """
    def list_models(self, refresh: bool = False) -> ModelRegistrySnapshot:
        """
        Return a snapshot of known models for this provider.

        - When refresh=True, implementations should attempt to re-fetch from source
          (network/API or local tool) and persist the provider's JSON registry.
        - When refresh=False, implementations may return a cached snapshot from JSON.
        """
        ...


# Optional convenience mixin for providers with a default model
@runtime_checkable
class HasDefaultModel(Protocol):
    def default_model(self) -> Optional[str]:
        return None