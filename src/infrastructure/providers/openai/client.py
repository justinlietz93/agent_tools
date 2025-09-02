"""
OpenAIProvider adapter

Implements the provider-agnostic interfaces defined in
[interfaces.py](Cogito/src/providers/base/interfaces.py) using the existing
OpenAI client implementation at [openai_client.py](Cogito/src/providers/openai_client.py).

This adapter:
- Accepts a normalized [ChatRequest](Cogito/src/providers/base/models.py:1)
- Invokes [call_openai_with_retry()](Cogito/src/providers/openai_client.py:51)
- Returns a normalized [ChatResponse](Cogito/src/providers/base/models.py:1)
- Exposes model registry listing via [ModelRegistryRepository](Cogito/src/providers/base/repositories/model_registry.py:1)

Notes
- Non-invasive: no changes to existing openai_client behavior.
- Stays contained within providers/ as requested.
"""

from __future__ import annotations

import json
import time
from typing import Optional, List

from src.providers.base.interfaces import (
    LLMProvider,
    SupportsJSONOutput,
    SupportsResponsesAPI,
    ModelListingProvider,
    HasDefaultModel,
)
from src.providers.base.models import (
    ChatRequest,
    ChatResponse,
    ProviderMetadata,
    Message,
    ContentPart,
)
from src.providers.base.repositories.model_registry import ModelRegistryRepository
from src.providers.model_config import get_openai_config
from src.providers.openai_client import call_openai_with_retry


class OpenAIProvider(LLMProvider, SupportsJSONOutput, SupportsResponsesAPI, ModelListingProvider, HasDefaultModel):
    """
    Adapter for OpenAI which conforms to provider-agnostic contracts.

    Construction:
      provider = OpenAIProvider()  # uses defaults from [get_openai_config()](Cogito/src/providers/model_config.py:69)
      response = provider.chat(ChatRequest(...))
    """

    def __init__(self, default_model: Optional[str] = None, registry: Optional[ModelRegistryRepository] = None) -> None:
        cfg = get_openai_config()
        self._default_model = default_model or cfg.get("model") or "o3-mini"
        self._registry = registry or ModelRegistryRepository()

    @property
    def provider_name(self) -> str:
        return "openai"

    def default_model(self) -> Optional[str]:
        return self._default_model

    def supports_json_output(self) -> bool:
        return True

    def uses_responses_api(self, model: str) -> bool:
        lower = (model or "").lower()
        return ("o1" in lower) or ("o3-mini" in lower)

    def list_models(self, refresh: bool = False):
        return self._registry.list_models("openai", refresh=refresh)

    # -------------------- Core Chat --------------------

    def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a single chat request via existing call_openai_with_retry, normalizing the result.
        """
        # Resolve model with request override
        model = (request.model or self._default_model)

        # Extract a single system message (first wins) and concatenate user content
        system_message: Optional[str] = None
        user_segments: List[str] = []

        for m in request.messages:
            if not isinstance(m, Message):
                continue
            if m.role == "system" and system_message is None:
                system_message = m.text_or_joined()
            elif m.role == "user":
                user_segments.append(m.text_or_joined())
            # assistant/tool roles are ignored for now in this minimal adapter; could be included as context

        user_content = "\n".join(s for s in user_segments if s)

        # Build config payload as expected by call_openai_with_retry
        api_config = {
            "api": {
                "openai": {
                    "model": model,
                    # request.max_tokens maps to the appropriate OpenAI param internally
                    **({"max_tokens": request.max_tokens} if request.max_tokens is not None else {}),
                    # temperature is omitted downstream if the chosen model family disallows it
                    **({"temperature": request.temperature} if request.temperature is not None else {}),
                    **({"system_message": system_message} if system_message else {}),
                }
            }
        }

        # Structured output?
        is_structured = (request.response_format == "json_object")

        # Invoke provider and measure latency
        t0 = time.perf_counter()
        try:
            resp, model_used = call_openai_with_retry(
                prompt_template="{content}",
                context={"content": user_content},
                config=api_config,
                is_structured=is_structured,
                # Also pass max_tokens explicitly to cooperate with responses.create path
                max_tokens=request.max_tokens if request.max_tokens is not None else None,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            # Normalize catastrophic failures into a ChatResponse with error metadata
            meta = ProviderMetadata(
                provider_name=self.provider_name,
                model_name=model,
                http_status=None,
                request_id=None,
                latency_ms=None,
                extra={"error": str(e), "phase": "call_openai_with_retry"},
            )
            return ChatResponse(text=None, parts=None, raw=None, meta=meta)

        # Build metadata (limited visibility since underlying client hides HTTP details)
        meta = ProviderMetadata(
            provider_name=self.provider_name,
            model_name=str(model_used),
            token_param_used=None,          # Not directly exposed by call_openai_with_retry
            temperature_included=None,      # Not directly exposed by call_openai_with_retry
            http_status=None,               # Not returned by the high-level client
            request_id=None,                # Not returned by the high-level client
            latency_ms=latency_ms,
            extra={
                "is_structured": is_structured,
                "used_responses_api": self.uses_responses_api(str(model_used)),
                "response_format": request.response_format,
            },
        )

        # Normalize output into text/parts
        if isinstance(resp, dict):
            # Structured JSON returned; represent as a single JSON part and provide a text serialization
            try:
                text = json.dumps(resp, ensure_ascii=False)
            except Exception:
                text = str(resp)
            parts = [ContentPart(type="json", text=text, data=None)]
            return ChatResponse(text=text, parts=parts, raw=None, meta=meta)

        # Fallback: plain text or unknown object
        if isinstance(resp, str):
            return ChatResponse(text=resp, parts=None, raw=None, meta=meta)

        # Last resort: string-coerce unknown types
        return ChatResponse(text=str(resp), parts=None, raw=None, meta=meta)