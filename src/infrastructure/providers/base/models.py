"""
Provider-agnostic domain models (DTOs) for the providers layer.

These dataclasses define a normalized contract for LLM requests/responses,
model registry items, and provider metadata. They are intentionally minimal
and JSON-serializable to allow easy logging, caching, and testing.

Design goals
- Pure data: no provider-specific behavior here.
- Provider adapters convert between SDK objects and these DTOs.
- Upstream layers depend only on these models and provider interfaces.

See interfaces in: src/providers/base/interfaces.py (to be added).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Union


# Message roles used across providers.
Role = Literal["system", "user", "assistant", "tool"]

# Known content part types seen across providers' structured messages.
ContentPartType = Literal[
    "text",          # Plain text content
    "json",          # JSON content as string
    "tool_call",     # Tool call metadata
    "image",         # Image content (path/URL/base64) - adapter-defined semantics
    "refusal",       # Refusal reason text
    "other"          # Catch-all (adapter may attach provider-specific type info)
]


@dataclass
class ContentPart:
    """
    A single piece of structured content from an assistant message.

    Providers may return structured "parts" instead of a flat string.
    This DTO captures the minimum cross-provider subset.
    """
    type: ContentPartType
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Message:
    """
    A chat message. Content may be a flat string or a list of ContentPart.

    Adapters should normalize provider payloads into one of:
      - content: str
      - content: List[ContentPart]
    """
    role: Role
    content: Union[str, List[ContentPart]]

    def is_structured(self) -> bool:
        return isinstance(self.content, list)

    def text_or_joined(self) -> str:
        """
        Returns a reasonable text representation of the content for logging or
        providers that require flattened text.
        """
        if isinstance(self.content, str):
            return self.content
        parts: List[str] = []
        for p in self.content:
            if p.text:
                parts.append(p.text)
            elif p.data is not None:
                # Compact JSON-ish preview
                parts.append(f"[{p.type}]")
            else:
                parts.append(f"[{p.type}]")
        return "\n".join(parts)


@dataclass
class ProviderMetadata:
    """
    Execution metadata for a provider call (for diagnostics and audits).
    """
    provider_name: str
    model_name: str
    token_param_used: Optional[str] = None  # e.g., "max_tokens", "max_completion_tokens", "max_output_tokens"
    temperature_included: Optional[bool] = None
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatRequest:
    """
    Normalized request to an LLM provider.

    Upstream specifies only these fields; provider adapters map to SDK calls.
    """
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    response_format: Optional[str] = None  # "text" | "json_object" | adapter-specific
    extra: Dict[str, Any] = field(default_factory=dict)  # Escape hatch for rare needs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": m.role,
                    "content": (
                        m.content if isinstance(m.content, str)
                        else [p.to_dict() for p in m.content]
                    ),
                }
                for m in self.messages
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_format": self.response_format,
            "extra": self.extra,
        }


@dataclass
class ChatResponse:
    """
    Normalized response from an LLM provider.

    At least one of (text, parts) should be present.
    The raw field may contain an SDK object or dict for debugging.
    """
    text: Optional[str]
    parts: Optional[List[ContentPart]]
    raw: Optional[Any]
    meta: ProviderMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "parts": [p.to_dict() for p in self.parts] if self.parts else None,
            "raw": None,  # Avoid serializing heavy raw objects by default
            "meta": self.meta.to_dict(),
        }


@dataclass
class ModelInfo:
    """
    Describes a model entry from a provider's "get models" listing.

    The JSON files in src/providers/{provider}/{provider}-models.json
    can be represented as a list[ModelInfo], potentially with provider-
    specific 'capabilities' fields preserved in 'capabilities'.
    """
    id: str
    name: str
    provider: str
    family: Optional[str] = None  # e.g., "gpt-4o", "o1", "claude-3"
    context_length: Optional[int] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None  # iso8601 string if sourced from an API

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelRegistrySnapshot:
    """
    A snapshot of the model registry for a single provider.
    """
    provider: str
    models: List[ModelInfo]
    fetched_via: Optional[str] = None  # "api", "local", "ollama_list", etc.
    fetched_at: Optional[str] = None   # iso8601 timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "models": [m.to_dict() for m in self.models],
            "fetched_via": self.fetched_via,
            "fetched_at": self.fetched_at,
            "metadata": self.metadata,
        }