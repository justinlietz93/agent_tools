"""
Providers Base Package

Exports provider-agnostic contracts, DTOs, repositories, and the provider factory
for use within the providers layer (and, later, by orchestrators).

Conforms to the hybrid clean architecture scaffolding:
- Interfaces: normalized provider boundaries
- Models (DTOs): serialization-friendly request/response objects
- Repositories: model registry and key resolution
- Factory: lazy creation of provider adapters by canonical name
"""

from .models import (
    Role,
    ContentPartType,
    ContentPart,
    Message,
    ProviderMetadata,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ModelRegistrySnapshot,
)

from .interfaces import (
    LLMProvider,
    SupportsJSONOutput,
    SupportsResponsesAPI,
    ModelListingProvider,
    HasDefaultModel,
)

from .repositories.model_registry import ModelRegistryRepository
from .repositories.keys import KeysRepository, KeyResolution
from .factory import ProviderFactory, UnknownProviderError

__all__ = [
    # Models
    "Role",
    "ContentPartType",
    "ContentPart",
    "Message",
    "ProviderMetadata",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "ModelRegistrySnapshot",
    # Interfaces
    "LLMProvider",
    "SupportsJSONOutput",
    "SupportsResponsesAPI",
    "ModelListingProvider",
    "HasDefaultModel",
    # Repositories
    "ModelRegistryRepository",
    "KeysRepository",
    "KeyResolution",
    # Factory
    "ProviderFactory",
    "UnknownProviderError",
]