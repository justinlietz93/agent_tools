Providers layer (Hybrid Clean Architecture)

Overview
- Goal: Normalize provider interactions behind small, typed contracts while keeping all provider-specific logic contained under src/providers/.
- Key benefits:
  - Provider-agnostic DTOs and interfaces
  - Central factory for adapter creation
  - Model registry repository to read/write per-provider model listings (JSON)
  - Pluggable “get models” fetchers per provider
  - Non-invasive: adapters wrap existing clients where possible

Core contracts and types
- Interfaces:
  - LLMProvider (core provider contract)
  - SupportsJSONOutput, SupportsResponsesAPI (capability flags)
  - ModelListingProvider (expose list_models)
  - HasDefaultModel (optional default model)
  See [interfaces.py](Cogito/src/providers/base/interfaces.py).

- DTOs (Provider-agnostic models):
  - Message, ChatRequest, ChatResponse, ProviderMetadata
  - ModelInfo, ModelRegistrySnapshot
  - References:
    - [Message](Cogito/src/providers/base/models.py:52)
    - [ChatRequest](Cogito/src/providers/base/models.py:104)
    - [ChatResponse](Cogito/src/providers/base/models.py:139)
    - [ProviderMetadata](Cogito/src/providers/base/models.py:86)
    - [ModelInfo](Cogito/src/providers/base/models.py:160)
    - [ModelRegistrySnapshot](Cogito/src/providers/base/models.py:181)
  See [models.py](Cogito/src/providers/base/models.py).

- Repositories:
  - Model registry I/O and refresh orchestration:
    - [ModelRegistryRepository](Cogito/src/providers/base/repositories/model_registry.py:64)
      - Key method: list_models(provider, refresh=False)
      - Reads canonical JSON files under src/providers/{provider}/{provider}-models.json
      - Optional refresh invokes provider “get models” script
      - Resilient JSON read for empty/whitespace files:
        [ModelRegistryRepository._read_provider_json()](Cogito/src/providers/base/repositories/model_registry.py:253)
  - Keys (API key resolution):
    - [KeysRepository.get_api_key()](Cogito/src/providers/base/repositories/keys.py:62)
    - Resolution order: env (OPENAI_API_KEY, etc.) → config.yaml → None

- Factory:
  - [ProviderFactory.create()](Cogito/src/providers/base/factory.py:47)
    - Maps canonical provider name (e.g., "openai") to adapter class by lazy import
    - Raises UnknownProviderError for unregistered/failed adapters

Provider adapters
- OpenAI
  - Adapter: [OpenAIProvider](Cogito/src/providers/openai/client.py:44)
    - Exposes:
      - provider_name: "openai"
      - default_model(): str
      - supports_json_output(): bool
      - uses_responses_api(model: str): bool (o3/o1 families)
      - list_models(refresh: bool): ModelRegistrySnapshot
      - chat(request: ChatRequest): ChatResponse
    - Internals:
      - Wraps existing call_openai_with_retry() with normalized ChatRequest/ChatResponse
      - model default derived from get_openai_config() with fallback (o3-mini)
      - Structured outputs: request.response_format == "json_object"

Model registry
- JSON location (per provider):
  - src/providers/{provider}/{provider}-models.json
  - Examples:
    - OpenAI: src/providers/openai/openai-models.json
    - Anthropic: src/providers/anthropic/anthropic-models.json
- Repository: [ModelRegistryRepository](Cogito/src/providers/base/repositories/model_registry.py:64)
  - list_models(provider, refresh=False)
    - refresh=True will attempt a provider get-models runner:
      src.providers.{provider}.get_{provider}_models
    - Accepts multiple entrypoint names: run(), get_models(), fetch_models(), update_models(), refresh_models(), main()
  - Resilient read:
    - [ModelRegistryRepository._read_provider_json()](Cogito/src/providers/base/repositories/model_registry.py:253)
    - Empty/whitespace-only files are treated as empty registry with a warning (stderr)

Provider “get models” fetchers (API-backed)
- OpenAI:
  - Module: [get_openai_models.py](Cogito/src/providers/openai/get_openai_models.py)
    - Entry point: [run()](Cogito/src/providers/openai/get_openai_models.py:49)
    - Behavior:
      - If OPENAI_API_KEY is present, fetch via OpenAI SDK:
        client = OpenAI(api_key=key)
        items = client.models.list()
      - Normalize and persist via save_provider_models()
      - Return list[dict] for ModelRegistryRepository convenience
    - Offline mode:
      - Falls back to load_cached_models(provider) and returns cached entries
      - Does not write empty registries (no false positives)
  - Key resolution:
    - [KeysRepository.get_api_key("openai")](Cogito/src/providers/base/repositories/keys.py:62)
    - Uses env (OPENAI_API_KEY) or config.yaml api.openai.api_key

Usage

1) Installing dependencies
- Base:
  - pip install -r requirements.txt
- Dev (pytest, etc.):
  - pip install -r requirements-dev.txt

2) API keys
- Preferred: .env at project root (Cogito/.env)
  - Example:
    OPENAI_API_KEY=sk-...
- Alternative: config.yaml
  - api:
      openai:
        api_key: "sk-..."
- Key resolution order:
  - Env → config.yaml → None. See [KeysRepository](Cogito/src/providers/base/repositories/keys.py:41).

3) Populate model registries (OpenAI example)
- Ensure key is loaded into the process environment (source .env).
- Run fetcher (no code changes required):
  - python -m src.providers.openai.get_openai_models
  - Expected: "[openai] loaded <N> models" and JSON persisted at:
    src/providers/openai/openai-models.json

4) Instantiate providers via factory
- Example:
  - from src.providers.base import ProviderFactory, ChatRequest, Message
  - provider = [ProviderFactory.create()](Cogito/src/providers/base/factory.py:47)("openai")
  - req = ChatRequest(
      model=provider.default_model() or "o3-mini",
      messages=[
        Message(role="system", content="You are a concise assistant."),
        Message(role="user", content="Say 'ok' once.")
      ],
      max_tokens=32,
      response_format="text",
    )
  - resp = provider.chat(req)
  - print(resp.text)

5) Listing models
- Example (OpenAI):
  - provider = ProviderFactory.create("openai")
  - snap = [OpenAIProvider.list_models()](Cogito/src/providers/openai/client.py:72)(refresh=False)
  - print(len(snap.models))
- To refresh from API:
  - snap = provider.list_models(refresh=True)  # requires OPENAI_API_KEY

Testing

Location and policy
- All tests live under Cogito/tests/... (no test scripts in src/ per architecture rules).

Unit smoke test (factory + OpenAI adapter)
- File: [tests/providers/test_provider_factory_smoke.py](Cogito/tests/providers/test_provider_factory_smoke.py)
- What it covers:
  - ProviderFactory creates OpenAI adapter
  - Adapter exposes provider_name and default_model
  - list_models(refresh=True) populates JSON when OPENAI_API_KEY is set; otherwise the test is skipped (no-network)
- Run:
  - python -m pytest -q tests/providers/test_provider_factory_smoke.py

Design notes and guardrails
- Non-invasive adapter design:
  - OpenAIProvider chat() wraps existing call_openai_with_retry()
  - No changes to legacy OpenAI client semantics
- JSON I/O resilience (important for offline/dev):
  - [ModelRegistryRepository._read_provider_json()](Cogito/src/providers/base/repositories/model_registry.py:253) gracefully handles empty/whitespace files by returning {} with a warning
  - Fetchers do not write empty registries in offline/cache mode
- Extensibility:
  - Register new providers by adding entries to ProviderFactory._PROVIDERS and implementing a client.py adapter plus get_{provider}_models.py
  - Keep provider-specific SDK imports in provider modules only
- Observability:
  - ProviderMetadata attached to ChatResponse supports audits and test logging

Metadata completeness and limitations
- context_length
  - Policy: Keep null when not provided by the API. Do not fabricate limits.
  - Enrichment flow: [get_openai_models.run()](Cogito/src/providers/openai/get_openai_models.py:49) fetches list and best-effort details (client.models.retrieve) and passes through numeric fields (e.g., input_token_limit/context_window) when present. Normalization in [normalize_items()](Cogito/src/providers/base/get_models_base.py:110) maps these into context_length if and only if explicitly available.
- capabilities
  - Derived from SDK “modalities” when present and from conservative id heuristics in [normalize_items()](Cogito/src/providers/base/get_models_base.py:110) (e.g., mark reasoning/responses_api for o1/o3 families; vision for gpt-4o/omni; embeddings for text-embedding ids; JSON structured output flagged by default).
- updated_at
  - Prefer explicit updated_at from the SDK. If absent, infer ISO date from the model id via [_infer_updated_at_from_id()](Cogito/src/providers/base/get_models_base.py:71). As a last resort, convert a numeric “created” epoch (UTC) to YYYY-MM-DD.
- provenance
  - The snapshot writer [save_provider_models()](Cogito/src/providers/base/get_models_base.py:145) persists fetched_via and metadata.source. The OpenAI fetcher records “api” and “openai_sdk_enriched” to indicate SDK-originated, enriched listings.

Note: If future endpoints expose explicit token limits per model id, the normalization path will automatically populate context_length without policy changes.

Extending to new providers (checklist)
- Create src/providers/{provider}/client.py implementing LLMProvider (+ capabilities)
- Add get_{provider}_models.py fetcher module with run() entrypoint
- Add factory mapping in [ProviderFactory](Cogito/src/providers/base/factory.py:32)
- Provide {provider}-models.json seed or rely on fetcher to create it
- Add unit tests under Cogito/tests/providers/

FAQ

Q: How do I run a minimal end-to-end provider call without wiring the orchestrator?
- Use ProviderFactory + ChatRequest in a small script or a unit test as shown above. No orchestrator changes are required.

Q: What if my JSON registry file is empty and I get errors?
- The repository now tolerates empty/whitespace files and returns an empty snapshot with a warning, but live API refresh is recommended to populate real models:
  python -m src.providers.openai.get_openai_models

Q: Where should tests go?
- Only under Cogito/tests/. Test-like scripts in src/ are not allowed by architecture rules.

References
- Factory: [ProviderFactory.create()](Cogito/src/providers/base/factory.py:47)
- OpenAI Adapter: [OpenAIProvider](Cogito/src/providers/openai/client.py:44)
- DTOs: [ChatRequest](Cogito/src/providers/base/models.py:104), [ChatResponse](Cogito/src/providers/base/models.py:139), [Message](Cogito/src/providers/base/models.py:52)
- Registry: [ModelRegistryRepository](Cogito/src/providers/base/repositories/model_registry.py:64)
- Key resolution: [KeysRepository.get_api_key()](Cogito/src/providers/base/repositories/keys.py:62)
- OpenAI fetcher: [get_openai_models.run()](Cogito/src/providers/openai/get_openai_models.py:49)