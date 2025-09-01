# ADR 0002 — Layered Alignment (Ports + DTOs only)

- Intent: Add contracts and DTOs to enforce Presentation→BL→Interfaces←Infra without behavior changes.
- Scope: New Protocols (IAgentRuntime, ITool, IMemoryStore, ILLM, IToolCatalog, IToolInvocationAdapter, ITracer, SettingsRepository, UnitOfWork) and DTOs (AgentTurn, AgentTranscript, ToolDescriptor, ToolInvocationResult).
- No modifications to existing modules. Moves and wiring will occur in Step 2–3 with shims and feature flags.