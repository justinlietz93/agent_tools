### Overview
Design a generic full-stack app template focusing on architectural strategies:
modular monolith with microservices decoupling
layered clean architecture (presentation, application, domain, infrastructure).
Bottom-up: Build from primitives (data structures, algorithms).
Top-down: Define high-level abstractions (interfaces, contracts, user stories, "what are we building" and how will it work?) first.
Ensure scalability, maintainability via loose coupling, high cohesion.


### Project Map Suggestions for agent_tools
agentic-tools/
├─ pyproject.toml
├─ README.md
├─ src/
│  ├─ ui/                       # presentation (clients only)
│  │  ├─ web/
│  │  │  ├─ adapters/           # calls api via interfaces DTOs
│  │  │  └─ state/
│  │  └─ cli/
│  │     └─ commands/
│  ├─ api/                      # presentation (server)
│  │  ├─ http/
│  │  │  ├─ controllers/
│  │  │  ├─ routers/
│  │  │  ├─ dto/
│  │  │  └─ middleware/
│  │  └─ di/                    # wire interfaces→impl at the edge
│  ├─ business_logic/           # application layer only
│  │  ├─ agents/
│  │  │  ├─ orchestrators/      # multi-agent planners
│  │  │  ├─ planners/           # task decomposition
│  │  │  ├─ tools/              # tool use protocols
│  │  │  ├─ memory/             # episodic/semantic policies (by iface)
│  │  │  └─ policies/           # safety, routing, retries
│  │  ├─ services/              # use-case services
│  │  ├─ use_cases/             # CQRS commands/queries
│  │  └─ validators/
│  ├─ domain/                   # pure models
│  │  ├─ entities/
│  │  ├─ value_objects/
│  │  └─ events/
│  ├─ interfaces/               # ports (contracts only)
│  │  ├─ repositories/          # e.g., IUserRepo, ITaskRepo
│  │  ├─ services/              # ILLM, IMessageBus, ICache, IVectorIndex
│  │  ├─ agents/                # IAgentRuntime, ITool, IMemoryStore
│  │  └─ unit_of_work.py
│  ├─ abstractions/             # shared types/base classes
│  │  ├─ dto/
│  │  ├─ errors/
│  │  ├─ result.py
│  │  └─ types.py
│  ├─ persistence/              # data plumbing (no business rules)
│  │  ├─ orm/
│  │  │  ├─ models/
│  │  │  ├─ mappers/
│  │  │  └─ migrations/
│  │  ├─ vector/
│  │  ├─ graph/
│  │  └─ connection/            # sessions, pools, UoW impl
│  ├─ repositories/             # adapters implementing repository ports
│  │  ├─ sql/
│  │  ├─ nosql/
│  │  ├─ vector/
│  │  └─ memory/
│  ├─ infrastructure/           # other adapters
│  │  ├─ llm/
│  │  ├─ message_bus/
│  │  ├─ cache/
│  │  ├─ http_clients/
│  │  └─ telemetry/
│  ├─ shared/
│  │  ├─ logging/
│  │  ├─ security/
│  │  └─ settings/
│  ├─ config/
│  │  ├─ settings.py
│  │  └─ env/
│  └─ tests/
│     ├─ unit/
│     ├─ integration/
│     └─ e2e/
└─ docs/
   └─ ADRs/




### Hybrid-Clean Architecture Implementation

This section documents the Hybrid-Clean Architecture approach, combining modular monolith design with Clean Architecture principles. This serves as a generic template for implementing scalable, maintainable applications across projects.

#### Core Principles
- **Modular Monolith**: Single deployable unit composed of multiple projects/modules for separation of concerns, enabling microservices-like decoupling without full microservices overhead.
- **Clean Architecture Layers**: Strict layering with dependency inversion to ensure testability, maintainability, and framework independence.
- **Dependency Rule**: Dependencies flow inward only; outer layers depend on inner layers via abstractions (interfaces).
- **File Size Limit**: No source code file shall exceed 500 lines of code (LOC) to maintain readability and facilitate refactoring.

#### Layer Structure

1. **Presentation Layer** (Outer)
   - **Projects**: API/Web projects, UI/Mobile clients, Console apps
   - **Responsibilities**: HTTP/REST endpoints, real-time hubs (e.g., SignalR), UI rendering, user interaction
   - **Dependencies**: Only on Business Logic interfaces and Shared utilities
   - **Rules**:
     - Controllers/handlers are thin; delegate business logic to services/managers
     - Use DTOs for data transfer (defined in Shared projects)
     - No direct database access or business rules

2. **Application/Business Logic Layer**
   - **Projects**: Core business logic projects and their interface abstractions
   - **Responsibilities**: Domain rules, business workflows, validations, service/manager classes
   - **Dependencies**: On Domain models, Repository interfaces, external service abstractions
   - **Rules**:
     - Services/managers implement interfaces from abstraction projects
     - Use repository pattern for data access
     - Contain all business rules and validations
     - Framework-agnostic (no direct ORM/framework references)

3. **Domain Layer**
   - **Projects**: Domain model projects
   - **Responsibilities**: Plain Old CLR Objects (POCOs), domain entities, value objects
   - **Dependencies**: None (pure domain models)
   - **Rules**:
     - No business logic (anemic models)
     - Framework-independent
     - Used across all layers

4. **Infrastructure/Persistence Layer** (Inner)
   - **Projects**: Data access projects, ORM implementations
   - **Responsibilities**: ORM DbContext, entity mappings, repository implementations, migrations
   - **Dependencies**: On Domain models for entities
   - **Rules**:
     - Implement repository interfaces from Business Logic abstractions
     - Handle data persistence and ORM details
     - No business logic

5. **Shared and Abstractions**
   - **Projects**: Common utilities, cross-cutting concerns, external service abstractions
   - **Responsibilities**: Common DTOs, utilities, external API interfaces (e.g., AI, payment)
   - **Dependencies**: Minimal; used across layers
   - **Rules**:
     - Define contracts for cross-cutting concerns
     - Abstractions enable pluggable implementations

#### Dependency Flow
```
[Presentation] → [Business Logic Services] → [Repository Interfaces] ← [Repository Implementations]
     ↓              ↓                              ↓
[Shared DTOs]   [Domain Models]                [ORM Framework]
```

- Arrows indicate allowed dependencies
- Interfaces act as contracts between layers
- DI container wires implementations at runtime

#### AI/External Service Integration Points
- Service abstractions in Business Logic layer for external dependencies
- Separate projects for AI/external services with abstraction layers
- Abstractions allow swapping implementations without touching core logic

#### Rules for AI Agents
1. **Maintain Layer Separation**: Outer layers depend on inner layers only through abstractions (interfaces), never on concrete implementations. Inner layers must not reference or depend on outer layers in any way.
2. **Use Interfaces**: All cross-layer communication via interfaces defined in abstraction projects
3. **Enforce 500 LOC Limit**: Break large files into smaller, focused classes
4. **Dependency Injection**: Register implementations in DI container, inject interfaces
5. **Repository Pattern**: Data access only through repository interfaces
6. **Testability**: Design for unit testing by depending on abstractions
7. **Framework Independence**: Keep domain models free of framework-specific code
8. **Modular Boundaries**: Changes in one module should not require changes in others

#### Example Implementation
- Controller injects `IService` (interface)
- `Service` implements `IService`, injects `IRepository`
- `Repository` implements `IRepository`, uses ORM
- All layers communicate via domain models

This architecture ensures applications remain maintainable, testable, and evolvable while supporting complex features like real-time communication and AI integration.

### AI Agent Development Guidelines

**Critical Rules for AI Agents:**

1. **File Size Enforcement**: Never create or modify files exceeding 500 lines of code. Break large classes into smaller, focused components.

2. **Dependency Direction**: Always maintain inward dependency flow. Outer layers (Presentation) depend on inner layers (Business Logic) only through interfaces. Inner layers never reference outer layers.

3. **Interface-First Design**: Define interfaces in abstraction projects before implementing concrete classes. All cross-layer communication must use interfaces.

4. **Layer Isolation**:
   - Presentation layer: Only HTTP handling, DTOs, and interface calls
   - Business Logic layer: Pure business rules, no framework references
   - Domain layer: POCO models only, no dependencies
   - Infrastructure layer: Only data access implementations

5. **Repository Pattern**: All data access must go through repository interfaces, never direct ORM calls from business logic.

6. **Dependency Injection**: Use constructor injection for all dependencies. Register implementations in DI container at startup.

7. **Framework Independence**: Keep domain models free of any framework-specific attributes or references.

8. **Modular Boundaries**: Each project/module should have a single responsibility. Changes in one module should not require changes in others.

**Code Review Checklist for AI Agents:**
- [ ] File size ≤ 500 LOC
- [ ] No outer-to-inner dependencies
- [ ] All public methods have interface contracts
- [ ] Business logic contains no UI/database code
- [ ] Domain models are framework-agnostic
- [ ] Dependencies injected via interfaces
- [ ] Repository pattern used for data access

### Architectural Concepts
- **Layered Architecture**: Separate concerns: UI layer (presentation), Business logic (application/domain), Data access (infrastructure), Cross-cutting (logging, security).
- **Patterns**: MVC for controllers; CQRS for read/write separation; Event-driven for real-time; Repository for DB abstraction.
- **Modularity**: Use dependency inversion; interfaces for decoupling. Modules as independent units with explicit dependencies.
- **Scalability Strategies**: Horizontal scaling; stateless services; load balancing. Vertical via optimization.
- **Fault Tolerance**: Circuit breakers; retries; graceful degradation.

### UI/UX Strategies
- **Responsive Design**: Fluid grids, flexible images, media queries. Progressive enhancement.
- **UX Principles**: User-centered; accessibility (semantic structure); minimalism for intuitiveness.
- **State Management**: Centralized store; immutable updates to prevent side effects.

### Performance Strategies (Speed & Power)
- **Speed**: Asynchronous processing; caching (in-memory, distributed); lazy/eager loading; compression.
- **Power Efficiency**: Throttle I/O; batch operations; profile hotspots (CPU, memory).
- **Optimization Techniques**: Algorithmic efficiency (O(n) bounds); indexing; pagination.

### Memory Optimization & Safety
- **Strategies**: Garbage collection tuning; object pooling; weak references. Avoid leaks via resource cleanup (RAII pattern).
- **Cleanup**: Use try-finally for resources; monitor heap usage; manual deallocation where applicable.
- **Pseudocode** (Safe Resource Handling):
```
function processData(data):
    resource = acquireResource()
    try:
        // Process
        result = compute(data, resource)
        return result
    finally:
        releaseResource(resource)  // Guarantee cleanup
```

### Logging Strategies
- **Levels & Structure**: Hierarchical (debug/info/warn/error); structured formats (JSON) for querying.
- **Centralization**: Aggregate logs; correlation IDs for tracing requests.
- **Rotation**: Time/size-based; alerting on anomalies.

### Security Strategies
- **Principles**: Least privilege; defense in depth; input validation everywhere.
- **Auth/Access**: Token-based (stateless); role-based access control (RBAC); encryption in transit/rest.
- **Vulnerabilities Mitigation**: Sanitize inputs; secure headers; rate limiting; auditing.

### Networking / API Strategies
- **Design**: RESTful principles (stateless, cacheable); GraphQL for flexible queries; versioning.
- **Optimization**: Keep-alive connections; payload minimization; error handling (standard codes).
- **Reliability**: Idempotency; timeouts; backoff retries.

### Controller / Backend Logic Strategies
- **Separation**: Thin controllers (orchestrate); fat services (logic). Use middleware for cross-concerns.
- **Error Handling**: Centralized handler; custom exceptions; logging integration.
- **Pseudocode** (Controller):
```
controller getItem(id):
    try:
        item = service.fetchItem(id)
        return response(200, item)
    catch NotFoundError:
        return response(404, "Not found")
    catch Error:
        log(error)
        return response(500, "Internal error")
```

### Data Activities & ACID Strategies
- **Transactions**: Atomic operations; two-phase commit for distributed.
- **Consistency Models**: Strong vs. eventual; use locking/optimistic concurrency.
- **Data Integrity**: Constraints (PK/FK); validation at boundaries.

### Database Design & Optimizations
- **When to Use**:
  | Type        | Use Case                          | Optimizations                  |
  |-------------|-----------------------------------|--------------------------------|
  | Relational | Structured data, joins, ACID     | Indexes, normalization (3NF), partitioning |
  | Hierarchical | Nested/JSON docs, flexible schema| Denormalization, embedding     |
  | Vector     | Embeddings, similarity search    | Approximate nearest neighbors  |
  | Graph      | Relationships, traversals        | Indexing edges, query planning |
- **Strategies**: Sharding for scale; replication for availability; query optimization (EXPLAIN plans).

### Source Control Strategies
- **Branching**: Feature branches; release/hotfix. Semantic versioning.
- **Collaboration**: Pull requests; code reviews; hooks for linting.

### CI/CD Strategies
- **Pipeline**: Build/test/deploy stages; parallel jobs. Blue-green deployments for zero downtime.
- **Automation**: Infrastructure as code; monitoring integration.

### Engineering Approach: Bottom-Up & Top-Down
- **Top-Down**: Start with ideal system model (e.g., UML diagrams for entities/flows); define APIs/contracts.
- **Bottom-Up**: Implement low-level primitives (e.g., custom data structures); integrate upwards, validating against high-level.
- **Iteration**: Prototype; refactor; discard mismatches. Measure impact via metrics (latency, throughput).

Apply these strategies agnostically; adapt to context for optimal template.


### Testing Strategies
Adopt comprehensive, automated testing pyramid: Heavy unit tests (fast, isolated), moderate integration (components interaction), light end-to-end (full flow). Use TDD/BDD for development. Coverage goal: 80%+. Tools-agnostic: Mock dependencies; parallel execution; mutation testing for robustness.

- **Unit Testing**: Isolate functions/modules. Test pure logic, edge cases, errors. Strategy: Arrange-Act-Assert; parameterize for variations.
- **Integration Testing**: Verify module interactions (e.g., API-DB). Use in-memory mocks; test contracts/interfaces.
- **End-to-End (E2E)**: Simulate user flows. Browser automation; API smoke tests.
- **Performance Testing**: Load/stress (throughput, latency); benchmark baselines.
- **Security Testing**: Static analysis (SAST); dynamic (DAST); penetration simulations.
- **Accessibility Testing**: Automated audits (WCAG); manual reviews.
- **Chaos Engineering**: Inject failures (e.g., network delays) to test resilience.

Integration in CI/CD: Run on every commit; fail-fast. Flaky test quarantine.

### Validation Domains & Best Combination
Validation ensures correctness across layers. Best combo: Layered defense—client-side (immediate feedback), server-side (authoritative), DB-level (integrity). Combine preventive (static types, schemas) with runtime checks. Use fail-early principle.

| Domain          | Description & Strategies                          | Best Practices & Combo Integration |
|-----------------|---------------------------------------------------|------------------------------------|
| Input Validation | Sanitize/validate user inputs (forms, APIs). Prevent injections, overflows. | Client (regex, schemas) + Server (validators). Combo: With security (OWASP rules); tie to unit tests. |
| Data Validation | Ensure data integrity (formats, ranges, relations). ACID enforcement. | Schema validation + Constraints (DB). Combo: With integration tests; optimistic locking for concurrency. |
| Business Logic Validation | Check rules/constraints (e.g., auth, workflows). | Domain-driven invariants; assertions in services. Combo: BDD scenarios + E2E; audit logs for post-validation. |
| UI/UX Validation | Verify rendering, responsiveness, accessibility. | Snapshot testing; visual regression. Combo: With performance (load times); user session replays. |
| Security Validation | Auth, encryption, access controls. | Token expiry checks; role validations. Combo: Pen tests + Runtime monitoring; integrate with logging. |
| Performance Validation | Resource usage, scalability thresholds. | Metrics assertions (e.g., <100ms response). Combo: With chaos; continuous profiling in prod. |

Overall Strategy: Orthogonal validation—cross-cut domains via aspects (e.g., AOP for logging validations). Automate 90%; manual for exploratory. Measure effectiveness via defect escape rate.

