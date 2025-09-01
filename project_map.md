agentic-ai-app/
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
