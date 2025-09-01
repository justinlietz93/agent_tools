"""
Agents runtime port (contract only). No implementations here.
Clean Architecture: Presentation → (IAgentRuntime) → BL, infra implements ports.
"""
from __future__ import annotations
from typing import Protocol, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.entities.agent_turn import AgentTurn
    from src.domain.entities.agent_transcript import AgentTranscript

class IAgentRuntime(Protocol):
    def run_once(self, model: str, provider: str, messages: List[Dict]) -> "AgentTurn":
        ...
    def run_until_done(self, model: str, provider: str, messages: List[Dict], max_steps: int = 3) -> "AgentTranscript":
        ...

__all__ = ["IAgentRuntime"]