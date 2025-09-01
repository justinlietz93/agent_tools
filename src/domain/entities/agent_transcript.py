"""
POCO DTO for a multi-turn agent transcript. No framework dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from src.domain.entities.agent_turn import AgentTurn

@dataclass
class AgentTranscript:
    turns: List[AgentTurn] = field(default_factory=list)
    final_response: str = ""
    used_tools: List[str] = field(default_factory=list)
    cancelled: bool = False

__all__ = ["AgentTranscript"]