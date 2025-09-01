"""
POCO DTO for a single agent turn. No framework dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class AgentTurn:
    reasoning: str = ""
    tool_call: Optional[Dict[str, Any]] = None   # {"tool": str, "input_schema": dict}
    tool_result: Optional[str] = None
    response: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

__all__ = ["AgentTurn"]