"""
LLM service port. BL depends on this; infra implements.
"""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaces.agents.tool import ITool

class ILLM(Protocol):
    @property
    def model(self) -> str:
        ...
    def execute(self, user_input: str) -> str:
        ...
    def register_tool(self, tool: "ITool") -> None:
        ...

__all__ = ["ILLM"]