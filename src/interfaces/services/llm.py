"""
LLM service port. BL depends on this; infra implements.
"""
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING, Callable, Optional

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
    def stream_and_collect(self, user_input: str, on_delta: Callable[[str], None]) -> str:
        """
        Optional streaming interface. Implementations should:
        - Call on_delta(token_or_text_chunk) as new content arrives
        - Return the same final formatted string as execute()
        """
        ...

__all__ = ["ILLM"]