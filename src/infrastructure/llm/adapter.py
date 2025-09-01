"""
ILLM adapter for wrapper executors (infra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interfaces.agents.tool import ITool

from src.interfaces.services.llm import ILLM


class LLMWrapperAdapter(ILLM):
    """
    Thin adapter implementing ILLM by delegating to existing wrapper instances.
    """

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    @property
    def model(self) -> str:
        return getattr(self.wrapper, "model", "")
    
    @model.setter
    def model(self, value: str) -> None:
        try:
            setattr(self.wrapper, "model", value)
        except Exception:
            # Fallback attach (defensive; underlying wrappers already expose 'model')
            self.wrapper.model = value  # type: ignore[attr-defined]

    @property
    def base_url(self) -> str:
        return getattr(self.wrapper, "base_url", "")
    
    @base_url.setter
    def base_url(self, value: str) -> None:
        try:
            setattr(self.wrapper, "base_url", value)
        except Exception:
            self.wrapper.base_url = value  # type: ignore[attr-defined]
    
    def execute(self, user_input: str) -> str:
        return self.wrapper.execute(user_input)

    def register_tool(self, tool: "ITool") -> None:
        self.wrapper.register_tool(tool)