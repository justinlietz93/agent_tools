"""
Tool port (contract only) used by BL and infra adapters.
"""
from __future__ import annotations
from typing import Protocol, Dict, Any, Union

class ITool(Protocol):
    name: str
    description: str
    input_schema: Dict[str, Any]

    def run(self, input: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        ...

__all__ = ["ITool"]