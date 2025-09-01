"""
Memory store port for agent state (contract only).
"""
from __future__ import annotations
from typing import Protocol, Dict, Optional

class IMemoryStore(Protocol):
    def get(self, key: str) -> Optional[Dict]:
        ...
    def put(self, key: str, value: Dict) -> None:
        ...

__all__ = ["IMemoryStore"]