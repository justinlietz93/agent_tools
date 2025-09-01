"""
Tracing/observability port for parse and invocation traces.
"""
from __future__ import annotations
from typing import Protocol, Dict, Any, Optional

class ITracer(Protocol):
    def record(self, event: str, data: Dict[str, Any]) -> None:
        ...
    def get_last(self) -> Optional[Dict[str, Any]]:
        ...

__all__ = ["ITracer"]