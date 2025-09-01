"""
Shared tool DTOs for catalogs and invocation results.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

@dataclass
class ToolDescriptor:
    name: str
    description: str
    raw_schema: Dict[str, Any]

@dataclass
class ToolInvocationResult:
    ok: bool
    value: Optional[Union[str, Dict[str, Any]]]
    error: Optional[str]
    tool_name: str
    signature_used: str  # "run(input)" | "run(call_id, **kwargs)"

__all__ = ["ToolDescriptor", "ToolInvocationResult"]