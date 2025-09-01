"""
Unit of Work contract for transaction boundaries.
"""
from __future__ import annotations
from typing import Protocol

class UnitOfWork(Protocol):
    def begin(self) -> None:
        ...
    def commit(self) -> None:
        ...
    def rollback(self) -> None:
        ...

__all__ = ["UnitOfWork"]