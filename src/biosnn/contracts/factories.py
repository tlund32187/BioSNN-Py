"""Factory/registry contracts.

These are lightweight abstractions intended to make it easy to add, swap, or deprecate
implementations without changing callers (Open/Closed Principle).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class IRegistry(Protocol[T]):
    """Named registry mapping string keys to constructors."""

    def register(self, key: str, ctor: Callable[..., T]) -> None:
        ...

    def create(self, key: str, **kwargs: Any) -> T:
        ...

    def keys(self) -> list[str]:
        ...

    def mapping(self) -> Mapping[str, Callable[..., T]]:
        ...
