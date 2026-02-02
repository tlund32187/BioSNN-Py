"""Factory/registry contracts.

These are lightweight abstractions intended to make it easy to add, swap, or deprecate
implementations without changing callers (Open/Closed Principle).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRegistry[T](Protocol):
    """Named registry mapping string keys to constructors."""

    def register(self, key: str, ctor: Callable[..., T]) -> None:
        ...

    def create(self, key: str, **kwargs: Any) -> T:
        ...

    def keys(self) -> list[str]:
        ...

    def mapping(self) -> Mapping[str, Callable[..., T]]:
        ...


class Registry[T](IRegistry[T]):
    """Simple named registry with optional aliasing and deprecations."""

    def __init__(self, *, label: str | None = None) -> None:
        self._label = label or "registry"
        self._ctors: dict[str, Callable[..., T]] = {}
        self._aliases: dict[str, str] = {}
        self._deprecated: dict[str, str | None] = {}

    def register(self, key: str, ctor: Callable[..., T]) -> None:
        if key in self._ctors or key in self._aliases:
            raise KeyError(f"{self._label} already has key '{key}'.")
        self._ctors[key] = ctor

    def register_alias(self, alias: str, target: str, *, deprecated: bool = False) -> None:
        if alias in self._ctors or alias in self._aliases:
            raise KeyError(f"{self._label} already has key '{alias}'.")
        if target not in self._ctors:
            raise KeyError(f"{self._label} has no target '{target}' for alias '{alias}'.")
        self._aliases[alias] = target
        if deprecated:
            self._deprecated[alias] = target

    def deprecate(self, key: str, *, replacement: str | None = None) -> None:
        if key not in self._ctors and key not in self._aliases:
            raise KeyError(f"{self._label} has no key '{key}' to deprecate.")
        self._deprecated[key] = replacement

    def create(self, key: str, **kwargs: Any) -> T:
        resolved = self._aliases.get(key, key)
        if resolved not in self._ctors:
            raise KeyError(f"{self._label} has no key '{key}'.")
        self._warn_if_deprecated(key, resolved)
        return self._ctors[resolved](**kwargs)

    def keys(self) -> list[str]:
        return sorted(set(self._ctors) | set(self._aliases))

    def mapping(self) -> Mapping[str, Callable[..., T]]:
        return dict(self._ctors)

    def _warn_if_deprecated(self, key: str, resolved: str) -> None:
        if key in self._deprecated:
            replacement = self._deprecated[key]
            _warn_deprecated(self._label, key, replacement)
        elif resolved in self._deprecated:
            replacement = self._deprecated[resolved]
            _warn_deprecated(self._label, resolved, replacement)


def _warn_deprecated(label: str, key: str, replacement: str | None) -> None:
    if replacement:
        message = f"{label} key '{key}' is deprecated; use '{replacement}' instead."
    else:
        message = f"{label} key '{key}' is deprecated."
    warnings.warn(message, DeprecationWarning, stacklevel=3)
