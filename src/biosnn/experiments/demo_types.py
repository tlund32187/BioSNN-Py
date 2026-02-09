"""Shared demo specification types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from biosnn.contracts.homeostasis import IHomeostasisRule
from biosnn.contracts.modulators import ModulatorRelease
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec


class ExternalDriveFn(Protocol):
    def __call__(
        self, t: float, step: int, pop_name: str, ctx: StepContext
    ) -> Mapping[Compartment, Any]:
        ...


class ReleasesFn(Protocol):
    def __call__(self, t: float, step: int, ctx: StepContext) -> Sequence[ModulatorRelease]:
        ...


@dataclass(frozen=True, slots=True)
class DemoModelSpec:
    populations: Sequence[PopulationSpec]
    projections: Sequence[ProjectionSpec]
    modulators: Sequence[ModulatorSpec] = ()
    homeostasis: IHomeostasisRule | None = None
    external_drive_fn: ExternalDriveFn | None = None
    releases_fn: ReleasesFn | None = None
    include_neuron_topology: bool = True


class DemoRuntimeConfig(Protocol):
    out_dir: Path
    mode: Literal["dashboard", "fast"]
    steps: int
    dt: float
    seed: int | None
    device: str
    max_ring_mib: float | None
    profile: bool
    profile_steps: int
    allow_cuda_monitor_sync: bool | None
    parallel_compile: Literal["auto", "on", "off"]
    parallel_compile_workers: int | None
    parallel_compile_torch_threads: int


__all__ = ["DemoModelSpec", "DemoRuntimeConfig", "ExternalDriveFn", "ReleasesFn"]
