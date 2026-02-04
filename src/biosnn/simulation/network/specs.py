"""Network specification DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from biosnn.contracts.learning import ILearningRule
from biosnn.contracts.modulators import IModulatorField, ModulatorKind
from biosnn.contracts.neurons import INeuronModel
from biosnn.contracts.synapses import ISynapseModel, SynapseTopology
from biosnn.contracts.tensor import Tensor


@dataclass(frozen=True, slots=True)
class PopulationSpec:
    name: str
    model: INeuronModel
    n: int
    positions: Tensor | None = None
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    name: str
    synapse: ISynapseModel
    topology: SynapseTopology
    pre: str
    post: str
    learning: ILearningRule | None = None
    learn_every: int = 1
    sparse_learning: bool = False
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ModulatorSpec:
    name: str
    field: IModulatorField
    kinds: tuple[ModulatorKind, ...]
    meta: Mapping[str, Any] | None = None


__all__ = ["PopulationSpec", "ProjectionSpec", "ModulatorSpec"]
