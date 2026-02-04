"""Synapse model contracts."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from biosnn.contracts.neurons import Compartment
from biosnn.contracts.tensor import Tensor


class ReceptorKind(StrEnum):
    AMPA = "ampa"
    NMDA = "nmda"
    GABA = "gaba"


@dataclass(frozen=True, slots=True)
class SynapseTopology:
    """Static synapse graph topology.

    pre_idx/post_idx are edge-indexed arrays mapping edges to neuron indices.
    delay_steps is optional and can be used by the simulation engine's delay buffer.
    target_compartments/receptor are optional per-edge annotations (int-coded).
    """

    pre_idx: Tensor      # shape [E] (int64)
    post_idx: Tensor     # shape [E] (int64)
    delay_steps: Tensor | None = None  # shape [E] (int32)
    target_compartment: Compartment = Compartment.DENDRITE
    target_compartments: Tensor | None = None  # shape [E] (int64; encoded Compartment ids)
    receptor: Tensor | None = None  # shape [E] (int64; receptor id per edge)
    receptor_kinds: tuple[ReceptorKind, ...] | None = None
    weights: Tensor | None = None  # shape [E] (float)
    pre_pos: Tensor | None = None  # shape [Npre, 3]
    post_pos: Tensor | None = None  # shape [Npost, 3]
    myelin: Tensor | None = None  # shape [E] (float; [0,1] typical)
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SynapseInputs:
    """Inputs to synapse stepping for one dt."""

    pre_spikes: Tensor   # shape [Npre] or [N] (population spikes for this dt)
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SynapseStepResult:
    """Outputs of a synapse model step."""

    post_drive: Mapping[Compartment, Tensor]  # typically current into post neurons
    extras: Mapping[str, Tensor] | None = None


@runtime_checkable
class ISynapseModel(Protocol):
    """Edge-population synapse model."""

    name: str

    def init_state(self, e: int, *, ctx: Any) -> Any:
        ...

    def reset_state(self, state: Any, *, ctx: Any, edge_indices: Tensor | None = None) -> Any:
        ...

    def step(
        self,
        state: Any,
        topology: SynapseTopology,
        inputs: SynapseInputs,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[Any, SynapseStepResult]:
        ...

    def state_tensors(self, state: Any) -> Mapping[str, Tensor]:
        ...


@runtime_checkable
class ISynapseModelInplace(Protocol):
    """Optional in-place synapse stepping API."""

    name: str

    def step_into(
        self,
        state: Any,
        pre_spikes: Tensor,
        out_drive: MutableMapping[Compartment, Tensor],
        t: int,
        **kwargs: Any,
    ) -> None:
        ...
