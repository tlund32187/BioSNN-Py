"""Neuron model contracts.

Design goals:
- **Population-first**: stepping operates on N neurons at once (GPU-friendly).
- **No spike history buffers in the model**: spike trains over time are a monitor concern.
- **Compartment-aware**: soma/dendrite/AIS/axon are addressed explicitly.
- **SOLID-ish**: implementations depend on contracts, not vice-versa.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.tensor import Tensor


class Compartment(StrEnum):
    SOMA = "soma"
    DENDRITE = "dendrite"
    AIS = "ais"
    AXON = "axon"


@dataclass(frozen=True, slots=True)
class StepContext:
    """Cross-cutting runtime context passed into step functions.

    Keep this flexible; it is the "escape hatch" for backend specifics (device, dtype,
    determinism toggles, debug flags) without forcing every interface to grow forever.
    """

    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    is_training: bool = True
    extras: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NeuronInputs:
    """Inputs to a neuron model for one step.

    drive: per-compartment external drive (e.g., injected current).
    modulators: per-modulator scalar values at neuron locations (shape [N]).
    """

    drive: Mapping[Compartment, Tensor]
    modulators: Mapping[ModulatorKind, Tensor] | None = None
    meta: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class NeuronStepResult:
    """Outputs of a neuron model for one step."""

    spikes: Tensor  # shape [N], bool or {0,1} float
    membrane: Mapping[Compartment, Tensor] | None = None
    extras: Mapping[str, Tensor] | None = None


@runtime_checkable
class INeuronModel(Protocol):
    """Population-level neuron model interface."""

    name: str
    compartments: frozenset[Compartment]

    def init_state(self, n: int, *, ctx: StepContext) -> Any:
        """Create model state for n neurons."""
        ...

    def reset_state(self, state: Any, *, ctx: StepContext, indices: Tensor | None = None) -> Any:
        """Reset all or a subset of neuron state."""
        ...

    def step(
        self,
        state: Any,
        inputs: NeuronInputs,
        *,
        dt: float,
        t: float,
        ctx: StepContext,
    ) -> tuple[Any, NeuronStepResult]:
        """Advance the neuron population by one timestep."""
        ...

    def state_tensors(self, state: Any) -> Mapping[str, Tensor]:
        """Expose tensors for monitoring/debugging (voltages, traces, etc.)."""
        ...


# ---------------------------------------------------------------------
# Neuron model specifications (DTOs). These are NOT exported from biosnn.api yet.
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GLIFParams:
    """Generalized LIF parameters.

    This is a pragmatic, simulation-oriented GLIF:
    - leaky integration towards v_rest
    - refractory lockout after spikes
    - optional dynamic threshold (theta) with spike-triggered increments and decay

    Units: assume SI seconds/volts/amps where applicable, but keep the system consistent.
    """

    v_rest: float = -0.065
    v_reset: float = -0.070
    v_thresh0: float = -0.050
    tau_m: float = 0.020
    r_m: float = 10e6  # ohms (used to map current -> voltage contribution)
    refrac_period: float = 0.002

    # "Spike width" as a held-high indicator (for motor eligibility / pulse shaping).
    # The neuron model returns spikes per step; spike_hold_time keeps spikes==1 for a
    # short duration after threshold crossing. Set to 0 for single-step spikes.
    spike_hold_time: float = 0.001

    # Dynamic threshold (optional)
    enable_threshold_adaptation: bool = True
    theta_tau: float = 0.050
    theta_increment: float = 0.005
    theta_min: float = 0.0
    theta_max: float = 0.050


@dataclass(slots=True)
class GLIFState:
    """GLIF state tensors (all shape [N])."""

    v_soma: Tensor
    refrac_left: Tensor  # seconds remaining (float tensor)
    spike_hold_left: Tensor  # seconds remaining (float tensor)
    theta: Tensor  # dynamic threshold offset (>=0)


@dataclass(frozen=True, slots=True)
class AdEx2CompParams:
    """Adaptive Exponential Integrate-and-Fire (AdEx), 2-compartment (soma+dend).

    This DTO describes the parameters; an implementation will define the discrete
    update scheme (Euler, RK, etc.) and the exact spike/reset logic.
    """

    # Soma
    c_s: float = 200e-12
    g_l_s: float = 10e-9
    e_l_s: float = -0.070

    # Dendrite
    c_d: float = 200e-12
    g_l_d: float = 10e-9
    e_l_d: float = -0.070

    # Coupling conductance between soma and dendrite
    g_c: float = 5e-9

    # AdEx nonlinearity (applied at soma)
    v_t: float = -0.050
    delta_t: float = 0.002
    v_reset: float = -0.065
    v_spike: float = 0.020  # spike detection level (implementation-defined)

    # Adaptation variable w (soma)
    a: float = 2e-9
    b: float = 0.05e-9
    tau_w: float = 0.200

    refrac_period: float = 0.002
    spike_hold_time: float = 0.001


@dataclass(slots=True)
class AdEx2CompState:
    """AdEx 2-compartment state tensors (all shape [N])."""

    v_soma: Tensor
    v_dend: Tensor
    w: Tensor
    refrac_left: Tensor
    spike_hold_left: Tensor
