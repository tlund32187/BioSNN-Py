from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype
from biosnn.learning import ThreeFactorHebbianParams, ThreeFactorHebbianRule
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

from .test_models import DeterministicLIFModel, SpikeInputModel

ExternalDriveFn = Callable[[float, int, str, StepContext], Mapping[Compartment, Tensor]]
ReleasesFn = Callable[[float, int, StepContext], Sequence[ModulatorRelease]]


def build_prop_chain_engine(
    *,
    compiled_mode: bool,
    device: str = "cpu",
    dtype: object | None = None,
) -> tuple[TorchNetworkEngine, tuple[str, ...], ExternalDriveFn]:
    torch = require_torch()
    n = 2
    dtype = dtype or torch.float64
    positions = _zeros_positions(n, dtype=dtype, device=device)

    input_model = SpikeInputModel()
    relay_model = DeterministicLIFModel(leak=0.0, gain=1.0, thresh=0.5, reset=0.0)
    hidden_model = DeterministicLIFModel(leak=0.0, gain=1.0, thresh=0.5, reset=0.0)
    out_model = DeterministicLIFModel(leak=0.0, gain=1.0, thresh=0.5, reset=0.0)

    populations = [
        PopulationSpec(name="Input", model=input_model, n=n, positions=positions),
        PopulationSpec(name="Relay", model=relay_model, n=n, positions=positions.clone()),
        PopulationSpec(name="Hidden", model=hidden_model, n=n, positions=positions.clone()),
        PopulationSpec(name="Out", model=out_model, n=n, positions=positions.clone()),
    ]

    proj_input_relay = ProjectionSpec(
        name="Input_to_Relay",
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0)),
        topology=_identity_topology(n, weight_value=1.0, delay_steps=0, dtype=dtype),
        pre="Input",
        post="Relay",
    )
    proj_relay_hidden = ProjectionSpec(
        name="Relay_to_Hidden",
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0)),
        topology=_identity_topology(n, weight_value=1.0, delay_steps=0, dtype=dtype),
        pre="Relay",
        post="Hidden",
    )
    proj_hidden_out = ProjectionSpec(
        name="Hidden_to_Out",
        synapse=DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0)),
        topology=_identity_topology(n, weight_value=1.0, delay_steps=0, dtype=dtype),
        pre="Hidden",
        post="Out",
    )

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext):
        _ = t
        if pop_name != "Input":
            return {}
        device, ctx_dtype = resolve_device_dtype(ctx)
        drive = _pulse(step, n, device=device, dtype=ctx_dtype, start=5, end=7, value=1.0)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    engine = TorchNetworkEngine(
        populations=populations,
        projections=[proj_input_relay, proj_relay_hidden, proj_hidden_out],
        external_drive_fn=external_drive_fn,
        fast_mode=True,
        compiled_mode=compiled_mode,
    )
    _reset_engine(engine, device=device, dtype=_dtype_to_str(dtype))

    tap_keys = (
        "pop/Input/spikes",
        "pop/Relay/spikes",
        "pop/Relay/v_soma_raw",
        "pop/Relay/last_drive_dendrite",
        "pop/Hidden/spikes",
        "pop/Hidden/v_soma_raw",
        "pop/Hidden/last_drive_dendrite",
        "pop/Out/spikes",
        "pop/Out/v_soma_raw",
        "pop/Out/last_drive_dendrite",
        f"proj/{proj_input_relay.name}/weights",
    )

    return engine, tap_keys, external_drive_fn


def build_delay_impulse_engine(
    delay_steps: int,
    *,
    compiled_mode: bool,
    device: str = "cpu",
    dtype: object | None = None,
) -> tuple[TorchNetworkEngine, str, ExternalDriveFn]:
    torch = require_torch()
    dtype = dtype or torch.float64
    positions = _zeros_positions(1, dtype=dtype, device=device)

    populations = [
        PopulationSpec(name="Input", model=SpikeInputModel(), n=1, positions=positions),
        PopulationSpec(
            name="Post",
            model=DeterministicLIFModel(leak=0.0, gain=1.0, thresh=0.5, reset=0.0),
            n=1,
            positions=positions.clone(),
        ),
    ]

    synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=1.0))
    topology = _identity_topology(1, weight_value=1.0, delay_steps=delay_steps, dtype=dtype)
    proj_name = "Input_to_Post"
    projection = ProjectionSpec(
        name=proj_name,
        synapse=synapse,
        topology=topology,
        pre="Input",
        post="Post",
    )

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext):
        _ = t
        if pop_name != "Input":
            return {}
        device, ctx_dtype = resolve_device_dtype(ctx)
        drive = _pulse(step, 1, device=device, dtype=ctx_dtype, start=5, end=5, value=1.0)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    engine = TorchNetworkEngine(
        populations=populations,
        projections=[projection],
        external_drive_fn=external_drive_fn,
        fast_mode=True,
        compiled_mode=compiled_mode,
    )
    _reset_engine(engine, device=device, dtype=_dtype_to_str(dtype))

    return engine, proj_name, external_drive_fn


def build_learning_gate_engine(
    *,
    dopamine_on: bool,
    compiled_mode: bool,
    device: str = "cpu",
    dtype: object | None = None,
) -> tuple[TorchNetworkEngine, str, ReleasesFn, ExternalDriveFn]:
    torch = require_torch()
    dtype = dtype or torch.float64
    positions = _zeros_positions(1, dtype=dtype, device=device)

    populations = [
        PopulationSpec(name="Pre", model=SpikeInputModel(), n=1, positions=positions),
        PopulationSpec(name="Post", model=SpikeInputModel(), n=1, positions=positions.clone()),
    ]

    synapse = DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.0))
    topology = _identity_topology(1, weight_value=0.0, delay_steps=2, dtype=dtype)
    proj_name = "Pre_to_Post"
    learning = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=0.1))
    projection = ProjectionSpec(
        name=proj_name,
        synapse=synapse,
        topology=topology,
        pre="Pre",
        post="Post",
        learning=learning,
        sparse_learning=True,
    )

    field = GlobalScalarField(
        kinds=(ModulatorKind.DOPAMINE,),
        params=GlobalScalarParams(decay_tau=10.0),
    )
    mod_spec = ModulatorSpec(name="dopamine", field=field, kinds=(ModulatorKind.DOPAMINE,))

    def releases_fn(t: float, step: int, ctx: StepContext):
        _ = (t, ctx)
        if not dopamine_on or step != 5:
            return []
        positions_rel = _zeros_positions(1, dtype=dtype, device=device)
        amount = torch.tensor([1.0], dtype=dtype, device=device)
        return [
            ModulatorRelease(
                kind=ModulatorKind.DOPAMINE,
                positions=positions_rel,
                amount=amount,
            )
        ]

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext):
        _ = t
        if pop_name not in {"Pre", "Post"}:
            return {}
        device, ctx_dtype = resolve_device_dtype(ctx)
        drive = _pulse(step, 1, device=device, dtype=ctx_dtype, start=5, end=5, value=1.0)
        if drive is None:
            return {}
        return {Compartment.SOMA: drive}

    engine = TorchNetworkEngine(
        populations=populations,
        projections=[projection],
        modulators=[mod_spec],
        external_drive_fn=external_drive_fn,
        releases_fn=releases_fn,
        fast_mode=True,
        compiled_mode=compiled_mode,
    )
    _reset_engine(engine, device=device, dtype=_dtype_to_str(dtype))

    return engine, proj_name, releases_fn, external_drive_fn


def _identity_topology(
    n: int,
    *,
    weight_value: float,
    delay_steps: int,
    dtype: object,
) -> SynapseTopology:
    torch = require_torch()
    device = torch.device("cpu")
    pre_idx = torch.arange(n, device=device, dtype=torch.long)
    post_idx = torch.arange(n, device=device, dtype=torch.long)
    weights = torch.full((n,), float(weight_value), device=device, dtype=dtype)
    delays = torch.full((n,), int(delay_steps), device=device, dtype=torch.int32)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delays,
        weights=weights,
        target_compartment=Compartment.DENDRITE,
    )


def _zeros_positions(n: int, *, dtype: object, device: str) -> Tensor:
    torch = require_torch()
    return torch.zeros((n, 3), device=device, dtype=dtype)


def _pulse(
    step: int,
    n: int,
    *,
    device: object,
    dtype: object,
    start: int,
    end: int,
    value: float,
) -> Tensor | None:
    if step < start or step > end:
        return None
    torch = require_torch()
    return torch.full((n,), float(value), device=device, dtype=dtype)


def _reset_engine(engine: TorchNetworkEngine, *, device: str, dtype: str) -> None:
    engine.reset(config=SimulationConfig(dt=1e-3, device=device, dtype=dtype))


def _dtype_to_str(dtype: object) -> str:
    if isinstance(dtype, str):
        return dtype
    return str(dtype)
