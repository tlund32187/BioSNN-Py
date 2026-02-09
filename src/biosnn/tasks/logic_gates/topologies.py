"""Canonical sparse topology builders for logic-gate tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)

from .datasets import LogicGate, coerce_gate
from .encoding import INPUT_NEURON_INDICES, OUTPUT_NEURON_INDICES


@dataclass(frozen=True, slots=True)
class LogicGateTopology:
    populations: tuple[PopulationSpec, ...]
    projections: tuple[ProjectionSpec, ...]
    projection_topologies: Mapping[str, SynapseTopology]


@dataclass(frozen=True, slots=True)
class LogicGateHandles:
    input_population: str
    output_population: str
    hidden_populations: tuple[str, ...]
    input_neuron_indices: Mapping[str, int]
    output_neuron_indices: Mapping[str, int]
    hidden_excit_indices: tuple[int, ...]
    hidden_inhib_indices: tuple[int, ...]
    projection_names: tuple[str, ...]


def build_logic_gate_ff(
    gate: LogicGate | str,
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Build canonical single-hidden-layer sparse FF topology for logic gates."""

    torch = require_torch()
    gate_enum = coerce_gate(gate)
    resolved_device = _resolve_device(torch, device)
    dtype = torch.float32
    generator = _make_generator(torch, resolved_device, seed)

    in_pos = _line_positions(n=4, x=0.0, device=resolved_device, dtype=dtype)
    hidden_pos = _line_positions(n=16, x=1.0, device=resolved_device, dtype=dtype)
    out_pos = _line_positions(n=2, x=2.0, device=resolved_device, dtype=dtype)

    populations = (
        PopulationSpec(name="In", model=GLIFModel(), n=4, positions=in_pos, meta={"role": "input"}),
        PopulationSpec(name="Hidden", model=GLIFModel(), n=16, positions=hidden_pos, meta={"role": "hidden"}),
        PopulationSpec(name="Out", model=GLIFModel(), n=2, positions=out_pos, meta={"role": "output"}),
    )

    in_to_hidden_topo = _build_fixed_fanin_topology(
        n_pre=4,
        n_post=16,
        fan_in=2,
        pre_positions=in_pos,
        post_positions=hidden_pos,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.03,
        delay_min=0,
        delay_max=3,
    )
    hidden_to_out_topo = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=6,
        pre_positions=hidden_pos,
        post_positions=out_pos,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=0,
        delay_max=1,
    )
    hidden_to_out_topo = _apply_hidden_ei_signs(hidden_to_out_topo, inhib_start=8)

    projections: list[ProjectionSpec] = [
        ProjectionSpec(
            name="In->Hidden",
            synapse=_make_sparse_synapse(),
            topology=in_to_hidden_topo,
            pre="In",
            post="Hidden",
        ),
        ProjectionSpec(
            name="Hidden->Out",
            synapse=_make_sparse_synapse(),
            topology=hidden_to_out_topo,
            pre="Hidden",
            post="Out",
        ),
    ]

    # Weak skip helps AND/OR convergence but is disabled for XOR-style gates.
    if gate_enum not in {LogicGate.XOR, LogicGate.XNOR}:
        in_to_out_skip = _build_fixed_fanin_topology(
            n_pre=4,
            n_post=2,
            fan_in=2,
            pre_positions=in_pos,
            post_positions=out_pos,
            generator=generator,
            device=resolved_device,
            dtype=dtype,
            weight_scale=0.01,
            delay_min=0,
            delay_max=0,
        )
        projections.append(
            ProjectionSpec(
                name="In->OutSkip",
                synapse=_make_sparse_synapse(),
                topology=in_to_out_skip,
                pre="In",
                post="Out",
            )
        )

    # Hidden inhibitory stabilization: Hidden[inhib] -> Hidden[excit].
    inhib_to_excit = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=16,
        fan_in=2,
        pre_positions=hidden_pos,
        post_positions=hidden_pos,
        pre_pool=_range_tensor(torch, start=8, end=16, device=resolved_device),
        post_pool=_range_tensor(torch, start=0, end=8, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.02,
        delay_min=1,
        delay_max=3,
    )
    inhib_to_excit = _force_negative_weights(inhib_to_excit)
    projections.append(
        ProjectionSpec(
            name="HiddenInhib->HiddenExcit",
            synapse=_make_sparse_synapse(),
            topology=inhib_to_excit,
            pre="Hidden",
            post="Hidden",
        )
    )

    return _bundle_and_engine(
        populations=populations,
        projections=tuple(projections),
        input_population="In",
        hidden_populations=("Hidden",),
        output_population="Out",
    )


def build_logic_gate_xor_variant(
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Backwards-compatible alias for the canonical XOR topology."""

    return build_logic_gate_xor(device=device, seed=seed)


def build_logic_gate_xor(
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Build canonical XOR topology with two hidden layers."""

    torch = require_torch()
    resolved_device = _resolve_device(torch, device)
    dtype = torch.float32
    generator = _make_generator(torch, resolved_device, seed)

    in_pos = _line_positions(n=4, x=0.0, device=resolved_device, dtype=dtype)
    h1_pos = _line_positions(n=16, x=1.0, device=resolved_device, dtype=dtype)
    h2_pos = _line_positions(n=16, x=2.0, device=resolved_device, dtype=dtype)
    out_pos = _line_positions(n=2, x=3.0, device=resolved_device, dtype=dtype)

    populations = (
        PopulationSpec(name="In", model=GLIFModel(), n=4, positions=in_pos, meta={"role": "input"}),
        PopulationSpec(name="Hidden0", model=GLIFModel(), n=16, positions=h1_pos, meta={"role": "hidden"}),
        PopulationSpec(name="Hidden1", model=GLIFModel(), n=16, positions=h2_pos, meta={"role": "hidden"}),
        PopulationSpec(name="Out", model=GLIFModel(), n=2, positions=out_pos, meta={"role": "output"}),
    )

    in_to_h1 = _build_fixed_fanin_topology(
        n_pre=4,
        n_post=16,
        fan_in=2,
        pre_positions=in_pos,
        post_positions=h1_pos,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.03,
        delay_min=0,
        delay_max=3,
    )
    h1_to_h2 = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=16,
        fan_in=4,
        pre_positions=h1_pos,
        post_positions=h2_pos,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.03,
        delay_min=1,
        delay_max=3,
    )
    h2_to_out = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=6,
        pre_positions=h2_pos,
        post_positions=out_pos,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=0,
        delay_max=2,
    )
    h2_to_out = _apply_hidden_ei_signs(h2_to_out, inhib_start=8)

    projections: tuple[ProjectionSpec, ...] = (
        ProjectionSpec(
            name="In->Hidden0",
            synapse=_make_sparse_synapse(),
            topology=in_to_h1,
            pre="In",
            post="Hidden0",
        ),
        ProjectionSpec(
            name="Hidden0->Hidden1",
            synapse=_make_sparse_synapse(),
            topology=h1_to_h2,
            pre="Hidden0",
            post="Hidden1",
        ),
        ProjectionSpec(
            name="Hidden1->Out",
            synapse=_make_sparse_synapse(),
            topology=h2_to_out,
            pre="Hidden1",
            post="Out",
        ),
    )

    # Optional inhibitory stabilization within each hidden layer.
    h0_inhib_to_excit = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=16,
        fan_in=2,
        pre_positions=h1_pos,
        post_positions=h1_pos,
        pre_pool=_range_tensor(torch, start=8, end=16, device=resolved_device),
        post_pool=_range_tensor(torch, start=0, end=8, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.02,
        delay_min=1,
        delay_max=3,
    )
    h0_inhib_to_excit = _force_negative_weights(h0_inhib_to_excit)
    h1_inhib_to_excit = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=16,
        fan_in=2,
        pre_positions=h2_pos,
        post_positions=h2_pos,
        pre_pool=_range_tensor(torch, start=8, end=16, device=resolved_device),
        post_pool=_range_tensor(torch, start=0, end=8, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.02,
        delay_min=1,
        delay_max=3,
    )
    h1_inhib_to_excit = _force_negative_weights(h1_inhib_to_excit)

    projections = projections + (
        ProjectionSpec(
            name="Hidden0Inhib->Hidden0Excit",
            synapse=_make_sparse_synapse(),
            topology=h0_inhib_to_excit,
            pre="Hidden0",
            post="Hidden0",
        ),
        ProjectionSpec(
            name="Hidden1Inhib->Hidden1Excit",
            synapse=_make_sparse_synapse(),
            topology=h1_inhib_to_excit,
            pre="Hidden1",
            post="Hidden1",
        ),
    )

    return _bundle_and_engine(
        populations=populations,
        projections=projections,
        input_population="In",
        hidden_populations=("Hidden0", "Hidden1"),
        output_population="Out",
    )


def _bundle_and_engine(
    *,
    populations: tuple[PopulationSpec, ...],
    projections: tuple[ProjectionSpec, ...],
    input_population: str,
    hidden_populations: tuple[str, ...],
    output_population: str,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    projection_topologies = {proj.name: proj.topology for proj in projections}
    topology = LogicGateTopology(
        populations=populations,
        projections=projections,
        projection_topologies=projection_topologies,
    )
    handles = LogicGateHandles(
        input_population=input_population,
        output_population=output_population,
        hidden_populations=hidden_populations,
        input_neuron_indices=dict(INPUT_NEURON_INDICES),
        output_neuron_indices=dict(OUTPUT_NEURON_INDICES),
        hidden_excit_indices=tuple(range(0, 8)),
        hidden_inhib_indices=tuple(range(8, 16)),
        projection_names=tuple(projection_topologies.keys()),
    )

    engine = TorchNetworkEngine(
        populations=populations,
        projections=projections,
        fast_mode=True,
        compiled_mode=True,
        learning_use_scratch=True,
    )
    return engine, topology, handles


def _make_sparse_synapse() -> DelayedSparseMatmulSynapse:
    return DelayedSparseMatmulSynapse(DelayedSparseMatmulParams(init_weight=0.0))


def _build_fixed_fanin_topology(
    *,
    n_pre: int,
    n_post: int,
    fan_in: int,
    pre_positions: Any,
    post_positions: Any,
    generator: Any,
    device: str,
    dtype: Any,
    weight_scale: float,
    delay_min: int = 0,
    delay_max: int = 0,
    pre_pool: Any | None = None,
    post_pool: Any | None = None,
) -> SynapseTopology:
    torch = require_torch()
    device_obj = torch.device(device)
    pre_candidates = (
        pre_pool.to(device=device_obj, dtype=torch.long)
        if pre_pool is not None
        else torch.arange(n_pre, device=device_obj, dtype=torch.long)
    )
    post_candidates = (
        post_pool.to(device=device_obj, dtype=torch.long)
        if post_pool is not None
        else torch.arange(n_post, device=device_obj, dtype=torch.long)
    )
    if pre_candidates.numel() == 0 or post_candidates.numel() == 0 or fan_in <= 0:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return SynapseTopology(
            pre_idx=empty,
            post_idx=empty,
            delay_steps=empty.to(dtype=torch.int32),
            weights=empty.to(dtype=dtype),
            pre_pos=pre_positions,
            post_pos=post_positions,
            target_compartment=Compartment.DENDRITE,
        )

    fan_in = min(int(fan_in), int(pre_candidates.numel()))
    pre_chunks: list[Any] = []
    post_chunks: list[Any] = []
    for post_idx in post_candidates:
        perm = torch.randperm(pre_candidates.numel(), generator=generator, device=device_obj)
        chosen = pre_candidates.index_select(0, perm[:fan_in])
        pre_chunks.append(chosen)
        post_chunks.append(torch.full((fan_in,), int(post_idx.item()), device=device_obj, dtype=torch.long))

    pre_idx = torch.cat(pre_chunks)
    post_idx = torch.cat(post_chunks)
    edge_count = int(pre_idx.numel())

    weights = (torch.rand((edge_count,), generator=generator, device=device_obj, dtype=dtype) * 2.0 - 1.0)
    weights.mul_(float(weight_scale))

    if delay_max < delay_min:
        raise ValueError("delay_max must be >= delay_min")
    if delay_max == delay_min:
        delay_steps = torch.full((edge_count,), int(delay_min), device=device_obj, dtype=torch.int32)
    else:
        delay_steps = torch.randint(
            low=int(delay_min),
            high=int(delay_max) + 1,
            size=(edge_count,),
            generator=generator,
            device=device_obj,
            dtype=torch.int32,
        )

    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        weights=weights,
        pre_pos=pre_positions,
        post_pos=post_positions,
        target_compartment=Compartment.DENDRITE,
    )


def _apply_hidden_ei_signs(topology: SynapseTopology, *, inhib_start: int) -> SynapseTopology:
    weights = topology.weights
    if weights is None or weights.numel() == 0:
        return topology
    signed = weights.clone()
    inhib_mask = topology.pre_idx >= int(inhib_start)
    if bool(inhib_mask.any()):
        signed[inhib_mask] = -signed[inhib_mask].abs()
    excit_mask = ~inhib_mask
    if bool(excit_mask.any()):
        signed[excit_mask] = signed[excit_mask].abs()
    return _replace_topology(topology, weights=signed)


def _force_negative_weights(topology: SynapseTopology) -> SynapseTopology:
    weights = topology.weights
    if weights is None:
        return topology
    return _replace_topology(topology, weights=-weights.abs())


def _line_positions(*, n: int, x: float, device: str, dtype: Any) -> Any:
    torch = require_torch()
    device_obj = torch.device(device)
    y = torch.linspace(0.0, 1.0, max(1, n), device=device_obj, dtype=dtype)
    pos = torch.zeros((n, 3), device=device_obj, dtype=dtype)
    pos[:, 0] = float(x)
    pos[:, 1] = y[:n]
    return pos


def _replace_topology(topology: SynapseTopology, **kwargs: Any) -> SynapseTopology:
    return SynapseTopology(
        pre_idx=kwargs.get("pre_idx", topology.pre_idx),
        post_idx=kwargs.get("post_idx", topology.post_idx),
        delay_steps=kwargs.get("delay_steps", topology.delay_steps),
        edge_dist=kwargs.get("edge_dist", topology.edge_dist),
        target_compartment=kwargs.get("target_compartment", topology.target_compartment),
        target_compartments=kwargs.get("target_compartments", topology.target_compartments),
        receptor=kwargs.get("receptor", topology.receptor),
        receptor_kinds=kwargs.get("receptor_kinds", topology.receptor_kinds),
        weights=kwargs.get("weights", topology.weights),
        pre_pos=kwargs.get("pre_pos", topology.pre_pos),
        post_pos=kwargs.get("post_pos", topology.post_pos),
        myelin=kwargs.get("myelin", topology.myelin),
        meta=kwargs.get("meta", topology.meta),
    )


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _make_generator(torch: Any, device: str, seed: int | None) -> Any:
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj)
    generator.manual_seed(int(seed) if seed is not None else 0)
    return generator


def _range_tensor(torch: Any, *, start: int, end: int, device: str) -> Any:
    return torch.arange(start, end, device=torch.device(device), dtype=torch.long)


__all__ = [
    "LogicGateHandles",
    "LogicGateTopology",
    "build_logic_gate_ff",
    "build_logic_gate_xor",
    "build_logic_gate_xor_variant",
]
