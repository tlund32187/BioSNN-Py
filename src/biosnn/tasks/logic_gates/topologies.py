"""Canonical sparse topology builders for logic-gate tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp
from typing import Any, Literal, cast

from biosnn.biophysics.models.adex_3c import AdEx3CompModel
from biosnn.biophysics.models.lif_3c import LIF3CompModel
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import ReceptorKind, SynapseTopology
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)
from biosnn.synapses.receptors import (
    ReceptorProfile,
    profile_exc_ampa_nmda,
    profile_inh_gabaa_gabab,
)

from .configs import AdvancedPostVoltageSource, AdvancedSynapseConfig, LogicGateNeuronModel
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
    neuron_model: LogicGateNeuronModel = "adex_3c",
    advanced_synapse: AdvancedSynapseConfig | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Build canonical single-hidden-layer sparse FF topology for logic gates."""

    torch = require_torch()
    gate_enum = coerce_gate(gate)
    resolved_device = _resolve_device(torch, device)
    dtype = torch.float32
    generator = _make_generator(torch, resolved_device, seed)
    syn_cfg = _resolve_synapse_cfg(run_spec)
    adv_cfg = _resolve_advanced_synapse_cfg(run_spec, advanced_synapse=advanced_synapse)
    excit_profile = profile_exc_ampa_nmda()
    inhib_profile = _inhibitory_profile_for_mode(syn_cfg["receptor_mode"])
    delay_steps = syn_cfg["delay_steps"]

    in_pos = _line_positions(n=4, x=0.0, device=resolved_device, dtype=dtype)
    hidden_pos = _line_positions(n=16, x=1.0, device=resolved_device, dtype=dtype)
    out_pos = _line_positions(n=2, x=2.0, device=resolved_device, dtype=dtype)

    populations = (
        PopulationSpec(
            name="In",
            model=_make_neuron_model(neuron_model),
            n=4,
            positions=in_pos,
            meta={"role": "input"},
        ),
        PopulationSpec(
            name="Hidden",
            model=_make_neuron_model(neuron_model),
            n=16,
            positions=hidden_pos,
            meta={"role": "hidden"},
        ),
        PopulationSpec(
            name="Out",
            model=_make_neuron_model(neuron_model),
            n=2,
            positions=out_pos,
            meta={"role": "output"},
        ),
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
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.DENDRITE,
    )
    hidden_excit_to_out_topo = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=3,
        pre_positions=hidden_pos,
        post_positions=out_pos,
        pre_pool=_range_tensor(torch, start=0, end=8, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=1),
        target_compartment=Compartment.DENDRITE,
    )
    hidden_inhib_to_out_topo = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=3,
        pre_positions=hidden_pos,
        post_positions=out_pos,
        pre_pool=_range_tensor(torch, start=8, end=16, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=1),
        target_compartment=Compartment.SOMA,
    )

    projections: list[ProjectionSpec] = [
        ProjectionSpec(
            name="In->Hidden",
            synapse=_make_sparse_synapse(
                excit_profile,
                adv_cfg,
                target_compartment=Compartment.DENDRITE,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=in_to_hidden_topo,
            pre="In",
            post="Hidden",
        ),
        ProjectionSpec(
            name="HiddenExcit->Out",
            synapse=_make_sparse_synapse(
                excit_profile,
                adv_cfg,
                target_compartment=Compartment.DENDRITE,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=hidden_excit_to_out_topo,
            pre="Hidden",
            post="Out",
        ),
        ProjectionSpec(
            name="HiddenInhib->Out",
            synapse=_make_sparse_synapse(
                inhib_profile,
                adv_cfg,
                target_compartment=Compartment.SOMA,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=hidden_inhib_to_out_topo,
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
            delay_min=_delay_or_default(delay_steps, default=0),
            delay_max=_delay_or_default(delay_steps, default=0),
            target_compartment=Compartment.DENDRITE,
        )
        projections.append(
            ProjectionSpec(
                name="In->OutSkip",
                synapse=_make_sparse_synapse(
                    excit_profile,
                    adv_cfg,
                    target_compartment=Compartment.DENDRITE,
                    backend=syn_cfg["backend"],
                    ring_strategy=syn_cfg["ring_strategy"],
                    fused_layout=syn_cfg["fused_layout"],
                    ring_dtype=syn_cfg["ring_dtype"],
                    store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
                ),
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
        delay_min=_delay_or_default(delay_steps, default=1),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.SOMA,
    )
    projections.append(
        ProjectionSpec(
            name="HiddenInhib->HiddenExcit",
            synapse=_make_sparse_synapse(
                inhib_profile,
                adv_cfg,
                target_compartment=Compartment.SOMA,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
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
    neuron_model: LogicGateNeuronModel = "adex_3c",
    advanced_synapse: AdvancedSynapseConfig | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Backwards-compatible alias for the canonical XOR topology."""

    return build_logic_gate_xor(
        device=device,
        seed=seed,
        neuron_model=neuron_model,
        advanced_synapse=advanced_synapse,
        run_spec=run_spec,
    )


def build_logic_gate_xor(
    *,
    device: str = "cpu",
    seed: int | None = None,
    neuron_model: LogicGateNeuronModel = "adex_3c",
    advanced_synapse: AdvancedSynapseConfig | None = None,
    run_spec: Mapping[str, Any] | None = None,
) -> tuple[TorchNetworkEngine, LogicGateTopology, LogicGateHandles]:
    """Build canonical XOR topology with two hidden layers."""

    torch = require_torch()
    resolved_device = _resolve_device(torch, device)
    dtype = torch.float32
    generator = _make_generator(torch, resolved_device, seed)
    syn_cfg = _resolve_synapse_cfg(run_spec)
    adv_cfg = _resolve_advanced_synapse_cfg(run_spec, advanced_synapse=advanced_synapse)
    excit_profile = profile_exc_ampa_nmda()
    inhib_profile = _inhibitory_profile_for_mode(syn_cfg["receptor_mode"])
    delay_steps = syn_cfg["delay_steps"]

    in_pos = _line_positions(n=4, x=0.0, device=resolved_device, dtype=dtype)
    h1_pos = _line_positions(n=16, x=1.0, device=resolved_device, dtype=dtype)
    h2_pos = _line_positions(n=16, x=2.0, device=resolved_device, dtype=dtype)
    out_pos = _line_positions(n=2, x=3.0, device=resolved_device, dtype=dtype)

    populations = (
        PopulationSpec(
            name="In",
            model=_make_neuron_model(neuron_model),
            n=4,
            positions=in_pos,
            meta={"role": "input"},
        ),
        PopulationSpec(
            name="Hidden0",
            model=_make_neuron_model(neuron_model),
            n=16,
            positions=h1_pos,
            meta={"role": "hidden"},
        ),
        PopulationSpec(
            name="Hidden1",
            model=_make_neuron_model(neuron_model),
            n=16,
            positions=h2_pos,
            meta={"role": "hidden"},
        ),
        PopulationSpec(
            name="Out",
            model=_make_neuron_model(neuron_model),
            n=2,
            positions=out_pos,
            meta={"role": "output"},
        ),
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
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.DENDRITE,
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
        delay_min=_delay_or_default(delay_steps, default=1),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.DENDRITE,
    )
    h2_excit_to_out = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=3,
        pre_positions=h2_pos,
        post_positions=out_pos,
        pre_pool=_range_tensor(torch, start=0, end=8, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=2),
        target_compartment=Compartment.DENDRITE,
    )
    h2_inhib_to_out = _build_fixed_fanin_topology(
        n_pre=16,
        n_post=2,
        fan_in=3,
        pre_positions=h2_pos,
        post_positions=out_pos,
        pre_pool=_range_tensor(torch, start=8, end=16, device=resolved_device),
        generator=generator,
        device=resolved_device,
        dtype=dtype,
        weight_scale=0.04,
        delay_min=_delay_or_default(delay_steps, default=0),
        delay_max=_delay_or_default(delay_steps, default=2),
        target_compartment=Compartment.SOMA,
    )

    projections: tuple[ProjectionSpec, ...] = (
        ProjectionSpec(
            name="In->Hidden0",
            synapse=_make_sparse_synapse(
                excit_profile,
                adv_cfg,
                target_compartment=Compartment.DENDRITE,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=in_to_h1,
            pre="In",
            post="Hidden0",
        ),
        ProjectionSpec(
            name="Hidden0->Hidden1",
            synapse=_make_sparse_synapse(
                excit_profile,
                adv_cfg,
                target_compartment=Compartment.DENDRITE,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=h1_to_h2,
            pre="Hidden0",
            post="Hidden1",
        ),
        ProjectionSpec(
            name="Hidden1Excit->Out",
            synapse=_make_sparse_synapse(
                excit_profile,
                adv_cfg,
                target_compartment=Compartment.DENDRITE,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=h2_excit_to_out,
            pre="Hidden1",
            post="Out",
        ),
        ProjectionSpec(
            name="Hidden1Inhib->Out",
            synapse=_make_sparse_synapse(
                inhib_profile,
                adv_cfg,
                target_compartment=Compartment.SOMA,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=h2_inhib_to_out,
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
        delay_min=_delay_or_default(delay_steps, default=1),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.SOMA,
    )
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
        delay_min=_delay_or_default(delay_steps, default=1),
        delay_max=_delay_or_default(delay_steps, default=3),
        target_compartment=Compartment.SOMA,
    )

    projections = projections + (
        ProjectionSpec(
            name="Hidden0Inhib->Hidden0Excit",
            synapse=_make_sparse_synapse(
                inhib_profile,
                adv_cfg,
                target_compartment=Compartment.SOMA,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
            topology=h0_inhib_to_excit,
            pre="Hidden0",
            post="Hidden0",
        ),
        ProjectionSpec(
            name="Hidden1Inhib->Hidden1Excit",
            synapse=_make_sparse_synapse(
                inhib_profile,
                adv_cfg,
                target_compartment=Compartment.SOMA,
                backend=syn_cfg["backend"],
                ring_strategy=syn_cfg["ring_strategy"],
                fused_layout=syn_cfg["fused_layout"],
                ring_dtype=syn_cfg["ring_dtype"],
                store_sparse_by_delay=syn_cfg["store_sparse_by_delay"],
            ),
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


def _make_neuron_model(neuron_model: LogicGateNeuronModel) -> Any:
    if neuron_model == "lif_3c":
        return LIF3CompModel()
    if neuron_model == "adex_3c":
        return AdEx3CompModel()
    raise ValueError(f"Unsupported logic-gate neuron model: {neuron_model}")


def _make_sparse_synapse(
    profile: ReceptorProfile,
    adv: AdvancedSynapseConfig,
    *,
    target_compartment: Compartment,
    backend: str,
    ring_strategy: str,
    fused_layout: str,
    ring_dtype: str | None,
    store_sparse_by_delay: bool | None,
) -> DelayedSparseMatmulSynapse:
    adv_effective = _advanced_enabled(adv)
    post_voltage_compartment = _resolve_post_voltage_compartment(
        source=adv.post_voltage_source,
        target_compartment=target_compartment,
    )
    nmda_mg_slope = _resolve_nmda_mg_slope(adv.nmda_slope_v)
    nmda_mg_scale = _resolve_nmda_mg_scale(
        mg_mM=adv.nmda_mg_mM,
        nmda_mg_slope=nmda_mg_slope,
        v_half_v=adv.nmda_v_half_v,
    )
    conductance_mode = bool(adv_effective and (adv.conductance_mode or adv.nmda_voltage_block))
    stp_enabled = bool(adv_effective and adv.stp_enabled)
    resolved_backend = "event_driven" if stp_enabled else backend
    resolved_ring_strategy = ring_strategy
    if stp_enabled and resolved_ring_strategy not in {"event_bucketed", "dense"}:
        resolved_ring_strategy = "event_bucketed"

    return DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=0.0,
            backend=cast(Literal["spmm_fused", "event_driven"], resolved_backend),
            ring_strategy=cast(Literal["dense", "event_bucketed"], resolved_ring_strategy),
            fused_layout=cast(Literal["auto", "coo", "csr"], fused_layout),
            ring_dtype=ring_dtype,
            store_sparse_by_delay=store_sparse_by_delay,
            receptor_profile=profile,
            conductance_mode=conductance_mode,
            reversal_potential=_map_reversal_potentials(adv.reversal_potential_v),
            nmda_voltage_block=bool(adv_effective and adv.nmda_voltage_block and conductance_mode),
            nmda_mg_concentration=float(adv.nmda_mg_mM),
            nmda_mg_slope=nmda_mg_slope,
            nmda_mg_scale=nmda_mg_scale,
            post_voltage_compartment=post_voltage_compartment,
            stp_enabled=stp_enabled,
            stp_u=float(adv.stp_u),
            stp_tau_rec=float(adv.stp_tau_rec_s),
            stp_tau_facil=float(adv.stp_tau_facil_s),
            stp_state_dtype=adv.stp_state_dtype,
        )
    )


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
    target_compartment: Compartment = Compartment.DENDRITE,
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
            target_compartment=target_compartment,
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

    weights = torch.rand((edge_count,), generator=generator, device=device_obj, dtype=dtype)
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
        target_compartment=target_compartment,
    )


def _resolve_synapse_cfg(run_spec: Mapping[str, Any] | None) -> dict[str, Any]:
    spec = _as_mapping(run_spec)
    syn = _as_mapping(spec.get("synapse"))
    backend = _coerce_choice(
        _first_non_none(syn.get("backend"), spec.get("synapse_backend")),
        allowed={"spmm_fused", "event_driven"},
        default="spmm_fused",
    )
    ring_strategy = _coerce_choice(
        _first_non_none(syn.get("ring_strategy"), spec.get("ring_strategy")),
        allowed={"dense", "event_bucketed"},
        default="dense",
    )
    fused_layout = _coerce_choice(
        _first_non_none(syn.get("fused_layout"), spec.get("fused_layout")),
        allowed={"auto", "coo", "csr"},
        default="auto",
    )
    ring_dtype_raw = _coerce_choice(
        _first_non_none(syn.get("ring_dtype"), spec.get("ring_dtype")),
        allowed={"none", "float32", "float16", "bfloat16"},
        default="none",
    )
    receptor_mode = _coerce_choice(
        _first_non_none(syn.get("receptor_mode"), spec.get("receptor_mode")),
        allowed={"exc_only", "ei_ampa_nmda_gabaa", "ei_ampa_nmda_gabaa_gabab"},
        default="exc_only",
    )
    store_sparse_by_delay = _coerce_optional_bool(
        _first_non_none(syn.get("store_sparse_by_delay"), spec.get("store_sparse_by_delay"))
    )
    delay_steps = _coerce_optional_nonnegative_int(
        _first_non_none(syn.get("delay_steps"), spec.get("delay_steps"))
    )
    return {
        "backend": backend,
        "ring_strategy": ring_strategy,
        "fused_layout": fused_layout,
        "ring_dtype": None if ring_dtype_raw == "none" else ring_dtype_raw,
        "receptor_mode": receptor_mode,
        "store_sparse_by_delay": store_sparse_by_delay,
        "delay_steps": delay_steps,
    }


def _resolve_advanced_synapse_cfg(
    run_spec: Mapping[str, Any] | None,
    *,
    advanced_synapse: AdvancedSynapseConfig | None,
) -> AdvancedSynapseConfig:
    if advanced_synapse is not None:
        base = AdvancedSynapseConfig(
            enabled=advanced_synapse.enabled,
            conductance_mode=advanced_synapse.conductance_mode,
            reversal_potential_v=dict(advanced_synapse.reversal_potential_v),
            bio_synapse=advanced_synapse.bio_synapse,
            bio_nmda_block=advanced_synapse.bio_nmda_block,
            bio_stp=advanced_synapse.bio_stp,
            nmda_voltage_block=advanced_synapse.nmda_voltage_block,
            nmda_mg_mM=advanced_synapse.nmda_mg_mM,
            nmda_v_half_v=advanced_synapse.nmda_v_half_v,
            nmda_slope_v=advanced_synapse.nmda_slope_v,
            stp_enabled=advanced_synapse.stp_enabled,
            stp_u=advanced_synapse.stp_u,
            stp_tau_rec_s=advanced_synapse.stp_tau_rec_s,
            stp_tau_facil_s=advanced_synapse.stp_tau_facil_s,
            stp_state_dtype=advanced_synapse.stp_state_dtype,
            post_voltage_source=advanced_synapse.post_voltage_source,
        )
    else:
        base = AdvancedSynapseConfig()

    spec = _as_mapping(run_spec)
    syn = _as_mapping(spec.get("synapse"))
    raw_adv = _first_non_none(
        spec.get("advanced_synapse"),
        syn.get("advanced_synapse"),
        syn.get("advanced"),
    )
    adv = _as_mapping(raw_adv)
    if not adv:
        return base

    merged_reversal = dict(base.reversal_potential_v)
    rev_raw = _as_mapping(adv.get("reversal_potential_v"))
    for key, value in rev_raw.items():
        merged_reversal[str(key).strip().lower()] = _coerce_float(value, merged_reversal.get(str(key), 0.0))

    post_voltage_source_raw = str(
        adv.get("post_voltage_source", base.post_voltage_source)
    ).strip().lower()
    if post_voltage_source_raw not in {"auto", "soma", "dendrite"}:
        post_voltage_source_raw = base.post_voltage_source
    post_voltage_source = cast(AdvancedPostVoltageSource, post_voltage_source_raw)

    return AdvancedSynapseConfig(
        enabled=bool(adv.get("enabled", base.enabled)),
        conductance_mode=bool(adv.get("conductance_mode", base.conductance_mode)),
        reversal_potential_v=merged_reversal,
        bio_synapse=bool(_first_non_none(adv.get("bio_synapse"), syn.get("bio_synapse"), spec.get("bio_synapse"), base.bio_synapse)),
        bio_nmda_block=bool(_first_non_none(adv.get("bio_nmda_block"), syn.get("bio_nmda_block"), spec.get("bio_nmda_block"), base.bio_nmda_block)),
        bio_stp=bool(_first_non_none(adv.get("bio_stp"), syn.get("bio_stp"), spec.get("bio_stp"), base.bio_stp)),
        nmda_voltage_block=bool(adv.get("nmda_voltage_block", base.nmda_voltage_block)),
        nmda_mg_mM=_coerce_float(adv.get("nmda_mg_mM"), base.nmda_mg_mM),
        nmda_v_half_v=_coerce_float(adv.get("nmda_v_half_v"), base.nmda_v_half_v),
        nmda_slope_v=_coerce_float(adv.get("nmda_slope_v"), base.nmda_slope_v),
        stp_enabled=bool(adv.get("stp_enabled", base.stp_enabled)),
        stp_u=_coerce_float(adv.get("stp_u"), base.stp_u),
        stp_tau_rec_s=max(0.0, _coerce_float(adv.get("stp_tau_rec_s"), base.stp_tau_rec_s)),
        stp_tau_facil_s=max(0.0, _coerce_float(adv.get("stp_tau_facil_s"), base.stp_tau_facil_s)),
        stp_state_dtype=_coerce_optional_str(adv.get("stp_state_dtype"), base.stp_state_dtype),
        post_voltage_source=post_voltage_source,
    )


def _advanced_enabled(adv: AdvancedSynapseConfig) -> bool:
    return bool(
        adv.enabled
        or adv.conductance_mode
        or adv.nmda_voltage_block
        or adv.stp_enabled
    )


def _resolve_post_voltage_compartment(
    *,
    source: str,
    target_compartment: Compartment,
) -> Compartment:
    source_norm = str(source).strip().lower()
    if source_norm == "soma":
        return Compartment.SOMA
    if source_norm == "dendrite":
        return Compartment.DENDRITE
    if target_compartment == Compartment.SOMA:
        return Compartment.SOMA
    return Compartment.DENDRITE


def _map_reversal_potentials(values: Mapping[str, float]) -> dict[ReceptorKind, float]:
    mapped: dict[ReceptorKind, float] = {}
    key_map: dict[str, ReceptorKind] = {
        "ampa": ReceptorKind.AMPA,
        "nmda": ReceptorKind.NMDA,
        "gaba": ReceptorKind.GABA,
        "gabaa": ReceptorKind.GABA_A,
        "gaba_a": ReceptorKind.GABA_A,
        "gabab": ReceptorKind.GABA_B,
        "gaba_b": ReceptorKind.GABA_B,
    }
    for key, value in values.items():
        receptor = key_map.get(str(key).strip().lower())
        if receptor is None:
            continue
        mapped[receptor] = float(value)
    return mapped


def _resolve_nmda_mg_slope(nmda_slope_v: float) -> float:
    slope_v = max(abs(float(nmda_slope_v)), 1e-6)
    return 1.0 / (1000.0 * slope_v)


def _resolve_nmda_mg_scale(*, mg_mM: float, nmda_mg_slope: float, v_half_v: float) -> float:
    mg = max(float(mg_mM), 1e-9)
    # Map the configurable half-activation voltage onto the backend scale constant.
    return max(mg * exp(-float(nmda_mg_slope) * 1000.0 * float(v_half_v)), 1e-9)


def _delay_or_default(value: int | None, *, default: int) -> int:
    if value is None:
        return int(default)
    return int(max(0, value))


def _inhibitory_profile_for_mode(receptor_mode: str) -> ReceptorProfile:
    mode = str(receptor_mode).strip().lower()
    if mode == "ei_ampa_nmda_gabaa_gabab":
        return profile_inh_gabaa_gabab()
    return _profile_inh_gabaa_only()


def _profile_inh_gabaa_only() -> ReceptorProfile:
    return ReceptorProfile(
        kinds=(ReceptorKind.GABA_A,),
        mix={ReceptorKind.GABA_A: 1.0},
        tau={ReceptorKind.GABA_A: 10e-3},
        sign={ReceptorKind.GABA_A: -1.0},
    )


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_choice(value: Any, *, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except Exception:
        return None
    return max(0, out)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_optional_str(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    token = str(value).strip()
    if not token:
        return None
    if token.lower() == "none":
        return None
    return token


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


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
