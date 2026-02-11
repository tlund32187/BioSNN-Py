from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.simulation import SimulationConfig
from biosnn.contracts.synapses import ReceptorKind, SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
)
from biosnn.synapses.receptors import ReceptorProfile
from biosnn.tasks.logic_gates import (
    AdvancedSynapseConfig,
    LogicGate,
    LogicGateRunConfig,
    build_logic_gate_ff,
    run_logic_gate_engine,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_engine_logic_gate_builds_and_steps_with_advanced_synapses() -> None:
    adv = AdvancedSynapseConfig(
        enabled=True,
        conductance_mode=True,
        nmda_voltage_block=True,
        stp_enabled=True,
        post_voltage_source="auto",
    )
    run_spec = {
        "delay_steps": 1,
        "synapse": {
            "backend": "spmm_fused",
            "ring_strategy": "event_bucketed",
            "fused_layout": "auto",
            "ring_dtype": "none",
            "receptor_mode": "ei_ampa_nmda_gabaa_gabab",
        },
    }

    engine, _, handles = build_logic_gate_ff(
        LogicGate.AND,
        device="cpu",
        seed=7,
        advanced_synapse=adv,
        run_spec=run_spec,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=7))

    input_drive = torch.tensor([1.0, 0.0, 1.0, 0.0], device="cpu", dtype=torch.float32)

    def drive_fn(t: float, step: int, pop_name: str, ctx: StepContext):
        _ = (t, step, ctx)
        if pop_name != handles.input_population:
            return {}
        return {Compartment.SOMA: input_drive}

    engine._external_drive_fn = drive_fn  # noqa: SLF001 - integration test probes runtime plumbing.

    for _ in range(8):
        engine.step()

    assert engine.last_projection_drive
    for post_drive in engine.last_projection_drive.values():
        for tensor in post_drive.values():
            assert bool(torch.isfinite(tensor).all())


def test_missing_post_voltage_meta_errors_in_conductance_mode() -> None:
    profile = ReceptorProfile(
        kinds=(ReceptorKind.AMPA,),
        mix={ReceptorKind.AMPA: 1.0},
        tau={ReceptorKind.AMPA: 5e-3},
        sign={ReceptorKind.AMPA: 1.0},
    )
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            init_weight=1.0,
            receptor_profile=profile,
            conductance_mode=True,
            reversal_potential={ReceptorKind.AMPA: 0.0},
        )
    )
    ctx = StepContext(device="cpu", dtype="float32")
    topology = compile_topology(
        _single_edge_topology(weight=1.0),
        device="cpu",
        dtype="float32",
        build_sparse_delay_mats=True,
        build_pre_adjacency=False,
    )
    state = model.init_state(1, ctx=ctx)
    if topology.weights is not None:
        state.weights.copy_(topology.weights)

    with pytest.raises(
        ValueError,
        match="conductance_mode/nmda_voltage_block requires post-membrane voltage meta",
    ):
        model.step(
            state,
            topology,
            SynapseInputs(pre_spikes=torch.tensor([1.0], dtype=torch.float32)),
            dt=1e-3,
            t=0.0,
            ctx=ctx,
        )


def test_stp_forces_event_path_and_pre_adjacency_requirement() -> None:
    model = DelayedSparseMatmulSynapse(
        DelayedSparseMatmulParams(
            backend="spmm_fused",
            ring_strategy="dense",
            stp_enabled=True,
        )
    )
    reqs = model.compilation_requirements()
    assert reqs["needs_pre_adjacency"] is True
    assert reqs["needs_sparse_delay_mats"] is False
    assert reqs["ring_strategy"] in {"dense", "event_bucketed"}

    adv = AdvancedSynapseConfig(
        bio_synapse=True,
        bio_stp=True,
    )
    run_spec = {
        "synapse": {
            "backend": "spmm_fused",
            "ring_strategy": "event_bucketed",
            "fused_layout": "auto",
            "ring_dtype": "none",
            "receptor_mode": "ei_ampa_nmda_gabaa_gabab",
        }
    }
    engine, _, _ = build_logic_gate_ff(
        LogicGate.AND,
        device="cpu",
        seed=13,
        advanced_synapse=adv,
        run_spec=run_spec,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=13))
    for projection in engine._proj_specs:  # noqa: SLF001 - integration guard for compiled topology meta.
        if not isinstance(projection.synapse, DelayedSparseMatmulSynapse):
            continue
        if not projection.synapse.params.stp_enabled:
            continue
        assert projection.topology.meta is not None
        assert projection.topology.meta.get("pre_ptr") is not None
        assert projection.topology.meta.get("edge_idx") is not None


def test_bio_bundle_presets_enable_advanced_components() -> None:
    adv = AdvancedSynapseConfig(
        bio_synapse=True,
        bio_nmda_block=True,
        bio_stp=True,
    )
    assert adv.enabled is True
    assert adv.conductance_mode is True
    assert adv.nmda_voltage_block is True
    assert adv.stp_enabled is True


@pytest.mark.parametrize("gate", [LogicGate.AND, LogicGate.OR, LogicGate.XOR])
def test_engine_logic_gate_artifacts_with_advanced_synapses(tmp_path, gate: LogicGate) -> None:
    cfg = LogicGateRunConfig(
        gate=gate,
        seed=17,
        steps=3,
        sim_steps_per_trial=1,
        device="cpu",
        learning_mode="none",
        out_dir=tmp_path / f"logic_{gate.value}_advanced",
        export_every=1,
        advanced_synapse=AdvancedSynapseConfig(
            bio_synapse=True,
            bio_nmda_block=True,
            bio_stp=True,
        ),
    )
    run_spec = {
        "dtype": "float32",
        "delay_steps": 1,
        "synapse": {
            "backend": "spmm_fused",
            "fused_layout": "auto",
            "ring_strategy": "event_bucketed",
            "ring_dtype": "none",
            "receptor_mode": "ei_ampa_nmda_gabaa_gabab",
        },
        "learning": {
            "enabled": False,
            "rule": "none",
            "lr": 0.0,
        },
    }
    result = run_logic_gate_engine(cfg, run_spec)
    out_dir = result["out_dir"]
    assert (out_dir / "topology.json").exists()
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "eval.csv").exists()
    assert (out_dir / "confusion.csv").exists()


def _single_edge_topology(*, weight: float = 1.0) -> SynapseTopology:
    return SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        delay_steps=torch.tensor([0], dtype=torch.int32),
        weights=torch.tensor([weight], dtype=torch.float32),
        target_compartment=Compartment.SOMA,
    )
