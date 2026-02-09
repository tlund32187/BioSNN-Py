from __future__ import annotations

import pytest

from biosnn.contracts.simulation import SimulationConfig
from biosnn.tasks.logic_gates import (
    LogicGate,
    build_logic_gate_ff,
    build_logic_gate_xor,
    build_logic_gate_xor_variant,
)

pytestmark = pytest.mark.acceptance

torch = pytest.importorskip("torch")


def test_logic_gate_topology_build_ff_runs_cpu() -> None:
    engine, topology, handles = build_logic_gate_ff(LogicGate.AND, device="cpu", seed=13)
    _assert_sparse_projection_topologies(topology)
    assert handles.input_population == "In"
    assert handles.output_population == "Out"
    assert "bit0_0" in handles.input_neuron_indices
    assert "class_1" in handles.output_neuron_indices

    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=13))
    engine.run(steps=10)


def test_logic_gate_topology_build_xor_variant_runs_cpu() -> None:
    engine, topology, handles = build_logic_gate_xor_variant(device="cpu", seed=17)
    _assert_sparse_projection_topologies(topology)
    assert handles.hidden_populations == ("Hidden0", "Hidden1")

    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=17))
    engine.run(steps=10)


def test_logic_gate_topology_build_xor_runs_cpu() -> None:
    engine, topology, handles = build_logic_gate_xor(device="cpu", seed=21)
    _assert_sparse_projection_topologies(topology)
    assert handles.hidden_populations == ("Hidden0", "Hidden1")

    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=21))
    engine.run(steps=10)


def _assert_sparse_projection_topologies(topology) -> None:
    pop_sizes = {pop.name: pop.n for pop in topology.populations}

    for proj in topology.projections:
        topo = proj.topology
        n_pre = pop_sizes[proj.pre]
        n_post = pop_sizes[proj.post]

        assert topo.pre_idx.dim() == 1
        assert topo.post_idx.dim() == 1
        assert topo.pre_idx.numel() == topo.post_idx.numel()
        assert topo.pre_idx.numel() > 0

        if topo.weights is not None:
            assert topo.weights.dim() == 1
            assert topo.weights.numel() == topo.pre_idx.numel()
        if topo.delay_steps is not None:
            assert topo.delay_steps.dim() == 1
            assert topo.delay_steps.numel() == topo.pre_idx.numel()

        if topo.meta:
            for value in topo.meta.values():
                if not hasattr(value, "dim"):
                    continue
                if int(value.dim()) != 2:
                    continue
                # Sparse layouts are expected in compiled metadata.
                is_sparse = bool(getattr(value, "is_sparse", False))
                if is_sparse:
                    continue
                assert tuple(value.shape) != (n_post, n_pre)
