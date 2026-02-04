from __future__ import annotations

import pytest

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.synapses import SynapseInputs, SynapseTopology
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def test_compile_topology_casts_and_meta():
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.int32)
    post_idx = torch.tensor([1, 2, 0], dtype=torch.int32)
    delay_steps = torch.tensor([0, 2, 1], dtype=torch.int16)
    receptor = torch.tensor([0, 1, 2], dtype=torch.int8)
    target_compartments = torch.tensor([0, 1, 0], dtype=torch.int8)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        receptor=receptor,
        target_compartments=target_compartments,
        weights=weights,
        target_compartment=Compartment.SOMA,
    )

    compiled = compile_topology(topology, device="cpu", dtype="float32")

    assert compiled.pre_idx.dtype == torch.long
    assert compiled.post_idx.dtype == torch.long
    assert compiled.pre_idx.device.type == "cpu"
    assert compiled.post_idx.device.type == "cpu"
    assert compiled.delay_steps is not None
    assert compiled.delay_steps.dtype == torch.long
    assert compiled.receptor is not None
    assert compiled.receptor.dtype == torch.long
    assert compiled.target_compartments is not None
    assert compiled.target_compartments.dtype == torch.long
    assert compiled.weights is not None
    assert compiled.weights.dtype == torch.float32
    assert compiled.meta is not None
    assert compiled.meta["n_pre"] == 3
    assert compiled.meta["n_post"] == 3
    assert compiled.meta["max_delay_steps"] == 2


def test_synapse_step_no_device_transfers(monkeypatch):
    ctx = StepContext(device="cpu", dtype="float32")
    model = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=1.0))
    state = model.init_state(2, ctx=ctx)
    pre_idx = torch.tensor([0, 1], dtype=torch.long)
    post_idx = torch.tensor([1, 0], dtype=torch.long)
    delay_steps = torch.tensor([0, 1], dtype=torch.int64)
    topology = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartment=Compartment.SOMA,
    )
    compile_topology(
        topology,
        device=ctx.device,
        dtype=ctx.dtype,
        build_edges_by_delay=True,
    )

    pre_spikes = torch.tensor([1.0, 0.0], dtype=state.weights.dtype)

    def _fail(*args, **kwargs):
        raise AssertionError("Tensor.to should not be called during step")

    monkeypatch.setattr(torch.Tensor, "to", _fail, raising=False)

    model.step(
        state,
        topology,
        SynapseInputs(pre_spikes=pre_spikes),
        dt=1e-3,
        t=0.0,
        ctx=ctx,
    )


def test_compile_topology_skips_edges_by_delay_by_default():
    pre_idx = torch.tensor([0, 1], dtype=torch.int32)
    post_idx = torch.tensor([1, 0], dtype=torch.int32)
    topology = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx)

    compiled = compile_topology(topology, device="cpu", dtype="float32")

    assert compiled.meta is not None
    assert "edges_by_delay" not in compiled.meta
    assert "pre_ptr" not in compiled.meta
    assert "edge_idx" not in compiled.meta
