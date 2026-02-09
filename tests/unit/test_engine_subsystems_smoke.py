from __future__ import annotations

import pytest

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

pytestmark = pytest.mark.unit


def test_engine_subsystems_smoke() -> None:
    pytest.importorskip("torch")

    pop = PopulationSpec(name="Pop", model=GLIFModel(), n=16)
    topo = build_erdos_renyi_topology(n=16, p=0.2, allow_self=False, dt=1e-3)
    proj = ProjectionSpec(
        name="Pop->Pop",
        synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05)),
        topology=topo,
        pre="Pop",
        post="Pop",
    )

    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[proj],
        fast_mode=True,
        compiled_mode=True,
    )

    assert engine._topology_subsystem is not None
    assert engine._buffer_subsystem is not None
    assert engine._modulator_subsystem is not None
    assert engine._learning_subsystem is not None
    assert engine._monitor_subsystem is not None
    assert engine._event_subsystem is not None

    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu", dtype="float32", seed=11))
    for _ in range(3):
        engine.step()

    assert engine._compiled_network_plan is not None
