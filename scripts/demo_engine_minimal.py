"""Minimal end-to-end engine demo (GLIF + delayed synapses)."""

from __future__ import annotations

from pathlib import Path

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.simulation import SimulationConfig
from biosnn.io.dashboard_export import export_neuron_csv, export_synapse_csv, export_topology_json
from biosnn.monitors.csv import GLIFCSVMonitor, SynapseCSVMonitor
from biosnn.simulation.engine import TorchSimulationEngine
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentSynapse


def main() -> None:
    n = 50
    dt = 1e-3

    neuron_model = GLIFModel()
    synapse_model = DelayedCurrentSynapse()

    topology = build_erdos_renyi_topology(
        n=n,
        p=0.1,
        allow_self=False,
        dt=dt,
        positions=None,
        weight_init=1e-9,
    )

    engine = TorchSimulationEngine(
        neuron_model=neuron_model,
        synapse_model=synapse_model,
        topology=topology,
        n=n,
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    neuron_monitor = GLIFCSVMonitor(artifacts / "neuron.csv", sample_indices=list(range(16)))
    synapse_monitor = SynapseCSVMonitor(artifacts / "synapse.csv", sample_indices=list(range(32)))

    engine.attach_monitors([neuron_monitor, synapse_monitor])
    engine.reset(config=SimulationConfig(dt=dt))
    engine.run(steps=200)

    export_topology_json(topology, path=artifacts / "topology.json")
    export_neuron_csv(neuron_model.state_tensors(engine._neuron_state), path=artifacts / "neuron_snapshot.csv")
    export_synapse_csv(engine._synapse_state.weights, path=artifacts / "synapse_snapshot.csv")


if __name__ == "__main__":
    main()
