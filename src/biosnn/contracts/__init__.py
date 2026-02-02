"""Contracts (interfaces/protocols) for BioSNN-Py.

This package is *not* the public API surface. Only symbols re-exported from
:mod:`biosnn.api` are considered semver-stable.
"""

from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.synapses import (
    ISynapseModel,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from biosnn.contracts.learning import (
    ILearningRule,
    LearningBatch,
    LearningStepResult,
)
from biosnn.contracts.modulators import (
    IModulatorField,
    ModulatorKind,
    ModulatorRelease,
)
from biosnn.contracts.monitors import (
    IMonitor,
    StepEvent,
)

__all__ = [
    # common
    "StepContext",
    # neurons
    "Compartment",
    "NeuronInputs",
    "NeuronStepResult",
    "INeuronModel",
    # synapses
    "SynapseTopology",
    "SynapseInputs",
    "SynapseStepResult",
    "ISynapseModel",
    # learning
    "LearningBatch",
    "LearningStepResult",
    "ILearningRule",
    # neuromodulation
    "ModulatorKind",
    "ModulatorRelease",
    "IModulatorField",
    # monitors
    "StepEvent",
    "IMonitor",
]
