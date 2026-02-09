"""Contracts (interfaces/protocols) for BioSNN-Py.

This package is *not* the public API surface. Only symbols re-exported from
:mod:`biosnn.api` are considered semver-stable.
"""

from biosnn.contracts.homeostasis import (
    HomeostasisPopulation,
    IHomeostasisRule,
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
from biosnn.contracts.neurons import (
    Compartment,
    INeuronModel,
    NeuronInputs,
    NeuronStepResult,
    StepContext,
)
from biosnn.contracts.synapses import (
    ISynapseModel,
    ReceptorKind,
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
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
    "ReceptorKind",
    # learning
    "LearningBatch",
    "LearningStepResult",
    "ILearningRule",
    # homeostasis
    "HomeostasisPopulation",
    "IHomeostasisRule",
    # neuromodulation
    "ModulatorKind",
    "ModulatorRelease",
    "IModulatorField",
    # monitors
    "StepEvent",
    "IMonitor",
]
