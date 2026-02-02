"""Public façade (stable API surface).

Only symbols re-exported from here are considered public and semver-stable.
Internal modules may change without notice.
"""

from biosnn.api.version import __version__
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
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)

__all__ = [
    "__version__",
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
    # neuromodulators
    "ModulatorKind",
    "ModulatorRelease",
    "IModulatorField",
    # monitors
    "StepEvent",
    "IMonitor",
]
