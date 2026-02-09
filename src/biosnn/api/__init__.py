"""Public façade (stable API surface).

Only symbols re-exported from here are considered public and semver-stable.
Internal modules may change without notice.
"""

from biosnn.api import presets
from biosnn.api.builders.network_builder import ErdosRenyi, Init, NetworkBuilder, NetworkSpec
from biosnn.api.training.trainer import EngineConfig, Trainer, TrainReport
from biosnn.api.version import __version__
from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.base import NeuronModelBase, StateTensorSpec
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.biophysics.models.template_neuron import TemplateNeuronModel
from biosnn.contracts.homeostasis import HomeostasisPopulation, IHomeostasisRule
from biosnn.contracts.learning import ILearningRule, LearningBatch, LearningStepResult
from biosnn.contracts.modulators import IModulatorField, ModulatorKind, ModulatorRelease
from biosnn.contracts.monitors import IMonitor, StepEvent
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
from biosnn.learning.homeostasis import (
    HomeostasisScope,
    RateEmaThresholdHomeostasis,
    RateEmaThresholdHomeostasisConfig,
)
from biosnn.learning.rules import ThreeFactorHebbianParams, ThreeFactorHebbianRule
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse
from biosnn.synapses.dynamics.delayed_sparse_matmul import (
    DelayedSparseMatmulParams,
    DelayedSparseMatmulSynapse,
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
    "ReceptorKind",
    # learning
    "LearningBatch",
    "LearningStepResult",
    "ILearningRule",
    # homeostasis
    "HomeostasisPopulation",
    "IHomeostasisRule",
    # neuromodulators
    "ModulatorKind",
    "ModulatorRelease",
    "IModulatorField",
    # monitors
    "StepEvent",
    "IMonitor",
    # builders
    "NetworkBuilder",
    "NetworkSpec",
    "ErdosRenyi",
    "Init",
    "presets",
    "Trainer",
    "EngineConfig",
    "TrainReport",
    # neuron models
    "NeuronModelBase",
    "StateTensorSpec",
    "TemplateNeuronModel",
    "GLIFModel",
    "AdEx2CompModel",
    # synapses
    "DelayedCurrentParams",
    "DelayedCurrentSynapse",
    "DelayedSparseMatmulParams",
    "DelayedSparseMatmulSynapse",
    # learning rules
    "ThreeFactorHebbianParams",
    "ThreeFactorHebbianRule",
    # homeostasis rules
    "HomeostasisScope",
    "RateEmaThresholdHomeostasis",
    "RateEmaThresholdHomeostasisConfig",
]
