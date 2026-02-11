"""Logic-gate task harness for deterministic CPU-friendly experiments."""

from .configs import (
    AdvancedPostVoltageSource,
    AdvancedSynapseConfig,
    CurriculumContextCompartment,
    CurriculumGateContextConfig,
    LogicGateNeuronModel,
    LogicGateRunConfig,
)
from .datasets import LogicGate, SamplingMethod, make_truth_table, sample_case_indices
from .encoding import (
    INPUT_NEURON_INDICES,
    OUTPUT_NEURON_INDICES,
    decode_output,
    encode_inputs,
    gate_context_level_for_gate,
)
from .engine_runner import run_logic_gate_curriculum_engine, run_logic_gate_engine
from .evaluators import PassCriterion, PassTracker, eval_accuracy, gate_pass_criterion
from .runner import run_logic_gate, run_logic_gate_curriculum
from .surrogate_train import SurrogateTrainResult, train_logic_gate_surrogate
from .topologies import (
    LogicGateHandles,
    LogicGateTopology,
    build_logic_gate_ff,
    build_logic_gate_xor,
    build_logic_gate_xor_variant,
)

__all__ = [
    "INPUT_NEURON_INDICES",
    "AdvancedPostVoltageSource",
    "AdvancedSynapseConfig",
    "CurriculumContextCompartment",
    "CurriculumGateContextConfig",
    "LogicGate",
    "LogicGateNeuronModel",
    "LogicGateRunConfig",
    "OUTPUT_NEURON_INDICES",
    "PassCriterion",
    "PassTracker",
    "SamplingMethod",
    "decode_output",
    "encode_inputs",
    "gate_context_level_for_gate",
    "eval_accuracy",
    "gate_pass_criterion",
    "LogicGateHandles",
    "LogicGateTopology",
    "build_logic_gate_ff",
    "build_logic_gate_xor",
    "build_logic_gate_xor_variant",
    "make_truth_table",
    "run_logic_gate",
    "run_logic_gate_curriculum",
    "run_logic_gate_engine",
    "run_logic_gate_curriculum_engine",
    "sample_case_indices",
    "SurrogateTrainResult",
    "train_logic_gate_surrogate",
]
