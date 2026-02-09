"""Experiment builders and demo runners."""

from biosnn.experiments.demo_minimal import DemoMinimalConfig, run_demo_minimal
from biosnn.experiments.demo_network import DemoNetworkConfig, build_network_demo, run_demo_network
from biosnn.experiments.demo_runner import run_demo_from_spec
from biosnn.experiments.demo_types import DemoModelSpec, DemoRuntimeConfig
from biosnn.experiments.demos import (
    FEATURE_DEMO_BUILDERS,
    FeatureDemoBuilder,
    FeatureDemoBuildResult,
    FeatureDemoConfig,
    FeatureDemoName,
    build_delay_impulse_demo,
    build_dopamine_plasticity_demo,
    build_learning_gate_demo,
    build_propagation_impulse_demo,
)

__all__ = [
    "DemoMinimalConfig",
    "DemoModelSpec",
    "DemoNetworkConfig",
    "DemoRuntimeConfig",
    "FEATURE_DEMO_BUILDERS",
    "FeatureDemoBuildResult",
    "FeatureDemoBuilder",
    "FeatureDemoConfig",
    "FeatureDemoName",
    "build_delay_impulse_demo",
    "build_dopamine_plasticity_demo",
    "build_learning_gate_demo",
    "build_network_demo",
    "build_propagation_impulse_demo",
    "run_demo_from_spec",
    "run_demo_minimal",
    "run_demo_network",
]
