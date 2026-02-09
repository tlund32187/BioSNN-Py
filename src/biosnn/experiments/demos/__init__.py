"""Feature demo builders used by the CLI."""

from biosnn.experiments.demos.feature_demos import (
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
    "FEATURE_DEMO_BUILDERS",
    "FeatureDemoBuildResult",
    "FeatureDemoBuilder",
    "FeatureDemoConfig",
    "FeatureDemoName",
    "build_delay_impulse_demo",
    "build_dopamine_plasticity_demo",
    "build_learning_gate_demo",
    "build_propagation_impulse_demo",
]
