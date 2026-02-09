"""Learning package."""

from biosnn.learning.homeostasis import (
    HomeostasisScope,
    RateEmaThresholdHomeostasis,
    RateEmaThresholdHomeostasisConfig,
)
from biosnn.learning.rules import (
    EligibilityTraceHebbianParams,
    EligibilityTraceHebbianRule,
    EligibilityTraceHebbianState,
    MetaplasticProjectionParams,
    MetaplasticProjectionRule,
    MetaplasticProjectionState,
    RStdpEligibilityParams,
    RStdpEligibilityRule,
    RStdpEligibilityState,
    ThreeFactorEligibilityStdpParams,
    ThreeFactorEligibilityStdpRule,
    ThreeFactorEligibilityStdpState,
    ThreeFactorHebbianParams,
    ThreeFactorHebbianRule,
    ThreeFactorHebbianState,
)

__all__ = [
    "HomeostasisScope",
    "RateEmaThresholdHomeostasis",
    "RateEmaThresholdHomeostasisConfig",
    "EligibilityTraceHebbianParams",
    "EligibilityTraceHebbianRule",
    "EligibilityTraceHebbianState",
    "MetaplasticProjectionParams",
    "MetaplasticProjectionRule",
    "MetaplasticProjectionState",
    "RStdpEligibilityParams",
    "RStdpEligibilityRule",
    "RStdpEligibilityState",
    "ThreeFactorEligibilityStdpParams",
    "ThreeFactorEligibilityStdpRule",
    "ThreeFactorEligibilityStdpState",
    "ThreeFactorHebbianParams",
    "ThreeFactorHebbianRule",
    "ThreeFactorHebbianState",
]
