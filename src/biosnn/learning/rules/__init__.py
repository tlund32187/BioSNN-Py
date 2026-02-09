"""Learning rules."""

from biosnn.learning.rules.eligibility_trace_hebbian import (
    EligibilityTraceHebbianParams,
    EligibilityTraceHebbianRule,
    EligibilityTraceHebbianState,
)
from biosnn.learning.rules.metaplastic_projection import (
    MetaplasticProjectionParams,
    MetaplasticProjectionRule,
    MetaplasticProjectionState,
)
from biosnn.learning.rules.rstdp_eligibility import (
    RStdpEligibilityParams,
    RStdpEligibilityRule,
    RStdpEligibilityState,
)
from biosnn.learning.rules.three_factor_eligibility_stdp import (
    ThreeFactorEligibilityStdpParams,
    ThreeFactorEligibilityStdpRule,
    ThreeFactorEligibilityStdpState,
)
from biosnn.learning.rules.three_factor_hebbian import (
    ThreeFactorHebbianParams,
    ThreeFactorHebbianRule,
    ThreeFactorHebbianState,
)

__all__ = [
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
