"""Receptor profile definitions for per-post synaptic kinetics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from biosnn.contracts.synapses import ReceptorKind


@dataclass(frozen=True, slots=True)
class ReceptorProfile:
    """Per-projection receptor kinetics profile."""

    kinds: tuple[ReceptorKind, ...]
    mix: Mapping[ReceptorKind, float]
    tau: Mapping[ReceptorKind, float]
    sign: Mapping[ReceptorKind, float]


def canonical_receptor_kind(kind: ReceptorKind) -> ReceptorKind:
    """Normalize legacy receptor aliases."""

    if kind == ReceptorKind.GABA:
        return ReceptorKind.GABA_A
    return kind


def resolve_profile_value(
    values: Mapping[ReceptorKind, float],
    kind: ReceptorKind,
    *,
    default: float,
) -> float:
    """Resolve receptor values with legacy ``GABA`` alias fallback."""

    if kind in values:
        return float(values[kind])
    canonical = canonical_receptor_kind(kind)
    if canonical in values:
        return float(values[canonical])
    if canonical == ReceptorKind.GABA_A and ReceptorKind.GABA in values:
        return float(values[ReceptorKind.GABA])
    return float(default)


def profile_exc_ampa_nmda(
    *,
    ampa_mix: float = 1.0,
    nmda_mix: float = 0.35,
    ampa_tau: float = 5e-3,
    nmda_tau: float = 100e-3,
) -> ReceptorProfile:
    """Excitatory AMPA/NMDA profile."""

    return ReceptorProfile(
        kinds=(ReceptorKind.AMPA, ReceptorKind.NMDA),
        mix={
            ReceptorKind.AMPA: float(ampa_mix),
            ReceptorKind.NMDA: float(nmda_mix),
        },
        tau={
            ReceptorKind.AMPA: float(ampa_tau),
            ReceptorKind.NMDA: float(nmda_tau),
        },
        sign={
            ReceptorKind.AMPA: 1.0,
            ReceptorKind.NMDA: 1.0,
        },
    )


def profile_inh_gabaa_gabab(
    *,
    gabaa_mix: float = 1.0,
    gabab_mix: float = 0.25,
    gabaa_tau: float = 10e-3,
    gabab_tau: float = 150e-3,
) -> ReceptorProfile:
    """Inhibitory GABA_A/GABA_B profile."""

    return ReceptorProfile(
        kinds=(ReceptorKind.GABA_A, ReceptorKind.GABA_B),
        mix={
            ReceptorKind.GABA_A: float(gabaa_mix),
            ReceptorKind.GABA_B: float(gabab_mix),
        },
        tau={
            ReceptorKind.GABA_A: float(gabaa_tau),
            ReceptorKind.GABA_B: float(gabab_tau),
        },
        sign={
            ReceptorKind.GABA_A: -1.0,
            ReceptorKind.GABA_B: -1.0,
        },
    )


__all__ = [
    "ReceptorProfile",
    "canonical_receptor_kind",
    "profile_exc_ampa_nmda",
    "profile_inh_gabaa_gabab",
    "resolve_profile_value",
]
