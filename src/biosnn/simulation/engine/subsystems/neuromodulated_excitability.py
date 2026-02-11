"""Injects neuromodulator-driven excitability bias into population drive buffers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.simulation import ExcitabilityModulationConfig
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.network.specs import PopulationSpec

if TYPE_CHECKING:
    from biosnn.simulation.engine.subsystems.modulators import ModulatorSubsystem


class NeuromodulatedExcitabilitySubsystem:
    """Adds per-neuron bias from sampled ACh/NE/5HT levels."""

    def __init__(self) -> None:
        self._enabled = False
        self._targets: tuple[str, ...] = ()
        self._target_compartment = Compartment.SOMA
        self._ach_gain = 0.0
        self._ne_gain = 0.0
        self._ht_gain = 0.0
        self._clamp_abs = 1.0
        self._bias_buffers: dict[str, Tensor] = {}

    def configure(
        self,
        *,
        config: ExcitabilityModulationConfig | Mapping[str, Any] | None,
        pop_specs: Sequence[PopulationSpec],
        device: Any,
        dtype: Any,
    ) -> None:
        resolved = _coerce_config(config)
        self._enabled = bool(resolved.enabled)
        self._target_compartment = _coerce_compartment(resolved.compartment)
        self._ach_gain = float(resolved.ach_gain)
        self._ne_gain = float(resolved.ne_gain)
        self._ht_gain = float(resolved.ht_gain)
        self._clamp_abs = abs(float(resolved.clamp_abs))
        self._targets = _resolve_target_populations(pop_specs=pop_specs, targets=resolved.targets)
        self._bias_buffers = {}
        if not self._enabled:
            return
        torch = _require_torch()
        for pop_name in self._targets:
            pop_spec = next((spec for spec in pop_specs if spec.name == pop_name), None)
            if pop_spec is None:
                continue
            self._bias_buffers[pop_name] = torch.zeros((pop_spec.n,), device=device, dtype=dtype)

    def apply(
        self,
        *,
        drive_by_pop: Mapping[str, Mapping[Compartment, Tensor]] | dict[str, dict[Compartment, Tensor]],
        mod_by_pop: Mapping[str, Mapping[ModulatorKind, Tensor]] | None,
        modulator_subsystem: ModulatorSubsystem,
    ) -> None:
        if not self._enabled or not self._targets:
            return
        if self._ach_gain == 0.0 and self._ne_gain == 0.0 and self._ht_gain == 0.0:
            return

        for pop_name in self._targets:
            drive = drive_by_pop.get(pop_name)
            if drive is None:
                continue
            drive_target = drive.get(self._target_compartment)
            if drive_target is None:
                continue

            bias = self._ensure_bias_buffer(pop_name=pop_name, like=drive_target)
            bias.zero_()
            has_term = False

            if self._ach_gain != 0.0:
                ach = modulator_subsystem.get_population_levels(
                    kind=ModulatorKind.ACETYLCHOLINE,
                    population_name=pop_name,
                    mod_by_pop=mod_by_pop,
                    like=drive_target,
                )
                if ach is not None:
                    bias.add_(ach, alpha=self._ach_gain)
                    has_term = True
            if self._ne_gain != 0.0:
                ne = modulator_subsystem.get_population_levels(
                    kind=ModulatorKind.NORADRENALINE,
                    population_name=pop_name,
                    mod_by_pop=mod_by_pop,
                    like=drive_target,
                )
                if ne is not None:
                    bias.add_(ne, alpha=self._ne_gain)
                    has_term = True
            if self._ht_gain != 0.0:
                ht = modulator_subsystem.get_population_levels(
                    kind=ModulatorKind.SEROTONIN,
                    population_name=pop_name,
                    mod_by_pop=mod_by_pop,
                    like=drive_target,
                )
                if ht is not None:
                    bias.add_(ht, alpha=-self._ht_gain)
                    has_term = True

            if not has_term:
                continue
            if self._clamp_abs > 0.0:
                clamp_abs = self._clamp_abs
                bias.clamp_(min=-clamp_abs, max=clamp_abs)
            drive_target.add_(bias)

    def _ensure_bias_buffer(self, *, pop_name: str, like: Tensor) -> Tensor:
        buffer = self._bias_buffers.get(pop_name)
        if (
            buffer is None
            or buffer.shape != like.shape
            or buffer.device != like.device
            or buffer.dtype != like.dtype
        ):
            torch = _require_torch()
            buffer = torch.zeros_like(like)
            self._bias_buffers[pop_name] = buffer
        return buffer


def _coerce_config(
    config: ExcitabilityModulationConfig | Mapping[str, Any] | None,
) -> ExcitabilityModulationConfig:
    if isinstance(config, ExcitabilityModulationConfig):
        return config
    if not isinstance(config, Mapping):
        return ExcitabilityModulationConfig()
    return ExcitabilityModulationConfig(
        enabled=bool(config.get("enabled", False)),
        targets=_coerce_targets(config.get("targets", ("hidden", "out"))),
        compartment=str(config.get("compartment", "soma")),
        ach_gain=_coerce_float(config.get("ach_gain"), 0.0),
        ne_gain=_coerce_float(config.get("ne_gain"), 0.0),
        ht_gain=_coerce_float(config.get("ht_gain"), 0.0),
        clamp_abs=abs(_coerce_float(config.get("clamp_abs"), 1.0)),
    )


def _coerce_targets(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence):
        parts = [str(part).strip() for part in raw]
    else:
        parts = []
    targets: list[str] = []
    for part in parts:
        token = _normalize_target_token(part)
        if not token or token in targets:
            continue
        targets.append(token)
    if targets:
        return tuple(targets)
    return ("hidden", "output")


def _normalize_target_token(value: str) -> str:
    token = str(value).strip().lower()
    aliases = {
        "out": "output",
        "outputs": "output",
        "hid": "hidden",
    }
    return aliases.get(token, token)


def _resolve_target_populations(
    *,
    pop_specs: Sequence[PopulationSpec],
    targets: Sequence[str],
) -> tuple[str, ...]:
    normalized = {_normalize_target_token(str(target)) for target in targets}
    if not normalized:
        return ()
    selected: list[str] = []
    for spec in pop_specs:
        name_token = _normalize_target_token(spec.name)
        role_token = _population_role(spec)
        if "all" in normalized:
            selected.append(spec.name)
            continue
        if name_token in normalized:
            selected.append(spec.name)
            continue
        if role_token and role_token in normalized:
            selected.append(spec.name)
    return tuple(selected)


def _population_role(spec: PopulationSpec) -> str:
    if not spec.meta:
        return ""
    role = spec.meta.get("role")
    if role is None:
        return ""
    token = _normalize_target_token(str(role))
    if token == "out":
        return "output"
    return token


def _coerce_compartment(raw: str) -> Compartment:
    token = str(raw).strip().lower()
    for comp in Compartment:
        if token == comp.value:
            return comp
    return Compartment.SOMA


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _require_torch() -> Any:
    from biosnn.core.torch_utils import require_torch

    return require_torch()


__all__ = ["NeuromodulatedExcitabilitySubsystem"]
