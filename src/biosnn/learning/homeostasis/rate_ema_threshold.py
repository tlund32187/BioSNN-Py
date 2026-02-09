"""EMA firing-rate homeostasis with threshold adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from biosnn.contracts.homeostasis import HomeostasisPopulation, IHomeostasisRule
from biosnn.contracts.neurons import StepContext, Tensor
from biosnn.core.torch_utils import require_torch

HomeostasisScope = Literal["per_population", "per_neuron"]


@dataclass(frozen=True, slots=True)
class RateEmaThresholdHomeostasisConfig:
    """Configuration for rate-EMA threshold homeostasis."""

    alpha: float = 0.01
    eta: float = 1e-3
    r_target: float | Mapping[str, float] = 0.05
    clamp_min: float = 0.0
    clamp_max: float = 0.050
    scope: HomeostasisScope = "per_neuron"

    def __post_init__(self) -> None:
        if not (0.0 < float(self.alpha) <= 1.0):
            raise ValueError("homeostasis alpha must be in (0, 1].")
        if float(self.eta) < 0.0:
            raise ValueError("homeostasis eta must be >= 0.")
        if float(self.clamp_max) < float(self.clamp_min):
            raise ValueError("homeostasis clamp_max must be >= clamp_min.")
        if self.scope not in {"per_population", "per_neuron"}:
            raise ValueError("homeostasis scope must be 'per_population' or 'per_neuron'.")


class RateEmaThresholdHomeostasis(IHomeostasisRule):
    """Vectorized homeostasis using spike-rate EMA and threshold adaptation."""

    name = "rate_ema_threshold"

    def __init__(self, config: RateEmaThresholdHomeostasisConfig | None = None) -> None:
        self.config = config or RateEmaThresholdHomeostasisConfig()
        self._scope = cast(HomeostasisScope, self.config.scope)
        self._one_minus_alpha = 1.0 - float(self.config.alpha)
        self._alpha = float(self.config.alpha)
        self._eta = float(self.config.eta)

        self._rate_ema: dict[str, Tensor] = {}
        self._target: dict[str, Tensor] = {}
        self._target_mean: dict[str, Tensor] = {}
        self._control: dict[str, Tensor | None] = {}
        self._spike_float: dict[str, Tensor] = {}
        self._delta: dict[str, Tensor] = {}
        self._spike_mean: dict[str, Tensor] = {}
        self._rate_mean: dict[str, Tensor] = {}
        self._control_mean: dict[str, Tensor] = {}

    def init(
        self,
        populations: Sequence[HomeostasisPopulation],
        *,
        device: Any,
        dtype: Any,
        ctx: StepContext,
    ) -> None:
        _ = (device, dtype, ctx)
        torch = require_torch()
        self._clear()

        for pop in populations:
            theta = _threshold_tensor(pop.state)
            tensor_device: Any | None
            tensor_dtype: Any
            if theta is not None:
                tensor_dtype = theta.dtype
                tensor_device = theta.device
                n = int(theta.numel())
            else:
                tensor_device = torch.device(device) if device is not None else None
                tensor_dtype = _dtype_or_default(torch, dtype)
                n = max(1, int(pop.n))

            spike_float = torch.zeros((n,), device=tensor_device, dtype=tensor_dtype)
            target_value = float(_target_for_population(self.config.r_target, pop.name))
            if self._scope == "per_population":
                rate_ema = torch.zeros((1,), device=tensor_device, dtype=tensor_dtype)
                target = torch.full((1,), target_value, device=tensor_device, dtype=tensor_dtype)
                delta = torch.zeros((1,), device=tensor_device, dtype=tensor_dtype)
            else:
                rate_ema = torch.zeros((n,), device=tensor_device, dtype=tensor_dtype)
                target = torch.full((n,), target_value, device=tensor_device, dtype=tensor_dtype)
                delta = torch.zeros((n,), device=tensor_device, dtype=tensor_dtype)

            self._rate_ema[pop.name] = rate_ema
            self._target[pop.name] = target
            self._target_mean[pop.name] = torch.full(
                (), target_value, device=tensor_device, dtype=tensor_dtype
            )
            self._control[pop.name] = theta
            self._spike_float[pop.name] = spike_float
            self._delta[pop.name] = delta
            self._spike_mean[pop.name] = torch.zeros((), device=tensor_device, dtype=tensor_dtype)
            self._rate_mean[pop.name] = torch.zeros((), device=tensor_device, dtype=tensor_dtype)
            self._control_mean[pop.name] = torch.full(
                (), float("nan"), device=tensor_device, dtype=tensor_dtype
            )

    def step(
        self,
        spikes_by_pop: Mapping[str, Tensor],
        *,
        dt: float,
        ctx: StepContext,
    ) -> Mapping[str, Tensor]:
        _ = (dt, ctx)
        for pop_name, rate_ema in self._rate_ema.items():
            spikes = spikes_by_pop.get(pop_name)
            if spikes is None:
                continue
            spike_float = self._spike_float[pop_name]
            _copy_spikes_into(spike_float, spikes)
            target = self._target[pop_name]
            delta = self._delta[pop_name]

            if self._scope == "per_population":
                spike_mean = self._spike_mean[pop_name]
                spike_mean.copy_(spike_float.sum())
                spike_mean.div_(float(max(1, spike_float.numel())))
                rate_ema.mul_(self._one_minus_alpha)
                rate_ema.add_(spike_mean, alpha=self._alpha)
            else:
                rate_ema.mul_(self._one_minus_alpha)
                rate_ema.add_(spike_float, alpha=self._alpha)

            control = self._control[pop_name]
            if control is not None:
                delta.copy_(rate_ema)
                delta.sub_(target)
                delta.mul_(self._eta)
                control.add_(delta)
                control.clamp_(min=float(self.config.clamp_min), max=float(self.config.clamp_max))

            rate_mean = self._rate_mean[pop_name]
            rate_mean.copy_(rate_ema.sum())
            rate_mean.div_(float(max(1, rate_ema.numel())))

            control_mean = self._control_mean[pop_name]
            if control is None:
                control_mean.fill_(float("nan"))
            else:
                control_mean.copy_(control.sum())
                control_mean.div_(float(max(1, control.numel())))

        return self.scalars()

    def scalars(self) -> Mapping[str, Tensor]:
        out: dict[str, Tensor] = {}
        for pop_name in self._rate_ema:
            out[f"homeostasis/rate_mean/{pop_name}"] = self._rate_mean[pop_name]
            out[f"homeostasis/target_rate/{pop_name}"] = self._target_mean[pop_name]
            out[f"homeostasis/control_mean/{pop_name}"] = self._control_mean[pop_name]
            out[f"homeostasis/theta_mean/{pop_name}"] = self._control_mean[pop_name]
        return out

    def state_tensors(self) -> Mapping[str, Tensor]:
        out: dict[str, Tensor] = {}
        for pop_name, rate_ema in self._rate_ema.items():
            out[f"homeostasis/{pop_name}/rate_ema"] = rate_ema
            control = self._control.get(pop_name)
            if control is not None:
                out[f"homeostasis/{pop_name}/theta"] = control
        return out

    def _clear(self) -> None:
        self._rate_ema.clear()
        self._target.clear()
        self._target_mean.clear()
        self._control.clear()
        self._spike_float.clear()
        self._delta.clear()
        self._spike_mean.clear()
        self._rate_mean.clear()
        self._control_mean.clear()


def _target_for_population(target: float | Mapping[str, float], pop_name: str) -> float:
    if isinstance(target, Mapping):
        if pop_name in target:
            return float(target[pop_name])
        if "*" in target:
            return float(target["*"])
        return 0.0
    return float(target)


def _threshold_tensor(state: Any) -> Tensor | None:
    tensor = getattr(state, "theta", None)
    if tensor is None:
        return None
    if not hasattr(tensor, "shape") or not hasattr(tensor, "add_"):
        return None
    return cast(Tensor, tensor)


def _copy_spikes_into(target: Tensor, spikes: Tensor) -> None:
    torch = require_torch()
    if spikes.device != target.device:
        spikes = spikes.to(device=target.device)
    if spikes.dtype == torch.bool:
        target.copy_(spikes)
        return
    target.copy_(spikes)
    target.clamp_(min=0.0, max=1.0)


def _dtype_or_default(torch: Any, dtype: Any) -> Any:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, str):
        maybe_dtype = getattr(torch, dtype, None)
        if maybe_dtype is None:
            return torch.float32
        return maybe_dtype
    return dtype


__all__ = [
    "HomeostasisScope",
    "RateEmaThresholdHomeostasis",
    "RateEmaThresholdHomeostasisConfig",
]
