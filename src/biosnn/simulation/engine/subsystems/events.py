"""StepEvent payload construction subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from biosnn.contracts.homeostasis import IHomeostasisRule
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec

from .models import (
    STEP_EVENT_KEY_HOMEOSTASIS_STATE,
    STEP_EVENT_KEY_LEARNING_STATE,
    STEP_EVENT_KEY_MODULATORS,
    STEP_EVENT_KEY_POPULATION_SLICES,
    STEP_EVENT_KEY_POPULATION_STATE,
    STEP_EVENT_KEY_PROJECTION_WEIGHTS,
    STEP_EVENT_KEY_SCALARS,
    STEP_EVENT_KEY_SPIKES,
    STEP_EVENT_KEY_SYNAPSE_STATE,
    STEP_EVENT_KEY_V_SOMA,
    NetworkRequirements,
)


@dataclass(frozen=True, slots=True)
class StepEventPayloadPlan:
    needs_spike_tensors: bool
    needs_event_spikes: bool
    needs_population_state: bool
    needs_v_soma: bool
    needs_population_slices: bool
    needs_scalars: bool
    needs_projection_weights: bool
    needs_synapse_state: bool
    needs_learning_state: bool
    needs_homeostasis_state: bool
    needs_modulator_state: bool

    @property
    def needs_population_tensors(self) -> bool:
        return bool(self.needs_population_state or self.needs_v_soma)


class StepEventSubsystem:
    """Builds minimal StepEvent tensor/scalar payloads based on monitor requirements."""

    def payload_plan(
        self,
        requirements: NetworkRequirements,
        *,
        fast_mode: bool,
    ) -> StepEventPayloadPlan:
        needs_synapse_state = requirements.needs_step_event(STEP_EVENT_KEY_SYNAPSE_STATE)
        needs_projection_weights = (
            requirements.needs_step_event(STEP_EVENT_KEY_PROJECTION_WEIGHTS)
            and not needs_synapse_state
        )
        needs_spike_tensors = requirements.needs_step_event(STEP_EVENT_KEY_SPIKES)
        return StepEventPayloadPlan(
            needs_spike_tensors=needs_spike_tensors,
            needs_event_spikes=bool(needs_spike_tensors and not fast_mode),
            needs_population_state=requirements.needs_step_event(STEP_EVENT_KEY_POPULATION_STATE),
            needs_v_soma=requirements.needs_step_event(STEP_EVENT_KEY_V_SOMA),
            needs_population_slices=requirements.needs_step_event(STEP_EVENT_KEY_POPULATION_SLICES),
            needs_scalars=requirements.needs_step_event(STEP_EVENT_KEY_SCALARS),
            needs_projection_weights=needs_projection_weights,
            needs_synapse_state=needs_synapse_state,
            needs_learning_state=requirements.needs_step_event(STEP_EVENT_KEY_LEARNING_STATE),
            needs_homeostasis_state=requirements.needs_step_event(STEP_EVENT_KEY_HOMEOSTASIS_STATE),
            needs_modulator_state=requirements.needs_step_event(STEP_EVENT_KEY_MODULATORS),
        )

    def merge_population_tensors(
        self,
        tensors_by_pop: Mapping[str, Mapping[str, Tensor]],
        pop_specs: Sequence[PopulationSpec],
        device: Any,
    ) -> dict[str, Tensor]:
        torch = require_torch()
        keys: set[str] = set()
        for tensors in tensors_by_pop.values():
            keys.update(tensors.keys())

        merged: dict[str, Tensor] = {}
        for key in keys:
            parts: list[Tensor] = []
            for spec in pop_specs:
                value = tensors_by_pop.get(spec.name, {}).get(key)
                if value is None:
                    parts.append(torch.full((spec.n,), float("nan"), device=device, dtype=torch.float32))
                else:
                    parts.append(value)
            merged[key] = torch.cat(parts)

        return merged

    def build_population_tensors(
        self,
        tensors_by_pop: Mapping[str, Mapping[str, Tensor]],
        pop_specs: Sequence[PopulationSpec],
        pop_states: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for spec in pop_specs:
            pop_name = spec.name
            spikes = cast(Tensor, pop_states[pop_name].spikes)
            tensors[f"pop/{pop_name}/spikes"] = spikes
            for key, value in tensors_by_pop.get(pop_name, {}).items():
                if key == "spikes":
                    continue
                tensors[f"pop/{pop_name}/{key}"] = value
        return tensors

    def summarize_spikes(
        self,
        pop_specs: Sequence[PopulationSpec],
        pop_states: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        torch = require_torch()
        total_spikes = None
        total_neurons = 0
        for spec in pop_specs:
            total_neurons += spec.n
            spikes = cast(Tensor, pop_states[spec.name].spikes)
            count = spikes.sum()
            total_spikes = count if total_spikes is None else total_spikes + count
        if total_neurons <= 0 or total_spikes is None:
            device = None
            dtype = None
            if pop_specs:
                sample = cast(Tensor, pop_states[pop_specs[0].name].spikes)
                device = sample.device
                dtype = sample.dtype
            return (
                torch.zeros((), device=device, dtype=dtype or torch.float32),
                torch.zeros((), device=device, dtype=dtype or torch.float32),
            )
        spike_fraction = total_spikes / float(total_neurons)
        return total_spikes, spike_fraction

    def projection_tensors(
        self,
        proj_specs: Sequence[ProjectionSpec],
        proj_states: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for proj in proj_specs:
            state = proj_states[proj.name].state
            for key, value in proj.synapse.state_tensors(state).items():
                tensors[f"proj/{proj.name}/{key}"] = value
        return tensors

    def projection_weight_tensors(
        self,
        proj_specs: Sequence[ProjectionSpec],
        proj_states: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for proj in proj_specs:
            state = proj_states[proj.name].state
            weights = getattr(state, "weights", None)
            if weights is not None:
                tensors[f"proj/{proj.name}/weights"] = cast(Tensor, weights)
        return tensors

    def learning_tensors(
        self,
        proj_specs: Sequence[ProjectionSpec],
        proj_states: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for proj in proj_specs:
            if proj.learning is None:
                continue
            state = proj_states[proj.name].learning_state
            if state is None:
                continue
            for key, value in proj.learning.state_tensors(state).items():
                tensors[f"learn/{proj.name}/{key}"] = value
        return tensors

    def homeostasis_tensors(self, homeostasis: IHomeostasisRule | None) -> dict[str, Tensor]:
        if homeostasis is None:
            return {}
        return dict(homeostasis.state_tensors())

    def modulator_tensors(
        self,
        mod_specs: Sequence[ModulatorSpec],
        mod_states: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {}
        for spec in mod_specs:
            state = mod_states[spec.name]
            for key, value in spec.field.state_tensors(state).items():
                tensors[f"mod/{spec.name}/{key}"] = value
        return tensors

    def accumulate_drive(
        self,
        target: dict[Any, Tensor],
        update: Mapping[Any, Tensor],
    ) -> None:
        for comp, tensor in update.items():
            if comp not in target:
                raise KeyError(f"Drive accumulator missing compartment {comp}")
            existing = target[comp]
            if hasattr(existing, "add_"):
                existing.add_(tensor)
            else:
                target[comp] = existing + tensor


__all__ = ["StepEventPayloadPlan", "StepEventSubsystem"]
