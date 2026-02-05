"""Fluent network builder API."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from biosnn.connectivity.builders.random_topology import build_erdos_renyi_edges
from biosnn.contracts.learning import ILearningRule
from biosnn.contracts.modulators import IModulatorField, ModulatorKind
from biosnn.contracts.neurons import Compartment, INeuronModel
from biosnn.contracts.synapses import ISynapseModel, ReceptorKind, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch
from biosnn.simulation.network.specs import ModulatorSpec, PopulationSpec, ProjectionSpec

TopologyFactory: TypeAlias = Callable[..., SynapseTopology]
WeightsFactory: TypeAlias = Callable[..., Tensor]

_COMPARTMENT_ORDER = tuple(Compartment)
_COMPARTMENT_TO_ID = {comp: idx for idx, comp in enumerate(_COMPARTMENT_ORDER)}


class TopologySpec(Protocol):
    def build(
        self,
        n_pre: int,
        n_post: int,
        *,
        device: str | None = None,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> SynapseTopology:
        ...


@dataclass(frozen=True, slots=True)
class ErdosRenyi:
    p: float
    allow_self: bool = False

    def build(
        self,
        n_pre: int,
        n_post: int,
        *,
        device: str | None = None,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> SynapseTopology:
        torch = require_torch()
        _ = _resolve_dtype(torch, dtype)

        pre_idx, post_idx = build_erdos_renyi_edges(
            n_pre=n_pre,
            n_post=n_post,
            p=self.p,
            device=device,
            allow_self=self.allow_self,
            seed=seed,
        )
        return SynapseTopology(pre_idx=pre_idx, post_idx=post_idx)


@dataclass(frozen=True, slots=True)
class InitSpec:
    kind: str
    params: Mapping[str, float]

    def build(self, e: int, *, device: str | None, dtype: str | None, seed: int | None) -> Tensor:
        torch = require_torch()
        device_obj = torch.device(device) if device else None
        dtype_obj = _resolve_dtype(torch, dtype)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device_obj) if device_obj is not None else torch.Generator()
            generator.manual_seed(seed)

        if self.kind == "normal":
            mean = float(self.params.get("mean", 0.0))
            std = float(self.params.get("std", 1.0))
            return cast(
                Tensor,
                torch.randn((e,), device=device_obj, dtype=dtype_obj, generator=generator) * std + mean,
            )
        if self.kind == "uniform":
            low = float(self.params.get("low", -0.1))
            high = float(self.params.get("high", 0.1))
            return cast(
                Tensor,
                torch.empty((e,), device=device_obj, dtype=dtype_obj).uniform_(
                    low, high, generator=generator
                ),
            )
        if self.kind == "constant":
            value = float(self.params.get("value", 0.0))
            return cast(Tensor, torch.full((e,), value, device=device_obj, dtype=dtype_obj))
        raise ValueError(f"Unknown InitSpec kind: {self.kind}")


class Init:
    @staticmethod
    def normal(mean: float = 0.0, std: float = 0.1) -> InitSpec:
        return InitSpec(kind="normal", params={"mean": mean, "std": std})

    @staticmethod
    def uniform(low: float = -0.1, high: float = 0.1) -> InitSpec:
        return InitSpec(kind="uniform", params={"low": low, "high": high})

    @staticmethod
    def constant(value: float) -> InitSpec:
        return InitSpec(kind="constant", params={"value": value})


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    populations: tuple[PopulationSpec, ...]
    projections: tuple[ProjectionSpec, ...]
    modulators: tuple[ModulatorSpec, ...]
    monitors: tuple[Any, ...] = ()


@dataclass(slots=True)
class _PopulationDraft:
    name: str
    n: int
    neuron: INeuronModel
    tags: Mapping[str, Any] | None


@dataclass(slots=True)
class _ProjectionDraft:
    name: str
    pre: str
    post: str
    synapse: ISynapseModel
    topology: SynapseTopology | TopologySpec | TopologyFactory
    weights: InitSpec | Tensor | WeightsFactory | None
    compartments: Compartment | str | Tensor | Sequence[Compartment | str] | None
    receptor_map: Tensor | Sequence[ReceptorKind | str] | None
    meta: Mapping[str, Any] | None


class NetworkBuilder:
    """Fluent builder for network specifications."""

    def __init__(self) -> None:
        self._device: str | None = None
        self._dtype: Any | None = None
        self._seed: int | None = None
        self._seed_offset = 0
        self._populations: dict[str, _PopulationDraft] = {}
        self._projections: list[_ProjectionDraft] = []
        self._learning: dict[str, ILearningRule] = {}
        self._modulators: list[ModulatorSpec] = []
        self._monitors: list[Any] = []

    def device(self, device: str | Any) -> NetworkBuilder:
        self._device = str(device)
        return self

    def dtype(self, dtype: str | Any) -> NetworkBuilder:
        self._dtype = dtype
        return self

    def seed(self, seed: int) -> NetworkBuilder:
        self._seed = int(seed)
        return self

    def population(
        self,
        name: str,
        *,
        n: int,
        neuron: INeuronModel,
        tags: Mapping[str, Any] | None = None,
    ) -> NetworkBuilder:
        if name in self._populations:
            raise ValueError(f"Population '{name}' already exists")
        self._populations[name] = _PopulationDraft(name=name, n=n, neuron=neuron, tags=tags)
        return self

    def projection(
        self,
        pre: str,
        post: str,
        *,
        name: str | None = None,
        synapse: ISynapseModel,
        topology: SynapseTopology | TopologySpec | TopologyFactory,
        weights: InitSpec | Tensor | WeightsFactory | None = None,
        compartments: Compartment | str | Tensor | Sequence[Compartment | str] | None = None,
        receptor_map: Tensor | Sequence[ReceptorKind | str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> NetworkBuilder:
        proj_name = name or f"{pre}->{post}"
        self._projections.append(
            _ProjectionDraft(
                name=proj_name,
                pre=pre,
                post=post,
                synapse=synapse,
                topology=topology,
                weights=weights,
                compartments=compartments,
                receptor_map=receptor_map,
                meta=meta,
            )
        )
        return self

    def learning(self, projection: str | tuple[str, str], *, rule: ILearningRule) -> NetworkBuilder:
        proj_name = projection if isinstance(projection, str) else f"{projection[0]}->{projection[1]}"
        self._learning[proj_name] = rule
        return self

    def modulator(
        self,
        name: str,
        *,
        field: IModulatorField,
        kinds: Sequence[ModulatorKind],
        tags: Mapping[str, Any] | None = None,
    ) -> NetworkBuilder:
        self._modulators.append(
            ModulatorSpec(name=name, field=field, kinds=tuple(kinds), meta=tags)
        )
        return self

    def monitor(self, monitor: Any) -> NetworkBuilder:
        self._monitors.append(monitor)
        return self

    def build(self) -> NetworkSpec:
        if not self._populations:
            raise ValueError("No populations defined")

        torch = require_torch()
        _ = torch

        populations = []
        for pop_draft in self._populations.values():
            populations.append(
                PopulationSpec(
                    name=pop_draft.name,
                    model=pop_draft.neuron,
                    n=pop_draft.n,
                    positions=None,
                    meta=pop_draft.tags,
                )
            )

        projections: list[ProjectionSpec] = []
        used_names: set[str] = set()
        for proj_draft in self._projections:
            if proj_draft.pre not in self._populations:
                raise ValueError(
                    f"Projection '{proj_draft.name}' pre population '{proj_draft.pre}' not found"
                )
            if proj_draft.post not in self._populations:
                raise ValueError(
                    f"Projection '{proj_draft.name}' post population '{proj_draft.post}' not found"
                )
            if proj_draft.name in used_names:
                raise ValueError(f"Projection name '{proj_draft.name}' already used")
            used_names.add(proj_draft.name)

            n_pre = self._populations[proj_draft.pre].n
            n_post = self._populations[proj_draft.post].n
            topology = self._resolve_topology(proj_draft.topology, n_pre, n_post)
            topology = self._apply_compartments(topology, proj_draft.compartments)
            topology = self._apply_receptors(topology, proj_draft.receptor_map)
            topology = self._apply_weights(topology, proj_draft.weights)
            _validate_topology_dims(topology, n_pre, n_post, proj_draft.name)

            projections.append(
                ProjectionSpec(
                    name=proj_draft.name,
                    synapse=proj_draft.synapse,
                    topology=topology,
                    pre=proj_draft.pre,
                    post=proj_draft.post,
                    learning=self._learning.get(proj_draft.name),
                    meta=proj_draft.meta,
                )
            )

        return NetworkSpec(
            populations=tuple(populations),
            projections=tuple(projections),
            modulators=tuple(self._modulators),
            monitors=tuple(self._monitors),
        )

    def _resolve_topology(
        self, topology: SynapseTopology | TopologySpec | TopologyFactory, n_pre: int, n_post: int
    ) -> SynapseTopology:
        if isinstance(topology, SynapseTopology):
            return topology
        if hasattr(topology, "build"):
            return topology.build(
                n_pre,
                n_post,
                device=self._device,
                dtype=self._dtype,
                seed=self._next_seed(),
            )
        if callable(topology):
            try:
                return topology(
                    n_pre,
                    n_post,
                    device=self._device,
                    dtype=self._dtype,
                    seed=self._next_seed(),
                )
            except TypeError:
                return topology(n_pre, n_post)
        raise ValueError("Unsupported topology type")

    def _apply_weights(
        self, topology: SynapseTopology, weights: InitSpec | Tensor | WeightsFactory | None
    ) -> SynapseTopology:
        if weights is None:
            return topology
        torch = require_torch()
        e = topology.pre_idx.numel()
        if e == 0:
            return topology
        values: Tensor
        if isinstance(weights, InitSpec):
            values = weights.build(e, device=self._device, dtype=self._dtype, seed=self._next_seed())
        elif hasattr(weights, "to") and hasattr(weights, "shape"):
            values = cast(Tensor, weights)
        elif callable(weights):
            try:
                values = weights(
                    e,
                    device=self._device,
                    dtype=self._dtype,
                    seed=self._next_seed(),
                )
            except TypeError:
                values = weights(e)
            values = cast(Tensor, values)
        else:
            raise ValueError("Unsupported weights spec")

        if hasattr(values, "numel") and values.numel() != e:
            raise ValueError(f"Weights size mismatch: expected {e}, got {values.numel()}")

        values = values.to(
            device=torch.device(self._device) if self._device else None,
            dtype=_resolve_dtype(torch, self._dtype),
        )
        return _replace_topology(topology, weights=values)

    def _apply_compartments(
        self, topology: SynapseTopology, compartments: Compartment | str | Tensor | Sequence[Compartment | str] | None
    ) -> SynapseTopology:
        if compartments is None:
            return topology
        torch = require_torch()
        e = topology.pre_idx.numel()
        if isinstance(compartments, Compartment):
            return _replace_topology(topology, target_compartment=compartments)
        if isinstance(compartments, str):
            return _replace_topology(topology, target_compartment=_coerce_compartment(compartments))
        if hasattr(compartments, "to"):
            if hasattr(compartments, "numel") and compartments.numel() != e:
                raise ValueError("compartments tensor length must match number of edges")
            comp_tensor = compartments.to(
                device=torch.device(self._device) if self._device else None, dtype=torch.long
            )
            return _replace_topology(topology, target_compartments=comp_tensor)
        if isinstance(compartments, Sequence):
            if len(compartments) != e:
                raise ValueError("compartments length must match number of edges")
            comp_ids = torch.tensor(
                [_COMPARTMENT_TO_ID[_coerce_compartment(value)] for value in compartments],
                device=torch.device(self._device) if self._device else None,
                dtype=torch.long,
            )
            return _replace_topology(topology, target_compartments=comp_ids)
        raise ValueError("Unsupported compartments spec")

    def _apply_receptors(
        self, topology: SynapseTopology, receptor_map: Tensor | Sequence[ReceptorKind | str] | None
    ) -> SynapseTopology:
        if receptor_map is None:
            return topology
        torch = require_torch()
        e = topology.pre_idx.numel()
        if hasattr(receptor_map, "to"):
            if hasattr(receptor_map, "numel") and receptor_map.numel() != e:
                raise ValueError("receptor_map tensor length must match number of edges")
            receptor = receptor_map.to(
                device=torch.device(self._device) if self._device else None, dtype=torch.long
            )
            return _replace_topology(topology, receptor=receptor)
        if isinstance(receptor_map, Sequence):
            if len(receptor_map) != e:
                raise ValueError("receptor_map length must match number of edges")
            receptor_kinds: list[ReceptorKind] = []
            receptor_ids: list[int] = []
            for value in receptor_map:
                kind = _coerce_receptor(value)
                if kind not in receptor_kinds:
                    receptor_kinds.append(kind)
                receptor_ids.append(receptor_kinds.index(kind))
            receptor = torch.tensor(
                receptor_ids,
                device=torch.device(self._device) if self._device else None,
                dtype=torch.long,
            )
            return _replace_topology(
                topology,
                receptor=receptor,
                receptor_kinds=tuple(receptor_kinds),
            )
        raise ValueError("Unsupported receptor_map spec")

    def _next_seed(self) -> int | None:
        if self._seed is None:
            return None
        value = self._seed + self._seed_offset
        self._seed_offset += 1
        return value


def _resolve_dtype(torch: Any, dtype: Any | None) -> Any:
    if dtype is None:
        return torch.get_default_dtype()
    if isinstance(dtype, str):
        if hasattr(torch, dtype):
            return getattr(torch, dtype)
        raise ValueError(f"Unknown dtype '{dtype}'")
    return dtype


def _coerce_compartment(value: Compartment | str) -> Compartment:
    if isinstance(value, Compartment):
        return value
    try:
        return Compartment(value)
    except Exception as exc:
        raise ValueError(f"Unknown compartment '{value}'") from exc


def _coerce_receptor(value: ReceptorKind | str) -> ReceptorKind:
    if isinstance(value, ReceptorKind):
        return value
    try:
        return ReceptorKind(value)
    except Exception as exc:
        raise ValueError(f"Unknown receptor '{value}'") from exc


def _replace_topology(topology: SynapseTopology, **kwargs: Any) -> SynapseTopology:
    data: dict[str, Any] = {
        "pre_idx": topology.pre_idx,
        "post_idx": topology.post_idx,
        "delay_steps": topology.delay_steps,
        "edge_dist": topology.edge_dist,
        "target_compartment": topology.target_compartment,
        "target_compartments": topology.target_compartments,
        "receptor": topology.receptor,
        "receptor_kinds": topology.receptor_kinds,
        "weights": topology.weights,
        "pre_pos": topology.pre_pos,
        "post_pos": topology.post_pos,
        "myelin": topology.myelin,
        "meta": topology.meta,
    }
    data.update(kwargs)
    return SynapseTopology(**cast(dict[str, Any], data))


def _validate_topology_dims(topology: SynapseTopology, n_pre: int, n_post: int, name: str) -> None:
    if topology.pre_idx.numel() > 0 and int(topology.pre_idx.max().detach().cpu()) >= n_pre:
        raise ValueError(f"Projection '{name}' pre_idx exceeds n_pre={n_pre}")
    if topology.post_idx.numel() > 0 and int(topology.post_idx.max().detach().cpu()) >= n_post:
        raise ValueError(f"Projection '{name}' post_idx exceeds n_post={n_post}")
