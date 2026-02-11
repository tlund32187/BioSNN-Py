"""Minimal torch-backed simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, SupportsInt, cast

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.monitors import IMonitor, Scalar, StepEvent
from biosnn.contracts.neurons import Compartment, INeuronModel, NeuronInputs, StepContext
from biosnn.contracts.simulation import ISimulationEngine, SimulationConfig
from biosnn.contracts.synapses import ISynapseModel, SynapseInputs, SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype

DriveFn = Callable[[float, int, StepContext], Mapping[Compartment, Tensor]]


class TorchSimulationEngine(ISimulationEngine):
    """Minimal single-population simulation engine using torch."""

    name = "torch"

    def __init__(
        self,
        *,
        neuron_model: INeuronModel,
        synapse_model: ISynapseModel,
        topology: SynapseTopology,
        n: int,
        drive_fn: DriveFn | None = None,
    ) -> None:
        self._neuron_model = neuron_model
        self._synapse_model = synapse_model
        self._topology = topology
        self._n = n
        self._drive_fn = drive_fn

        self._ctx = StepContext()
        self._dt = 0.0
        self._t = 0.0
        self._step = 0
        self._device = None
        self._dtype = None
        self._n_pre: int | None = None

        self._neuron_state: Any = None
        self._synapse_state: Any = None
        self._spikes: Tensor | None = None
        self._monitors: list[IMonitor] = []

        self.last_post_drive: Mapping[Compartment, Tensor] | None = None
        self.last_spikes: Tensor | None = None

    def reset(self, *, config: SimulationConfig) -> None:
        torch = require_torch()

        self._dt = float(config.dt)
        self._t = 0.0
        self._step = 0
        self._ctx = StepContext(
            device=config.device,
            dtype=config.dtype,
            seed=config.seed,
            is_training=True,
            extras=config.meta or None,
        )
        self._device, self._dtype = resolve_device_dtype(self._ctx)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

        self._neuron_state = self._neuron_model.init_state(self._n, ctx=self._ctx)

        edge_count = _edge_count(self._topology)
        self._synapse_state = self._synapse_model.init_state(edge_count, ctx=self._ctx)

        self._spikes = torch.zeros((self._n,), device=self._device, dtype=torch.bool)
        _apply_initial_spikes(self._spikes, config.meta)

        topology = self._topology
        if topology.weights is None and getattr(self._synapse_state, "bind_weights_to_topology", False):
            topology = replace(topology, weights=self._synapse_state.weights)
        _copy_topology_weights(self._synapse_state, topology.weights)
        topology = _ensure_topology_meta(topology, n_pre=self._n, n_post=self._n)
        build_edges_by_delay = False
        build_pre_adjacency = False
        build_sparse_delay_mats = False
        build_bucket_edge_mapping = False
        reqs = None
        if hasattr(self._synapse_model, "compilation_requirements"):
            try:
                reqs = self._synapse_model.compilation_requirements()
            except Exception:
                reqs = None
        if isinstance(reqs, Mapping):
            build_edges_by_delay = bool(reqs.get("needs_edges_by_delay", False))
            build_pre_adjacency = bool(reqs.get("needs_pre_adjacency", False))
            build_sparse_delay_mats = bool(reqs.get("needs_sparse_delay_mats", False))
            build_bucket_edge_mapping = bool(reqs.get("needs_bucket_edge_mapping", False))

        if build_sparse_delay_mats and hasattr(self._synapse_model, "params"):
            params = getattr(self._synapse_model, "params", None)
            receptor_scale = getattr(params, "receptor_scale", None) if params is not None else None
            if receptor_scale is not None:
                meta = dict(topology.meta) if topology.meta else {}
                meta.setdefault("receptor_scale", receptor_scale)
                topology = replace(topology, meta=meta)

        topology = compile_topology(
            topology,
            device=self._device,
            dtype=self._dtype,
            build_edges_by_delay=build_edges_by_delay,
            build_pre_adjacency=build_pre_adjacency,
            build_sparse_delay_mats=build_sparse_delay_mats,
            build_bucket_edge_mapping=build_bucket_edge_mapping,
            fuse_delay_buckets=build_sparse_delay_mats,
        )
        self._topology = topology
        self._n_pre = _infer_n_pre(self._topology)

    def attach_monitors(self, monitors: Sequence[IMonitor]) -> None:
        """Replace the current monitor list with the provided sequence."""

        self._monitors = list(monitors)

    def step(self) -> Mapping[str, Any]:
        if self._spikes is None:
            raise RuntimeError("Engine must be reset before stepping.")

        torch = require_torch()
        pre_spikes = self._spikes
        weights = getattr(self._synapse_state, "weights", None)
        if weights is not None and hasattr(pre_spikes, "to") and pre_spikes.device != weights.device:
            pre_spikes = pre_spikes.to(device=weights.device)

        n_pre = self._n_pre
        if n_pre is not None and hasattr(pre_spikes, "shape"):
            if pre_spikes.shape[0] < n_pre:
                raise ValueError(f"pre_spikes must have at least {n_pre} entries")
            if pre_spikes.shape[0] > n_pre:
                pre_spikes = pre_spikes[:n_pre]

        syn_inputs_meta: Mapping[str, Any] | None = None
        if _synapse_needs_post_membrane(self._synapse_model):
            membrane = _extract_membrane_views(self._neuron_model.state_tensors(self._neuron_state))
            if not membrane:
                raise RuntimeError(
                    "Synapse requires post membrane tensors but neuron model does not expose them."
                )
            syn_inputs_meta = {"post_membrane": membrane}

        self._synapse_state, synapse_result = self._synapse_model.step(
            self._synapse_state,
            self._topology,
            SynapseInputs(pre_spikes=pre_spikes, meta=syn_inputs_meta),
            dt=self._dt,
            t=self._t,
            ctx=self._ctx,
        )

        combined_drive = _merge_drive(
            synapse_result.post_drive,
            _call_drive_fn(self._drive_fn, self._t, self._step, self._ctx),
            device=self._device,
            dtype=self._dtype,
        )

        self._neuron_state, neuron_result = self._neuron_model.step(
            self._neuron_state,
            NeuronInputs(drive=combined_drive),
            dt=self._dt,
            t=self._t,
            ctx=self._ctx,
        )

        spikes_next = neuron_result.spikes
        spikes_next_bool = spikes_next if spikes_next.dtype == torch.bool else (spikes_next > 0)
        if self._spikes is not None and self._spikes.shape == spikes_next_bool.shape:
            self._spikes.copy_(spikes_next_bool.to(device=self._spikes.device))
        else:
            self._spikes = spikes_next_bool.to(device=self._spikes.device)

        if self._spikes.numel():
            spike_count = self._spikes.sum()
            spike_fraction = spike_count / float(self._spikes.numel())
        else:
            spike_count = self._spikes.new_zeros(())
            spike_fraction = torch.zeros((), device=self._spikes.device, dtype=torch.float32)

        scalars: dict[str, Scalar] = {
            "step": float(self._step),
            "spike_count": spike_count,
            "spike_fraction": spike_fraction,
        }

        event = StepEvent(
            t=self._t,
            dt=self._dt,
            spikes=self._spikes,
            tensors=_merge_tensors(
                self._neuron_model.state_tensors(self._neuron_state),
                self._synapse_model.state_tensors(self._synapse_state),
            ),
            scalars=cast(Mapping[str, Scalar], scalars),
        )

        for monitor in self._monitors:
            monitor.on_step(event)

        self.last_post_drive = synapse_result.post_drive
        self.last_spikes = self._spikes

        self._t += self._dt
        self._step += 1

        return {
            "t": float(event.t),
            "t_next": float(self._t),
            "step": float(self._step),
            "spike_count": spike_count,
            "spike_fraction": spike_fraction,
        }

    def run(self, steps: int) -> None:
        try:
            for _ in range(steps):
                self.step()
        finally:
            for monitor in self._monitors:
                monitor.flush()
            for monitor in self._monitors:
                monitor.close()


def _edge_count(topology: SynapseTopology) -> int:
    pre_idx = topology.pre_idx
    if hasattr(pre_idx, "numel"):
        return int(pre_idx.numel())
    try:
        return len(pre_idx)
    except TypeError:
        return 0


def _apply_initial_spikes(spikes: Tensor, meta: Mapping[str, Any] | None) -> None:
    if meta is None:
        return
    indices = meta.get("initial_spike_indices")
    if indices is None:
        indices = meta.get("initial_spikes")
    if indices is None:
        return

    if hasattr(indices, "numel"):
        if indices.numel() == spikes.numel():
            spikes.copy_(indices.to(device=spikes.device, dtype=spikes.dtype))
            return
        idx_tensor = indices.to(device=spikes.device, dtype=require_torch().long)
        spikes[idx_tensor] = True
        return

    if isinstance(indices, (list, tuple)):
        for idx in indices:
            spikes[int(idx)] = True
        return

    spikes[int(indices)] = True


def _copy_topology_weights(state: Any, weights: Tensor | None) -> None:
    if weights is None or not hasattr(state, "weights"):
        if weights is None and getattr(state, "bind_weights_to_topology", False):
            return
        return
    state_weights = state.weights
    if not hasattr(state_weights, "shape"):
        return
    if getattr(state, "bind_weights_to_topology", False):
        state.weights = weights
        return
    if state_weights.shape != weights.shape:
        return
    if hasattr(weights, "to"):
        weights = weights.to(device=state_weights.device, dtype=state_weights.dtype)
    state_weights.copy_(weights)


def _call_drive_fn(
    drive_fn: DriveFn | None,
    t: float,
    step_index: int,
    ctx: StepContext,
) -> Mapping[Compartment, Tensor] | None:
    if drive_fn is None:
        return None
    return drive_fn(t, step_index, ctx)


def _merge_drive(
    base: Mapping[Compartment, Tensor],
    extra: Mapping[Compartment, Tensor] | None,
    *,
    device: Any,
    dtype: Any,
) -> Mapping[Compartment, Tensor]:
    torch = require_torch()
    merged: dict[Compartment, Tensor] = {}

    for comp, tensor in base.items():
        merged[comp] = tensor

    if extra is None:
        return merged

    for comp, tensor in extra.items():
        if comp in merged:
            other = tensor
            if hasattr(other, "to"):
                other = other.to(device=merged[comp].device, dtype=merged[comp].dtype)
            merged[comp] = merged[comp] + other
        else:
            if hasattr(tensor, "to"):
                tensor = tensor.to(device=device, dtype=dtype)
            else:
                tensor = torch.tensor(tensor, device=device, dtype=dtype)
            merged[comp] = tensor

    return merged


def _merge_tensors(
    neuron_tensors: Mapping[str, Tensor],
    synapse_tensors: Mapping[str, Tensor],
) -> Mapping[str, Tensor]:
    merged: dict[str, Tensor] = {}
    merged.update(neuron_tensors)
    for key, value in synapse_tensors.items():
        if key in merged:
            merged[f"synapse_{key}"] = value
        else:
            merged[key] = value
    return merged


def _synapse_needs_post_membrane(synapse: ISynapseModel) -> bool:
    reqs = None
    if hasattr(synapse, "compilation_requirements"):
        try:
            reqs = synapse.compilation_requirements()
        except Exception:
            reqs = None
    if isinstance(reqs, Mapping) and "needs_post_membrane" in reqs:
        return bool(reqs.get("needs_post_membrane"))
    params = getattr(synapse, "params", None)
    return bool(getattr(params, "conductance_mode", False))


def _extract_membrane_views(tensors: Mapping[str, Tensor]) -> dict[Compartment, Tensor]:
    membrane: dict[Compartment, Tensor] = {}

    v_soma = tensors.get("v_soma")
    if v_soma is not None:
        membrane[Compartment.SOMA] = v_soma
    v_dend = tensors.get("v_dend")
    if v_dend is not None:
        membrane[Compartment.DENDRITE] = v_dend
    v_axon = tensors.get("v_axon")
    if v_axon is not None:
        membrane[Compartment.AXON] = v_axon
    v_ais = tensors.get("v_ais")
    if v_ais is not None:
        membrane[Compartment.AIS] = v_ais

    packed = tensors.get("v")
    if packed is not None and packed.dim() == 2:
        cols = int(packed.shape[1])
        if cols >= 1 and Compartment.SOMA not in membrane:
            membrane[Compartment.SOMA] = packed[:, 0]
        if cols >= 2 and Compartment.DENDRITE not in membrane:
            membrane[Compartment.DENDRITE] = packed[:, 1]
        if cols >= 3 and Compartment.AXON not in membrane:
            membrane[Compartment.AXON] = packed[:, 2]
    return membrane


__all__ = ["TorchSimulationEngine"]


def _infer_n_pre(topology: SynapseTopology) -> int | None:
    if topology.meta and "n_pre" in topology.meta:
        return int(topology.meta["n_pre"])
    pre_idx = topology.pre_idx
    if hasattr(pre_idx, "numel"):
        if pre_idx.numel():
            max_val = pre_idx.detach()
            if hasattr(max_val, "max"):
                max_val = max_val.max()
            if hasattr(max_val, "cpu"):
                max_val = max_val.cpu()
            if hasattr(max_val, "tolist"):
                max_list = max_val.tolist()
                if isinstance(max_list, list):
                    scalar = max_list[0] if max_list else 0
                    return int(cast(SupportsInt, scalar)) + 1
                return int(cast(SupportsInt, max_list)) + 1
            return int(max_val) + 1
        return None
    try:
        return int(max(pre_idx)) + 1
    except Exception:
        return None


def _ensure_topology_meta(
    topology: SynapseTopology,
    *,
    n_pre: int | None = None,
    n_post: int | None = None,
) -> SynapseTopology:
    meta = dict(topology.meta) if topology.meta else {}
    updated = False
    if n_pre is not None and "n_pre" not in meta:
        meta["n_pre"] = int(n_pre)
        updated = True
    if n_post is not None and "n_post" not in meta:
        meta["n_post"] = int(n_post)
        updated = True

    if topology.delay_steps is not None and "max_delay_steps" not in meta:
        delay_steps = topology.delay_steps
        max_delay = 0
        if hasattr(delay_steps, "numel") and delay_steps.numel():
            max_val = delay_steps.detach()
            if hasattr(max_val, "max"):
                max_val = max_val.max()
            if hasattr(max_val, "cpu"):
                max_val = max_val.cpu()
            if hasattr(max_val, "tolist"):
                max_list = max_val.tolist()
                if isinstance(max_list, list):
                    scalar = max_list[0] if max_list else 0
                    max_delay = int(cast(SupportsInt, scalar))
                else:
                    max_delay = int(cast(SupportsInt, max_list))
            else:
                max_delay = int(max_val)
        meta["max_delay_steps"] = max_delay
        updated = True

    if updated:
        return replace(topology, meta=meta)
    return topology
