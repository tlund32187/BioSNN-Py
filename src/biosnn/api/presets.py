"""Opinionated network presets and engine defaults."""

from __future__ import annotations

from biosnn.api.builders.network_builder import ErdosRenyi, Init, NetworkBuilder, NetworkSpec
from biosnn.api.training.trainer import EngineConfig, Trainer
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse


def default_engine_config(*, compiled_mode: bool = False) -> EngineConfig:
    return EngineConfig(compiled_mode=compiled_mode)


def make_two_pop_delayed_demo(
    *,
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
    n_pre: int = 32,
    n_post: int = 8,
    p: float = 0.25,
    weight_init: float = 0.05,
) -> NetworkSpec:
    builder = NetworkBuilder()
    if device is not None:
        builder = builder.device(device)
    if dtype is not None:
        builder = builder.dtype(dtype)
    if seed is not None:
        builder = builder.seed(seed)

    builder = (
        builder.population("input", n=n_pre, neuron=GLIFModel(), tags={"role": "input"})
        .population("output", n=n_post, neuron=GLIFModel(), tags={"role": "output"})
        .projection(
            "input",
            "output",
            synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=weight_init)),
            topology=ErdosRenyi(p=p),
            weights=Init.constant(weight_init),
        )
    )
    return builder.build()


def make_minimal_logic_gate_network(
    gate: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
    n_input: int = 2,
    n_hidden: int = 8,
    n_output: int = 1,
) -> NetworkSpec:
    gate_name = gate.strip().lower()
    if gate_name not in {"and", "or", "xor"}:
        raise ValueError("gate must be one of: and, or, xor")

    builder = NetworkBuilder()
    if device is not None:
        builder = builder.device(device)
    if dtype is not None:
        builder = builder.dtype(dtype)
    if seed is not None:
        builder = builder.seed(seed)

    builder = builder.population(
        "input",
        n=n_input,
        neuron=GLIFModel(),
        tags={"role": "input", "gate": gate_name},
    )

    if gate_name == "xor":
        builder = (
            builder.population(
                "hidden",
                n=n_hidden,
                neuron=GLIFModel(),
                tags={"role": "hidden", "gate": gate_name},
            )
            .population(
                "output",
                n=n_output,
                neuron=GLIFModel(),
                tags={"role": "output", "gate": gate_name},
            )
            .projection(
                "input",
                "hidden",
                synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05)),
                topology=ErdosRenyi(p=1.0),
            )
            .projection(
                "hidden",
                "output",
                synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05)),
                topology=ErdosRenyi(p=1.0),
            )
        )
    else:
        builder = builder.population(
            "output",
            n=n_output,
            neuron=GLIFModel(),
            tags={"role": "output", "gate": gate_name},
        ).projection(
            "input",
            "output",
            synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05)),
            topology=ErdosRenyi(p=1.0),
        )

    return builder.build()


__all__ = [
    "EngineConfig",
    "Trainer",
    "default_engine_config",
    "make_minimal_logic_gate_network",
    "make_two_pop_delayed_demo",
]
