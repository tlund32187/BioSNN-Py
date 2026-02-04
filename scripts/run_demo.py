"""Minimal demo using the public biosnn.api entrypoint."""

from __future__ import annotations

from biosnn.api import (
    DelayedCurrentParams,
    DelayedCurrentSynapse,
    ErdosRenyi,
    GLIFModel,
    NetworkBuilder,
    Trainer,
    presets,
)


def main() -> None:
    net = (
        NetworkBuilder()
        .device("cpu")
        .dtype("float32")
        .seed(123)
        .population("sensory", n=64, neuron=GLIFModel())
        .population("motor", n=8, neuron=GLIFModel())
        .projection(
            "sensory",
            "motor",
            synapse=DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.02)),
            topology=ErdosRenyi(p=0.2),
        )
        .build()
    )

    Trainer(net).run(steps=10, log_every=5)

    # Optional preset example (kept to show available entrypoint)
    _ = presets.make_two_pop_delayed_demo(device="cpu")


if __name__ == "__main__":
    main()
