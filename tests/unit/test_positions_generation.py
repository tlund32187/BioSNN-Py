from __future__ import annotations

from typing import Literal, cast

import pytest

from biosnn.api.builders.network_builder import NetworkBuilder
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.positions import generate_positions, positions_tensor
from biosnn.simulation.network.specs import PopulationFrame

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("layout", ["grid", "random", "ring", "line"])
def test_positions_tensor_shape_for_layouts(layout: str) -> None:
    layout_literal = cast(Literal["grid", "random", "ring", "line"], layout)
    frame = PopulationFrame(
        origin=(1.0, 2.0, 3.0),
        extent=(4.0, 5.0, 6.0),
        layout=layout_literal,
        seed=11,
    )
    pos = positions_tensor(frame, 17, device="cpu", dtype="float32")
    assert tuple(pos.shape) == (17, 3)


def test_positions_generation_is_deterministic_for_seeded_frame() -> None:
    frame = PopulationFrame(
        origin=(0.0, 0.0, 0.0),
        extent=(1.0, 1.0, 1.0),
        layout="random",
        seed=123,
    )
    a = positions_tensor(frame, 32, device="cpu", dtype="float32")
    b = positions_tensor(frame, 32, device="cpu", dtype="float32")
    assert torch.allclose(a, b)

    rows_a = generate_positions(frame, 32)
    rows_b = generate_positions(frame, 32)
    assert rows_a == rows_b


@pytest.mark.parametrize("layout", ["grid", "random", "line"])
def test_positions_within_frame_bounds(layout: str) -> None:
    layout_literal = cast(Literal["grid", "random", "line"], layout)
    origin = (2.0, -3.0, 5.0)
    extent = (4.0, 6.0, 2.0)
    frame = PopulationFrame(
        origin=origin,
        extent=extent,
        layout=layout_literal,
        seed=19,
    )
    pos = positions_tensor(frame, 25, device="cpu", dtype="float32")

    mins = torch.tensor(
        [min(origin[idx], origin[idx] + extent[idx]) for idx in range(3)],
        dtype=pos.dtype,
    )
    maxs = torch.tensor(
        [max(origin[idx], origin[idx] + extent[idx]) for idx in range(3)],
        dtype=pos.dtype,
    )
    assert torch.all(pos >= (mins - 1e-6))
    assert torch.all(pos <= (maxs + 1e-6))


def test_network_builder_generates_positions_from_population_frame() -> None:
    frame = PopulationFrame(
        origin=(0.0, 0.0, 0.0),
        extent=(1.0, 1.0, 0.0),
        layout="line",
        seed=7,
    )
    spec = (
        NetworkBuilder()
        .device("cpu")
        .dtype("float32")
        .population("input", n=4, neuron=GLIFModel(), frame=frame)
        .build()
    )

    pop = spec.populations[0]
    assert pop.frame == frame
    assert pop.positions is not None
    assert tuple(pop.positions.shape) == (4, 3)


def test_network_builder_population_frame_fluent_method() -> None:
    frame = {
        "origin": (1.0, 2.0, 3.0),
        "extent": (4.0, 5.0, 0.0),
        "layout": "grid",
        "seed": 5,
    }
    spec = (
        NetworkBuilder()
        .device("cpu")
        .dtype("float32")
        .population("hidden", n=6, neuron=GLIFModel())
        .population_frame("hidden", frame)
        .build()
    )

    pop = spec.populations[0]
    assert pop.frame is not None
    assert pop.positions is not None
    assert tuple(pop.positions.shape) == (6, 3)
