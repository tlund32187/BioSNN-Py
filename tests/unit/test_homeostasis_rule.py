from __future__ import annotations

import pytest

from biosnn.biophysics.models.adex_2c import AdEx2CompModel
from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.homeostasis import HomeostasisPopulation
from biosnn.contracts.neurons import StepContext
from biosnn.learning.homeostasis import (
    RateEmaThresholdHomeostasis,
    RateEmaThresholdHomeostasisConfig,
)

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_rate_ema_threshold_homeostasis_updates_theta_per_neuron() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    model = GLIFModel()
    state = model.init_state(4, ctx=ctx)
    pop = HomeostasisPopulation(name="Pop", model=model, state=state, n=4)

    rule = RateEmaThresholdHomeostasis(
        RateEmaThresholdHomeostasisConfig(
            alpha=0.5,
            eta=0.2,
            r_target=0.0,
            clamp_min=0.0,
            clamp_max=0.02,
            scope="per_neuron",
        )
    )
    rule.init([pop], device="cpu", dtype=torch.float32, ctx=ctx)

    spikes = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    for _ in range(4):
        rule.step({"Pop": spikes}, dt=1e-3, ctx=ctx)

    assert float(state.theta.mean().item()) > 0.0
    assert float(state.theta.max().item()) <= 0.020001


def test_rate_ema_threshold_homeostasis_per_population_produces_uniform_theta() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    model = GLIFModel()
    state = model.init_state(6, ctx=ctx)
    pop = HomeostasisPopulation(name="Pop", model=model, state=state, n=6)

    rule = RateEmaThresholdHomeostasis(
        RateEmaThresholdHomeostasisConfig(
            alpha=0.5,
            eta=0.1,
            r_target=0.0,
            clamp_min=0.0,
            clamp_max=0.05,
            scope="per_population",
        )
    )
    rule.init([pop], device="cpu", dtype=torch.float32, ctx=ctx)

    spikes = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
    for _ in range(3):
        rule.step({"Pop": spikes}, dt=1e-3, ctx=ctx)

    first = state.theta[0].expand_as(state.theta)
    assert torch.allclose(state.theta, first)


def test_rate_ema_threshold_homeostasis_tracks_rate_without_theta_control() -> None:
    ctx = StepContext(device="cpu", dtype="float32")
    model = AdEx2CompModel()
    state = model.init_state(5, ctx=ctx)
    pop = HomeostasisPopulation(name="Hidden", model=model, state=state, n=5)

    rule = RateEmaThresholdHomeostasis(
        RateEmaThresholdHomeostasisConfig(alpha=0.5, eta=0.1, r_target=0.2)
    )
    rule.init([pop], device="cpu", dtype=torch.float32, ctx=ctx)
    spikes = torch.tensor([1, 0, 0, 1, 0], dtype=torch.bool)
    scalars = rule.step({"Hidden": spikes}, dt=1e-3, ctx=ctx)

    assert "homeostasis/rate_mean/Hidden" in scalars
    assert "homeostasis/control_mean/Hidden" in scalars
    assert torch.isnan(scalars["homeostasis/control_mean/Hidden"])
