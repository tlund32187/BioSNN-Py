import pytest

from biosnn.api import Trainer, presets

torch = pytest.importorskip("torch")


def test_api_demo_runs_cpu() -> None:
    net = presets.make_two_pop_delayed_demo(device="cpu", seed=11)
    report = Trainer(net).run(steps=10, log_every=5, progress=False)
    assert report.steps == 10
