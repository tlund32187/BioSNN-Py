import pytest

from biosnn.api import presets
from biosnn.api.training.trainer import Trainer

torch = pytest.importorskip("torch")


def test_trainer_runs_cpu() -> None:
    net = presets.make_two_pop_delayed_demo(device="cpu", seed=4)
    trainer = Trainer(net)
    report = trainer.run(steps=50, log_every=25, progress=False)
    assert report.steps == 50
    assert report.elapsed_s >= 0.0
    assert report.steps_per_sec >= 0.0


def test_trainer_logging_no_monitors() -> None:
    net = presets.make_minimal_logic_gate_network("and", device="cpu", seed=2)
    trainer = Trainer(net)
    report = trainer.run(steps=10, log_every=5, progress=True)
    assert report.steps == 10
