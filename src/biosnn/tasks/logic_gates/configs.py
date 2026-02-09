"""Configuration for logic-gate harness runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from .datasets import LogicGate, SamplingMethod, coerce_gate

LearningMode = Literal["rstdp", "surrogate", "none"]


@dataclass(slots=True)
class LogicGateRunConfig:
    gate: LogicGate | str = LogicGate.AND
    seed: int = 7
    steps: int = 5000
    dt: float = 1e-3
    sim_steps_per_trial: int = 10
    device: str = "cpu"
    learning_mode: LearningMode = "rstdp"
    debug: bool = False
    debug_every: int = 25
    debug_top_k: int = 3
    export_every: int = 25
    dump_last_trials_csv: bool = False
    dump_last_trials_n: int = 50
    sampling_method: SamplingMethod = "sequential"
    out_dir: Path | None = None
    artifacts_root: Path | None = None

    def __post_init__(self) -> None:
        self.gate = coerce_gate(self.gate)
        self.device = str(self.device).lower().strip()
        self.learning_mode = cast_learning_mode(self.learning_mode)
        self.sampling_method = cast_sampling_method(self.sampling_method)
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.sim_steps_per_trial <= 0:
            raise ValueError("sim_steps_per_trial must be > 0")
        if self.debug_every <= 0:
            raise ValueError("debug_every must be > 0")
        if self.debug_top_k <= 0:
            raise ValueError("debug_top_k must be > 0")
        if self.export_every <= 0:
            raise ValueError("export_every must be > 0")
        if self.dump_last_trials_n <= 0:
            raise ValueError("dump_last_trials_n must be > 0")


def cast_learning_mode(value: str) -> LearningMode:
    mode = str(value).lower().strip()
    if mode not in {"rstdp", "surrogate", "none"}:
        raise ValueError("learning_mode must be one of: rstdp, surrogate, none")
    return cast(LearningMode, mode)


def cast_sampling_method(value: str) -> SamplingMethod:
    mode = str(value).lower().strip()
    if mode not in {"sequential", "random_balanced"}:
        raise ValueError("sampling_method must be one of: sequential, random_balanced")
    return cast(SamplingMethod, mode)


__all__ = ["LearningMode", "LogicGateRunConfig", "cast_learning_mode", "cast_sampling_method"]
