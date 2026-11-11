"""Held-out-model folds: step difficulty, cost and information from the other models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Benchmark
from .model import logit_rate, step_information, task_variance


@dataclass(frozen=True)
class Fold:
    """What a planner is allowed to know about a model whose results it never sees."""

    model: str
    d: np.ndarray
    e: np.ndarray
    cost: np.ndarray
    info: np.ndarray
    tau2: float


def others_of(bench: Benchmark, model: str) -> list[int]:
    """Indices of every model except the held-out one."""
    return [i for i, name in enumerate(bench.models) if name != model]


def make_fold(bench: Benchmark, model: str) -> Fold:
    """Pool the other models' counts and derive the held-out model's fold."""
    others = others_of(bench, model)
    passes = bench.passes[others].sum(axis=0)
    runs = bench.runs[others].sum(axis=0)
    on_track = bench.on_track[others].sum(axis=0)
    reached = bench.reached[others].sum(axis=0)
    seconds = bench.seconds[others].sum(axis=0)
    timed = bench.timed[others].sum(axis=0)

    d = np.asarray(logit_rate(passes, runs), dtype=float)
    e = np.asarray(logit_rate(on_track, reached), dtype=float)
    cost = np.divide(seconds, timed, out=np.zeros_like(seconds, dtype=float), where=timed > 0)

    return Fold(
        model=model,
        d=d,
        e=e,
        cost=cost,
        info=np.asarray(step_information(bench, d, e), dtype=float),
        tau2=float(task_variance(bench, others, d)),
    )
