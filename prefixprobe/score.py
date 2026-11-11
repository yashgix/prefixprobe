"""Turn a probe plan into estimated and true step and solve rates."""

from __future__ import annotations

import numpy as np

from .data import Benchmark
from .fold import Fold
from .model import fit_ability

_TINY = 1e-12


def _sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def _probe_mask(bench: Benchmark, depths) -> np.ndarray:
    depth_of_step = np.asarray(depths, dtype=int)[bench.task_of]
    return bench.position < depth_of_step


def _solve_rate(bench: Benchmark, model_idx: int) -> float:
    """Runs on track at each task's last step over the runs at each task's first."""
    on_track = bench.on_track[model_idx]
    clean = sum(float(on_track[int(bench.start[t + 1]) - 1]) for t in range(len(bench.tasks)))
    return _percent(clean, float(bench.runs[model_idx].sum()))


def _solve_rate_hat(bench, fold: Fold, depths, model_idx: int, on_track_ability: float) -> float:
    on_track = bench.on_track[model_idx]
    runs = bench.runs[model_idx]
    total = 0.0
    denominators = 0.0
    for t in range(len(bench.tasks)):
        first = int(bench.start[t])
        last = int(bench.start[t + 1]) - 1
        depth = int(min(max(int(depths[t]), 0), last - first + 1))
        denominators += float(runs[first])
        if depth == 0:
            clean = float(runs[first])
        else:
            clean = float(on_track[first + depth - 1])
        future = 1.0
        for j in range(first + depth, last + 1):
            future *= float(_sigmoid(on_track_ability + fold.e[j]))
        total += clean * future
    return _percent(total, denominators)


def score(bench: Benchmark, fold: Fold, depths) -> dict:
    """Score a probe plan for ``fold.model``; its results outside the probe are used only as truth."""
    model_idx = bench.models.index(fold.model)
    passes = bench.passes[model_idx]
    runs = bench.runs[model_idx]
    on_track = bench.on_track[model_idx]
    reached = bench.reached[model_idx]

    probe = _probe_mask(bench, depths)
    other = ~probe
    total_runs = float(runs.sum())

    ability = float(fit_ability(fold.d[probe], passes[probe], runs[probe]))
    on_track_ability = float(fit_ability(fold.e[probe], on_track[probe], reached[probe]))

    predicted = runs[other] * _sigmoid(ability + fold.d[other])
    step_rate_hat = _percent(float(passes[probe].sum()) + float(predicted.sum()), total_runs)

    if other.any() and float(runs[other].sum()) > 0:
        p = np.clip(_sigmoid(ability + fold.d[other]), _TINY, 1.0 - _TINY)
        n = runs[other]
        loss = -(passes[other] * np.log(p) + (n - passes[other]) * np.log1p(-p))
        logloss = float(loss.sum() / float(n.sum()))
    else:
        logloss = float("nan")

    cost_total = float(fold.cost.sum())
    spend = float(fold.cost[probe].sum()) / cost_total if cost_total else float("nan")

    return {
        "ability": ability,
        "on_track_ability": on_track_ability,
        "step_rate": _percent(float(passes.sum()), total_runs),
        "step_rate_hat": step_rate_hat,
        "solve_rate": _solve_rate(bench, model_idx),
        "solve_rate_hat": _solve_rate_hat(bench, fold, depths, model_idx, on_track_ability),
        "logloss": logloss,
        "probed_steps": int(probe.sum()),
        "probed_tasks": int((np.asarray(depths, dtype=int) > 0).sum()),
        "spend": spend,
    }
