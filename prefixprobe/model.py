'''One-parameter item-response utilities for prefix probing.'''
from __future__ import annotations

import numpy as np

from .data import Benchmark

PRIOR_SD = 2.0
RUNS_PER_TASK = 4


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0.0, -x))


def logit_rate(passes, trials):
    p = (np.asarray(passes, dtype=float) + 0.5) / (np.asarray(trials, dtype=float) + 1.0)
    return np.log(p / (1.0 - p))


def fit_ability(d, passes, trials, prior_sd=PRIOR_SD) -> float:
    d = np.asarray(d, dtype=float).ravel()
    passes = np.asarray(passes, dtype=float).ravel()
    trials = np.asarray(trials, dtype=float).ravel()
    if d.size == 0 or passes.size == 0 or trials.size == 0:
        return 0.0
    prior_var = float(prior_sd) ** 2
    a = 0.0
    for _ in range(100):
        p = _sigmoid(a + d)
        grad = float(np.sum(passes - trials * p) - a / prior_var)
        hess = float(np.sum(trials * p * (1.0 - p)) + 1.0 / prior_var)
        step = grad / hess
        a += step
        if abs(step) < 1e-12:
            break
    return float(a)


def task_variance(bench: Benchmark, others, d) -> float:
    d = np.asarray(d, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for i in others:
        i = int(i)
        a = fit_ability(d, bench.passes[i], bench.runs[i])
        p = _sigmoid(a + d)
        for t in range(len(bench.tasks)):
            s = int(bench.start[t])
            e = int(bench.start[t + 1])
            ps = p[s:e]
            r = float(np.sum(bench.passes[i, s:e] - bench.runs[i, s:e] * ps))
            info = float(np.sum(bench.runs[i, s:e] * ps * (1.0 - ps)))
            numerator += r * r - info
            denominator += info * info
    if denominator <= 0.0:
        return 0.0
    return float(max(0.0, numerator / denominator))


def step_information(bench: Benchmark, d, e) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    e = np.asarray(e, dtype=float)
    p = _sigmoid(d)
    q = _sigmoid(e)
    info = np.empty(p.shape, dtype=float)
    for t in range(len(bench.tasks)):
        s = int(bench.start[t])
        end = int(bench.start[t + 1])
        reach = 1.0
        for j in range(s, end):
            info[j] = RUNS_PER_TASK * (p[j] * (1.0 - p[j]) + reach * q[j] * (1.0 - q[j]))
            reach *= q[j]
    return info