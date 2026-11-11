from __future__ import annotations

import numpy as np

from .model import fit_ability


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _probe_mask(bench, depths):
    depths = np.asarray(depths, dtype=int)
    if len(bench.task_of) == 0:
        return np.zeros(0, dtype=bool)
    return bench.position < depths[bench.task_of]


def _rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(100.0 * numerator / denominator)


def score(bench, fold, depths) -> dict:
    model_index = bench.models.index(fold.model)
    passes = bench.passes[model_index]
    runs = bench.runs[model_index]
    on_track = bench.on_track[model_index]
    reached = bench.reached[model_index]

    probe = _probe_mask(bench, depths)
    other = ~probe
    d = np.asarray(fold.d, dtype=float)
    e = np.asarray(fold.e, dtype=float)

    ability = float(fit_ability(d[probe], passes[probe], runs[probe]))
    on_track_ability = float(fit_ability(e[probe], on_track[probe], reached[probe]))

    total_runs = float(runs.sum())
    step_rate = _rate(float(passes.sum()), total_runs)
    step_rate_hat = _rate(
        float(passes[probe].sum()) + float(np.sum(runs[other] * _sigmoid(ability + d[other]))),
        total_runs,
    )

    task_starts = bench.start[:-1]
    task_lasts = bench.start[1:] - 1
    first_runs = float(runs[task_starts].sum())
    solve_rate = _rate(float(on_track[task_lasts].sum()), first_runs)

    solve_num = 0.0
    solve_den = 0.0
    for task_index in range(len(bench.tasks)):
        first = int(bench.start[task_index])
        last = int(bench.start[task_index + 1])
        length = last - first
        if length == 0:
            continue
        solve_den += float(runs[first])
        depth = int(depths[task_index]) if task_index < len(depths) else 0
        depth = max(0, min(depth, length))
        if depth == 0:
            clean = float(runs[first])
            after = np.arange(first, last, dtype=int)
        else:
            clean = float(on_track[first + depth - 1])
            after = np.arange(first + depth, last, dtype=int)
        factor = float(np.prod(_sigmoid(on_track_ability + e[after]))) if after.size else 1.0
        solve_num += clean * factor
    solve_rate_hat = _rate(solve_num, solve_den)

    if float(runs[other].sum()) == 0:
        logloss = float('nan')
    else:
        q = np.clip(_sigmoid(ability + d[other]), 1e-15, 1.0 - 1e-15)
        logloss = float(
            -np.sum(passes[other] * np.log(q) + (runs[other] - passes[other]) * np.log1p(-q))
            / float(runs[other].sum())
        )

    cost = np.asarray(fold.cost, dtype=float)
    total_cost = float(np.sum(cost))
    spend = float(np.sum(cost[probe]) / total_cost) if total_cost != 0 else 0.0

    return {
        'ability': ability,
        'on_track_ability': on_track_ability,
        'step_rate': step_rate,
        'step_rate_hat': step_rate_hat,
        'solve_rate': solve_rate,
        'solve_rate_hat': solve_rate_hat,
        'logloss': logloss,
        'probed_steps': int(probe.sum()),
        'probed_tasks': int(np.sum(np.asarray(depths) > 0)),
        'spend': spend,
    }
