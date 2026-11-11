"""Scoring a held-out model against a prefix probe."""

from __future__ import annotations

import numpy as np
from scipy.special import expit as sigmoid

from prefixprobe.model import fit_ability


def _probe_mask(bench, depths: np.ndarray) -> np.ndarray:
    depth_per_step = np.repeat(depths, np.diff(bench.start))
    return bench.position < depth_per_step


def score(bench, fold, depths) -> dict:
    """Score the held-out model using only its probed prefixes."""
    depths = np.asarray(depths, dtype=int)
    if depths.shape != (len(bench.tasks),):
        raise ValueError("depths must have one entry per task")

    probe = _probe_mask(bench, depths)
    unprobed = ~probe

    model = bench.models.index(fold.model)
    passes = bench.passes[model]
    runs = bench.runs[model]
    on_track = bench.on_track[model]
    reached = bench.reached[model]

    ability = fit_ability(fold.d[probe], passes[probe], runs[probe])
    on_track_ability = fit_ability(fold.e[probe], on_track[probe], reached[probe])

    total_runs = float(runs.sum())
    total_passes = float(passes.sum())
    step_rate = 100.0 * total_passes / total_runs if total_runs else 0.0

    probed_passes = float(passes[probe].sum())
    predicted_other_passes = float((runs[unprobed] * sigmoid(ability + fold.d[unprobed])).sum())
    step_rate_hat = 100.0 * (probed_passes + predicted_other_passes) / total_runs if total_runs else 0.0

    solve_num = 0.0
    solve_den = 0.0
    solve_hat_num = 0.0

    for task_idx in range(len(bench.tasks)):
        start = int(bench.start[task_idx])
        end = int(bench.start[task_idx + 1])
        task_length = end - start
        first_runs = float(runs[start])
        if first_runs == 0:
            continue

        solve_den += first_runs
        solve_num += float(on_track[end - 1])

        depth = int(depths[task_idx])
        if depth == 0:
            clean = first_runs
            suffix = slice(start, end)
        else:
            last_probed = min(depth, task_length) - 1
            clean = float(on_track[start + last_probed])
            suffix = slice(start + last_probed + 1, end)

        suffix_probability = 1.0
        if suffix.start < suffix.stop:
            suffix_probability = float(np.prod(sigmoid(on_track_ability + fold.e[suffix])))
        solve_hat_num += clean * suffix_probability

    solve_rate = 100.0 * solve_num / solve_den if solve_den else 0.0
    solve_rate_hat = 100.0 * solve_hat_num / solve_den if solve_den else 0.0

    unprobed_runs = runs[unprobed]
    if unprobed.sum() == 0 or unprobed_runs.sum() == 0:
        logloss = float("nan")
    else:
        p = sigmoid(ability + fold.d[unprobed])
        losses = -(passes[unprobed] * np.log(p) + (unprobed_runs - passes[unprobed]) * np.log1p(-p))
        logloss = float(losses.sum() / unprobed_runs.sum())

    denominator = float(fold.cost.sum())
    spend = float(fold.cost[probe].sum() / denominator) if denominator else 0.0

    return {
        "ability": float(ability),
        "on_track_ability": float(on_track_ability),
        "step_rate": float(step_rate),
        "step_rate_hat": float(step_rate_hat),
        "solve_rate": float(solve_rate),
        "solve_rate_hat": float(solve_rate_hat),
        "logloss": logloss,
        "probed_steps": int(probe.sum()),
        "probed_tasks": int(np.count_nonzero(depths > 0)),
        "spend": spend,
    }
