"""The budgeted-probe experiment: point estimates, task bootstrap and the metric file."""

import json
from pathlib import Path

import numpy as np

from .data import load
from .fold import make_fold
from .score import score
from .select import STRATEGIES, make_plan

SEED = 20261110
BUDGETS = (0.05, 0.10, 0.20, 0.30)
RANDOM_DRAWS = 200
REPLICATES = 2000
TARGETS = ("step", "solve")

CELLS = [(budget, strategy) for budget in BUDGETS for strategy in STRATEGIES]
PAIRS = [(budget, strategy) for budget in BUDGETS for strategy in STRATEGIES if strategy != "greedy"]
FIELDS = ("step", "solve", "logloss", "spend", "probed_steps", "probed_tasks")


def errors(bench, fold, strategy, budget, rng):
    """Plan one probe worth ``budget`` of the full cost, score it, and return its errors."""
    depths = make_plan(strategy, bench, fold, budget * float(fold.cost.sum()), rng)
    result = score(bench, fold, depths)
    return {
        "step": float(result["step_rate_hat"] - result["step_rate"]),
        "solve": float(result["solve_rate_hat"] - result["solve_rate"]),
        "logloss": float(result["logloss"]),
        "spend": float(result["spend"]),
        "probed_steps": int(result["probed_steps"]),
        "probed_tasks": int(result["probed_tasks"]),
    }


def grid(bench, run_one):
    """results[budget][strategy][model], with one fold per model of ``bench``."""
    folds = [make_fold(bench, model) for model in bench.models]
    results = {}
    for j, budget in enumerate(BUDGETS):
        results[budget] = {}
        for strategy in STRATEGIES:
            results[budget][strategy] = {
                model: run_one(folds[i], strategy, budget, i, j) for i, model in enumerate(bench.models)
            }
    return results


def point_results(bench):
    """Real-task point estimates: RANDOM_DRAWS draws for random plans, one draw for every other plan."""

    def run_one(fold, strategy, budget, i, j):
        draws = RANDOM_DRAWS if strategy == "random" else 1
        cells = [
            errors(bench, fold, strategy, budget, np.random.default_rng([SEED, 1, i, j, k])) for k in range(draws)
        ]
        cell = {field: float(np.mean([c[field] for c in cells])) for field in FIELDS}
        cell["step_abs"] = float(np.mean([abs(c["step"]) for c in cells]))
        cell["solve_abs"] = float(np.mean([abs(c["solve"]) for c in cells]))
        return cell

    return grid(bench, run_one)


def replicate(bench, b):
    """One task bootstrap replicate of every cell, with the folds refit on the resampled tasks."""
    tasks = len(bench.start) - 1
    drawn = np.random.default_rng([SEED, 2, b]).integers(0, tasks, tasks)
    sample = bench.resample(drawn)

    def run_one(fold, strategy, budget, i, j):
        return errors(sample, fold, strategy, budget, np.random.default_rng([SEED, 3, b, i, j]))

    return grid(sample, run_one)


def summarize(bench, point, reps):
    """Every reported metric, rounded, in ``CELLS`` and ``PAIRS`` order."""
    models = list(bench.models)
    tasks = len(bench.start) - 1
    folds = [make_fold(bench, model) for model in models]
    plain = [score(bench, fold, np.zeros(tasks, dtype=int)) for fold in folds]

    summary = {
        "tasks": tasks,
        "steps": int(np.shape(bench.passes)[1]),
        "models": len(models),
        "replicates": len(reps),
        "random_draws": RANDOM_DRAWS,
        "budgets": [float(budget) for budget in BUDGETS],
        "model_tau2": [float(fold.tau2) for fold in folds],
        "model_step_rate": [float(result["step_rate"]) for result in plain],
        "model_solve_rate": [float(result["solve_rate"]) for result in plain],
    }

    for field in ("logloss", "spend", "probed_steps", "probed_tasks"):
        summary[field] = [float(np.mean([point[b][s][m][field] for m in models])) for b, s in CELLS]

    for target in TARGETS:
        _add_target(summary, point, reps, models, target)
    for target in TARGETS:
        _add_pairs(summary, point, reps, models, target)

    return _rounded(summary)


def _add_target(summary, point, reps, models, target):
    """The per-cell and per-cell-and-model numbers of one target rate."""
    absolute = f"{target}_abs"
    summary[f"{target}_mae"] = [float(np.mean([point[b][s][m][absolute] for m in models])) for b, s in CELLS]
    summary[f"{target}_err"] = [point[b][s][m][target] for b, s in CELLS for m in models]
    summary[f"{target}_abs"] = [point[b][s][m][absolute] for b, s in CELLS for m in models]

    replicate_mae = [
        [float(np.mean([abs(rep[b][s][m][target]) for m in models])) for b, s in CELLS] for rep in reps
    ]
    replicate_err = [[rep[b][s][m][target] for b, s in CELLS for m in models] for rep in reps]
    summary[f"{target}_mae_lo"], summary[f"{target}_mae_hi"] = _interval(replicate_mae)
    summary[f"{target}_err_lo"], summary[f"{target}_err_hi"] = _interval(replicate_err)


def _add_pairs(summary, point, reps, models, target):
    """Greedy minus each other strategy at the same budget, point and replicate wise."""
    absolute = f"{target}_abs"
    diff = [
        float(
            np.mean([point[b]["greedy"][m][absolute] for m in models])
            - np.mean([point[b][s][m][absolute] for m in models])
        )
        for b, s in PAIRS
    ]
    better = [
        int(sum(1 for m in models if point[b]["greedy"][m][absolute] < point[b][s][m][absolute])) for b, s in PAIRS
    ]
    replicate_diff = [
        [
            float(
                np.mean([abs(rep[b]["greedy"][m][target]) for m in models])
                - np.mean([abs(rep[b][s][m][target]) for m in models])
            )
            for b, s in PAIRS
        ]
        for rep in reps
    ]
    summary[f"greedy_vs_{target}_mae_diff"] = diff
    summary[f"greedy_vs_{target}_mae_diff_lo"], summary[f"greedy_vs_{target}_mae_diff_hi"] = _interval(replicate_diff)
    summary[f"greedy_vs_{target}_models_better"] = better
    summary[f"greedy_vs_{target}_share_replicates_better"] = [
        float(np.mean([row[k] < 0 for row in replicate_diff])) for k in range(len(PAIRS))
    ]


def _interval(series):
    """2.5th and 97.5th percentiles, position by position, NumPy's default interpolation."""
    if not series:
        return [], []
    values = np.asarray(series, dtype=float)
    return (
        np.percentile(values, 2.5, axis=0).tolist(),
        np.percentile(values, 97.5, axis=0).tolist(),
    )


def _rounded(value):
    """Six decimal places everywhere, plain Python types for JSON."""
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def main(data_dir, out, replicates=REPLICATES):
    """Load the data, run the point estimates and ``replicates`` bootstraps, and write the metric file."""
    bench = load(Path(data_dir))
    point = point_results(bench)
    reps = [replicate(bench, b) for b in range(replicates)]
    metrics = summarize(bench, point, reps)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return metrics
