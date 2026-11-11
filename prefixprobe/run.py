"""Held-out-model experiment: plans, scoring, bootstrap summarization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prefixprobe.data import load
from prefixprobe.fold import make_fold
from prefixprobe.score import score
from prefixprobe.select import make_plan, STRATEGIES

SEED = 20261110
BUDGETS = (0.05, 0.10, 0.20, 0.30)
RANDOM_DRAWS = 200
REPLICATES = 2000
TARGETS = ("step", "solve")
CELLS = [(budget, strategy) for budget in BUDGETS for strategy in STRATEGIES]
PAIRS = [(budget, strategy) for budget in BUDGETS for strategy in STRATEGIES if strategy != "greedy"]


def _mean(values) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _percentile(values):
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _round_numbers(value):
    if isinstance(value, dict):
        return {key: _round_numbers(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_numbers(val) for val in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def errors(bench, fold, strategy: str, budget: float, rng) -> dict:
    plan = make_plan(strategy, bench, fold, budget * float(fold.cost.sum()), rng)
    result = score(bench, fold, plan)
    return {
        "step": result["step_rate_hat"] - result["step_rate"],
        "solve": result["solve_rate_hat"] - result["solve_rate"],
        "logloss": result["logloss"],
        "spend": result["spend"],
        "probed_steps": result["probed_steps"],
        "probed_tasks": result["probed_tasks"],
    }


def grid(bench, run_one):
    folds = {model: make_fold(bench, model) for model in bench.models}
    results = {}
    for budget_index, budget in enumerate(BUDGETS):
        results[budget] = {}
        for strategy in STRATEGIES:
            results[budget][strategy] = {}
            for model_index, model in enumerate(bench.models):
                results[budget][strategy][model] = run_one(
                    folds[model], strategy, budget, model_index, budget_index
                )
    return results


def point_results(bench) -> dict:
    folds = {model: make_fold(bench, model) for model in bench.models}
    point = {}
    for budget_index, budget in enumerate(BUDGETS):
        point[budget] = {}
        for strategy in STRATEGIES:
            point[budget][strategy] = {}
            for model_index, model in enumerate(bench.models):
                draw_count = RANDOM_DRAWS if strategy == "random" else 1
                totals = {key: 0.0 for key in (
                    "step", "solve", "logloss", "spend", "probed_steps", "probed_tasks",
                    "step_abs", "solve_abs",
                )}
                for draw in range(draw_count):
                    rng = np.random.default_rng([SEED, 1, model_index, budget_index, draw])
                    result = errors(bench, folds[model], strategy, budget, rng)
                    for key in ("step", "solve", "logloss", "spend", "probed_steps", "probed_tasks"):
                        totals[key] += result[key]
                    totals["step_abs"] += abs(result["step"])
                    totals["solve_abs"] += abs(result["solve"])
                point[budget][strategy][model] = {
                    key: totals[key] / draw_count for key in totals
                }
    return point


def replicate(bench, replication_index: int) -> dict:
    rng = np.random.default_rng([SEED, 2, replication_index])
    task_count = len(bench.tasks)
    chosen = rng.integers(0, task_count, task_count)
    sample = bench.resample(chosen)
    folds = {model: make_fold(sample, model) for model in sample.models}

    rep = {}
    for budget_index, budget in enumerate(BUDGETS):
        rep[budget] = {}
        for strategy in STRATEGIES:
            rep[budget][strategy] = {}
            for model_index, model in enumerate(sample.models):
                cell_rng = np.random.default_rng([SEED, 3, replication_index, model_index, budget_index])
                rep[budget][strategy][model] = errors(sample, folds[model], strategy, budget, cell_rng)
    return rep


def _cell_mae(point, target: str, budget: float, strategy: str, models) -> float:
    return _mean([point[budget][strategy][model][f"{target}_abs"] for model in models])


def _replicate_mae(rep, target: str, budget: float, strategy: str, models) -> float:
    return _mean([abs(rep[budget][strategy][model][target]) for model in models])


def summarize(bench, point, reps) -> dict:
    models = tuple(bench.models)
    n_models = len(models)
    n_tasks = len(bench.tasks)

    summary = {
        "tasks": int(n_tasks),
        "steps": int(bench.start[-1]),
        "models": int(n_models),
        "replicates": len(reps),
        "random_draws": RANDOM_DRAWS,
        "budgets": list(BUDGETS),
    }

    model_tau2 = []
    model_step_rate = []
    model_solve_rate = []
    zero_plan = np.zeros(n_tasks, dtype=int)
    for model in models:
        fold = make_fold(bench, model)
        result = score(bench, fold, zero_plan)
        model_tau2.append(float(fold.tau2))
        model_step_rate.append(float(result["step_rate"]))
        model_solve_rate.append(float(result["solve_rate"]))
    summary["model_tau2"] = model_tau2
    summary["model_step_rate"] = model_step_rate
    summary["model_solve_rate"] = model_solve_rate

    cell_lists = {
        "logloss": [],
        "spend": [],
        "probed_steps": [],
        "probed_tasks": [],
        "step_mae": [],
        "step_mae_lo": [],
        "step_mae_hi": [],
        "solve_mae": [],
        "solve_mae_lo": [],
        "solve_mae_hi": [],
        "step_err": [],
        "step_err_lo": [],
        "step_err_hi": [],
        "step_abs": [],
        "solve_err": [],
        "solve_err_lo": [],
        "solve_err_hi": [],
        "solve_abs": [],
    }

    for budget, strategy in CELLS:
        point_cells = [point[budget][strategy][model] for model in models]
        cell_lists["logloss"].append(_mean([cell["logloss"] for cell in point_cells]))
        cell_lists["spend"].append(_mean([cell["spend"] for cell in point_cells]))
        cell_lists["probed_steps"].append(_mean([cell["probed_steps"] for cell in point_cells]))
        cell_lists["probed_tasks"].append(_mean([cell["probed_tasks"] for cell in point_cells]))

        step_abs_points = [cell["step_abs"] for cell in point_cells]
        solve_abs_points = [cell["solve_abs"] for cell in point_cells]
        cell_lists["step_mae"].append(_mean(step_abs_points))
        cell_lists["solve_mae"].append(_mean(solve_abs_points))

        step_rep_maes = [_replicate_mae(rep, "step", budget, strategy, models) for rep in reps]
        solve_rep_maes = [_replicate_mae(rep, "solve", budget, strategy, models) for rep in reps]
        step_lo, step_hi = _percentile(step_rep_maes)
        solve_lo, solve_hi = _percentile(solve_rep_maes)
        cell_lists["step_mae_lo"].append(step_lo)
        cell_lists["step_mae_hi"].append(step_hi)
        cell_lists["solve_mae_lo"].append(solve_lo)
        cell_lists["solve_mae_hi"].append(solve_hi)

        for model in models:
            cell = point[budget][strategy][model]
            cell_lists["step_err"].append(cell["step"])
            cell_lists["solve_err"].append(cell["solve"])
            cell_lists["step_abs"].append(cell["step_abs"])
            cell_lists["solve_abs"].append(cell["solve_abs"])

            step_replicate_errs = [rep[budget][strategy][model]["step"] for rep in reps]
            solve_replicate_errs = [rep[budget][strategy][model]["solve"] for rep in reps]
            step_lo, step_hi = _percentile(step_replicate_errs)
            solve_lo, solve_hi = _percentile(solve_replicate_errs)
            cell_lists["step_err_lo"].append(step_lo)
            cell_lists["step_err_hi"].append(step_hi)
            cell_lists["solve_err_lo"].append(solve_lo)
            cell_lists["solve_err_hi"].append(solve_hi)

    summary.update(cell_lists)

    for target in TARGETS:
        diffs = []
        diff_los = []
        diff_his = []
        models_better = []
        share_replicates_better = []
        for budget, other in PAIRS:
            greedy_mae = _cell_mae(point, target, budget, "greedy", models)
            other_mae = _cell_mae(point, target, budget, other, models)
            diffs.append(greedy_mae - other_mae)

            rep_diffs = [
                _replicate_mae(rep, target, budget, "greedy", models)
                - _replicate_mae(rep, target, budget, other, models)
                for rep in reps
            ]
            lo, hi = _percentile(rep_diffs)
            diff_los.append(lo)
            diff_his.append(hi)

            models_better.append(sum(
                point[budget]["greedy"][model][f"{target}_abs"]
                < point[budget][other][model][f"{target}_abs"]
                for model in models
            ))
            share_replicates_better.append(_mean([diff < 0 for diff in rep_diffs]))

        summary[f"greedy_vs_{target}_mae_diff"] = diffs
        summary[f"greedy_vs_{target}_mae_diff_lo"] = diff_los
        summary[f"greedy_vs_{target}_mae_diff_hi"] = diff_his
        summary[f"greedy_vs_{target}_models_better"] = models_better
        summary[f"greedy_vs_{target}_share_replicates_better"] = share_replicates_better

    return summary


def main(data_dir, out, replicates: int = REPLICATES) -> None:
    bench = load(Path(data_dir))
    point = point_results(bench)
    reps = [replicate(bench, replication_index) for replication_index in range(replicates)]
    summary = summarize(bench, point, reps)

    rounded = _round_numbers(summary)
    text = json.dumps(rounded, sort_keys=True, indent=1)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
