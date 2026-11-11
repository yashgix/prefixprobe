"""Tests for the experiment runner, on a small synthetic benchmark."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from prefixprobe.data import Benchmark
from prefixprobe.fold import make_fold
from prefixprobe.run import (
    BUDGETS,
    CELLS,
    FIELDS,
    PAIRS,
    RANDOM_DRAWS,
    TARGETS,
    errors,
    grid,
    main,
    point_results,
    replicate,
    summarize,
)
from prefixprobe.select import STRATEGIES

MODELS = ("a", "b", "c")
TASKS = ("t1", "t2")
# SCORES[model][task] is the list of runs of that task, each a list of 0/1 step scores.
SCORES = {
    "a": [[[1, 1], [1, 0]], [[1, 0], [0, 0]]],
    "b": [[[1, 1], [0, 1]], [[0, 0], [0, 0]]],
    "c": [[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
}

BENCH = POINT = REPS = SUMMARY = None


def make_bench():
    """A three-model, two-task benchmark with two runs per step."""
    lengths = [len(SCORES[MODELS[0]][t][0]) for t in range(len(TASKS))]
    start = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    shape = (len(MODELS), int(start[-1]))
    passes, runs = np.zeros(shape), np.zeros(shape)
    on_track, reached = np.zeros(shape), np.zeros(shape)
    for m, model in enumerate(MODELS):
        for t in range(len(TASKS)):
            for run in SCORES[model][t]:
                clean = True
                for k, score in enumerate(run):
                    step = start[t] + k
                    runs[m, step] += 1
                    if clean:
                        reached[m, step] += 1
                    if score:
                        passes[m, step] += 1
                    if clean and score:
                        on_track[m, step] += 1
                    clean = clean and bool(score)
    return Benchmark(
        models=MODELS,
        tasks=TASKS,
        start=start,
        passes=passes,
        runs=runs,
        on_track=on_track,
        reached=reached,
        seconds=np.full(shape, 60.0),
        timed=np.ones(shape),
    )


def setUpModule():
    global BENCH, POINT, REPS, SUMMARY
    BENCH = make_bench()
    POINT = point_results(BENCH)
    REPS = [replicate(BENCH, b) for b in range(2)]
    SUMMARY = summarize(BENCH, POINT, REPS)


def _floats(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _floats(item)
    elif isinstance(value, float):
        yield value


class ErrorsTest(unittest.TestCase):
    def setUp(self):
        self.fold = make_fold(BENCH, "a")

    def test_fields(self):
        result = errors(BENCH, self.fold, "uniform", 0.30, np.random.default_rng(0))
        self.assertEqual(set(result), set(FIELDS))

    def test_no_probe_spends_nothing(self):
        result = errors(BENCH, self.fold, "none", 0.30, np.random.default_rng(0))
        self.assertEqual(result["spend"], 0.0)
        self.assertEqual(result["probed_steps"], 0)
        self.assertEqual(result["probed_tasks"], 0)
        self.assertIsInstance(result["step"], float)

    def test_full_probe_reproduces_the_true_rates(self):
        result = errors(BENCH, self.fold, "cheapest_tasks", 1e6, np.random.default_rng(0))
        self.assertAlmostEqual(result["step"], 0.0, places=9)
        self.assertAlmostEqual(result["solve"], 0.0, places=6)
        self.assertAlmostEqual(result["spend"], 1.0, places=9)
        self.assertEqual(result["probed_tasks"], len(TASKS))
        self.assertEqual(result["probed_steps"], 4)

    def test_same_seed_gives_the_same_errors(self):
        first = errors(BENCH, self.fold, "random", 0.30, np.random.default_rng(7))
        second = errors(BENCH, self.fold, "random", 0.30, np.random.default_rng(7))
        self.assertEqual(first, second)


class GridTest(unittest.TestCase):
    def test_names_every_cell_and_model(self):
        seen = []

        def run_one(fold, strategy, budget, i, j):
            seen.append((fold.model, strategy, budget, i, j))
            return {"step": 0.0}

        results = grid(BENCH, run_one)
        self.assertEqual(list(results), list(BUDGETS))
        for budget_index, budget in enumerate(BUDGETS):
            self.assertEqual(list(results[budget]), list(STRATEGIES))
            for strategy in STRATEGIES:
                self.assertEqual(tuple(results[budget][strategy]), MODELS)
        self.assertEqual(len(seen), len(BUDGETS) * len(STRATEGIES) * len(MODELS))
        for model, strategy, budget, i, j in seen:
            self.assertEqual(i, MODELS.index(model))
            self.assertEqual(j, BUDGETS.index(budget))
            self.assertIn(strategy, STRATEGIES)


class PointTest(unittest.TestCase):
    def test_cells_average_every_field(self):
        cell = POINT[BUDGETS[0]]["greedy"]["b"]
        self.assertEqual(set(cell), set(FIELDS) | {"step_abs", "solve_abs"})

    def test_unprobed_plan_has_no_absolute_gap(self):
        cell = POINT[BUDGETS[0]]["none"]["a"]
        self.assertEqual(cell["probed_steps"], 0.0)
        self.assertAlmostEqual(cell["step_abs"], abs(cell["step"]), places=9)
        self.assertAlmostEqual(cell["solve_abs"], abs(cell["solve"]), places=9)

    def test_random_average_is_not_smaller_than_its_mean(self):
        cell = POINT[BUDGETS[-1]]["random"]["c"]
        self.assertGreaterEqual(cell["step_abs"], abs(cell["step"]) - 1e-9)

    def test_no_cell_overspends_its_budget(self):
        for budget in BUDGETS:
            for strategy in STRATEGIES:
                for model in MODELS:
                    self.assertLessEqual(POINT[budget][strategy][model]["spend"], budget + 1e-9)

    def test_random_draws_repeat(self):
        self.assertEqual(point_results(BENCH)[BUDGETS[0]]["random"]["a"], POINT[BUDGETS[0]]["random"]["a"])


class SummaryTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(SUMMARY["tasks"], len(TASKS))
        self.assertEqual(SUMMARY["steps"], 4)
        self.assertEqual(SUMMARY["models"], len(MODELS))
        self.assertEqual(SUMMARY["replicates"], 2)
        self.assertEqual(SUMMARY["random_draws"], RANDOM_DRAWS)
        self.assertEqual(SUMMARY["budgets"], list(BUDGETS))

    def test_per_model_lists(self):
        for key in ("model_tau2", "model_step_rate", "model_solve_rate"):
            self.assertEqual(len(SUMMARY[key]), len(MODELS))
        for tau2 in SUMMARY["model_tau2"]:
            self.assertGreaterEqual(tau2, 0.0)
        for rate in SUMMARY["model_step_rate"] + SUMMARY["model_solve_rate"]:
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 100.0)

    def test_per_cell_lists(self):
        for key in ("logloss", "spend", "probed_steps", "probed_tasks"):
            self.assertEqual(len(SUMMARY[key]), len(CELLS))
        for target in TARGETS:
            for key in (f"{target}_mae", f"{target}_mae_lo", f"{target}_mae_hi"):
                self.assertEqual(len(SUMMARY[key]), len(CELLS))
            for key in (f"{target}_err", f"{target}_err_lo", f"{target}_err_hi", f"{target}_abs"):
                self.assertEqual(len(SUMMARY[key]), len(CELLS))
                for row in SUMMARY[key]:
                    self.assertEqual(len(row), len(MODELS))

    def test_per_pair_lists(self):
        for target in TARGETS:
            for suffix in ("mae_diff", "mae_diff_lo", "mae_diff_hi", "models_better", "share_replicates_better"):
                self.assertEqual(len(SUMMARY[f"greedy_vs_{target}_{suffix}"]), len(PAIRS))

    def test_pair_difference_is_greedy_minus_the_other(self):
        for index, (budget, strategy) in enumerate(PAIRS):
            greedy = SUMMARY["step_mae"][CELLS.index((budget, "greedy"))]
            other = SUMMARY["step_mae"][CELLS.index((budget, strategy))]
            self.assertAlmostEqual(SUMMARY["greedy_vs_step_mae_diff"][index], greedy - other, places=5)

    def test_models_better_counts_models(self):
        for count in SUMMARY["greedy_vs_solve_models_better"]:
            self.assertGreaterEqual(count, 0)
            self.assertLessEqual(count, len(MODELS))

    def test_share_replicates_better_is_a_share(self):
        for share in SUMMARY["greedy_vs_step_share_replicates_better"]:
            self.assertGreaterEqual(share, 0.0)
            self.assertLessEqual(share, 1.0)

    def test_every_float_is_rounded(self):
        numbers = list(_floats(SUMMARY))
        self.assertGreater(len(numbers), 0)
        for number in numbers:
            self.assertEqual(round(number, 6), number)


class MainTest(unittest.TestCase):
    def test_main_writes_the_metrics_it_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "metrics.json"
            with mock.patch("prefixprobe.run.load", return_value=BENCH):
                metrics = main(Path(tmp) / "data", out, replicates=1)
            text = out.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text), metrics)
        self.assertEqual(metrics["replicates"], 1)
        self.assertEqual(metrics["steps"], 4)
