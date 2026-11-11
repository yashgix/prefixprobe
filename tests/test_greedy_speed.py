"""The fast greedy planner must choose exactly what the original one chose, and stay fast."""

import time
import unittest

import numpy as np

from prefixprobe.data import Benchmark, load
from prefixprobe.fold import Fold, make_fold
from prefixprobe.run import BUDGETS
from prefixprobe.select import _g, greedy_plan


def original_greedy_plan(bench: Benchmark, fold: Fold, budget: float) -> np.ndarray:
    """The planner as built, before prefix sums were cached: the oracle."""
    lengths = np.diff(bench.start).astype(int)
    depths = np.zeros(len(bench.tasks), dtype=int)
    cost, info, tau2, left = np.asarray(fold.cost, float), np.asarray(fold.info, float), float(fold.tau2), float(budget)
    while True:
        best = (-np.inf, -1, 0, 0.0)
        for task in range(len(bench.tasks)):
            old, length = int(depths[task]), int(lengths[task])
            if old >= length:
                continue
            lo, hi = int(bench.start[task]), int(bench.start[task + 1])
            cost_prefix, info_prefix = np.cumsum(cost[lo:hi]), np.cumsum(info[lo:hi])
            old_gain = _g(float(info_prefix[old - 1]) if old else 0.0, tau2)
            old_cost = float(cost_prefix[old - 1]) if old else 0.0
            for depth in range(old + 1, length + 1):
                add_cost = float(cost_prefix[depth - 1] - old_cost)
                if add_cost > left:
                    break
                gain = _g(float(info_prefix[depth - 1]), tau2) - old_gain
                if gain <= 0:
                    continue
                ratio = gain / add_cost if add_cost > 0 else np.inf
                if ratio > best[0]:
                    best = (ratio, task, depth, add_cost)
        if best[1] < 0:
            return depths
        depths[best[1]] = best[2]
        left -= best[3]


def synthetic(rng, n_tasks: int, ties: bool) -> tuple[Benchmark, Fold]:
    lengths = rng.integers(1, 7, n_tasks)
    start = np.concatenate(([0], np.cumsum(lengths))).astype(int)
    steps = int(start[-1])
    zeros, ones = np.zeros((1, steps)), np.ones((1, steps))
    bench = Benchmark(("m",), tuple(f"t{i}" for i in range(n_tasks)), start, zeros, ones, zeros, ones, zeros, ones)
    cost = np.ones(steps) if ties else rng.uniform(0.0, 3.0, steps)
    info = np.full(steps, 0.5) if ties else rng.uniform(0.0, 2.0, steps)
    return bench, Fold("m", zeros[0], zeros[0], cost, info, float(rng.uniform(0.0, 1.0)))


class GreedySpeedTest(unittest.TestCase):
    def assert_same(self, bench: Benchmark, fold: Fold, budget: float) -> None:
        np.testing.assert_array_equal(greedy_plan(bench, fold, budget), original_greedy_plan(bench, fold, budget))

    def test_same_plans_as_the_original_on_the_real_data(self) -> None:
        bench = load()
        for model in bench.models:
            fold = make_fold(bench, model)
            for share in BUDGETS:
                budget = share * fold.cost.sum()
                self.assert_same(bench, fold, budget)

    def test_same_plans_as_the_original_on_synthetic_and_tied_cases(self) -> None:
        rng = np.random.default_rng(20261111)
        for case in range(60):
            bench, fold = synthetic(rng, int(rng.integers(1, 12)), ties=case % 3 == 0)
            for budget in (0.0, 0.5, 3.0, float(fold.cost.sum()), float(fold.cost.sum()) * 2):
                self.assert_same(bench, fold, budget)

    def test_real_data_plans_stay_fast(self) -> None:
        bench = load()
        folds = [make_fold(bench, model) for model in bench.models]
        started = time.perf_counter()
        for fold in folds:
            for share in BUDGETS:
                greedy_plan(bench, fold, share * fold.cost.sum())
        # 36 plans took 0.13 s after the change (0.68 s before) on an M4 Pro under load; 2 s leaves room for a busy Mac.
        self.assertLess(time.perf_counter() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
