import unittest

import numpy as np

from prefixprobe.data import Benchmark
from prefixprobe.fold import Fold
from prefixprobe.select import (
    cheapest_tasks_plan,
    fill_in_order,
    greedy_plan,
    make_plan,
    prefix_cost,
    random_plan,
    uniform_plan,
)


COST = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 5.0])


def make_benchmark() -> Benchmark:
    start = np.array([0, 2, 5, 6], dtype=int)
    zeros = np.zeros((2, 6), dtype=float)
    ones = np.ones((2, 6), dtype=float)
    return Benchmark(
        models=("m0", "m1"),
        tasks=("t0", "t1", "t2"),
        start=start,
        passes=zeros.copy(),
        runs=ones.copy(),
        on_track=zeros.copy(),
        reached=ones.copy(),
        seconds=zeros.copy(),
        timed=ones.copy(),
    )


def make_fold(cost: np.ndarray | None = None, info: np.ndarray | None = None, tau2: float = 0.0) -> Fold:
    return Fold(
        model="m0",
        d=np.zeros(6),
        e=np.zeros(6),
        cost=np.ones(6) if cost is None else cost,
        info=np.ones(6) if info is None else info,
        tau2=tau2,
    )


class SelectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = make_benchmark()

    def test_prefix_cost_pays_for_prefixes(self) -> None:
        depths = np.array([1, 2, 0])
        self.assertAlmostEqual(prefix_cost(self.bench, COST, depths), 3.0)

    def test_fill_in_order_uses_remaining_budget(self) -> None:
        depths = fill_in_order(self.bench, COST, [0, 1, 2], 5.0)
        self.assertEqual(depths.tolist(), [2, 2, 0])
        self.assertAlmostEqual(prefix_cost(self.bench, COST, depths), 5.0)

    def test_random_plan_is_deterministic_and_within_budget(self) -> None:
        first = random_plan(self.bench, COST, 4.0, np.random.default_rng(123))
        second = random_plan(self.bench, COST, 4.0, np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(prefix_cost(self.bench, COST, first), 4.0)

    def test_uniform_plan_uses_the_largest_equal_depth(self) -> None:
        self.assertEqual(uniform_plan(self.bench, COST, 10.0).tolist(), [2, 2, 1])
        self.assertEqual(uniform_plan(self.bench, COST, 4.0).tolist(), [0, 0, 0])
        self.assertEqual(uniform_plan(self.bench, COST, 11.0).tolist(), [2, 3, 1])

    def test_cheapest_tasks_plan_fills_cheap_tasks_first(self) -> None:
        self.assertEqual(cheapest_tasks_plan(self.bench, COST, 6.0).tolist(), [2, 3, 0])

    def test_greedy_plan_ties_go_to_the_lowest_flat_step(self) -> None:
        cost = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
        fold = make_fold(cost=cost, info=np.ones(6), tau2=0.0)
        plan = greedy_plan(self.bench, fold, 2.0)
        self.assertEqual(plan.tolist(), [2, 0, 0])
        self.assertAlmostEqual(prefix_cost(self.bench, cost, plan), 2.0)

    def test_greedy_plan_uses_gain_per_cost(self) -> None:
        cost = np.array([1.0, 1.0, 4.0, 4.0, 4.0, 10.0])
        info = np.array([1.0, 1.0, 10.0, 10.0, 10.0, 100.0])
        fold = make_fold(cost=cost, info=info, tau2=0.0)
        plan = greedy_plan(self.bench, fold, 4.0)
        self.assertEqual(plan.tolist(), [0, 1, 0])

    def test_make_plan_dispatches_and_rejects_unknown_names(self) -> None:
        fold = make_fold(cost=COST)
        none = make_plan("none", self.bench, fold, 5.0, np.random.default_rng(0))
        self.assertEqual(none.tolist(), [0, 0, 0])

        random = make_plan("random", self.bench, fold, 5.0, np.random.default_rng(0))
        self.assertEqual(random.shape, (3,))
        self.assertLessEqual(prefix_cost(self.bench, fold.cost, random), 5.0)

        with self.assertRaises(ValueError):
            make_plan("bogus", self.bench, fold, 5.0, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
