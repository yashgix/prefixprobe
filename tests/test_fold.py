import unittest
from dataclasses import replace

import numpy as np

from prefixprobe.data import Benchmark
from prefixprobe.fold import Fold, make_fold
from prefixprobe.model import step_information


def make_bench() -> Benchmark:
    """Three models, two tasks (3 and 2 steps) with hand-checkable counts."""
    models = ("a", "b", "c")
    tasks = ("t0", "t1")
    start = np.array([0, 3, 5], dtype=int)
    passes = np.array([[3, 3, 2, 4, 3], [2, 1, 1, 2, 1], [1, 1, 0, 1, 0]], dtype=float)
    runs = np.full((3, 5), 4.0)
    on_track = np.array([[3, 2, 1, 3, 2], [2, 1, 1, 2, 1], [1, 1, 0, 1, 0]], dtype=float)
    reached = np.array([[4, 3, 2, 4, 3], [4, 2, 1, 4, 2], [4, 1, 1, 4, 1]], dtype=float)
    seconds = np.array(
        [[40.0, 40.0, 40.0, 20.0, 20.0], [8.0, 8.0, 0.0, 4.0, 4.0], [4.0, 0.0, 0.0, 2.0, 2.0]]
    )
    timed = np.array([[4.0, 4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 0.0, 4.0, 4.0], [4.0, 0.0, 0.0, 4.0, 4.0]])
    return Benchmark(
        models=models,
        tasks=tasks,
        start=start,
        passes=passes,
        runs=runs,
        on_track=on_track,
        reached=reached,
        seconds=seconds,
        timed=timed,
    )


class MakeFoldTest(unittest.TestCase):
    def setUp(self):
        self.bench = make_bench()

    def test_fields(self):
        fold = make_fold(self.bench, "a")
        self.assertIsInstance(fold, Fold)
        self.assertEqual(fold.model, "a")
        for array in (fold.d, fold.e, fold.cost, fold.info):
            self.assertEqual(array.shape, (5,))
            self.assertEqual(array.dtype, np.float64)
        self.assertIsInstance(fold.tau2, float)

    def test_difficulties_are_log_odds_of_the_other_models(self):
        fold = make_fold(self.bench, "a")
        passes = self.bench.passes[[1, 2]].sum(axis=0)
        runs = self.bench.runs[[1, 2]].sum(axis=0)
        p = (passes + 0.5) / (runs + 1.0)
        np.testing.assert_allclose(fold.d, np.log(p / (1.0 - p)))

        on_track = self.bench.on_track[[1, 2]].sum(axis=0)
        reached = self.bench.reached[[1, 2]].sum(axis=0)
        q = (on_track + 0.5) / (reached + 1.0)
        np.testing.assert_allclose(fold.e, np.log(q / (1.0 - q)))

    def test_cost_is_mean_seconds_of_the_other_models(self):
        fold = make_fold(self.bench, "a")
        seconds = self.bench.seconds[[1, 2]].sum(axis=0)
        timed = self.bench.timed[[1, 2]].sum(axis=0)
        expected = np.divide(seconds, timed, out=np.zeros_like(seconds), where=timed > 0)
        np.testing.assert_allclose(fold.cost, expected)
        self.assertEqual(fold.cost[2], 0.0)

    def test_information_and_variance(self):
        fold = make_fold(self.bench, "a")
        np.testing.assert_allclose(fold.info, step_information(self.bench, fold.d, fold.e))
        self.assertGreaterEqual(fold.tau2, 0.0)
        self.assertTrue(np.isfinite(fold.tau2))

    def test_held_out_results_do_not_enter_the_fold(self):
        before = make_fold(self.bench, "a")
        passes = self.bench.passes.copy()
        passes[0] = [0.0, 1.0, 0.0, 1.0, 0.0]
        after = make_fold(replace(self.bench, passes=passes), "a")
        for name in ("d", "e", "cost", "info"):
            np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
        self.assertEqual(before.tau2, after.tau2)

    def test_each_model_gets_its_own_fold(self):
        fold_a = make_fold(self.bench, "a")
        fold_b = make_fold(self.bench, "b")
        self.assertEqual(fold_a.model, "a")
        self.assertEqual(fold_b.model, "b")
        self.assertFalse(np.array_equal(fold_a.d, fold_b.d))


if __name__ == "__main__":
    unittest.main()
