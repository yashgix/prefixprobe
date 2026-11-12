import unittest
from dataclasses import replace

import numpy as np

from prefixprobe.data import Benchmark
from prefixprobe.fold import make_fold
from prefixprobe.model import fit_ability
from prefixprobe.score import score


def make_bench() -> Benchmark:
    """Three models, two tasks (3 and 2 steps) with hand-checkable counts."""
    models = ("a", "b", "c")
    tasks = ("t0", "t1")
    start = np.array([0, 3, 5], dtype=int)
    passes = np.array([[3, 3, 2, 4, 3], [2, 1, 1, 2, 1], [1, 1, 0, 1, 0]], dtype=float)
    runs = np.full((3, 5), 4.0)
    on_track = np.array([[3, 2, 1, 3, 2], [2, 1, 1, 2, 1], [1, 1, 0, 1, 0]], dtype=float)
    reached = np.array([[4, 3, 2, 4, 3], [4, 2, 1, 4, 2], [4, 1, 1, 4, 1]], dtype=float)
    seconds = np.array([[40.0, 40.0, 40.0, 20.0, 20.0], [8.0, 8.0, 0.0, 4.0, 4.0], [4.0, 0.0, 0.0, 2.0, 2.0]])
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.bench = make_bench()
        self.fold = make_fold(self.bench, "a")

    def test_returns_plain_python_scalars(self):
        out = score(self.bench, self.fold, [1, 1])
        expected = {
            "ability",
            "on_track_ability",
            "step_rate",
            "step_rate_hat",
            "solve_rate",
            "solve_rate_hat",
            "logloss",
            "probed_steps",
            "probed_tasks",
            "spend",
        }
        self.assertEqual(set(out), expected)
        for name, value in out.items():
            self.assertNotIsInstance(value, np.generic, msg=name)
            self.assertIsInstance(value, (int, float), msg=name)

    def test_full_probe_reproduces_the_true_rates(self):
        out = score(self.bench, self.fold, [3, 2])
        self.assertAlmostEqual(out["step_rate_hat"], out["step_rate"], places=9)
        self.assertAlmostEqual(out["solve_rate_hat"], out["solve_rate"], places=9)
        self.assertEqual(out["probed_steps"], 5)
        self.assertEqual(out["probed_tasks"], 2)
        self.assertAlmostEqual(out["spend"], 1.0)
        self.assertTrue(np.isnan(out["logloss"]))

    def test_depths_longer_than_tasks_behave_like_a_full_probe(self):
        full = score(self.bench, self.fold, [3, 2])
        over = score(self.bench, self.fold, [99, 99])
        self.assertEqual(over["probed_steps"], 5)
        self.assertAlmostEqual(over["step_rate_hat"], full["step_rate_hat"])
        self.assertAlmostEqual(over["solve_rate_hat"], full["solve_rate_hat"])

    def test_empty_probe(self):
        out = score(self.bench, self.fold, [0, 0])
        self.assertEqual(out["probed_steps"], 0)
        self.assertEqual(out["probed_tasks"], 0)
        self.assertEqual(out["spend"], 0.0)
        self.assertEqual(out["ability"], 0.0)
        self.assertEqual(out["on_track_ability"], 0.0)
        self.assertFalse(np.isnan(out["logloss"]))
        runs = self.bench.runs[0]
        self.assertAlmostEqual(out["step_rate"], 100.0 * self.bench.passes[0].sum() / runs.sum())
        expected = 100.0 * (runs * sigmoid(self.fold.d)).sum() / runs.sum()
        self.assertAlmostEqual(out["step_rate_hat"], expected)

    def test_partial_probe_counts_and_spend(self):
        out = score(self.bench, self.fold, [2, 0])
        self.assertEqual(out["probed_steps"], 2)
        self.assertEqual(out["probed_tasks"], 1)
        self.assertAlmostEqual(out["spend"], self.fold.cost[:2].sum() / self.fold.cost.sum())

    def test_step_rate_hat_uses_the_probe_and_the_pooled_difficulty(self):
        depths = [2, 0]
        out = score(self.bench, self.fold, depths)
        passes = self.bench.passes[0]
        runs = self.bench.runs[0]
        ability = fit_ability(self.fold.d[:2], passes[:2], runs[:2])
        self.assertAlmostEqual(out["ability"], ability)
        predicted = runs[2:] * sigmoid(ability + self.fold.d[2:])
        expected = 100.0 * (passes[:2].sum() + predicted.sum()) / runs.sum()
        self.assertAlmostEqual(out["step_rate_hat"], expected)

    def test_logloss_covers_only_the_unprobed_steps(self):
        out = score(self.bench, self.fold, [2, 0])
        passes = self.bench.passes[0]
        runs = self.bench.runs[0]
        ability = fit_ability(self.fold.d[:2], passes[:2], runs[:2])
        p = np.clip(sigmoid(ability + self.fold.d[2:]), 1e-12, 1.0 - 1e-12)
        n = runs[2:]
        loss = -(passes[2:] * np.log(p) + (n - passes[2:]) * np.log1p(-p))
        self.assertAlmostEqual(out["logloss"], loss.sum() / n.sum())

    def test_solve_rates(self):
        out = score(self.bench, self.fold, [2, 0])
        on_track = self.bench.on_track[0]
        runs = self.bench.runs[0]
        self.assertAlmostEqual(out["solve_rate"], 100.0 * (on_track[2] + on_track[4]) / (runs[0] + runs[3]))

        on_track_ability = fit_ability(self.fold.e[:2], on_track[:2], self.bench.reached[0, :2])
        self.assertAlmostEqual(out["on_track_ability"], on_track_ability)
        clean0 = on_track[1]
        clean1 = runs[3]
        rest0 = sigmoid(on_track_ability + self.fold.e[2])
        future = sigmoid(on_track_ability + self.fold.e[3:]).prod()
        expected = 100.0 * (clean0 * rest0 + clean1 * future) / (runs[0] + runs[3])
        self.assertAlmostEqual(out["solve_rate_hat"], expected)

    def test_estimation_ignores_unprobed_results(self):
        depths = [1, 1]  # probes steps 0 and 3; steps 1, 2 and 4 stay unprobed
        before = score(self.bench, self.fold, depths)
        for field, value in (("passes", 0.0), ("on_track", 0.0), ("reached", 4.0)):
            with self.subTest(field=field):
                counts = getattr(self.bench, field).copy()
                counts[0, [1, 2, 4]] = value
                after = score(replace(self.bench, **{field: counts}), self.fold, depths)
                for name in ("ability", "on_track_ability", "step_rate_hat", "solve_rate_hat"):
                    self.assertEqual(before[name], after[name])


def unequal_runs_bench() -> Benchmark:
    """make_bench with model "a" run 3 times on t0 and 5 times on t1, so run counts vary across tasks."""
    bench = make_bench()
    rows = {
        "runs": [3, 3, 3, 5, 5],
        "passes": [2, 3, 1, 4, 2],
        "on_track": [2, 1, 1, 4, 2],
        "reached": [3, 2, 1, 5, 4],
    }
    changed = {}
    for field, row in rows.items():
        counts = getattr(bench, field).copy()
        counts[0] = row
        changed[field] = counts
    return replace(bench, **changed)


def with_t1_runs(bench: Benchmark, n: float) -> Benchmark:
    runs, reached = bench.runs.copy(), bench.reached.copy()
    runs[0, 3:] = n
    reached[0, 3] = n
    return replace(bench, runs=runs, reached=reached)


class RunCountWeightTest(unittest.TestCase):
    """Unprobed outcomes never enter the estimate; run counts, fixed before probing, weight it."""

    def setUp(self):
        self.bench = unequal_runs_bench()
        self.fold = make_fold(self.bench, "a")

    def test_unprobed_outcomes_never_move_the_estimate(self):
        for depths, unprobed in (([1, 1], [1, 2, 4]), ([0, 2], [0, 1, 2]), ([2, 0], [2, 3, 4])):
            before = score(self.bench, self.fold, depths)
            for field in ("passes", "on_track", "reached"):
                with self.subTest(depths=depths, field=field):
                    counts = getattr(self.bench, field).copy()
                    counts[0, unprobed] = 0.0
                    after = score(replace(self.bench, **{field: counts}), self.fold, depths)
                    for name in ("ability", "on_track_ability", "step_rate_hat", "solve_rate_hat"):
                        self.assertEqual(before[name], after[name])

    def test_run_counts_weight_the_extrapolation(self):
        fits = []
        for n in (5.0, 8.0):
            with self.subTest(t1_runs=n):
                bench = with_t1_runs(self.bench, n)
                out = score(bench, self.fold, [1, 0])  # probes step 0 only; t1 is wholly unprobed
                a, a_e = out["ability"], out["on_track_ability"]
                fits.append((a, a_e))
                runs, d, e = bench.runs[0], self.fold.d, self.fold.e
                step = (bench.passes[0, 0] + np.sum(runs[1:] * sigmoid(a + d[1:]))) / runs.sum()
                clean = bench.on_track[0, 0] * sigmoid(a_e + e[1:3]).prod() + n * sigmoid(a_e + e[3:]).prod()
                solve = clean / (runs[0] + n)
                self.assertAlmostEqual(out["step_rate_hat"], 100.0 * step)
                self.assertAlmostEqual(out["solve_rate_hat"], 100.0 * solve)
        self.assertEqual(fits[0], fits[1])

    def test_full_probe_reproduces_the_true_rates(self):
        out = score(self.bench, self.fold, [3, 2])
        self.assertAlmostEqual(out["step_rate_hat"], out["step_rate"])
        self.assertAlmostEqual(out["solve_rate_hat"], out["solve_rate"])


if __name__ == "__main__":
    unittest.main()
