"""Tests for prefixprobe.model."""

from __future__ import annotations

import unittest

import numpy as np

from prefixprobe.data import Benchmark
from prefixprobe.model import (
    PRIOR_SD,
    RUNS_PER_TASK,
    fit_ability,
    logit_rate,
    step_information,
    task_variance,
)


def _benchmark(passes, runs, start) -> Benchmark:
    passes = np.asarray(passes, dtype=float)
    runs = np.asarray(runs, dtype=float)
    n_models, n_steps = passes.shape
    return Benchmark(
        models=tuple(f"m{i}" for i in range(n_models)),
        tasks=tuple(f"t{i}" for i in range(len(start) - 1)),
        start=np.asarray(start, dtype=int),
        passes=passes,
        runs=runs,
        on_track=np.zeros((n_models, n_steps)),
        reached=np.zeros((n_models, n_steps)),
        seconds=np.zeros((n_models, n_steps)),
        timed=np.zeros((n_models, n_steps)),
    )


class ModelTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(PRIOR_SD, 2.0)
        self.assertEqual(RUNS_PER_TASK, 4)

    def test_logit_rate(self):
        passes = np.array([0.0, 1.0, 5.0])
        trials = np.array([0.0, 2.0, 5.0])
        p = (passes + 0.5) / (trials + 1.0)
        expected = np.log(p / (1.0 - p))
        np.testing.assert_allclose(logit_rate(passes, trials), expected)
        self.assertAlmostEqual(float(logit_rate(0, 0)), 0.0)

    def test_fit_ability_empty_and_balanced(self):
        empty = np.array([])
        self.assertEqual(fit_ability(empty, empty, empty), 0.0)
        balanced = fit_ability(np.zeros(4), np.full(4, 2.0), np.full(4, 4.0))
        self.assertAlmostEqual(balanced, 0.0, places=12)
        above = fit_ability(np.zeros(4), np.full(4, 4.0), np.full(4, 4.0))
        self.assertGreater(above, 0.0)

    def test_task_variance(self):
        d = np.array([0.0])
        bench = _benchmark([[1.0]], [[1.0]], [0, 1])
        self.assertEqual(task_variance(bench, [], d), 0.0)
        a = fit_ability(d, bench.passes[0], bench.runs[0])
        p = 1.0 / (1.0 + np.exp(-a))
        r = 1.0 - p
        info = p * (1.0 - p)
        expected = max(0.0, (r * r - info) / (info * info))
        self.assertAlmostEqual(task_variance(bench, [0], d), expected, places=12)
        bench2 = _benchmark([[1, 0], [1, 1]], [[1, 1], [1, 1]], [0, 2])
        tau2 = task_variance(bench2, [0, 1], np.zeros(2))
        self.assertGreaterEqual(tau2, 0.0)
        self.assertTrue(np.isfinite(tau2))

    def test_step_information(self):
        bench = _benchmark(np.zeros((1, 5)), np.ones((1, 5)), [0, 3, 5])
        info = step_information(bench, np.zeros(5), np.zeros(5))
        np.testing.assert_allclose(info, [2.0, 1.5, 1.25, 2.0, 1.5])
