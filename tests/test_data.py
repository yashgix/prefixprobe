"""Tests for prefixprobe.data."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prefixprobe.data import DATA_DIR, Benchmark, load, read_module


def _write_module(path: Path, obj) -> None:
    path.write_text("export default " + json.dumps(obj) + ";\n", encoding="utf-8")


class ReadModuleTests(unittest.TestCase):
    def test_reads_exported_object(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "module.js"
            _write_module(path, {"a": 1, "b": [2, 3]})
            self.assertEqual(read_module(path), {"a": 1, "b": [2, 3]})


class BenchmarkTests(unittest.TestCase):
    def _benchmark(self) -> Benchmark:
        return Benchmark(
            models=("m0", "m1"),
            tasks=("a", "b"),
            start=np.array([0, 3, 5], dtype=int),
            passes=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float),
            runs=np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]], dtype=float),
            on_track=np.array([[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]], dtype=float),
            reached=np.array([[31, 32, 33, 34, 35], [36, 37, 38, 39, 40]], dtype=float),
            seconds=np.array([[41, 42, 43, 44, 45], [46, 47, 48, 49, 50]], dtype=float),
            timed=np.array([[51, 52, 53, 54, 55], [56, 57, 58, 59, 60]], dtype=float),
        )

    def test_positions_and_within_task_cumsum(self):
        bench = self._benchmark()
        np.testing.assert_array_equal(bench.task_of, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(bench.position, [0, 1, 2, 0, 1])
        np.testing.assert_array_equal(
            bench.within_task_cumsum(np.array([1, 1, 1, 2, 2])),
            np.array([1, 2, 3, 2, 4]),
        )

    def test_resample_repeats_tasks(self):
        bench = self._benchmark()
        sample = bench.resample([1, 0, 1])
        self.assertEqual(sample.models, bench.models)
        self.assertEqual(sample.tasks, ("b", "a", "b"))
        np.testing.assert_array_equal(sample.start, [0, 2, 5, 7])
        expected = np.array([[4, 5, 1, 2, 3, 4, 5], [9, 10, 6, 7, 8, 9, 10]])
        for offset, field in enumerate(("passes", "runs", "on_track", "reached", "seconds", "timed")):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(sample, field), expected + 10 * offset)

    def test_load_synthetic_data(self):
        benchmark = {
            "data": {
                "models": [{"id": "m1"}, {"id": "m2"}],
                "tasks": [
                    {
                        "id": "a",
                        "steps": ["a1", "a2"],
                        "runs": {
                            "m1": [
                                {"id": "run1", "scores": [1, 1]},
                                {"id": "run2", "scores": [1, 0]},
                            ],
                            "m2": [{"id": "run1", "scores": [0, 0]}],
                        },
                    },
                    {
                        "id": "b",
                        "steps": ["b1", "b2", "b3"],
                        "runs": {
                            "m1": [{"id": "run1", "scores": [1, 0, 1]}],
                            "m2": [],
                        },
                    },
                ],
            }
        }
        traces_a = {
            "traces": {
                "m1": {
                    "run1": [{"seconds": 1.5}, {"seconds": 2.0}],
                    "run2": [{"seconds": None}, {"seconds": 3.0}],
                },
                "m2": {"run1": [{"seconds": 4.0}, {"seconds": None}]},
            }
        }
        traces_b = {
            "traces": {
                "m1": {"run1": [{"seconds": 1.0}, {"seconds": None}, {"seconds": 2.0}]},
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tasks").mkdir()
            _write_module(root / "benchmark.js", benchmark)
            _write_module(root / "tasks" / "a.js", traces_a)
            _write_module(root / "tasks" / "b.js", traces_b)
            bench = load(root)

        self.assertEqual(bench.models, ("m1", "m2"))
        self.assertEqual(bench.tasks, ("a", "b"))
        np.testing.assert_array_equal(bench.start, [0, 2, 5])
        np.testing.assert_allclose(bench.passes, [[2, 1, 1, 0, 1], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.runs, [[2, 2, 1, 1, 1], [1, 1, 0, 0, 0]])
        np.testing.assert_allclose(bench.on_track, [[2, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.reached, [[2, 2, 1, 1, 0], [1, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.seconds, [[1.5, 5.0, 1.0, 0.0, 2.0], [4.0, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(bench.timed, [[1, 2, 1, 0, 1], [1, 0, 0, 0, 0]])

    @unittest.skipUnless(
        (DATA_DIR / "benchmark.js").exists(),
        "data/ is missing; run python tools/fetch_data.py",
    )
    def test_real_data_shape(self):
        bench = load(DATA_DIR)
        self.assertEqual(len(bench.models), 9)
        self.assertEqual(len(bench.tasks), 69)
        self.assertEqual(bench.passes.shape, (9, 574))
        self.assertEqual(int(bench.runs.sum()), 20651)
        task_runs = bench.runs[:, bench.start[:-1]]
        expected = np.full(task_runs.shape, 4.0)
        expected[bench.models.index("qwen"), bench.tasks.index("pcre2-regex-engine-multiversion")] = 3.0
        np.testing.assert_array_equal(task_runs, expected)
        self.assertEqual(task_runs.sum(axis=1).tolist(), [276.0] * 6 + [275.0] + [276.0] * 2)
        for s in bench.start[:-1]:
            np.testing.assert_array_equal(bench.reached[:, s], bench.runs[:, s])
