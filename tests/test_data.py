'''Tests for prefixprobe.data.'''
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prefixprobe.data import DATA_DIR, Benchmark, load, read_module


def _write_module(path: Path, obj) -> None:
    path.write_text('export default ' + json.dumps(obj) + ';\n', encoding='utf-8')


class ReadModuleTests(unittest.TestCase):
    def test_reads_exported_object(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'module.js'
            _write_module(path, {'a': 1, 'b': [2, 3]})
            self.assertEqual(read_module(path), {'a': 1, 'b': [2, 3]})


class BenchmarkTests(unittest.TestCase):
    def _benchmark(self) -> Benchmark:
        return Benchmark(
            models=('m0', 'm1'),
            tasks=('a', 'b'),
            start=np.array([0, 3, 5], dtype=int),
            passes=np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float),
            runs=np.ones((2, 5)),
            on_track=np.zeros((2, 5)),
            reached=np.ones((2, 5)),
            seconds=np.zeros((2, 5)),
            timed=np.zeros((2, 5)),
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
        self.assertEqual(sample.tasks, ('b', 'a', 'b'))
        np.testing.assert_array_equal(sample.start, [0, 2, 5, 7])
        expected = np.array([[4, 5, 1, 2, 3, 4, 5], [9, 10, 6, 7, 8, 9, 10]])
        np.testing.assert_array_equal(sample.passes, expected)

    def test_load_synthetic_data(self):
        benchmark = {
            'data': {
                'models': [{'id': 'm1'}, {'id': 'm2'}],
                'tasks': [
                    {
                        'id': 'a',
                        'steps': ['a1', 'a2'],
                        'runs': {
                            'm1': [
                                {'id': 'run1', 'scores': [1, 1]},
                                {'id': 'run2', 'scores': [1, 0]},
                            ],
                            'm2': [{'id': 'run1', 'scores': [0, 0]}],
                        },
                    },
                    {
                        'id': 'b',
                        'steps': ['b1', 'b2', 'b3'],
                        'runs': {
                            'm1': [{'id': 'run1', 'scores': [1, 0, 1]}],
                            'm2': [],
                        },
                    },
                ],
            }
        }
        traces_a = {
            'traces': {
                'm1': {
                    'run1': [{'seconds': 1.5}, {'seconds': 2.0}],
                    'run2': [{'seconds': None}, {'seconds': 3.0}],
                },
                'm2': {'run1': [{'seconds': 4.0}, {'seconds': None}]},
            }
        }
        traces_b = {
            'traces': {
                'm1': {'run1': [{'seconds': 1.0}, {'seconds': None}, {'seconds': 2.0}]},
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'tasks').mkdir()
            _write_module(root / 'benchmark.js', benchmark)
            _write_module(root / 'tasks' / 'a.js', traces_a)
            _write_module(root / 'tasks' / 'b.js', traces_b)
            bench = load(root)

        self.assertEqual(bench.models, ('m1', 'm2'))
        self.assertEqual(bench.tasks, ('a', 'b'))
        np.testing.assert_array_equal(bench.start, [0, 2, 5])
        np.testing.assert_allclose(bench.passes, [[2, 1, 1, 0, 1], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.runs, [[2, 2, 1, 1, 1], [1, 1, 0, 0, 0]])
        np.testing.assert_allclose(bench.on_track, [[2, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.reached, [[2, 2, 1, 1, 0], [1, 0, 0, 0, 0]])
        np.testing.assert_allclose(bench.seconds, [[1.5, 5.0, 1.0, 0.0, 2.0], [4.0, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(bench.timed, [[1, 2, 1, 0, 1], [1, 0, 0, 0, 0]])

    @unittest.skipUnless(
        (DATA_DIR / 'benchmark.js').exists(),
        'data/ is missing; run python tools/fetch_data.py',
    )
    def test_real_data_shape(self):
        bench = load(DATA_DIR)
        self.assertEqual(len(bench.models), 9)
        self.assertEqual(len(bench.tasks), 69)
        self.assertEqual(bench.passes.shape, (9, 574))
        self.assertEqual(int(bench.runs.sum()), 20651)
        for s in bench.start[:-1]:
            np.testing.assert_array_equal(bench.reached[:, s], bench.runs[:, s])