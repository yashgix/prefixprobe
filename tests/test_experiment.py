"""Tests for the experiment command line."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import experiment
from prefixprobe import DATA_DIR
from prefixprobe.data import Benchmark

MODELS = ("a", "b", "c")
TASKS = ("t1", "t2")
SCORES = {
    "a": [[[1, 1], [1, 0]], [[1, 0], [0, 0]]],
    "b": [[[1, 1], [0, 1]], [[0, 0], [0, 0]]],
    "c": [[[1, 0], [0, 0]], [[0, 0], [0, 0]]],
}


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


class ParseArgsTest(unittest.TestCase):
    def test_out_is_required(self):
        with self.assertRaises(SystemExit):
            experiment.parse_args([])

    def test_defaults(self):
        args = experiment.parse_args(["--out", "metrics.json"])
        self.assertEqual(args.out, Path("metrics.json"))
        self.assertEqual(args.data, Path(DATA_DIR))
        self.assertEqual(args.replicates, experiment.run.REPLICATES)

    def test_overrides(self):
        args = experiment.parse_args(["--out", "m.json", "--data", "d", "--replicates", "7"])
        self.assertEqual(args.data, Path("d"))
        self.assertEqual(args.replicates, 7)


class MainTest(unittest.TestCase):
    def test_writes_sorted_indent_one_json_with_a_final_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.json"
            with mock.patch("prefixprobe.run.load", return_value=make_bench()):
                experiment.main(["--out", str(out), "--replicates", "1"])
            text = out.read_text(encoding="utf-8")
        self.assertTrue(text.startswith('{\n "'))
        self.assertTrue(text.endswith("}\n"))
        document = json.loads(text)
        self.assertEqual(list(document), sorted(document))
        self.assertEqual(document["replicates"], 1)

    def test_repeated_runs_are_byte_identical(self):
        bench = make_bench()
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.json", Path(tmp) / "b.json"
            with mock.patch("prefixprobe.run.load", return_value=bench):
                experiment.main(["--out", str(first), "--replicates", "1"])
                experiment.main(["--out", str(second), "--replicates", "1"])
            self.assertEqual(first.read_bytes(), second.read_bytes())
