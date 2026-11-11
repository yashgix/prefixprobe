from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


def read_module(path):
    text = Path(path).read_text(encoding='utf-8')
    return json.JSONDecoder().raw_decode(text, text.index('{'))[0]


@dataclass(frozen=True)
class Benchmark:
    models: tuple[str, ...]
    tasks: tuple[str, ...]
    start: np.ndarray
    passes: np.ndarray
    runs: np.ndarray
    on_track: np.ndarray
    reached: np.ndarray
    seconds: np.ndarray
    timed: np.ndarray

    @property
    def task_of(self) -> np.ndarray:
        lengths = np.diff(self.start)
        return np.repeat(np.arange(len(self.tasks), dtype=int), lengths)

    @property
    def position(self) -> np.ndarray:
        lengths = np.diff(self.start)
        starts = np.repeat(self.start[:-1], lengths)
        return np.arange(self.start[-1], dtype=int) - starts

    def within_task_cumsum(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        out = np.empty_like(x)
        for a, b in zip(self.start[:-1], self.start[1:]):
            out[a:b] = np.cumsum(x[a:b])
        return out

    def resample(self, task_idx) -> Benchmark:
        idx = np.asarray(task_idx, dtype=int)
        if idx.size:
            lengths = self.start[idx + 1] - self.start[idx]
            columns = np.concatenate([np.arange(self.start[t], self.start[t + 1], dtype=int) for t in idx])
        else:
            lengths = np.zeros(0, dtype=int)
            columns = np.zeros(0, dtype=int)
        start = np.concatenate(([0], np.cumsum(lengths))).astype(int)

        def take(a: np.ndarray) -> np.ndarray:
            return a[:, columns].copy()

        return Benchmark(
            models=self.models,
            tasks=tuple(self.tasks[t] for t in idx),
            start=start,
            passes=take(self.passes),
            runs=take(self.runs),
            on_track=take(self.on_track),
            reached=take(self.reached),
            seconds=take(self.seconds),
            timed=take(self.timed),
        )


def load(data_dir=DATA_DIR) -> Benchmark:
    data_dir = Path(data_dir)
    data = read_module(data_dir / 'benchmark.js')['data']
    models = tuple(m['id'] for m in data['models'])
    task_entries = data['tasks']
    tasks = tuple(t['id'] for t in task_entries)
    lengths = np.array([len(t['steps']) for t in task_entries], dtype=int)
    start = np.concatenate(([0], np.cumsum(lengths))).astype(int)

    m_count = len(models)
    step_count = int(start[-1])
    passes = np.zeros((m_count, step_count), dtype=float)
    runs = np.zeros((m_count, step_count), dtype=float)
    on_track = np.zeros((m_count, step_count), dtype=float)
    reached = np.zeros((m_count, step_count), dtype=float)
    seconds = np.zeros((m_count, step_count), dtype=float)
    timed = np.zeros((m_count, step_count), dtype=float)

    for task_index, task in enumerate(task_entries):
        first, last = int(start[task_index]), int(start[task_index + 1])
        task_id = task['id']
        traces = read_module(data_dir / 'tasks' / f'{task_id}.js').get('traces', {})
        for model_index, model in enumerate(models):
            run_records = task.get('runs', {}).get(model, [])
            model_traces = traces.get(model, {})
            for offset in range(last - first):
                col = first + offset
                n_runs = 0
                n_passed = 0
                n_reached = 0
                n_on_track = 0
                for record in run_records:
                    scores = record.get('scores') or []
                    if offset >= len(scores):
                        continue
                    n_runs += 1
                    before = all(scores[i] == 1 for i in range(offset))
                    current = scores[offset] == 1
                    if before:
                        n_reached += 1
                    if before and current:
                        n_on_track += 1
                    if current:
                        n_passed += 1
                runs[model_index, col] = n_runs
                passes[model_index, col] = n_passed
                reached[model_index, col] = n_reached
                on_track[model_index, col] = n_on_track

                for record in run_records:
                    entries = model_traces.get(record.get('id'), [])
                    if offset < len(entries):
                        value = entries[offset].get('seconds')
                        if value is not None:
                            seconds[model_index, col] += float(value)
                            timed[model_index, col] += 1.0

    return Benchmark(models, tasks, start, passes, runs, on_track, reached, seconds, timed)
