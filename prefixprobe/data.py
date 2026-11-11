'''Reading and shaping Multi-step SWE result modules.'''
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import DATA_DIR


def read_module(path: str | Path) -> Any:
    '''Return the JSON object exported by one generated JavaScript module.'''
    text = Path(path).read_text(encoding='utf-8')
    start = text.index('{')
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


@dataclass(frozen=True)
class Benchmark:
    '''Flat step-level benchmark counts, one column per step.'''

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
        return np.repeat(np.arange(len(self.tasks), dtype=int), np.diff(self.start))

    @property
    def position(self) -> np.ndarray:
        return np.arange(int(self.start[-1]), dtype=int) - self.start[self.task_of]

    def within_task_cumsum(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        cum = np.cumsum(x)
        out = cum.copy()
        for t in range(len(self.tasks)):
            s = int(self.start[t])
            e = int(self.start[t + 1])
            if s:
                out[s:e] = cum[s:e] - cum[s - 1]
        return out

    def resample(self, task_idx) -> Benchmark:
        idx = [int(t) for t in task_idx]
        selected = []
        lengths = []
        for t in idx:
            s = int(self.start[t])
            e = int(self.start[t + 1])
            selected.append(np.arange(s, e, dtype=int))
            lengths.append(e - s)
        if selected:
            cols = np.concatenate(selected)
            start = np.concatenate((np.array([0], dtype=int), np.cumsum(lengths, dtype=int)))
        else:
            cols = np.empty(0, dtype=int)
            start = np.array([0], dtype=int)
        return Benchmark(
            models=self.models,
            tasks=tuple(self.tasks[t] for t in idx),
            start=start,
            passes=self.passes[:, cols],
            runs=self.runs[:, cols],
            on_track=self.on_track[:, cols],
            reached=self.reached[:, cols],
            seconds=self.seconds[:, cols],
            timed=self.timed[:, cols],
        )


def load(data_dir: str | Path = DATA_DIR) -> Benchmark:
    '''Load the benchmark counts and recorded agent seconds from a data directory.'''
    data_dir = Path(data_dir)
    data = read_module(data_dir / 'benchmark.js')['data']
    models = tuple(model['id'] for model in data['models'])
    tasks_data = data['tasks']
    lengths = [len(task['steps']) for task in tasks_data]
    start = np.concatenate((np.array([0], dtype=int), np.cumsum(lengths, dtype=int)))
    model_index = {model: i for i, model in enumerate(models)}
    n_models = len(models)
    n_steps = int(start[-1])
    passes = np.zeros((n_models, n_steps), dtype=float)
    runs = np.zeros((n_models, n_steps), dtype=float)
    on_track = np.zeros((n_models, n_steps), dtype=float)
    reached = np.zeros((n_models, n_steps), dtype=float)
    seconds = np.zeros((n_models, n_steps), dtype=float)
    timed = np.zeros((n_models, n_steps), dtype=float)

    for t_idx, task in enumerate(tasks_data):
        s = int(start[t_idx])
        e = int(start[t_idx + 1])
        for model, run_list in task.get('runs', {}).items():
            i = model_index.get(model)
            if i is None:
                continue
            for run in run_list:
                clean = True
                for k, score in enumerate(run.get('scores', [])):
                    j = s + k
                    if j >= e:
                        break
                    runs[i, j] += 1.0
                    if score:
                        passes[i, j] += 1.0
                    if clean:
                        reached[i, j] += 1.0
                        if score:
                            on_track[i, j] += 1.0
                    clean = clean and bool(score)

        task_id = task['id']
        trace_path = data_dir / 'tasks' / f'{task_id}.js'
        if not trace_path.exists():
            continue
        traces = read_module(trace_path).get('traces', {})
        for model, run_map in traces.items():
            i = model_index.get(model)
            if i is None:
                continue
            for entries in run_map.values():
                for k, entry in enumerate(entries):
                    j = s + k
                    if j >= e:
                        break
                    if not isinstance(entry, dict):
                        continue
                    sec = entry.get('seconds')
                    if sec is not None:
                        seconds[i, j] += float(sec)
                        timed[i, j] += 1.0

    return Benchmark(
        models=models,
        tasks=tuple(task['id'] for task in tasks_data),
        start=start,
        passes=passes,
        runs=runs,
        on_track=on_track,
        reached=reached,
        seconds=seconds,
        timed=timed,
    )