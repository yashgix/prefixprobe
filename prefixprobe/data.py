"""Loading and reshaping Multi-step SWE benchmark data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_module(path: str | Path) -> object:
    """Return the exported JSON object from a JavaScript module file."""
    text = Path(path).read_text()
    start = text.index("{")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


@dataclass(frozen=True)
class Benchmark:
    models: tuple
    tasks: tuple
    start: np.ndarray
    passes: np.ndarray
    runs: np.ndarray
    on_track: np.ndarray
    reached: np.ndarray
    seconds: np.ndarray
    timed: np.ndarray

    @property
    def task_of(self) -> np.ndarray:
        return np.repeat(np.arange(self.start.size - 1), np.diff(self.start))

    @property
    def position(self) -> np.ndarray:
        return np.concatenate([np.arange(n) for n in np.diff(self.start)])

    def within_task_cumsum(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for start, end in zip(self.start[:-1], self.start[1:]):
            out[start:end] = np.cumsum(x[start:end])
        return out

    def resample(self, task_idx) -> "Benchmark":
        """Return a benchmark consisting of the selected tasks in order.

        Repeats are separate tasks: each occurrence gets its own task
        boundary and its own columns in every count array.
        """
        columns = []
        starts = [0]
        chosen_tasks = []
        for idx in task_idx:
            start = int(self.start[idx])
            end = int(self.start[idx + 1])
            columns.append(slice(start, end))
            starts.append(starts[-1] + (end - start))
            chosen_tasks.append(self.tasks[idx])

        names = ("passes", "runs", "on_track", "reached", "seconds", "timed")
        if columns:
            gathered = [
                np.concatenate([getattr(self, name)[:, sl] for sl in columns], axis=1)
                for name in names
            ]
        else:
            gathered = [np.empty((len(self.models), 0)) for _ in names]

        return Benchmark(
            models=self.models,
            tasks=tuple(chosen_tasks),
            start=np.asarray(starts, dtype=int),
            **dict(zip(names, gathered)),
        )


def load(data_dir: str | Path = DATA_DIR) -> Benchmark:
    data_dir = Path(data_dir)
    payload = read_module(data_dir / "benchmark.js")["data"]

    model_ids = tuple(model["id"] for model in payload["models"])
    model_index = {model_id: i for i, model_id in enumerate(model_ids)}
    task_infos = payload["tasks"]
    task_ids = tuple(task["id"] for task in task_infos)
    lengths = [len(task["steps"]) for task in task_infos]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(int)

    total_steps = int(starts[-1])
    n_models = len(model_ids)
    passes = np.zeros((n_models, total_steps))
    runs = np.zeros((n_models, total_steps))
    on_track = np.zeros((n_models, total_steps))
    reached = np.zeros((n_models, total_steps))
    seconds = np.zeros((n_models, total_steps))
    timed = np.zeros((n_models, total_steps))

    for task_idx, task in enumerate(task_infos):
        first = int(starts[task_idx])
        length = int(lengths[task_idx])

        for model_id, run_list in task.get("runs", {}).items():
            model = model_index.get(model_id)
            if model is None:
                continue
            for run in run_list:
                scores = run["scores"]
                clean_prefix = True
                for step_idx, score in enumerate(scores):
                    col = first + step_idx
                    runs[model, col] += 1
                    if clean_prefix:
                        reached[model, col] += 1
                    if score:
                        passes[model, col] += 1
                        if clean_prefix:
                            on_track[model, col] += 1
                    clean_prefix = clean_prefix and bool(score)

        module_path = data_dir / "tasks" / f"{task['id']}.js"
        if module_path.exists():
            traces = read_module(module_path).get("traces", {})
            for model_id, run_list in task.get("runs", {}).items():
                model = model_index.get(model_id)
                if model is None:
                    continue
                model_traces = traces.get(model_id, {})
                for run in run_list:
                    run_traces = model_traces.get(run.get("id"))
                    if not run_traces:
                        continue
                    for step_idx, record in enumerate(run_traces):
                        value = record.get("seconds")
                        if value is None:
                            continue
                        col = first + step_idx
                        seconds[model, col] += value
                        timed[model, col] += 1

    return Benchmark(
        models=model_ids,
        tasks=task_ids,
        start=starts,
        passes=passes,
        runs=runs,
        on_track=on_track,
        reached=reached,
        seconds=seconds,
        timed=timed,
    )
