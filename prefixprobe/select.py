"""Probe planners for budgeted prefix evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .data import Benchmark
    from .fold import Fold

STRATEGIES = ("none", "random", "uniform", "cheapest_tasks", "greedy")


def _task_lengths(bench: Benchmark) -> np.ndarray:
    return np.diff(bench.start).astype(int)


def prefix_cost(bench: Benchmark, cost: np.ndarray, depths: np.ndarray) -> float:
    """Cost of probing the first ``depths[task]`` steps of every task."""
    depths = np.asarray(depths)
    return float(cost[bench.position < depths[bench.task_of]].sum())


def fill_in_order(bench: Benchmark, cost: np.ndarray, order, budget: float) -> np.ndarray:
    """Give tasks their deepest affordable prefix, in the order supplied."""
    depths = np.zeros(len(bench.tasks), dtype=int)
    left = float(budget)
    for task in order:
        task = int(task)
        lo, hi = int(bench.start[task]), int(bench.start[task + 1])
        if lo == hi:
            continue
        cumulative = np.cumsum(cost[lo:hi])
        depth = int(np.searchsorted(cumulative, left, side="right"))
        if depth:
            left -= float(cumulative[depth - 1])
        depths[task] = depth
    return depths


def random_plan(bench: Benchmark, cost: np.ndarray, budget: float, rng) -> np.ndarray:
    return fill_in_order(bench, cost, rng.permutation(len(bench.tasks)), budget)


def uniform_plan(bench: Benchmark, cost: np.ndarray, budget: float) -> np.ndarray:
    lengths = _task_lengths(bench)
    depths = np.zeros_like(lengths)
    if not len(lengths):
        return depths
    for depth in range(int(lengths.max()), 0, -1):
        candidate = np.minimum(depth, lengths)
        if prefix_cost(bench, cost, candidate) <= budget:
            return candidate
    return depths


def cheapest_tasks_plan(bench: Benchmark, cost: np.ndarray, budget: float) -> np.ndarray:
    totals = np.array([cost[int(bench.start[t]) : int(bench.start[t + 1])].sum() for t in range(len(bench.tasks))])
    return fill_in_order(bench, cost, np.argsort(totals, kind="stable"), budget)


def _g(info: float, tau2: float) -> float:
    return info / (1.0 + tau2 * info)


def greedy_plan(bench: Benchmark, fold: Fold, budget: float) -> np.ndarray:
    """Grow the plan one cost-effective prefix extension at a time."""
    n_tasks = len(bench.tasks)
    lengths = _task_lengths(bench)
    depths = np.zeros(n_tasks, dtype=int)
    cost = np.asarray(fold.cost, dtype=float)
    info = np.asarray(fold.info, dtype=float)
    tau2 = float(fold.tau2)
    left = float(budget)
    # Prefix sums and their gains never change, so they are computed once per task and read as Python floats.
    cost_prefixes, gain_prefixes = [], []
    for task in range(n_tasks):
        lo, hi = int(bench.start[task]), int(bench.start[task + 1])
        cost_prefixes.append(np.cumsum(cost[lo:hi]).tolist())
        gain_prefixes.append([_g(x, tau2) for x in np.cumsum(info[lo:hi]).tolist()])

    while True:
        best_ratio = -np.inf
        best_task = -1
        best_depth = 0
        best_cost = 0.0

        for task in range(n_tasks):
            old = int(depths[task])
            length = int(lengths[task])
            if old >= length:
                continue

            cost_prefix = cost_prefixes[task]
            gain_prefix = gain_prefixes[task]
            old_gain = gain_prefix[old - 1] if old else _g(0.0, tau2)
            old_cost = cost_prefix[old - 1] if old else 0.0

            for depth in range(old + 1, length + 1):
                add_cost = cost_prefix[depth - 1] - old_cost
                if add_cost > left:
                    break
                gain = gain_prefix[depth - 1] - old_gain
                if gain <= 0:
                    continue
                ratio = gain / add_cost if add_cost > 0 else np.inf
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_task = task
                    best_depth = depth
                    best_cost = add_cost

        if best_task < 0:
            break
        depths[best_task] = best_depth
        left -= best_cost

    return depths


def make_plan(strategy: str, bench: Benchmark, fold: Fold, budget: float, rng) -> np.ndarray:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if strategy == "none":
        return np.zeros(len(bench.tasks), dtype=int)
    if strategy == "random":
        return random_plan(bench, fold.cost, budget, rng)
    if strategy == "uniform":
        return uniform_plan(bench, fold.cost, budget)
    if strategy == "cheapest_tasks":
        return cheapest_tasks_plan(bench, fold.cost, budget)
    return greedy_plan(bench, fold, budget)
