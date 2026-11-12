"""One-line summary and README rendering for the prefixprobe experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from prefixprobe import DATA_DIR
from prefixprobe.data import read_module
from prefixprobe.select import STRATEGIES

START = "<!-- results:start -->"
END = "<!-- results:end -->"
MODEL_BUDGET = 0.10
OTHERS = tuple(strategy for strategy in STRATEGIES if strategy != "greedy")
ABSENT = "n/a"


def load_metrics(results_dir):
    """The metrics file the experiment wrote."""
    return json.loads((Path(results_dir) / "metrics.json").read_text(encoding="utf-8"))


def model_ids(data_dir=DATA_DIR):
    """Model ids in file order, or an empty tuple when the benchmark module is not there."""
    try:
        models = read_module(Path(data_dir) / "benchmark.js")["data"]["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    return tuple(str(model["id"]) for model in models)


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _num(value, spec=".2f"):
    return ABSENT if _missing(value) else format(value, spec)


def _interval(center, low, high):
    return f"{_num(center)} [{_num(low)}, {_num(high)}]"


def _percent(value, spec=".1f", scale=100.0):
    """A share as a percentage; pass ``scale=1.0`` for values already in percent."""
    return ABSENT if _missing(value) else format(scale * value, spec) + "%"


def _cell(metrics, key, budget_index, strategy):
    """One value from a per-cell list, which runs budget by budget and strategy by strategy."""
    return metrics[key][budget_index * len(STRATEGIES) + STRATEGIES.index(strategy)]


def _cell_interval(metrics, key, budget_index, strategy):
    return _interval(
        _cell(metrics, key, budget_index, strategy),
        _cell(metrics, f"{key}_lo", budget_index, strategy),
        _cell(metrics, f"{key}_hi", budget_index, strategy),
    )


def _cell_model(metrics, key, budget_index, strategy, model_index):
    """One value from a per-cell-and-model list: the models of a cell sit together, in file order."""
    cell = budget_index * len(STRATEGIES) + STRATEGIES.index(strategy)
    return metrics[key][cell * metrics["models"] + model_index]


def _model_interval(metrics, key, budget_index, strategy, model_index):
    return _interval(
        _cell_model(metrics, key, budget_index, strategy, model_index),
        _cell_model(metrics, f"{key}_lo", budget_index, strategy, model_index),
        _cell_model(metrics, f"{key}_hi", budget_index, strategy, model_index),
    )


def _pair(metrics, key, budget_index, strategy):
    """One value from a per-pair list: budgets, and inside each budget every strategy but greedy."""
    return metrics[key][budget_index * len(OTHERS) + OTHERS.index(strategy)]


def _budget_index(budgets, target=MODEL_BUDGET):
    return min(range(len(budgets)), key=lambda index: abs(budgets[index] - target))


def summary(metrics):
    """One line: the models and tasks, then the step-rate error of every strategy at the model budget."""
    budgets = metrics["budgets"]
    index = _budget_index(budgets)
    rates = ", ".join(f"{strategy} {_num(_cell(metrics, 'step_mae', index, strategy))}" for strategy in STRATEGIES)
    return (
        f"prefixprobe: {metrics['models']} models, {metrics['tasks']} tasks, {metrics['steps']} steps, "
        f"{metrics['replicates']} task-bootstrap replicates; step-rate MAE at the {budgets[index]:.0%} budget: {rates}"
    )


def _strategy_table(metrics, budget_index):
    lines = [
        "| Strategy | Step-rate MAE | Solve-rate MAE | Log-loss (unprobed steps) | Spend | Steps probed "
        "| Tasks probed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for strategy in STRATEGIES:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{strategy}`",
                    _cell_interval(metrics, "step_mae", budget_index, strategy),
                    _cell_interval(metrics, "solve_mae", budget_index, strategy),
                    _num(_cell(metrics, "logloss", budget_index, strategy), ".3f"),
                    _percent(_cell(metrics, "spend", budget_index, strategy)),
                    _num(_cell(metrics, "probed_steps", budget_index, strategy), ".1f"),
                    _num(_cell(metrics, "probed_tasks", budget_index, strategy), ".1f"),
                ]
            )
            + " |"
        )
    return lines


def _pair_interval(metrics, key, budget_index, strategy):
    return _interval(
        _pair(metrics, key, budget_index, strategy),
        _pair(metrics, f"{key}_lo", budget_index, strategy),
        _pair(metrics, f"{key}_hi", budget_index, strategy),
    )


def _compare_table(metrics, budget_index):
    lines = [
        "| Greedy minus | Step-rate MAE difference | Solve-rate MAE difference | Models better on step rate "
        "| Replicates better on step rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strategy in OTHERS:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{strategy}`",
                    _pair_interval(metrics, "greedy_vs_step_mae_diff", budget_index, strategy),
                    _pair_interval(metrics, "greedy_vs_solve_mae_diff", budget_index, strategy),
                    f"{_num(_pair(metrics, 'greedy_vs_step_models_better', budget_index, strategy), '.0f')}"
                    f"/{metrics['models']}",
                    _percent(_pair(metrics, "greedy_vs_step_share_replicates_better", budget_index, strategy)),
                ]
            )
            + " |"
        )
    return lines


def _budget_section(metrics, budget_index, budget):
    return [
        f"### Budget {budget:.0%} of the full-benchmark cost",
        "",
        *_strategy_table(metrics, budget_index),
        "",
        f"Greedy minus each alternative at {budget:.0%}; a negative difference favours greedy.",
        "",
        *_compare_table(metrics, budget_index),
        "",
    ]


def _model_section(metrics, ids):
    budgets = metrics["budgets"]
    index = _budget_index(budgets)
    columns = ["Model", "True step rate", "True solve rate"] + [f"`{s}` step error" for s in STRATEGIES]
    lines = [
        f"### Per model at the {budgets[index]:.0%} budget",
        "",
        "Signed step-rate error: the probe's estimate minus the model's true step rate, negative where the probe "
        "underestimates.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for model_index in range(metrics["models"]):
        label = ids[model_index] if model_index < len(ids) else f"model {model_index + 1}"
        cells = [
            f"`{label}`",
            _percent(metrics["model_step_rate"][model_index], scale=1.0),
            _percent(metrics["model_solve_rate"][model_index], scale=1.0),
        ]
        for strategy in STRATEGIES:
            cells.append(_model_interval(metrics, "step_err", index, strategy, model_index))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_block(metrics, ids=()):
    """The generated results region: one section per budget, then the per-model table."""
    lines = [
        "Generated by `demo.py` from `results/metrics.json`. Errors and intervals are percentage points of the "
        "true rate; intervals are 95% percentiles over task-bootstrap replicates.",
        "",
    ]
    for budget_index, budget in enumerate(metrics["budgets"]):
        lines += _budget_section(metrics, budget_index, budget)
    lines += _model_section(metrics, ids)
    return "\n".join(lines).rstrip("\n") + "\n"


def apply_block(text, block):
    """Replace the marked results region of text with block, leaving everything else alone."""
    head, marker, rest = text.partition(START)
    if not marker:
        raise ValueError(f"missing {START} marker")
    _, marker, tail = rest.partition(END)
    if not marker:
        raise ValueError(f"missing {END} marker")
    return f"{head}{START}\n\n{block}{END}{tail}"


def main(argv=None):
    """Print the summary and, given a README, write it back with the results block rendered."""
    parser = argparse.ArgumentParser(description="Summarise and render the prefixprobe experiment.")
    parser.add_argument("--results", default="results", help="directory holding metrics.json")
    parser.add_argument("--readme", help="README whose results block to render")
    parser.add_argument("--out", help="path to write the rendered README to (default: --readme)")
    parser.add_argument("--data", default=str(DATA_DIR), help="directory holding benchmark.js, for model ids")
    args = parser.parse_args(argv)

    try:
        metrics = load_metrics(args.results)
    except OSError as error:
        raise SystemExit(f"cannot read metrics from {args.results}: run experiment.py first ({error})") from error

    print(summary(metrics))

    if args.readme:
        block = render_block(metrics, model_ids(args.data))
        source = Path(args.readme).read_text(encoding="utf-8")
        Path(args.out or args.readme).write_text(apply_block(source, block), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
