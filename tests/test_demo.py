import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from demo import END, START, apply_block, main, model_ids, render_block, summary
from prefixprobe.select import STRATEGIES

BUDGETS = [0.05, 0.10, 0.20, 0.30]
OTHERS = [strategy for strategy in STRATEGIES if strategy != "greedy"]


def _series(length, base):
    return [base + 0.01 * index for index in range(length)]


def _metrics(models=3, replicates=8):
    cells = len(BUDGETS) * len(STRATEGIES)
    pairs = len(BUDGETS) * len(OTHERS)
    metrics = {
        "tasks": 4,
        "steps": 10,
        "models": models,
        "replicates": replicates,
        "random_draws": 5,
        "budgets": list(BUDGETS),
        "model_step_rate": [50.0] * models,
        "model_solve_rate": [20.0] * models,
    }
    for key in ("step_mae", "step_mae_lo", "step_mae_hi", "solve_mae", "solve_mae_lo", "solve_mae_hi"):
        metrics[key] = _series(cells, 1.0)
    metrics["logloss"] = _series(cells, 0.5)
    metrics["spend"] = _series(cells, 0.02)
    metrics["probed_steps"] = _series(cells, 3.0)
    metrics["probed_tasks"] = _series(cells, 0.5)
    for key in ("step_err", "step_err_lo", "step_err_hi"):
        metrics[key] = _series(cells * models, -1.0)
    for key in ("greedy_vs_step_mae_diff", "greedy_vs_step_mae_diff_lo", "greedy_vs_step_mae_diff_hi"):
        metrics[key] = _series(pairs, -0.5)
    metrics["greedy_vs_step_models_better"] = [2] * pairs
    metrics["greedy_vs_step_share_replicates_better"] = [0.75] * pairs
    return metrics


class SummaryTest(unittest.TestCase):
    def test_one_line_naming_every_strategy(self):
        line = summary(_metrics())
        self.assertNotIn("\n", line)
        self.assertIn("3 models", line)
        self.assertIn("4 tasks", line)
        self.assertIn("10%", line)
        for strategy in STRATEGIES:
            self.assertIn(strategy, line)


class RenderTest(unittest.TestCase):
    def test_a_section_per_budget_and_a_model_table(self):
        block = render_block(_metrics(), ("alpha", "beta", "gamma"))
        for budget in BUDGETS:
            self.assertIn(f"### Budget {budget:.0%}", block)
        self.assertIn("Greedy minus", block)
        self.assertIn("### Per model at the 10% budget", block)
        self.assertIn("`alpha`", block)
        self.assertIn("`gamma`", block)
        self.assertIn("75.0%", block)
        self.assertTrue(block.endswith("\n"))

    def test_models_without_ids_are_numbered(self):
        block = render_block(_metrics())
        self.assertIn("`model 1`", block)
        self.assertIn("`model 3`", block)

    def test_missing_values_render_as_na(self):
        metrics = _metrics()
        metrics["logloss"][0] = float("nan")
        self.assertIn("n/a", render_block(metrics))


class ApplyBlockTest(unittest.TestCase):
    def test_replaces_only_the_marked_region(self):
        source = f"before\n{START}\nstale\n{END}\nafter\n"
        written = apply_block(source, "fresh\n")
        self.assertTrue(written.startswith("before\n"))
        self.assertTrue(written.endswith("\nafter\n"))
        self.assertIn("fresh", written)
        self.assertNotIn("stale", written)

    def test_missing_marker_raises(self):
        with self.assertRaises(ValueError):
            apply_block("no markers here", "fresh\n")
        with self.assertRaises(ValueError):
            apply_block(f"{START}\nunterminated\n", "fresh\n")


class ModelIdsTest(unittest.TestCase):
    def test_ids_come_from_the_benchmark_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "benchmark.js").write_text('export default {"data":{"models":[{"id":"a"},{"id":"b"}]}};\n')
            self.assertEqual(model_ids(tmp), ("a", "b"))

    def test_absent_module_gives_no_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(model_ids(tmp), ())


class MainTest(unittest.TestCase):
    def test_prints_one_line_and_writes_the_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp, "results")
            results.mkdir()
            (results / "metrics.json").write_text(json.dumps(_metrics()), encoding="utf-8")
            readme = Path(tmp, "README.md")
            readme.write_text(f"# Title\n\n{START}\nstale\n{END}\n\nfooter\n", encoding="utf-8")
            out = Path(tmp, "out.md")

            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = main(
                    [
                        "--results",
                        str(results),
                        "--readme",
                        str(readme),
                        "--out",
                        str(out),
                        "--data",
                        str(Path(tmp, "nodata")),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(len(printed.getvalue().splitlines()), 1)
            written = out.read_text(encoding="utf-8")
            self.assertTrue(written.startswith("# Title\n"))
            self.assertTrue(written.endswith("\nfooter\n"))
            self.assertIn("### Budget 5%", written)
            self.assertNotIn("stale", written)
            self.assertIn("`model 1`", written)

    def test_summary_only_without_a_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp, "results")
            results.mkdir()
            (results / "metrics.json").write_text(json.dumps(_metrics()), encoding="utf-8")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                self.assertEqual(main(["--results", str(results), "--data", str(Path(tmp, "nodata"))]), 0)
            self.assertEqual(len(printed.getvalue().splitlines()), 1)
            self.assertEqual(sorted(path.name for path in Path(tmp).iterdir()), ["results"])

    def test_missing_metrics_is_a_clear_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(["--results", str(Path(tmp, "absent"))])


if __name__ == "__main__":
    unittest.main()
