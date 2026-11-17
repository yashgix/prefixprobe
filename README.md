# prefixprobe

If a lab can afford only a slice of [Multi-step SWE](https://github.com/Klavis-AI/Multi-step-SWE) for a new model,
which slices should it run — and how well do those runs predict the model's full step rate and solve rate?

**Write-up: [yashgix.github.io/prefixprobe](https://yashgix.github.io/prefixprobe/)** (the same page is in
[`docs/index.html`](docs/index.html))

Multi-step SWE chains **5–15 dependent engineering requests** on one codebase, so the natural unit is not the task but
the *prefix*: step k of a task can only run after steps 1..k-1, and a model's own earlier work is what step k is
scored against. Probing the first k steps of a task therefore costs the sum of those k steps' costs and yields a
principled sample of everything behind them. `prefixprobe` uses exactly that structure: it pays for prefixes, fits two
ability offsets from them, and extrapolates to the steps and tasks it did not run.

Klavis AI reports that "The strongest model passes 56.8% of steps but solves only 18.5% of complete tasks" and that
"38 of 69 tasks remain unsolved" ([blog](https://www.klavis.ai/blog/multi-step-swe)), and names a limitation: the
benchmark does "not yet isolate accumulated errors from intrinsic step difficulty". Both facts argue for a cheap,
prefix-aware estimate of a new model rather than another complete run. This repository is an independent analysis; it
is not affiliated with Klavis AI and says nothing about Klavis's own process.

## What the experiment found

Each of the nine models in the benchmark's published results is held out in turn. A planner spends a budget, measured as a share
of the full benchmark's cost in recorded agent time, on task prefixes. The estimator then predicts the held-out
model's full step rate and solve rate, and the prediction is scored against its real results. The dependency-aware
`greedy` planner is compared with four simple baselines: `none` (no probe), `random` prefixes, `uniform` (the same
depth in every task) and `cheapest_tasks` (whole tasks, cheapest first).

- **Step rate, point estimates.** At a 10% budget, greedy's mean absolute error is 2.12 percentage points, against
  4.41 for random and 4.37 for cheapest_tasks. For the strongest model (`opus`, true step rate 56.8%) the greedy
  probe's estimate is 3.49 points low, and with no probe it is 23.83 points low.
- **Not significant against random.** The paired greedy-minus-random interval includes zero at every budget, for
  step rate and for solve rate (at 10%: step rate -2.29, interval -4.23 to 1.27). With nine models, this study does
  not show that greedy beats random selection.
- **No solve-rate gain over the simple baselines.** Greedy's solve-rate error is higher than cheapest_tasks' at
  every budget (2.08 against 0.96 at 10%, where cheapest_tasks' point error sits below its own interval, as *Limits*
  explains), and its difference from random includes zero. Its only solve-rate intervals that exclude zero are against `none`
  at 10%, 20% and 30% (and `uniform` at 10%, where it spends nothing).
- **Where greedy's intervals do exclude zero:** on step rate against `none` at every budget (and `uniform`, which
  cannot afford any depth at 5% and 10% and so spends nothing), and against cheapest_tasks at 20% and 30%.

So the result is modest: a better step-rate estimate than no probe and than whole cheap tasks at the larger
budgets, not a proven better selection method. The numbers are quoted from the generated tables under
[Results](#results).

## The estimate, and what is actually new

The estimator is a standard one-parameter item-response offset, fitted twice: a step's difficulty is the log-odds that
the other models pass it, and a new model gets one ability offset for passing a step and a second, separately fitted
offset for passing it after a clean prefix (which drives the solve rate), each by MAP under a normal prior. That is
textbook item-response theory, not a new statistical method; the prior art is cited below.

What is specific here is the **planning problem**. Items arrive in dependent chains, each item has its own cost, and
the bill for a chain is the sum of its prefix. The planner spends a fixed budget on prefixes, gain per unit cost
first, and the evaluation holds one model out entirely so the estimate never sees the results it is predicting. Of
the unprobed steps it uses only their run counts: how many runs each task gets is fixed before probing, so the
counts weight the extrapolation and no unprobed outcome enters it.

## Data

The benchmark's published results — `benchmark.js` plus one module per task — are fetched into `data/` and checked
against a sha256 manifest. Fetch and check them with:

```bash
python tools/fetch_data.py           # download from multistepswe.com
python tools/fetch_data.py --verify  # check against data/MANIFEST.json (also: make data-check)
```

Each module is a JavaScript file whose text carries one JSON object; it is read as text with `json` and never
imported or executed. The published results cover 69 tasks, 574 steps and nine models with four runs per task (one
model has three runs on one task), and runs continue after a failed step — so a run's first k scores are exactly what
a probe of that task's first k steps would have produced.

**Terms.** The site states no license for this data, so this repository does not include it. It carries only
code, `tools/fetch_data.py`, `data/MANIFEST.json` and the derived aggregate results in `results/metrics.json`, never
the benchmark text, and nothing here quotes task instructions or any other part of the data. To reproduce the
experiment, fetch the data yourself for your own analysis; `.gitignore` keeps the fetched files out of commits.

## The cost proxy

The published results carry a dollar cost per model (`costData` in `benchmark.js`), but the traces carry something
better suited to planning: the agent seconds each run spent on each step. The probe's cost unit is therefore the mean
recorded agent time for a run of a step (`seconds / timed`), summed over the prefix. It stands in for money; it is
not money, and it says nothing about how fast a *new* model would be.

## Running it

Python 3.12 or later:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy==2.5.4 scipy==1.18.1 ruff
python tools/fetch_data.py                         # needed by the experiment and the data tests
make test                                          # unittest suite (data tests skip without data/)
make lint                                          # ruff check + format check
python experiment.py --out results/metrics.json    # ~7 minutes on one core, ~23 KB
python demo.py                                     # one-line summary
python demo.py --readme README.md --out README.md  # refresh the results block below
```

`experiment.py --replicates N` shortens the task bootstrap for a quick look; the committed metrics use the default
2000. Two runs of the same command produce byte-identical files, so a rerun is checkable by `git diff`.

## Results

Everything below is generated by `demo.py`; the summary at the top quotes these tables by hand. Errors are
percentage points of the true rate, intervals are 95% percentiles over task-bootstrap replicates, and the fair
comparison between strategies is the paired *greedy minus alternative* difference rather than the absolute errors
(see *Limits*).

<!-- results:start -->

Generated by `demo.py` from `results/metrics.json`. Errors and intervals are percentage points of the true rate; intervals are 95% percentiles over task-bootstrap replicates.

### Budget 5% of the full-benchmark cost

| Strategy | Step-rate MAE | Solve-rate MAE | Log-loss (unprobed steps) | Spend | Steps probed | Tasks probed |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `random` | 6.23 [3.42, 10.15] | 4.21 [1.98, 7.25] | 0.412 | 5.0% | 29.0 | 4.5 |
| `uniform` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `cheapest_tasks` | 4.62 [3.37, 10.47] | 2.14 [1.41, 6.42] | 0.392 | 5.0% | 39.9 | 7.4 |
| `greedy` | 3.96 [2.76, 7.09] | 2.43 [1.58, 5.21] | 0.383 | 5.0% | 32.7 | 30.8 |

Greedy minus each alternative at 5%; a negative difference favours greedy.

| Greedy minus | Step-rate MAE difference | Solve-rate MAE difference | Models better on step rate | Replicates better on step rate |
| --- | --- | --- | --- | --- |
| `none` | -7.14 [-9.34, -3.89] | -2.75 [-4.40, 0.12] | 6/9 | 100.0% |
| `random` | -2.26 [-5.83, 2.05] | -1.78 [-4.45, 1.79] | 5/9 | 79.1% |
| `uniform` | -7.14 [-9.34, -3.89] | -2.75 [-4.40, 0.12] | 6/9 | 100.0% |
| `cheapest_tasks` | -0.66 [-5.94, 2.06] | 0.29 [-3.61, 2.50] | 5/9 | 78.9% |

### Budget 10% of the full-benchmark cost

| Strategy | Step-rate MAE | Solve-rate MAE | Log-loss (unprobed steps) | Spend | Steps probed | Tasks probed |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `random` | 4.41 [2.34, 6.83] | 3.08 [1.53, 5.35] | 0.398 | 10.0% | 57.9 | 7.9 |
| `uniform` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `cheapest_tasks` | 4.37 [2.64, 8.59] | 0.96 [0.99, 4.16] | 0.383 | 10.0% | 75.6 | 13.3 |
| `greedy` | 2.12 [1.80, 4.57] | 2.08 [1.27, 4.59] | 0.368 | 10.0% | 60.7 | 48.7 |

Greedy minus each alternative at 10%; a negative difference favours greedy.

| Greedy minus | Step-rate MAE difference | Solve-rate MAE difference | Models better on step rate | Replicates better on step rate |
| --- | --- | --- | --- | --- |
| `none` | -8.98 [-10.63, -5.88] | -3.10 [-4.61, -0.61] | 8/9 | 100.0% |
| `random` | -2.29 [-4.23, 1.27] | -1.00 [-2.89, 1.92] | 8/9 | 85.8% |
| `uniform` | -8.98 [-10.63, -5.88] | -3.10 [-4.61, -0.61] | 8/9 | 100.0% |
| `cheapest_tasks` | -2.25 [-5.57, 0.63] | 1.12 [-1.73, 2.58] | 6/9 | 93.2% |

### Budget 20% of the full-benchmark cost

| Strategy | Step-rate MAE | Solve-rate MAE | Log-loss (unprobed steps) | Spend | Steps probed | Tasks probed |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `random` | 3.02 [1.68, 4.79] | 2.12 [1.10, 3.70] | 0.391 | 20.0% | 115.3 | 14.8 |
| `uniform` | 1.60 [1.63, 4.18] | 2.21 [1.49, 4.27] | 0.386 | 14.0% | 69.0 | 69.0 |
| `cheapest_tasks` | 5.31 [2.87, 7.67] | 1.17 [0.67, 2.99] | 0.380 | 19.9% | 140.4 | 23.1 |
| `greedy` | 1.39 [1.29, 3.52] | 1.79 [1.12, 3.58] | 0.356 | 20.0% | 122.0 | 56.7 |

Greedy minus each alternative at 20%; a negative difference favours greedy.

| Greedy minus | Step-rate MAE difference | Solve-rate MAE difference | Models better on step rate | Replicates better on step rate |
| --- | --- | --- | --- | --- |
| `none` | -9.72 [-11.20, -6.92] | -3.39 [-5.08, -1.34] | 9/9 | 100.0% |
| `random` | -1.63 [-2.73, 1.08] | -0.33 [-1.70, 1.71] | 8/9 | 78.8% |
| `uniform` | -0.22 [-1.97, 0.89] | -0.42 [-1.74, 0.61] | 4/9 | 75.2% |
| `cheapest_tasks` | -3.92 [-5.39, -0.48] | 0.62 [-1.08, 2.02] | 7/9 | 99.1% |

### Budget 30% of the full-benchmark cost

| Strategy | Step-rate MAE | Solve-rate MAE | Log-loss (unprobed steps) | Spend | Steps probed | Tasks probed |
| --- | --- | --- | --- | --- | --- | --- |
| `none` | 11.11 [9.47, 13.29] | 5.18 [3.47, 7.43] | 0.458 | 0.0% | 0.0 | 0.0 |
| `random` | 2.32 [1.25, 3.68] | 1.72 [0.88, 2.89] | 0.389 | 30.0% | 172.2 | 21.7 |
| `uniform` | 1.14 [1.04, 3.16] | 1.29 [0.97, 3.14] | 0.381 | 26.7% | 138.0 | 69.0 |
| `cheapest_tasks` | 3.98 [2.07, 6.13] | 0.80 [0.42, 2.11] | 0.363 | 30.0% | 197.0 | 32.2 |
| `greedy` | 1.28 [1.07, 2.83] | 1.21 [0.84, 2.75] | 0.341 | 30.0% | 184.0 | 58.8 |

Greedy minus each alternative at 30%; a negative difference favours greedy.

| Greedy minus | Step-rate MAE difference | Solve-rate MAE difference | Models better on step rate | Replicates better on step rate |
| --- | --- | --- | --- | --- |
| `none` | -9.82 [-11.55, -7.51] | -3.97 [-5.53, -1.98] | 8/9 | 100.0% |
| `random` | -1.03 [-2.11, 1.01] | -0.50 [-1.41, 1.22] | 8/9 | 73.2% |
| `uniform` | 0.14 [-1.08, 0.77] | -0.08 [-1.05, 0.49] | 4/9 | 57.0% |
| `cheapest_tasks` | -2.70 [-4.32, -0.18] | 0.41 [-0.72, 1.75] | 8/9 | 98.4% |

### Per model at the 10% budget

Signed step-rate error: the probe's estimate minus the model's true step rate, negative where the probe underestimates.

| Model | True step rate | True solve rate | `none` step error | `random` step error | `uniform` step error | `cheapest_tasks` step error | `greedy` step error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `opus` | 56.8% | 18.5% | -23.83 [-28.84, -18.53] | 0.26 [-13.54, 11.08] | -23.83 [-28.84, -18.53] | -7.48 [-18.64, 4.00] | -3.49 [-11.38, 4.88] |
| `fable` | 49.1% | 17.8% | -15.13 [-19.17, -11.02] | -0.74 [-11.14, 9.50] | -15.13 [-19.17, -11.02] | -1.06 [-10.46, 7.76] | 2.03 [-4.12, 8.86] |
| `gemini` | 40.1% | 10.1% | -5.07 [-9.64, -0.66] | 0.33 [-13.91, 12.64] | -5.07 [-9.64, -0.66] | 1.58 [-11.18, 10.77] | 1.62 [-6.07, 8.61] |
| `sol` | 37.7% | 10.1% | -2.34 [-7.00, 1.70] | -0.64 [-12.32, 13.64] | -2.34 [-7.00, 1.70] | -2.68 [-10.38, 7.63] | 1.12 [-4.21, 6.50] |
| `astra` | 36.7% | 6.9% | -1.26 [-5.39, 2.91] | -0.24 [-11.49, 11.37] | -1.26 [-5.39, 2.91] | -3.98 [-12.10, 8.04] | 1.61 [-3.92, 7.03] |
| `deepseek` | 35.8% | 6.5% | -0.29 [-3.94, 3.31] | 0.55 [-11.52, 10.55] | -0.29 [-3.94, 3.31] | 8.07 [-1.48, 16.09] | 0.23 [-5.25, 5.09] |
| `qwen` | 24.4% | 4.0% | 12.65 [9.27, 16.08] | 0.36 [-10.35, 9.73] | 12.65 [9.27, 16.08] | 8.21 [0.99, 15.27] | 3.45 [-2.90, 7.26] |
| `mimo` | 22.1% | 2.9% | 15.10 [11.28, 18.83] | -0.57 [-11.27, 11.57] | 15.10 [11.28, 18.83] | 2.47 [-4.78, 8.37] | 4.77 [-0.08, 8.99] |
| `grok` | 13.9% | 1.4% | 24.28 [20.49, 28.08] | 0.00 [-7.31, 6.66] | 24.28 [20.49, 28.08] | 3.83 [-2.41, 9.18] | 0.77 [-3.73, 5.14] |
<!-- results:end -->

## Prior art

Choosing items by Fisher information is adaptive testing, the standard tool of item-response theory; selecting a
small, independent, equally priced item subset to estimate a benchmark score is what tinyBenchmarks does (Polo et
al., ICML 2024, [arXiv 2402.14992](https://arxiv.org/abs/2402.14992)); adaptive testing for LLMs is ATLAS
([arXiv 2511.04689](https://arxiv.org/abs/2511.04689)); PTA-IRT ([arXiv 2609.01603](https://arxiv.org/abs/2609.01603))
picks budgeted calibration subsets for SWE agents from their trajectories; budget-constrained subset selection for
agents ([arXiv 2605.10834](https://arxiv.org/abs/2605.10834)) and proxy selection such as PACE
([arXiv 2607.02032](https://arxiv.org/abs/2607.02032)) are close cousins; and the planner's gain-over-cost rule is
the classical heuristic for the precedence-constrained knapsack. Selecting items under a cost budget is established
work.

The part specific to this project is that items come in dependent chains — step k needs steps 1..k-1, each with its
own cost — together with a held-out-model evaluation of that on Multi-step SWE. A short search on 10 November 2026
found no work handling such prefix constraints; that is not a claim that none exists.

## Limits

- **Nine models.** Held-out results are nine folds, and the bootstrap resamples tasks, not models. A future model
generation may rank steps differently from these nine.
- **The cost proxy is agent time, not dollars,** taken from the other models, and a new model's own speed is unknown.
- **Log-loss is computed on each plan's own unprobed steps,** so it is not comparable across strategies. The
  step-rate and solve-rate errors are.
- **Absolute errors are biased upward by the resampling,** so a point MAE can sit below its own interval. The paired
  differences are the fair comparison; each budget has its own table of them.
- **No significant gain over random.** Every paired interval against `random`, for step rate and solve rate, includes
  zero, and greedy's solve-rate error is higher than `cheapest_tasks`'s at every budget. The paired tables show where
  greedy's intervals do exclude zero.
- **Planning is fixed in advance,** not adaptive, and the planner's design was checked on the same nine models it is
  evaluated on.
- **The estimator is deliberately simple** — two separately fitted offsets, one normal prior. It is a baseline that the prefix structure
  has to earn its keep against, not a claim about the best possible estimator.

## Layout

```
prefixprobe/   data loading, difficulty model, folds, planners, scoring, experiment runner
tests/         unittest suite; data-dependent tests skip with a message when data/ is absent
data/          MANIFEST.json; tools/fetch_data.py downloads the benchmark modules here (never committed)
tools/         fetch_data.py
experiment.py  command line: writes results/metrics.json
demo.py        one-line summary and README rendering
docs/          design notes and the write-up page
results/       metrics.json from the committed run
```

## License

MIT for this repository's code (`LICENSE`). The Multi-step SWE results data is not included and is not covered by
this license.
