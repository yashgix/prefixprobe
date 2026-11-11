# Design notes

## 1. What one row of the data means

The benchmark is a list of tasks; a task is a list of ordered steps; every step was attempted by four runs of every
model. The data is stored flat: steps are concatenated task by task in file order, and `Benchmark.start` holds the
index of each task's first step followed by the total step count, so `task_of` and `position` are pure arithmetic
over that array.

Per step the loader counts, per model:

- `runs` — runs that reached the step at all (a model with no run on a task contributes nothing there), and `passes` —
runs that passed it;
- `reached` — runs that passed every earlier step of the task (equal to `runs` at the first step);
- `on_track` — runs that passed the step *and* every earlier step.

Runs continue after a failure, and each step is scored on the agent's own earlier work. That is what makes a prefix a
valid sample: a run's first k scores are exactly what a probe of that task's first k steps would have produced.

`seconds` is the sum of the recorded durations of a step's runs and `timed` the number of runs that recorded one; a
null duration counts as nothing. A model that publishes no timings therefore contributes no time and no denominator,
and does not distort the cost proxy. Qwen has 275 runs rather than 276 because one task has three, so every rate is
taken over run counts rather than over `4 · steps`.

## 2. Two difficulty scales

**Marginal difficulty `d`** is the log-odds of a pass, `logit((passes + 0.5) / (runs + 1))`, over the models that are
*not* being held out. It describes how often a step is passed at all, luck after a broken prefix included.

**On-track difficulty `e`** is the same transform of `on_track / reached`: how often a step is passed given that
everything before it in the task worked. This is the scale the solve rate lives on, because solving a task is a chain
of conditional passes.

The half-pass and one-extra-trial smoothing keeps both finite on steps nobody or everybody passed, which happens often
with nine models on a five-step task.

## 3. The ability model

A model's pass probability on a step is `sigmoid(a + d)` with one offset `a` shared by all its steps, and a
`Normal(0, 2)` prior to keep it sane on small probes. `fit_ability` maximises the posterior with Newton's method from
zero: gradient `Σ(passes − trials·p) − a/σ²`, Hessian `Σ trials·p(1−p) + 1/σ²`, at most 100 steps, stopping once the
Newton step is below `1e-12`. Empty input returns `0.0`, which is the prior mode.

The same fitting routine is reused on the on-track scale with `e`, giving `a_e`: the conditional pass probability of a
step after a clean prefix.

`task_variance` estimates how much ability varies *across tasks*, `τ²`. For each other model it fits `a`, forms the
per-task residual `r = Σ(passes − runs·p)` and the per-task information `I = Σ runs·p(1−p)`, and pools
`τ² = max(0, Σ(r² − I) / Σ I²)`. It is a moment estimate, not a hierarchical fit; its job is only to say how quickly
extra steps of the *same* task stop telling you about the model.

`step_information` is the Fisher information about the two abilities for a model as strong as the others' average:
per step `4 · (p(1−p) + reach·q(1−q))` with `p = sigmoid(d)`, `q = sigmoid(e)` and `reach` the product of `q` over the
earlier steps of the task (1 at the first step). The first term is what the step says about the marginal scale; the
second is what it says about the on-track scale, and it is worth nothing on a step a clean prefix rarely reaches.

## 4. Held-out protocol

For a held-out model, `make_fold` uses **only the other models' counts**: `d`, `e`, `cost`, `info` and `τ²` are all
built from the summed counts of the remaining models. The held-out model's own results enter at scoring time and
nowhere else, through its probe. A test perturbs the held-out model's counts and checks that the fold is unchanged,
which is the property the whole evaluation rests on.

The probe is a per-task depth. Steps with `position < depth` are probed and are the only ones used for estimation;
every other step of the held-out model is used only to compute the true rates and the log-loss.

## 5. The cost proxy

The data has no per-step dollar cost — the published figure is one average per model — so the planner works in
agent time: `cost = seconds / timed`, the mean recorded agent seconds per run of a step, taken over all models. It is
a proxy: it prices an hour of a new model as an hour of the average old one, and it ignores that a weaker model may
need longer. Planning in dollars would need per-step billing the site does not publish.

## 6. Strategies

A plan is one depth per task; a probe of depth `j` runs all of the model's runs for steps 1..j. `prefix_cost` sums the
costs of the probed steps, so a deep prefix pays for its whole prefix.

- **none** — all zeros. The baseline that spends nothing and guesses from `d` alone.
- **random** — tasks visited in a random order, each taking the deepest prefix that still fits. Averaged over 200
draws, so it is not one lucky permutation.
- **uniform** — one depth `k` for every task, the largest that fits; it under-spends by construction, which is honest
about not being able to spend the remainder.
- **cheapest_tasks** — tasks sorted by total cost, cheapest first, each taking the deepest fitting prefix. This is the
"buy the most tasks" plan and the natural cost-only baseline.
- **greedy** — the dependency-aware, cost-aware planner. State is the current depth per task. A candidate extends one
task from its current depth to a deeper step `s`; it costs the steps it adds and gains
`g(I_new) − g(I_old)` with `g(I) = I / (1 + τ²·I)` and `I` the information of the task's probed steps. The best
feasible gain per unit cost is taken, ties going to the lowest flat step index, until nothing feasible is left.

The damping in `g` is the point of the whole exercise: information from the same task saturates, because a model's
ability varies from task to task by `τ²`. A planner that ignored `τ²` would keep deepening one long task; the damped
version spreads out once a task's information stops paying.

## 7. The experiment

Budgets are shares of the fold's full-benchmark cost (`fold.cost.sum()`), so 10% means "a tenth of running everything".
`SEED = 20261110`; the random streams are keyed by use — `[SEED, 1, i, j, k]` for point draws, `[SEED, 2, b]` for
the replicate's task sample, `[SEED, 3, b, i, j]` for a replicate's plan. Keys rather than one long stream means a
cell is reproducible on its own and adding a cell does not shift the others.

Point results are one plan per cell except for `random`, which averages 200 draws. Replicates are a task bootstrap:
draw `T` tasks with replacement, resample the benchmark, refit every fold on the sample, and score every cell on it.
Intervals are the 2.5th and 97.5th percentiles over 2000 replicates with NumPy's default interpolation.

Lists follow `CELLS = budgets × strategies` and `PAIRS = budgets × strategies without greedy`; inside a cell, models
are in file order. Greedy-versus-others differences are computed *within* a replicate before taking percentiles, so
the task sample cancels — that is why the paired difference is quoted rather than two overlapping intervals.

## 8. Trade-offs and what the numbers cannot say

- Absolute error intervals are shifted upward by resampling; only the paired differences are fair.
- Log-loss is evaluated on each plan's own unprobed steps, so a plan that probes little is scored on harder steps. It
  is a diagnostic of calibration, not a league table.
- Nine folds is nine points. The bootstrap says how much of the result is task sampling, not how much is model
  sampling; a tenth model is a genuinely new draw.
- Planning is fixed in advance. An adaptive planner that spends the next prefix depending on what it saw would be
  better and would need a different evaluation to stay honest.
- The planner is a classical greedy heuristic for a precedence-constrained knapsack, not an optimal solution; on a
  budget this small the gap to optimal is not measured here.
