# Interview notes

## The problem

If you can only run the first few steps of some of the tasks, `prefixprobe` tells you which prefixes to buy and how close the resulting estimate of step rate and solve rate gets to a full run.

Each Multi-step SWE task chains 5–15 dependent engineering requests on one codebase ([README](https://raw.githubusercontent.com/Klavis-AI/Multi-step-SWE/main/README.md)), and there are 69 tasks ([blog](https://www.klavis.ai/blog/multi-step-swe)). In the published results (`data/benchmark.js`) each of the nine models has four runs of every task, except one task where Qwen has three, and the file carries a dollar cost per model (`costData`). My inference, not a rule Klavis states: a new model placed the same way needs a comparable set of runs, which is not cheap to repeat every time a model ships. The question: with a small fraction of that budget, where should you spend it, and how wrong is the answer?

## Why this approach

Step k of a task cannot run without steps 1..k-1. So the cost of a step is never the step: it is the prefix behind it. That single fact rules out the usual "pick a random subset of items" answer — items are not independent, and they are not equally priced. Probing the first k steps of a task also matches the published runs: they continue after a failed step (every step of every run is scored), so a probe of a prefix is a truncation of a real run, not a simulation of one.

## Architecture

Two difficulty scales. For each step, the other eight models give two numbers. `d` is the log-odds of passing the step at all. `e` is the log-odds of passing it *given a clean prefix*. The second one is the scale the solve rate lives on, because solving a task is a chain of conditional passes.

The estimator. Two ability offsets per model, each fitted separately on the probe by Newton's method with a normal prior pulled to zero: `a` for passing a step, `sigmoid(a + d)`, and `a_e` for passing it after a clean prefix. Solving a task is then the product of `sigmoid(a_e + e)` over the steps after the probe. Unprobed steps enter only through their run counts, which are fixed before probing, as weights. That is textbook one-parameter item-response theory. I say so in the README: the statistics is not the contribution.

The planner. The contribution is the spending rule for dependent items. Start at depth zero everywhere. A candidate extends one task deeper; it costs the steps it adds and gains information, but the information of a task saturates at a rate set by how much ability varies across tasks (`τ²`). So take the best feasible gain per unit cost until the budget is gone. It is the greedy ratio rule for a precedence-constrained knapsack, with the saturation term added because steps inside one task are not independent evidence.

## Data flow

`tools/fetch_data.py` downloads `benchmark.js` and one module per task from multistepswe.com and checks each file against the sha256 manifest `data/MANIFEST.json`; the data is used for analysis only and never redistributed. `prefixprobe/data.py` reads each module as text with `json`, never importing or executing it.

For each held-out model, `fold.py` pools the other eight models' counts into `d`, `e`, a cost per step (their mean recorded agent seconds per run of the step), step information and `τ²`. `select.py` turns a budget into a probe depth per task under each strategy. `score.py` fits the ability offsets on the held-out model's probed steps only and compares the estimated step rate and solve rate with its full results. `run.py` repeats this for four budgets and five strategies, then over 2000 task-bootstrap replicates, and writes `results/metrics.json`. `demo.py` renders the README's results block from that file.

## Trade-offs

Agent time instead of dollars: the published results carry only a dollar cost per model, while the traces carry the agent seconds each run spent on each step, which is what a per-step plan needs. It stands in for money; it is not money.

A fixed plan instead of an adaptive one: it is easier to evaluate honestly with nine models, and it can be costed before it runs.

A weak normal prior on each ability offset: a probe can be a handful of steps, and the prior keeps a single lucky step from producing an ability of five; it stops mattering once the probe is deep.

A deliberately simple estimator (two separately fitted offsets, one prior, a moment estimate for `τ²`): it is the baseline the prefix structure has to earn its keep against, not a claim about the best possible estimator.

## Results

Hold one model out completely. Its fold — difficulties, costs, information, variance — is built from the other eight. Its own results enter only through the probe. Score the estimate against the truth it never saw, for four budgets and five strategies, then bootstrap over tasks (2000 replicates) to see how much of the result is just which tasks happened to be drawn. Compare strategies pairwise within a replicate, because absolute error intervals shift upward under resampling.

The generated table in the README is the answer: at each budget, the step-rate and solve-rate MAE per strategy, then the paired greedy-minus-alternative difference with its interval, how many models improve and in what share of replicates. I will read the numbers off the table rather than quote them from memory — they are regenerated by `demo.py` and never typed by hand.

## Limitations

Nine folds is nine points; the bootstrap is over tasks, not models. Against random, every paired interval includes zero, so greedy's gain over random is not significant, and greedy's solve-rate error is higher than cheapest-tasks' at every budget. The cost proxy is other models' agent time, not dollars. Log-loss is computed on each plan's own unprobed steps and is therefore a calibration diagnostic, not a ranking. Planning is fixed in advance rather than adaptive, and I designed the planner while looking at the same nine models it is evaluated on.

## What I learned

That the unit of cost in a chained benchmark is the prefix, not the step, and that this, not the statistics, is where the problem is: the estimator is textbook, the spending rule for dependent items is the part worth explaining. What would come next: sequential planning (spend the next prefix based on what the first ones showed), a hierarchical fit instead of a moment estimate for `τ²`, and per-step billing so the budget is measured in dollars.

## Talking points

- In five minutes, in this order: the question, why the prefix is the unit, the two difficulty scales, the estimator, the planner, the evaluation, the result, what I would say against it, what comes next.
- Step k cannot run without steps 1..k-1, so a step costs its whole prefix; random item subsets do not fit a chained, unequally priced benchmark.
- The estimator is textbook one-parameter item-response theory; the contribution is the greedy gain-per-cost planner over task extensions, with saturation set by `τ²`.
- The held-out model never informs its own fold: difficulties, costs, information and variance come from the other eight models.
- Strategies are compared by paired greedy-minus-alternative differences within a bootstrap replicate, because absolute error intervals shift upward under resampling.
- Every result number comes from the README's generated table (`demo.py`); none is typed by hand.

## Questions to expect

**Isn't this just IRT?**

The estimator is; the item structure is not. IRT assumes items you can pick independently. Here items are chained and priced by their prefix, which is why the planner is a knapsack-style greedy over extensions rather than a Fisher-information pick over items.

**Why not adaptive?**

Because it is harder to evaluate honestly with nine models, and because a fixed plan can be costed before it runs. It is listed as the next step, not as a limitation I am hiding.

**Why the prior?**

A probe can be a handful of steps; the prior keeps a single lucky step from producing an ability of five. It is weak enough not to matter once the probe is deep.

**Why percentage points?**

Because a lab cares about being off by two points of step rate, not by a log-odds unit. The log-odds model is internal; everything reported is on the scale people quote.
