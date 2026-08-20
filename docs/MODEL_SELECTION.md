# Model selection

Why this model serves the batch endpoint, what it was chosen over, and what the
choice costs. Every number here comes from the training run recorded in
`models/model_metadata.json`; nothing is illustrative.

---

## 1. The problem

Predict whether a user will engage with a short video, where engagement is
`liked OR shared OR commented OR followed_creator OR replayed`.

| | |
|---|---|
| Rows | 500,000 |
| Positive rate | 27.79% |
| Split | Chronological — earliest 70% train, next 15% validation, latest 15% test |
| Features | 2 (`watch_time_seconds`, `watch_ratio`) |

The split is chronological because production predicts the future from the
past. A random split would let the model train on interactions that happened
*after* the ones it is tested on, which is a form of leakage that flatters the
score without improving the product.

**A finding worth stating plainly:** on this data the chronological test AUC is
**0.5796** and the random-split AUC is **0.5770** — a difference of −0.0026.
The random split was not optimistic here. The chronological split is still the
right choice, because its correctness does not depend on the gap turning out
large; it just happens that on one month of data the distribution barely moved.

## 2. Four candidates

Each was fitted on the same training rows and scored on the same validation
rows.

| Model | Valid ROC-AUC | PR-AUC | Brier | Fit time | Artifact |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.5711 | 0.3374 | 0.1980 | 0.12 s | **0.002 MB** |
| Random forest | 0.5736 | 0.3366 | 0.1967 | 4.0 s | 3.916 MB |
| **Gradient boosting** | **0.5742** | 0.3360 | 0.1967 | 9.39 s | 0.196 MB |
| LightGBM | 0.5732 | 0.3353 | 0.1968 | 2.8 s | 0.692 MB |

Gradient boosting leads on AUC. Taking the leader would be the obvious move,
and it would be wrong.

## 3. The gap is inside the noise

The spread between best and worst is 0.0031 AUC. The standard error on a single
model's AUC is ~0.0023. Comparing two numbers that differ by less than their
own uncertainty is not a comparison.

So the models were compared *pairwise on the same resampled rows* — a paired
bootstrap, which cancels the shared variance from the sample itself and asks
only whether one model beats another on the rows they both saw.

| Model | AUC | Gap to leader | 95% CI on the gap | Tied? |
|---|---:|---:|---|---|
| Gradient boosting | 0.5742 | — | — | leader |
| Random forest | 0.5736 | 0.0006 | [−0.0011, +0.0023] | ✅ |
| LightGBM | 0.5732 | 0.0010 | [−0.0009, +0.0029] | ✅ |
| Logistic regression | 0.5711 | 0.0031 | [−0.0002, +0.0060] | ✅ |

Every interval crosses zero. **All four models are statistically
indistinguishable**, including the linear one against the boosted one.

This is not a shortcut to justify the simple model. It is the measurement
saying there is no high-order structure left for a tree to find — which is what
you would expect from two continuous features that are close to monotone in the
target.

## 4. The tie is broken by cost

When accuracy cannot separate the candidates, something else must. The rule
applied was: **among models statistically tied with the leader, take the
cheapest.**

Logistic regression wins on every cost axis at once:

- **Artifact size** — 0.002 MB against 3.916 MB for the random forest, ~2,000×
  smaller. This is why `models/best_model.joblib` is 3 KB and can be committed
  to the repository, so a clone runs with no training step and no model
  registry.
- **Fit time** — 0.12 s against 9.39 s, ~78× faster. Retraining is cheap enough
  to be routine.
- **Inference** — a dot product and a lookup. For a batch endpoint scoring up
  to 100 rows per call this is the difference between a millisecond and a
  measurable fraction of the response budget.
- **Explicability** — the decision rule is one line. When a caller asks why a
  row scored what it did, there is an answer.

### The honest counter-evidence

On the **test** set, the ordering shifts: random forest 0.5816, LightGBM 0.5808,
logistic regression 0.5794, gradient boosting 0.5790. The random forest is now
ahead by 0.0022, and the validation leader is now last.

That reshuffling is the tie test being *right*. These models are separated by
noise, and noise reorders them from one sample to the next. Choosing the model
that happens to lead on the test set would be selecting on the very set that is
supposed to provide an unbiased estimate — the score would stop meaning what it
claims to mean.

## 5. Calibration is applied, and it costs something

The raw model's probabilities were poorly calibrated. Isotonic regression,
fitted on validation and evaluated on test:

| | Raw | Calibrated | |
|---|---:|---:|---|
| ECE | 0.02725 | **0.00353** | 7.7× better |
| Brier | 0.19840 | **0.19694** | better |
| Log loss | 0.58532 | **0.58174** | better |
| ROC-AUC | 0.57942 | 0.57957 | unchanged |
| **PR-AUC** | **0.34504** | 0.32674 | **worse** |

Calibration was applied because a batch consumer reads the probability as a
number, not as a rank — averaging it, thresholding it, budgeting against it.
A mean probability of 0.268 across a batch is only meaningful if 0.268 means
0.268.

The PR-AUC drop is real and is the price. Isotonic regression is a step
function, so it maps ranges of raw scores onto a single calibrated value,
discarding fine-grained ordering *within* each step. That is also why
probabilities repeat across rows in the results table — many rows return
exactly 0.2366 or 0.3757. If this service were used purely for ranking, the
uncalibrated model would be the better choice.

## 6. The threshold is 0.381, and not for the usual reason

The model's maximum output on test data is **0.389**. The default threshold of
0.5 therefore flags **nothing at all** — every candidate is "degenerate at 0.5",
which is a threshold problem, not a model problem.

Optimising F1 does not help either: F1 is maximised at threshold 0.01, which
flags 100% of traffic. On a signal this weak, F1 optimisation collapses to
blanket treatment.

So the threshold was chosen by **budget**:

| Flag rate | Threshold | Precision | Recall | Lift |
|---:|---:|---:|---:|---:|
| 7.3% | 0.3890 | 39.6% | 10.2% | 1.41× |
| **23.9%** | **0.3812** | **39.3%** | **33.4%** | **1.40×** |
| 100% | 0.2366 | 28.1% | 100% | 1.00× |

Selected: **0.3812** — the highest lift among points that flag at least 10% of
traffic. Flagging 7.3% buys 0.01× more lift for two thirds less recall, which
is a bad trade unless the intervention is expensive.

Read plainly: against a 27.8% base rate, acting on the flagged quarter of
traffic finds engaged users **1.40× more often than acting at random**. That is
a real effect and a modest one.

## 7. What this means for a batch service

- **Throughput is not the constraint.** The model scores a 100-row batch in
  well under the time it takes to resolve the identifiers. Server-side
  processing for a 20-row batch measured between 30 ms and 110 ms across runs,
  and that figure is dominated by identifier resolution and response assembly,
  not by the estimator. Choosing the cheap model made throughput a non-issue
  rather than a tuning exercise.
- **Rank, do not threshold, when you can.** With a maximum output of 0.389 and
  1.40× lift, the useful question is "which rows in this batch are most likely
  to engage" — which is why the results view sorts a Highest Scoring Rows
  panel — rather than "is this row engaged, yes or no".
- **Treat the probability as a rate, not a verdict.** A row at 0.3757 means
  roughly 38 in 100 such rows engage. Calibration is what makes that sentence
  true.

## 8. What would change the decision

| If this changed | Then |
|---|---|
| More features with real signal | The tie would likely break; re-run the comparison rather than assuming the linear model still wins |
| Ranking replaces thresholding | Drop isotonic calibration — it costs PR-AUC and buys nothing a ranker uses |
| Batch size limit rises far above 100 | Re-measure; the artifact-size argument weakens as per-call overhead is amortised |
| The intervention becomes expensive | Move to the 7.3% operating point, trading recall for a slightly better rate |

## 9. Reproducing this

The model is trained in the
[SinglePrediction](https://github.com/PSCRedefine/SinglePrediction) project,
which holds the training pipeline, the feature-selection measurements and the
figures behind these tables:

```bash
python -m single_prediction.prepare_data
python -m single_prediction.train
```

That writes `models/best_model.joblib`, `models/model_metadata.json` and
`reports/training_report.json`. The first two are copied into this repository;
the service reads the threshold and model name from the metadata at startup
rather than hard-coding them.
