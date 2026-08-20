# Batch Prediction

Batch engagement prediction for **Cognitive Shorts**: a FastAPI service that
scores up to 100 user-video interactions per call, and a Streamlit console for
driving it from a CSV or by hand.

Built to [docs/SPEC.md](docs/SPEC.md). 77 tests.

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [How a batch is processed](#how-a-batch-is-processed)
- [Fault tolerance](#fault-tolerance)
- [The model](#the-model)
- [Repository layout](#repository-layout)
- [Verification](#verification)
- [Limitations](#limitations)

---

## What it does

```text
CSV upload  ─┐
             ├─→ app.py ──POST /predict/batch──→ api.py ──→ FeatureStore ──→ model
manual entry ─┘   (Streamlit)                    (FastAPI)   users.csv        │
                                                             videos.csv       │
                      ┌──────────────────────────────────────────────────────┘
                      ▼
        metrics · results table · CSV download · probability histogram
```

The page holds no model and no data. Everything it shows came back over HTTP,
so what you see is exactly what any API client would get.

## Quick start

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Start the API:

```bash
uvicorn batch_prediction.api:app --reload
```

Start the console in a second terminal:

```bash
streamlit run app.py
```

Then open http://localhost:8501 and upload
[`data/sample_batch_requests.csv`](data/sample_batch_requests.csv) — 20 rows,
two of which fail on purpose so the error path is visible on the first run.

Point the console at a different API with `BATCH_PREDICTION_API_URL`.

## How a batch is processed

1. **Validate columns.** `user_id`, `video_id` and `watch_time` are required;
   a missing one stops the upload with `Missing required columns: [...]`.
   `hour_of_day` is optional.
2. **Truncate to 100.** The service caps a batch at 100, so the page sends 100
   rather than sending 500 and getting a 400 back. Over-length files get a
   warning naming the real row count.
3. **Convert types.** Identifiers to string, `watch_time` to float,
   `hour_of_day` to a nullable integer. A value that will not convert becomes
   null rather than raising — a batch reports bad rows, it does not refuse the
   file.
4. **Build features.** Each row is resolved against the user and video tables,
   and `watch_ratio` is computed from `watch_time` and the video's duration.
5. **Score once.** Every row that survived step 4 is scored in a single call,
   not one call per row.
6. **Reassemble.** Results come back in submission order, each carrying the
   `index` it had in the request.

## Fault tolerance

The batch route's central property: **one bad row costs you that row, not the
batch.** Fault isolation happens during feature construction, which is where
the failures actually are — an identifier that does not resolve, a watch time
that is not a number. Those rows get an `error` string and null predictions;
everything else is scored normally, and `successful` counts only the scored
rows.

Running the shipped sample:

```text
batch_size=20 successful=18 failed=2 response_time_ms=29.6
index  user_id          video_id         probability  error
    3  user_018394      video_0009705       0.236613  None
    4  user_999999999   video_0017814            NaN  unknown user_id: user_999999999
    5  user_012831      video_0014304       0.236613  None
   11  user_000001      video_999999999          NaN  unknown video_id: video_999999999
```

A failure that is *not* one row's fault — the model itself raising — returns
500 for the whole batch instead, because attributing it to a row would be a
lie.

## The model

`models/best_model.joblib` is committed (3 KB) so the repository runs after a
clone with no training step. It is an isotonic-calibrated logistic regression
over two features, `watch_time_seconds` and `watch_ratio`, trained in the
[SinglePrediction](https://github.com/PSCRedefine/SinglePrediction) project.

Two things are worth knowing before reading anything into its output:

- **The signal is weak.** Test ROC-AUC is 0.5796. The lift at the recommended
  operating point is 1.40x — real, but modest.
- **The decision threshold is 0.381, not 0.5.** The model's maximum output on
  this data is 0.389, so at 0.5 it flags nothing at all. The threshold ships in
  `model_metadata.json` and the service reads it from there.

Isotonic calibration is a step function, so probabilities repeat across rows.
That is the calibrator, not a bug.

`hour_of_day` is part of the API contract because the specification defines it
as an input. It is not a model feature. Accepting a field and training on it
are separate decisions.

## Repository layout

```text
app.py                           Streamlit console
src/batch_prediction/
  api.py                         FastAPI: /predict/batch, /predict, /health, /model/info
  batching.py                    upload validation, type conversion, payload building
  features.py                    identifier resolution, watch_ratio, per-row isolation
  config.py                      paths, batch limit, confidence bands
models/                          committed model and metadata (3 KB)
data/
  users.csv, videos.csv          lookup tables the FeatureStore needs
  sample_batch_requests.csv      20 rows, two failing on purpose
docs/
  SPEC.md                        the requirements this was built to
  API.md                         endpoint contract
tests/                           77 tests
```

## Verification

```bash
python -m pytest -q
```

| Area | Tests | Covers |
|---|---:|---|
| `test_api.py` | 28 | routes, status codes, per-row errors, batch limits, clipping |
| `test_features.py` | 25 | identifier validation, `watch_ratio`, per-row isolation |
| `test_batching.py` | 24 | required columns, truncation, type conversion, payload shape |

The API tests run against the real shipped model and the real lookup tables. A
stubbed store would not exercise identifier resolution, which is where most
per-row errors come from.

## Limitations

- **This duplicates SinglePrediction's model and feature code.** The two
  repositories now hold separate copies of `features.py`, the API skeleton and
  the model artifact; a change to the feature contract has to be made twice.
  Keeping batch prediction as a page in that project would have avoided it.
- **No training here.** The model is inherited. Retraining happens in
  SinglePrediction, and the artifact is copied across.
- **The 100-row cap is a constant**, not a function of load. It protects the
  service from one caller, not from many.
- **`users.csv` and `videos.csv` are committed** (9 MB) so a clone runs. A real
  deployment would read them from a store rather than from the repository.
