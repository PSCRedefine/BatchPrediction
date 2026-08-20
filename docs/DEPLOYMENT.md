# Deployment

Two services, one image: the FastAPI scorer on 8000 and the Streamlit console
on 8501. The console holds no model — it talks to the API over HTTP — so the
two scale and fail independently.

> **Verification status.** The local path below was run end to end on the
> development machine: both services started, a 20-row batch was scored, and
> the results view rendered (see `image/ui_batch_results.png`). The Docker path
> was **not** executed, because Docker is not installed on that machine. The
> compose file parses and its health-check command was run against the live
> service and returned 0, but the image build itself is unverified. Treat the
> first `docker compose up` as the real test.

---

## 1. Local, no containers

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Terminal one:

```bash
uvicorn batch_prediction.api:app --host 127.0.0.1 --port 8000
```

Terminal two:

```bash
streamlit run app.py
```

Console at http://localhost:8501, API docs at http://localhost:8000/docs.
Append `?demo=1` to the console URL to load the bundled sample and score it in
one step.

## 2. Docker Compose

```bash
docker compose up --build
```

Compose builds one image and starts both services. The console waits for the
API's health check to pass before it starts, so it never comes up pointing at a
service that cannot score.

| Service | Port | Command |
|---|---|---|
| `api` | 8000 | `uvicorn ... --workers 2` |
| `console` | 8501 | `streamlit run app.py` |

Stop with `docker compose down`. Rebuild after a dependency change with
`docker compose build --no-cache`.

### What the image contains

Dependencies install before source is copied, so editing `app.py` does not
reinstall scikit-learn. The model (3 KB) and the lookup tables (9 MB) are baked
in, which is what makes the container self-contained: it needs no volume, no
model registry and no network access to start.

The container runs as uid 10001, not root. Nothing in the image needs write
access at runtime.

### Why one image and not two

The API and the console share every dependency except Streamlit's. Two images
would save a few megabytes and introduce a way for them to drift apart — a
model contract change landing in one and not the other. The compose file
selects the entry point instead.

## 3. Configuration

Every setting is an environment variable with a working default.

| Variable | Default | Purpose |
|---|---|---|
| `BATCH_PREDICTION_API_URL` | `http://127.0.0.1:8000` | Where the console sends requests. Compose sets it to `http://api:8000`. |
| `BATCH_PREDICTION_MODEL_PATH` | `models/best_model.joblib` | Serve a different model without rebuilding |
| `BATCH_PREDICTION_METADATA_PATH` | `models/model_metadata.json` | Threshold and model name are read from here, not hard-coded |
| `BATCH_PREDICTION_USERS_PATH` | `data/users.csv` | Identifier resolution |
| `BATCH_PREDICTION_VIDEOS_PATH` | `data/videos.csv` | Durations for `watch_ratio` |

There are no secrets, no database and no outbound calls. Nothing needs to be
injected at deploy time for the service to work.

### Swapping the model without a rebuild

```bash
docker compose run --rm \
  -v /path/to/new_model.joblib:/models/model.joblib:ro \
  -v /path/to/new_metadata.json:/models/metadata.json:ro \
  -e BATCH_PREDICTION_MODEL_PATH=/models/model.joblib \
  -e BATCH_PREDICTION_METADATA_PATH=/models/metadata.json \
  api
```

The threshold travels with the metadata, so a model whose useful threshold is
not 0.5 stays correct — which matters here, where it is 0.381.

## 4. Health and readiness

```bash
curl -s localhost:8000/health
```

```json
{"status":"ok","model_loaded":true,"store_loaded":true,
 "max_batch_size":100,"uptime_seconds":14.3,
 "model_name":"logistic_regression+isotonic","version":"1.0.0"}
```

A load failure does **not** kill the process. `/health` keeps serving and
reports `status: degraded` with `model_loaded: false`, and scoring routes
return 503. This distinguishes "the process is up but cannot score" from "the
process is down" — two conditions with different fixes.

The compose health check therefore tests `model_loaded` and `store_loaded`
rather than just the port, because a degraded service answers on the port
perfectly well.

For an orchestrator: use `/health` as the **readiness** probe (gated on
`model_loaded`), and a plain TCP check or `/health` without the gate as the
**liveness** probe. Gating liveness on the model would restart a container that
is up and honestly reporting a problem a restart will not fix.

## 5. Scaling

- **Workers.** `--workers 2` is the compose default. Each worker holds its own
  copy of the model (3 KB) and the lookup tables (~60 MB resident once parsed),
  so memory grows linearly with worker count. That table, not the model, is the
  memory cost.
- **Batch size.** The 100-row cap is a constant in `config.py`, enforced in
  three places: the console truncates, the request schema declares, and the
  route re-checks. Raising it means re-measuring — a 1,000-row batch changes
  the response-time budget and the JSON payload size, not the model cost.
- **Throughput.** Scoring is not the bottleneck. Server-side processing for 20
  rows measured 30–110 ms, dominated by identifier resolution and response
  assembly. Horizontal replicas behind a load balancer scale this cleanly
  because the service is stateless.
- **State.** The API keeps none. The console keeps per-session state in
  Streamlit's `session_state`, which means multiple console replicas need
  sticky sessions — or accept that a user's batch lives in one replica.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Console shows `API server is offline` | API not up, or wrong URL | Check `BATCH_PREDICTION_API_URL`; under compose it must be `http://api:8000`, not `localhost` |
| `/health` reports `degraded` | Model or CSVs missing from the image | Confirm `models/` and `data/` were copied; `.dockerignore` must not exclude them |
| Every row returns `unknown user_id` | Lookup tables do not match the request identifiers | The shipped tables use `user_000001` / `video_0000001` shapes |
| 400 on a batch | More than 100 rows, or zero | Split the batch; the console truncates automatically |
| 422 on a batch | A row fails schema validation | Check field names and that `watch_time` is a number |
| Console starts before the API is ready | `depends_on` condition missing | Keep `condition: service_healthy` |

## 7. Beyond one host

The image is a standard, stateless, non-root web service on a fixed port, so
any container platform takes it as-is. Two properties matter wherever it lands:

- **Startup does I/O.** Loading the lookup tables takes a moment, so set a
  start period or initial delay before the readiness probe counts failures.
  The compose file uses `start_period: 20s`.
- **The threshold lives in metadata, not code.** Deploying a new model means
  replacing two files, not editing a constant and rebuilding.
