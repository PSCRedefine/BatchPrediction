# Production readiness

What this service does not have yet, and what it would need before it carries
real traffic. Ordered by the risk of shipping without it.

Almost nothing here is a defect against the specification. The specification
stops where it stops; this document is where it would have to resume. It is
written to be argued with — each item states the risk, the remedy and the cost,
so that "accept this one" is a decision somebody can make on the record rather
than an omission nobody noticed.

## Blocking

### 1. Synchronous HTTP is the wrong shape for the real workload

A hundred rows per call is a demonstration size. The workload this exists to
serve — building a nightly candidate set, producing a settlement list — is
millions of rows, and the only way to reach that number through this interface
is tens of thousands of sequential HTTP calls, each of which can fail
independently and none of which can be resumed.

**Remedy.** An asynchronous job interface: submit, receive a job identifier,
poll or receive a callback, read results from object storage. The synchronous
route stays for interactive use. Below a few thousand rows the current shape is
correct; above it, the batch belongs in a pipeline, not in a request.

**Cost.** High. This is a second service, not a change to this one.

### 2. Retries are not safe

A caller that times out mid-batch has no way to know which rows were scored, and
no way to ask for the same batch again without producing a second set of
predictions. Nothing is persisted, so nothing can be reconciled afterwards.

**Remedy.** An idempotency key on the request, results persisted against it, and
a documented retry contract. Partial success needs the same treatment: the
response says 18 of 20 succeeded, but the contract does not say what a caller is
expected to do about the other two.

**Cost.** Low once results are persisted; impossible before that.

### 3. Predictions are returned and forgotten

Nothing records what was scored, with which artefact, or what came back. That
makes three later questions unanswerable: what did we predict for this user last
Tuesday, which artefact produced it, and how did those predictions compare to
what actually happened.

**Remedy.** Write predictions to durable storage with the model version and the
input features. This is also the prerequisite for any outcome-based monitoring.

### 4. The feature contract exists in two places

`features.py`, the API skeleton and the model artefact are copies of
SinglePrediction's. A change to the feature contract has to be made twice, and
nothing detects it when only one copy changes. The two services would then
disagree about what the same identifiers mean, silently, and only batch results
would be wrong.

**Remedy.** A shared, versioned feature package that both services depend on.
Until that exists, a contract test that loads both implementations and asserts
they produce identical features for a fixed set of inputs.

**Cost.** Medium, and it is the item most likely to cause a real incident,
because divergence is invisible until someone compares outputs.

### 5. The row cap protects against one caller, not many

The hundred-row limit bounds a single request. It does nothing about
concurrency: enough simultaneous callers exhaust the worker pool regardless of
how small each batch is. There is also no authentication, so there is no way to
identify or throttle the caller responsible.

**Remedy.** Authentication at the edge, per-caller quotas, and a concurrency
limit with explicit backpressure — a 429 with a retry hint is a better answer
than a timeout.

### 6. The artefact and the build carry SinglePrediction's risks

The committed model is loaded with `joblib.load`, which executes what it
deserialises, from a file with no checksum and no provenance. Dependencies are
ranges, so the container can resolve different libraries than the training run
used; the artefact already emits NumPy deprecation warnings on the unpickling
path.

**Remedy.** As in SinglePrediction: verified hashes, recorded provenance, a
lockfile for the image, and library versions compared at load.

---

## Required within the first quarter

- **Per-row observability.** Failed rows are reported to the caller and then
  discarded. A dead-letter record — the row, the error, the batch identifier —
  turns "two rows failed" into something that can be investigated.
- **Failure-rate alerting.** A batch where 60% of rows fail to resolve is a
  broken upstream, not a bad file, and nothing currently distinguishes them.
- **Cost accounting.** Batch is where spend concentrates. Rows scored per caller
  per day should be a number somebody can look up.
- **Structured logs and a request identifier**, for the same reasons as in
  SinglePrediction: a batch result that looks wrong is currently untraceable.

---

## Accepted, with reasons

| Item | Why it is acceptable |
|---|---|
| Fault isolation lives in feature construction | That is where the failures are. Isolating at the scoring step would catch nothing that happens today. |
| A whole-batch 500 when the model itself raises | Attributing a model failure to a particular row would be false. |
| `hour_of_day` accepted but not used | The API contract and the feature set are deliberately separate decisions. |

---

## What is already in place

Listed so this document reads as a review and not a confession.

| Concern | Where it is handled |
|---|---|
| Load failure does not kill the process | `/health` stays serviceable and reports the cause |
| Non-root container, dependency layer cached separately | `Dockerfile` |
| The console starts only behind a real health check | `docker-compose.yml` |
| Configuration is environment-backed, not hard-coded | `config.py` |
| The suite runs on every supported interpreter | `.github/workflows/tests.yml` |
