# API contract

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## POST /predict/batch

Score 1–100 interactions in one call.

**Request**

```json
{
  "requests": [
    {"user_id": "user_000001", "video_id": "video_0000001", "watch_time": 45.0, "hour_of_day": 14},
    {"user_id": "user_000002", "video_id": "video_0000003", "watch_time": 12.5}
  ]
}
```

`hour_of_day` is optional. It is accepted, validated (0–23) and echoed back, but
it is **not** a model feature — see [Model](#model).

**Response 200**

```json
{
  "results": [
    {"index": 0, "user_id": "user_000001", "video_id": "video_0000001",
     "watch_time": 45.0, "hour_of_day": 14,
     "probability": 0.375725, "confidence": "low", "predicted_engaged": false,
     "error": null}
  ],
  "batch_size": 2,
  "successful": 2,
  "failed": 0,
  "threshold": 0.381201,
  "model_name": "logistic_regression+isotonic",
  "model_version": "1.0.0",
  "response_time_ms": 29.623,
  "timestamp": "2026-08-20T08:02:33Z"
}
```

`results` always has one entry per submitted request, in the order submitted.
`index` is that position, so a caller can join the response back onto its input
without relying on ordering alone.

### Per-row failure

A row that cannot be scored comes back with `error` set and `probability`,
`confidence` and `predicted_engaged` all null. The other rows are unaffected,
and `successful` counts only the scored ones.

```json
{"index": 4, "user_id": "user_999999999", "video_id": "video_0017814",
 "watch_time": 12.0, "hour_of_day": 9,
 "probability": null, "confidence": null, "predicted_engaged": null,
 "error": "unknown user_id: user_999999999"}
```

Causes: an identifier that does not resolve, an identifier of the wrong shape,
a watch time outside 0–3600, a watch time that is not a number.

### Status codes

| Code | When |
|---|---|
| 200 | The batch was processed. Individual rows may still carry `error`. |
| 400 | `requests` is empty or holds more than 100 items. |
| 422 | A row fails schema validation (missing field, wrong type, out of range). |
| 500 | The model raised while scoring. Not attributable to one row. |
| 503 | The model or lookup tables failed to load. `/health` says why. |

**On 400 vs 422.** The specification asks for 400 when the batch exceeds 100.
FastAPI answers every schema violation with 422, so the size bound is declared
on the model (it appears in the OpenAPI schema) and a validation handler
translates that one case to 400. Every other schema violation keeps 422.

## POST /predict

One interaction, same fields without the wrapper. Returns 404 for an
unresolvable identifier and 422 for a malformed one — a single request has no
other row to protect, so it fails outright rather than reporting in place.

## GET /health

`status` is `ok` or `degraded`. Serviceable even when the model is missing, so
a probe can distinguish "process is up but cannot score" from "process is down".

## GET /model/info

Model name, version, the feature list, the decision threshold and the batch
limit, plus the full training metadata.

## Model

Two features: `watch_time_seconds` and `watch_ratio`. The request carries
neither directly — `watch_ratio` is derived from `watch_time` and the video's
duration, which is why the service needs `videos.csv` at startup.

The threshold of 0.381 is not 0.5. At 0.5 the model flags nothing, because its
maximum output on this data is 0.389. The threshold was chosen for lift at a
workable flag rate, and ships in `models/model_metadata.json`.
