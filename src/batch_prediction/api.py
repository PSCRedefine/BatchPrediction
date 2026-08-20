"""FastAPI service exposing batch and single engagement prediction.

The batch route is the reason this service exists. Its contract is defined by
section 4 of docs/SPEC.md and rests on one idea: a batch is a collection of
independent requests, so one bad row returns an error *for that row* and the
other rows still get scored.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    MAX_BATCH_SIZE,
    METADATA_PATH,
    MODEL_PATH,
    USERS_PATH,
    VIDEOS_PATH,
)
from .features import FEATURE_COLUMNS, MAX_WATCH_TIME_SECONDS, FeatureStore, validate_ids


class PredictionRequest(BaseModel):
    """One user-video-watch observation.

    ``hour_of_day`` is optional: the specification defines it as an input, and
    the service accepts and echoes it. It is not a model feature.
    """

    user_id: str = Field(min_length=6, max_length=80, examples=["user_000001"])
    video_id: str = Field(min_length=7, max_length=80, examples=["video_0000001"])
    watch_time: float = Field(ge=0, le=MAX_WATCH_TIME_SECONDS, examples=[45.0])
    hour_of_day: int | None = Field(default=None, ge=0, le=23, examples=[14])

    @field_validator("user_id", "video_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        prefix = "user_" if info.field_name == "user_id" else "video_"
        if not str(value).startswith(prefix):
            raise ValueError(f"{info.field_name} must start with '{prefix}'")
        return str(value)


class BatchPredictionRequest(BaseModel):
    """A batch of independent requests.

    The upper bound protects the service from a single caller monopolising it.
    It is declared here so it appears in the OpenAPI schema, and translated to
    400 rather than FastAPI's default 422 by the handler below, because the
    specification calls for 400.
    """

    requests: list[PredictionRequest] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE, description=f"1 to {MAX_BATCH_SIZE} requests"
    )


class BatchResultItem(BaseModel):
    """One row of the response, mirroring one row of the request.

    Either ``probability`` and ``confidence`` are set, or ``error`` is. The
    index ties the row back to its position in the submitted batch, which the
    caller needs because failed rows are reported in place rather than dropped.
    """

    index: int
    user_id: str
    video_id: str
    watch_time: float
    hour_of_day: int | None = None
    probability: float | None = None
    confidence: str | None = None
    predicted_engaged: bool | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    results: list[BatchResultItem]
    batch_size: int
    successful: int
    failed: int
    threshold: float
    model_name: str
    model_version: str
    response_time_ms: float
    timestamp: str


class PredictionResponse(BaseModel):
    user_id: str
    video_id: str
    watch_time: float
    hour_of_day: int
    probability: float
    confidence: str
    predicted_engaged: bool
    threshold: float
    model_name: str
    model_version: str
    response_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    store_loaded: bool
    max_batch_size: int
    uptime_seconds: float
    model_name: str
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    features: list[str] = Field(default_factory=list)
    threshold: float
    max_batch_size: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def confidence_label(probability: float) -> str:
    """Distance from a coin flip, not a claim about correctness."""
    distance = abs(probability - 0.5) * 2
    if distance >= CONFIDENCE_HIGH:
        return "high"
    if distance >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def predict_probabilities(model: Any, frame) -> np.ndarray:
    """Score a frame, tolerating a regressor in place of a classifier.

    ``predict_proba`` is the normal path. A LightGBM regressor exposes only
    ``predict``, and its output is not bounded, so both paths are clipped to
    [0, 1] as the specification requires.
    """
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(frame))[:, 1]
    else:
        raw = np.asarray(model.predict(frame)).ravel()
    return np.clip(raw.astype(float), 0.0, 1.0)


def create_app(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    users_path: Path = USERS_PATH,
    videos_path: Path = VIDEOS_PATH,
) -> FastAPI:
    app = FastAPI(
        title="Cognitive Shorts — Batch Prediction API",
        version=__version__,
        description="Batch engagement prediction with per-row fault tolerance.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.state.started_at = time.time()
    app.state.model = None
    app.state.metadata = {"model_name": "unknown", "model_version": __version__}
    app.state.store = None
    app.state.load_error = None
    try:
        app.state.model = joblib.load(model_path)
        app.state.store = FeatureStore.from_csv(str(users_path), str(videos_path))
        if Path(metadata_path).exists():
            app.state.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep /health serviceable
        app.state.load_error = f"{type(exc).__name__}: {exc}"

    @app.exception_handler(RequestValidationError)
    async def batch_size_is_a_bad_request(_, exc: RequestValidationError) -> JSONResponse:
        """Return 400 for a violated batch-size bound, 422 for everything else.

        FastAPI answers every schema violation with 422. Section 4.2 of the
        specification asks for 400 specifically when the batch is too large, so
        that one case is translated and the rest keep the framework default.
        """
        for error in exc.errors():
            if error.get("type") in {"too_long", "too_short"} and "requests" in error.get("loc", ()):
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"requests must contain between 1 and {MAX_BATCH_SIZE} items"
                    },
                )
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    def ready() -> None:
        if app.state.model is None or app.state.store is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.load_error or "model or feature store is not loaded",
            )

    def threshold() -> float:
        return float(app.state.metadata.get("recommended_threshold", 0.5))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if app.state.model is not None and app.state.store is not None
            else "degraded",
            model_loaded=app.state.model is not None,
            store_loaded=app.state.store is not None,
            max_batch_size=MAX_BATCH_SIZE,
            uptime_seconds=round(time.time() - app.state.started_at, 1),
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            version=__version__,
            timestamp=datetime.now(UTC).isoformat(),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        ready()
        started = time.perf_counter()
        hour = request.hour_of_day if request.hour_of_day is not None else datetime.now(UTC).hour
        try:
            frame = app.state.store.build_one(
                request.user_id, request.video_id, request.watch_time
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        probability = float(predict_probabilities(app.state.model, frame[FEATURE_COLUMNS])[0])
        cut = threshold()
        return PredictionResponse(
            user_id=request.user_id,
            video_id=request.video_id,
            watch_time=request.watch_time,
            hour_of_day=hour,
            probability=round(probability, 6),
            confidence=confidence_label(probability),
            predicted_engaged=probability >= cut,
            threshold=cut,
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            response_time_ms=round((time.perf_counter() - started) * 1000, 3),
            timestamp=datetime.now(UTC).isoformat(),
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(batch: BatchPredictionRequest) -> BatchPredictionResponse:
        """Score a batch, reporting per-row failures in place.

        Rows that survive feature construction are scored in one call rather
        than one per row: the fault isolation happens during feature building,
        which is where the failures actually are.
        """
        ready()
        started = time.perf_counter()
        if len(batch.requests) > MAX_BATCH_SIZE:  # belt and braces alongside the schema bound
            raise HTTPException(
                status_code=400,
                detail=f"requests must contain between 1 and {MAX_BATCH_SIZE} items",
            )

        pairs = [(r.user_id, r.video_id, r.watch_time) for r in batch.requests]
        frame, kept, errors = app.state.store.build_many(pairs)

        probabilities: dict[int, float] = {}
        if kept:
            try:
                scored = predict_probabilities(app.state.model, frame[FEATURE_COLUMNS])
            except Exception as exc:  # a model failure is not one row's fault
                raise HTTPException(
                    status_code=500, detail=f"prediction failed: {type(exc).__name__}: {exc}"
                ) from exc
            probabilities = {position: float(value) for position, value in zip(kept, scored,
                                                                              strict=True)}

        cut = threshold()
        results: list[BatchResultItem] = []
        for position, request in enumerate(batch.requests):
            common = {
                "index": position,
                "user_id": request.user_id,
                "video_id": request.video_id,
                "watch_time": request.watch_time,
                "hour_of_day": request.hour_of_day,
            }
            if position in probabilities:
                probability = probabilities[position]
                results.append(BatchResultItem(
                    **common,
                    probability=round(probability, 6),
                    confidence=confidence_label(probability),
                    predicted_engaged=probability >= cut,
                ))
            else:
                results.append(BatchResultItem(**common, error=errors[position]))

        successful = len(probabilities)
        return BatchPredictionResponse(
            results=results,
            batch_size=len(batch.requests),
            successful=successful,
            failed=len(batch.requests) - successful,
            threshold=cut,
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            response_time_ms=round((time.perf_counter() - started) * 1000, 3),
            timestamp=datetime.now(UTC).isoformat(),
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        ready()
        return ModelInfoResponse(
            model_name=str(app.state.metadata.get("model_name", "unknown")),
            model_version=str(app.state.metadata.get("model_version", __version__)),
            features=list(FEATURE_COLUMNS),
            threshold=threshold(),
            max_batch_size=MAX_BATCH_SIZE,
            metadata=app.state.metadata,
        )

    return app


app = create_app()
