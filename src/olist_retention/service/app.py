"""FastAPI application serving the churn model.

Run locally with:

    olist-retention serve

or in production style with the factory:

    uvicorn --factory olist_retention.service.app:create_app

Beyond scoring, the service reports on itself. Every request contributes
to a latency window and to a rolling comparison of incoming features
against the distribution the model was trained on. Those two numbers are
what turns a model that runs into a model somebody can be responsible
for: without them, a scorer that has quietly started seeing different
customers looks exactly like one that has not.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from olist_retention import __version__
from olist_retention.logging_utils import get_logger
from olist_retention.monitoring.cloudwatch import CloudWatchPublisher
from olist_retention.monitoring.telemetry import prometheus_text
from olist_retention.service.monitoring_state import MonitoringState
from olist_retention.service.schemas import (
    CustomerScore,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
)
from olist_retention.service.scorer import ModelNotReadyError, ModelScorer

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = "models/churn_model.joblib"


def json_safe(value: Any) -> Any:
    """Replace non-finite floats so a payload can always be serialised.

    A validation error echoes the offending input back to the caller, and
    `NaN` has no JSON representation, so rejecting a non-finite feature
    produced a 500 while building the 422 that explained it. Rendering
    such a value as its name keeps the message useful and the response
    valid.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def create_app(
    model_path: str | None = None,
    s3_uri: str | None = None,
    publisher: CloudWatchPublisher | None = None,
) -> FastAPI:
    """Application factory; every input can also come from the environment."""
    path = Path(model_path or os.environ.get("OLIST_MODEL_PATH", DEFAULT_MODEL_PATH))
    scorer = ModelScorer(path, s3_uri=s3_uri)
    state = MonitoringState(scorer)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        sender = publisher
        if sender is None and os.environ.get("OLIST_CLOUDWATCH", "").lower() in {"1", "true"}:
            sender = CloudWatchPublisher(region=os.environ.get("AWS_REGION"))
        if sender is not None:
            sender.start(state.snapshots)
        try:
            yield
        finally:
            if sender is not None:
                sender.stop()

    app = FastAPI(
        title="Olist retention scoring service",
        version=__version__,
        description="Repeat purchase propensity scores for Olist customers.",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"detail": json_safe(jsonable_encoder(error.errors()))}
        )

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness: the process is up. Does not touch the model."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz() -> dict:
        """Readiness: the model artefact is loaded and usable."""
        try:
            card = scorer.card()
        except ModelNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "ready", "model_name": card["model_name"]}

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        try:
            card = scorer.card()
        except ModelNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ModelInfo(
            model_name=card["model_name"],
            trained_at_utc=card["trained_at_utc"],
            sklearn_version=card["sklearn_version"],
            train_cutoff=card["train_cutoff"],
            horizon_days=card["horizon_days"],
            holdout_metrics=card["holdout_metrics"],
        )

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        rows = [customer.model_dump() for customer in request.customers]
        started = time.perf_counter()
        try:
            probabilities = scorer.score(rows)
        except ModelNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        duration_ms = (time.perf_counter() - started) * 1000.0

        model_name = scorer.card()["model_name"]
        state.record(rows, duration_ms, model_name)
        return ScoreResponse(
            model_name=model_name,
            scores=[CustomerScore(repeat_probability=round(float(p), 6)) for p in probabilities],
        )

    @app.get("/v1/monitoring")
    def monitoring() -> dict:
        """Latency and feature drift over the recent window, as JSON."""
        latency, drift = state.snapshots()
        return {"latency": latency, "drift": drift}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """The same numbers in the format a Prometheus scraper expects."""
        latency, drift = state.snapshots()
        try:
            model_name = scorer.card()["model_name"]
        except ModelNotReadyError:
            model_name = "unloaded"
        return prometheus_text(latency, drift, model_name)

    return app
