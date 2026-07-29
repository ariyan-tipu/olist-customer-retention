"""FastAPI application serving the churn model.

Run locally with:

    olist-retention serve

or in production style with the factory:

    uvicorn --factory olist_retention.service.app:create_app
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from olist_retention import __version__
from olist_retention.logging_utils import get_logger
from olist_retention.service.schemas import (
    CustomerScore,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
)
from olist_retention.service.scorer import ModelNotReadyError, ModelScorer

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = "models/churn_model.joblib"


def create_app(model_path: str | None = None) -> FastAPI:
    """Application factory; the model path can come from the environment."""
    path = Path(model_path or os.environ.get("OLIST_MODEL_PATH", DEFAULT_MODEL_PATH))
    scorer = ModelScorer(path)

    app = FastAPI(
        title="Olist retention scoring service",
        version=__version__,
        description="Repeat purchase propensity scores for Olist customers.",
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
        try:
            probabilities = scorer.score([customer.model_dump() for customer in request.customers])
        except ModelNotReadyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ScoreResponse(
            model_name=scorer.card()["model_name"],
            scores=[CustomerScore(repeat_probability=round(float(p), 6)) for p in probabilities],
        )

    return app
