"""BAZAR recommendation microservice (FastAPI)."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from recommender import Recommender

logger = logging.getLogger("bazar-ml")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommender = Recommender.load(MODEL_DIR)
        logger.info("Model loaded: %d products", len(app.state.recommender.product_ids))
    except Exception:
        logger.exception("Could not load model artifacts from %s", MODEL_DIR)
        app.state.recommender = None
    yield


app = FastAPI(title="BAZAR Recommendation Microservice", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int | None = Field(None, ge=0, description="Reserved for future personalisation; null for anonymous users")
    recent_product_ids: list[int] = Field(default_factory=list, max_length=100)
    top_n: int = Field(5, ge=1, le=50)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: int | None
    recommended_product_ids: list[int]
    strategy: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    n_products: int
    model_version: str


def _get_recommender(request: Request) -> Recommender:
    rec = request.app.state.recommender
    if rec is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return rec


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    rec = request.app.state.recommender
    if rec is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_loaded=True,
        n_products=len(rec.product_ids),
        model_version=rec.meta.get("trained_at", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    rec = _get_recommender(request)
    try:
        ids, strategy = rec.recommend(payload.recent_product_ids, top_n=payload.top_n)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return PredictionResponse(
        user_id=payload.user_id,
        recommended_product_ids=ids,
        strategy=strategy,
        model_version=rec.meta.get("trained_at", "unknown"),
    )
