
"""FastAPI inference service for ModelForge."""

from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def load_latest_model():
    """Load the model artifact from the latest baseline training run."""

    experiment = mlflow.get_experiment_by_name("ModelForge-Baseline")
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        return None

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"

    loaded_model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded MLflow run: {run_id}")
    return loaded_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the API starts."""
    app.state.model = None

    try:
        app.state.model = load_latest_model()
    except Exception as exc:
        print(f"Model loading failed: {exc}")

    yield


app = FastAPI(
    title="ModelForge Prediction API",
    description="MLflow-backed prediction service for a demonstration classifier.",
    version="1.1.0",
    lifespan=lifespan,
)


class ModelInput(BaseModel):
    features: list[float] = Field(
        ...,
        description="30 numerical features in the dataset's original feature order.",
    )


@app.get("/")
def read_root():
    return {
        "platform": "ModelForge",
        "status": "active",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health_check():
    model_loaded = app.state.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict")
def predict(input_data: ModelInput):
    if len(input_data.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 30 features, got {len(input_data.features)}.",
        )

    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train a model and restart the API.",
        )

    try:
        features = np.asarray(input_data.features, dtype=float).reshape(1, -1)

        if not np.isfinite(features).all():
            raise HTTPException(
                status_code=422,
                detail="All feature values must be finite numbers.",
            )

        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        # scikit-learn dataset convention: 0 = malignant, 1 = benign.
        labels = {0: "malignant", 1: "benign"}

        return {
            "prediction": prediction,
            "class_label": labels[prediction],
            "probability_benign": float(probabilities[1]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
