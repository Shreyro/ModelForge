"""
FastAPI inference service for ModelForge.
Loads the latest trained model from MLflow or local storage and serves predictions.
"""

import os
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="ModelForge Prediction API",
    description="Production-ready MLOps platform serving Breast Cancer classification models.",
    version="1.0.0"
)

# Define request schema with validation
class ModelInput(BaseModel):
    features: List[float] = Field(
        ..., 
        description="List of 30 numerical features corresponding to the breast cancer dataset."
    )

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    """
    Loads the trained model on startup by searching the mlruns directory.
    """
    global model
    try:
        mlruns_path = "mlruns"
        if os.path.exists(mlruns_path):
            # Recursively walk through mlruns to find the 'model' artifact folder
            for root, dirs, files in os.walk(mlruns_path):
                if "model" in dirs:
                    model_path = os.path.join(root, "model")
                    if os.path.exists(os.path.join(model_path, "MLmodel")):
                        model = mlflow.sklearn.load_model(model_path)
                        print(f"Successfully loaded model from: {model_path}")
                        return
        
        print("Warning: MLflow artifact not found automatically.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {
        "platform": "ModelForge",
        "status": "active",
        "endpoints": ["/health", "/predict"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes / Docker container probes."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(input_data: ModelInput):
    """
    Accepts 30 feature values, validates them, and returns the classification prediction.
    """
    # 1. Validate input feature length first
    if len(input_data.features) != 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feature count. Expected 30 features, got {len(input_data.features)}."
        )
    
    # 2. Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
    
    try:
        # Reshape input for sklearn prediction
        features_array = np.array(input_data.features).reshape(1, -1)
        prediction = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0][1])
        
        # Map label: 0 = Malignant, 1 = Benign
        label_mapping = {0: "malignant", 1: "benign"}

        return {
            "prediction": prediction,
            "class_label": label_mapping.get(prediction, "unknown"),
            "probability_benign": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))