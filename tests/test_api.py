
"""Tests for the ModelForge prediction API."""

from fastapi.testclient import TestClient

from src.api.main import app

# Using TestClient as a context manager runs the app's lifespan.
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_predict_invalid_input_length():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0]},
        )

    assert response.status_code == 400
    assert "Expected 30 features" in response.json()["detail"]
