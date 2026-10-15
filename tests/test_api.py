from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["platform"] == "ModelForge"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint_validation():
    # Test with invalid feature length (should throw 400 error)
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 400