```markdown
# ⚡ ModelForge: Production-Ready End-to-End MLOps Platform

ModelForge is a modular, production-grade MLOps platform built to demonstrate rigorous engineering discipline, reproducible experiments, automated testing, and active drift monitoring using the Breast Cancer Wisconsin dataset.

---

## 🏗️ Architecture & Project Structure

```text
ModelForge/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (automated testing)
├── docs/
│   └── data_drift_report.html # Evidently AI automated data drift report
├── src/
│   ├── api/
│   │   └── main.py            # FastAPI inference service with Pydantic validation
│   ├── models/
│   │   └── train.py           # Model training & MLflow experiment tracking script
│   ├── monitoring/
│   │   └── drift.py           # Evidently AI data drift detection script
│   └── pipeline/
│       └── data_loader.py     # Leakage-free data ingestion & preprocessing pipeline
├── tests/
│   ├── test_api.py            # Pytest test suite for API endpoints & validation
│   └── test_pipeline.py       # Pytest test suite for data loader logic
├── Dockerfile                 # Containerization definition for deployment
├── requirements.txt           # Pinned project dependencies
└── README.md

```

---

## 🚀 Key Features & Engineering Highlights

1. **Leakage-Free Data Engineering**: Stratified train-test splits with scalers fitted **exclusively** on training partitions (`StandardScaler` isolation).
2. **Experiment Tracking**: Full hyperparameter tuning, metric logging (ROC-AUC, Precision, Recall), and artifact versioning powered by **MLflow**.
3. **Robust Serving API**: High-performance REST API built using **FastAPI** featuring strict input payload length validation and health-check container probes.
4. **Automated Testing Suite**: Comprehensive unit and integration testing via **Pytest** and FastAPI's `TestClient`.
5. **Production Monitoring**: Automated data drift detection powered by **Evidently AI**, generating visual HTML drift reports.
6. **Containerization & CI/CD**: Fully containerized using **Docker** with automated CI testing pipelines configured via **GitHub Actions**.

---

## 🛠️ Getting Started & Local Development

### 1. Clone & Set Up Environment

```powershell
git clone [https://github.com/your-username/ModelForge.git](https://github.com/your-username/ModelForge.git)
cd ModelForge
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
pip install pytest httpx setuptools<70.0.0

```

### 2. Run Training & MLflow Tracking

```powershell
\$env:PYTHONPATH="."
python src/models/train.py

```

### 3. Launch FastAPI Inference Service

```powershell
\$env:PYTHONPATH="."
uvicorn src.api.main:app --reload

```

### 4. Run Automated Tests

```powershell
\$env:PYTHONPATH="."
.\venv\Scripts\python -m pytest

```

### 5. Generate Data Drift Report

```powershell
\$env:PYTHONPATH="."
python src/monitoring/drift.py

```

---

## 🐳 Running with Docker

Build and run the container locally:

```powershell
docker build -t modelforge:latest .
docker run -p 8000:8000 modelforge:latest

```

