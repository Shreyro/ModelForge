"""
Training script for ModelForge baseline model.
Trains a Logistic Regression classifier, evaluates performance,
and logs parameters, metrics, and artifacts to MLflow.
"""

import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from src.pipeline.data_loader import load_and_preprocess_data

def train_baseline_model(C: float = 1.0, max_iter: int = 1000):
    """
    Trains the baseline Logistic Regression model and logs to MLflow.
    """
    # 1. Load data using our leak-free pipeline
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()

    # 2. Set MLflow experiment
    mlflow.set_experiment("ModelForge-Baseline")

    with mlflow.start_run(run_name="logistic-regression-baseline") as run:
        print(f"\nStarting MLflow Run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", 0.2)

        # 3. Initialize and train model
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        # 4. Predict and evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        # 5. Save model artifact to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print("\nModel trained and logged successfully to MLflow!")
        return run.info.run_id

if __name__ == "__main__":
    train_baseline_model()