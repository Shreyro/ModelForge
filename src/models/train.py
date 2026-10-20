
"""Train and evaluate the ModelForge baseline model."""

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline.data_loader import load_and_preprocess_data


def train_baseline_model(C: float = 1.0, max_iter: int = 1000):
    """Train a leakage-safe model pipeline and log it to MLflow."""

    X_train, X_test, y_train, y_test, feature_names = (
        load_and_preprocess_data()
    )

    mlflow.set_experiment("ModelForge-Baseline")

    with mlflow.start_run(
        run_name="logistic-regression-baseline"
    ) as run:
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_count", len(feature_names))

        # Scaling is fitted only on training data.
        # The same scaler is saved with the trained classifier.
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
            print(f"{metric_name}: {metric_value:.4f}")

        # Log the complete pipeline, not just the classifier.
        mlflow.sklearn.log_model(model, "model")

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print("Model pipeline trained and logged successfully.")

        return run.info.run_id


if __name__ == "__main__":
    train_baseline_model()
