
"""Load and split the Breast Cancer Wisconsin dataset.

Preprocessing is fitted inside the model pipeline during training.
The loader returns raw features to avoid preprocessing inconsistencies.
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Return raw train/test features, labels, and feature names."""

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("Data loaded successfully:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Training class distribution: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, list(data.feature_names)


if __name__ == "__main__":
    load_and_preprocess_data()
