"""
Data loading and preprocessing pipeline for ModelForge.
Loads the Breast Cancer Wisconsin dataset from scikit-learn, splits it,
and applies standard scaling without data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_and_preprocess_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads breast cancer dataset, splits into stratified train/test sets,
    and scales features using statistics computed ONLY on the training set.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    """
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target  # 0 = malignant, 1 = benign (in sklearn convention, let's verify or keep as is)

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Data loaded successfully:")
    print(f"Training samples: {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    print(f"Class distribution in train: {np.bincount(y_train)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, list(data.feature_names)

if __name__ == "__main__":
    load_and_preprocess_data()