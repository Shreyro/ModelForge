from src.pipeline.data_loader import load_and_preprocess_data

def test_data_loader():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Assertions for shapes and data integrity
    assert X_train.shape[1] == 30, "Should have exactly 30 features"
    assert X_train.shape[0] > X_test.shape[0], "Training set should be larger than test set"
    assert len(feature_names) == 30, "Feature names list should match feature count"
    assert len(y_train) == X_train.shape[0], "Target and feature rows must match"