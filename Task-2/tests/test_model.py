"""
Automated Test Suite for KNN Iris Classification
DecodeLabs AI Project 2

Tests cover data integrity, pipeline architecture, no data leakage,
evaluation metrics, reproducibility, and invalid input validation.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from train import (
    FEATURE_NAMES,
    TARGET_NAMES,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_K_NEIGHBORS,
    load_iris_data,
    split_data,
    create_pipeline,
    train_model,
    evaluate_model,
    predict_sample,
    validate_input_measurements,
    compute_k_elbow_curve,
    save_model,
    load_model,
)


def test_load_iris_data():
    """Verify raw Iris dataset dimensions and schema."""
    X, y, feature_names, target_names = load_iris_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert feature_names == FEATURE_NAMES
    assert target_names == TARGET_NAMES
    assert set(y.unique()) == {0, 1, 2}


def test_split_data_integrity():
    """Verify 80:20 train-test split and no index overlap."""
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, shuffle=True
    )

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30

    # Ensure no data leakage between split indices
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(test_idx)


def test_pipeline_architecture_preventing_leakage():
    """Verify pipeline structure contains StandardScaler and KNeighborsClassifier."""
    pipeline = create_pipeline(n_neighbors=5)
    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], KNeighborsClassifier)
    assert pipeline.named_steps["classifier"].n_neighbors == 5

    # Verify scaler is not fitted before training
    assert not hasattr(pipeline.named_steps["scaler"], "mean_")


def test_model_training_and_metrics():
    """Train pipeline and verify comprehensive evaluation metrics."""
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_model(X_train, y_train, n_neighbors=5)

    metrics = evaluate_model(pipeline, X_test, y_test)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 3
    assert len(metrics["confusion_matrix"][0]) == 3

    assert "per_class" in metrics
    for name in TARGET_NAMES:
        assert name in metrics["per_class"]
        cls_m = metrics["per_class"][name]
        assert 0.0 <= cls_m["precision"] <= 1.0
        assert 0.0 <= cls_m["recall"] <= 1.0
        assert 0.0 <= cls_m["f1_score"] <= 1.0
        assert cls_m["support"] > 0


def test_reproducibility():
    """Ensure identical metrics and predictions with same random seed."""
    X, y, _, _ = load_iris_data()

    # Run 1
    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(X, y, random_state=42)
    p1 = train_model(X_train_1, y_train_1)
    m1 = evaluate_model(p1, X_test_1, y_test_1)

    # Run 2
    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(X, y, random_state=42)
    p2 = train_model(X_train_2, y_train_2)
    m2 = evaluate_model(p2, X_test_2, y_test_2)

    assert m1["accuracy"] == m2["accuracy"]
    assert m1["confusion_matrix"] == m2["confusion_matrix"]
    assert np.array_equal(p1.predict(X_test_1), p2.predict(X_test_2))


def test_prediction_workflow_and_probabilities():
    """Verify single-sample inference and probability calibration."""
    X, y, _, _ = load_iris_data()
    X_train, _, y_train, _ = split_data(X, y)
    pipeline = train_model(X_train, y_train)

    # Typical Setosa observation: short petal length and width
    setosa_features = [5.1, 3.5, 1.4, 0.2]
    result_setosa = predict_sample(pipeline, setosa_features)

    assert result_setosa["class_name"] == "setosa"
    assert result_setosa["class_index"] == 0
    assert "probabilities" in result_setosa
    assert np.isclose(sum(result_setosa["probabilities"].values()), 1.0)
    assert result_setosa["probabilities"]["setosa"] >= 0.8

    # Typical Virginica observation: long petal length and width
    virginica_features = [6.7, 3.3, 5.7, 2.1]
    result_virginica = predict_sample(pipeline, virginica_features)
    assert result_virginica["class_name"] == "virginica"
    assert result_virginica["class_index"] == 2


def test_invalid_inputs_handled_gracefully():
    """Verify that malformed or out-of-bound inputs raise explicit ValueError."""
    # Fewer or more than 4 features
    with pytest.raises(ValueError, match="Expected exactly 4 measurements"):
        validate_input_measurements([5.1, 3.5, 1.4])

    with pytest.raises(ValueError, match="Expected exactly 4 measurements"):
        validate_input_measurements([5.1, 3.5, 1.4, 0.2, 1.0])

    # Zero or negative measurements
    with pytest.raises(ValueError, match="must be positive"):
        validate_input_measurements([5.1, 0.0, 1.4, 0.2])

    with pytest.raises(ValueError, match="must be positive"):
        validate_input_measurements([-1.5, 3.5, 1.4, 0.2])

    # Unrealistic physical measurement (>30 cm)
    with pytest.raises(ValueError, match="exceeds realistic botanical bounds"):
        validate_input_measurements([50.0, 3.5, 1.4, 0.2])

    # Non-finite values
    with pytest.raises(ValueError, match="non-finite value"):
        validate_input_measurements([float("nan"), 3.5, 1.4, 0.2])


def test_save_and_load_model():
    """Verify trained pipeline persistence using joblib."""
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, _ = split_data(X, y)
    pipeline = train_model(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = os.path.join(tmp_dir, "test_pipeline.joblib")
        save_model(pipeline, tmp_model_path)
        assert os.path.exists(tmp_model_path)

        loaded_pipeline = load_model(tmp_model_path)
        orig_preds = pipeline.predict(X_test)
        loaded_preds = loaded_pipeline.predict(X_test)
        assert np.array_equal(orig_preds, loaded_preds)


def test_elbow_curve_calculation():
    """Verify calculation of K-error curve."""
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    elbow_df = compute_k_elbow_curve(X_train, y_train, X_test, y_test, k_range=range(1, 10))

    assert isinstance(elbow_df, pd.DataFrame)
    assert len(elbow_df) == 9
    assert "k" in elbow_df.columns
    assert "error_rate" in elbow_df.columns
    assert "accuracy" in elbow_df.columns
