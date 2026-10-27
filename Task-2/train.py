"""
AI-Based Iris Flower Classification Using K-Nearest Neighbors
DecodeLabs Artificial Intelligence Project 2

This module implements reproducible training, feature scaling via Pipeline
(preventing data leakage), model evaluation, and inference helpers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Default Configuration per Technical Blueprint
DEFAULT_TEST_SIZE: float = 0.20
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_K_NEIGHBORS: int = 5
MODEL_SAVE_PATH: str = os.path.join(os.path.dirname(__file__), "models", "iris_knn_pipeline.joblib")

FEATURE_NAMES: List[str] = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

TARGET_NAMES: List[str] = ["setosa", "versicolor", "virginica"]

# Realistic biological measurement ranges (in cm) for validation
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "sepal length (cm)": (4.0, 8.0),
    "sepal width (cm)": (2.0, 4.5),
    "petal length (cm)": (1.0, 7.0),
    "petal width (cm)": (0.1, 2.6),
}


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load the scikit-learn Iris flower dataset.

    Returns:
        X (pd.DataFrame): 150 samples with 4 feature dimensions.
        y (pd.Series): Target class integer labels (0, 1, 2).
        feature_names (List[str]): Names of the 4 features.
        target_names (List[str]): Names of the 3 flower classes.
    """
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)
    return X, y, feature_names, target_names


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
    stratify: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing subsets.

    Structural integrity requirement:
    Shuffling randomizes rows before splitting to remove ordering bias.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )


def create_pipeline(n_neighbors: int = DEFAULT_K_NEIGHBORS) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with StandardScaler and KNeighborsClassifier.

    The Gatekeeper Rule:
    The StandardScaler is included inside the Pipeline so it is fitted ONLY
    on the training partition during fit(), preventing data leakage into the test set.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski", p=2)),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> Pipeline:
    """
    Instantiate and train the KNN classification pipeline on the training split.
    """
    pipeline = create_pipeline(n_neighbors=n_neighbors)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the model on unseen test data.

    Returns:
        Dict containing accuracy, confusion matrix, per-class metrics,
        and macro/weighted precision, recall, and f1 scores.
    """
    if target_names is None:
        target_names = TARGET_NAMES

    y_pred = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    # Per-class metrics
    class_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = {}
    for name in target_names:
        per_class_metrics[name] = {
            "precision": float(class_report_dict[name]["precision"]),
            "recall": float(class_report_dict[name]["recall"]),
            "f1_score": float(class_report_dict[name]["f1-score"]),
            "support": int(class_report_dict[name]["support"]),
        }

    macro_precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_mat,
        "per_class": per_class_metrics,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report_text": classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        "test_sample_count": len(y_test),
        "y_true": y_test.tolist() if hasattr(y_test, "tolist") else list(y_test),
        "y_pred": y_pred.tolist(),
    }


def compute_k_elbow_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_range: range = range(1, 26),
) -> pd.DataFrame:
    """
    Compute error rates across different values of K to generate the Elbow Curve.
    Demonstrates overfitting at low K (k=1 noise sensitivity) vs underfitting at high K.
    """
    records = []
    for k in k_range:
        pipe = create_pipeline(n_neighbors=k)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        error_rate = 1.0 - acc
        records.append({"k": k, "accuracy": acc, "error_rate": error_rate})
    return pd.DataFrame(records)


def validate_input_measurements(measurements: Union[List[float], Dict[str, float]]) -> List[float]:
    """
    Validates flower dimensions. Raises ValueError if inputs are invalid or non-physical.
    """
    if isinstance(measurements, dict):
        ordered_keys = FEATURE_NAMES
        missing = [k for k in ordered_keys if k not in measurements]
        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")
        vals = [measurements[k] for k in ordered_keys]
    elif isinstance(measurements, (list, tuple, np.ndarray)):
        if len(measurements) != 4:
            raise ValueError(f"Expected exactly 4 measurements, received {len(measurements)}")
        vals = [float(v) for v in measurements]
    else:
        raise TypeError("Measurements must be a list of 4 numbers or a dictionary with feature names.")

    for i, (val, name) in enumerate(zip(vals, FEATURE_NAMES)):
        if not np.isfinite(val):
            raise ValueError(f"Feature '{name}' has non-finite value: {val}")
        if val <= 0:
            raise ValueError(f"Feature '{name}' must be positive. Received {val}")
        # Sanity check against absurd physical inputs
        if val > 30.0:
            raise ValueError(f"Feature '{name}' ({val} cm) exceeds realistic botanical bounds (<30 cm).")

    return vals


def predict_sample(
    pipeline: Pipeline,
    measurements: Union[List[float], Dict[str, float]],
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate and predict class and class probabilities for a single flower observation.
    """
    if target_names is None:
        target_names = TARGET_NAMES

    valid_vals = validate_input_measurements(measurements)
    input_df = pd.DataFrame([valid_vals], columns=FEATURE_NAMES)

    pred_idx = int(pipeline.predict(input_df)[0])
    pred_class = target_names[pred_idx]

    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        for idx, name in enumerate(target_names):
            probabilities[name] = float(probs[idx])

    return {
        "class_index": pred_idx,
        "class_name": pred_class,
        "probabilities": probabilities,
        "input_features": dict(zip(FEATURE_NAMES, valid_vals)),
    }


def save_model(pipeline: Pipeline, filepath: str = MODEL_SAVE_PATH) -> str:
    """Save the trained pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    return filepath


def load_model(filepath: str = MODEL_SAVE_PATH) -> Pipeline:
    """Load the trained pipeline from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}. Run train.py first.")
    return joblib.load(filepath)


def run_training_experiment(
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_neighbors: int = DEFAULT_K_NEIGHBORS,
    save_artifact: bool = True,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate full training and evaluation workflow.
    """
    X, y, _, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    pipeline = train_model(X_train, y_train, n_neighbors=n_neighbors)
    metrics = evaluate_model(pipeline, X_test, y_test)

    if save_artifact:
        save_model(pipeline)

    return pipeline, metrics, X_train, X_test


def main() -> None:
    print("=" * 65)
    print("DecodeLabs AI Project 2 — KNN Iris Flower Classification")
    print("=" * 65)
    print(f"Configuration: test_size={DEFAULT_TEST_SIZE}, random_state={DEFAULT_RANDOM_STATE}, k={DEFAULT_K_NEIGHBORS}")
    print("Architecture: StandardScaler -> KNeighborsClassifier (Scikit-Learn Pipeline)")
    print("-" * 65)

    pipeline, metrics, X_train, X_test = run_training_experiment()

    print(f"Dataset Size: 150 total samples (Training: {len(X_train)}, Testing: {len(X_test)})")
    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n")

    print("Confusion Matrix:")
    print("                 Predicted Setosa  Versicolor  Virginica")
    for row_name, row in zip(TARGET_NAMES, metrics["confusion_matrix"]):
        print(f"Actual {row_name:<10}: {row[0]:>14}  {row[1]:>10}  {row[2]:>9}")

    print("\nPer-Class Classification Report:")
    for cls_name, vals in metrics["per_class"].items():
        print(f" - {cls_name.capitalize():<11}: Precision={vals['precision']:.2f}, Recall={vals['recall']:.2f}, F1={vals['f1_score']:.2f} (Support: {vals['support']})")

    print("\nImportant Note on Model Evaluation:")
    print("These evaluation results reflect the chosen 80:20 shuffle split and fixed random seed (42).")
    print("Because Iris is a compact, balanced benchmark, accuracy varies slightly across alternative splits.")
    print("Never evaluate solely on a single split or raw accuracy without per-class diagnostic metrics.")
    print("=" * 65)


if __name__ == "__main__":
    main()
