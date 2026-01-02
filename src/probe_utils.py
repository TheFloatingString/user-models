"""Utilities for training and evaluating linear probes."""

from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def prepare_probe_data(
    activations: list[np.ndarray],
    labels: list[dict],
    feature_name: str,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Prepare activation data for probe training.

    Args:
        activations: List of activation arrays
        labels: List of label dictionaries
        feature_name: Feature to extract from labels
        test_size: Fraction for test split

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, classes)
    """
    # Extract feature labels
    y_raw = [label[feature_name] for label in labels]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Repeat activations for each turn if needed
    X = []
    y_expanded = []
    for i, acts in enumerate(activations):
        if isinstance(acts, list):
            for act in acts:
                X.append(act)
                y_expanded.append(y[i])
        else:
            X.append(acts)
            y_expanded.append(y[i])

    X = np.array(X)
    y_expanded = np.array(y_expanded)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_expanded, test_size=test_size, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder.classes_.tolist(),
    )


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a logistic regression probe.

    Args:
        X_train: Training features
        y_train: Training labels
        max_iter: Maximum iterations
        random_state: Random seed

    Returns:
        Trained LogisticRegression model
    """
    probe = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )
    probe.fit(X_train, y_train)
    return probe


def evaluate_probe(
    probe: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Evaluate probe performance.

    Args:
        probe: Trained probe model
        X_test: Test features
        y_test: Test labels
        classes: Class names

    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "classes": classes,
    }


def get_probe_weights(
    probe: LogisticRegression,
) -> np.ndarray:
    """Extract probe weights.

    Args:
        probe: Trained probe model

    Returns:
        Probe weight matrix
    """
    return probe.coef_


def compute_top_features(
    probe: LogisticRegression,
    feature_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[int, list[tuple[int, float]]]:
    """Get top contributing features per class.

    Args:
        probe: Trained probe model
        feature_names: Optional feature names
        top_k: Number of top features to return

    Returns:
        Dict mapping class idx to top features
    """
    weights = probe.coef_
    top_features = {}

    for class_idx, class_weights in enumerate(weights):
        top_indices = np.argsort(np.abs(class_weights))[-top_k:]
        top_values = class_weights[top_indices]

        if feature_names:
            top_features[class_idx] = [
                (feature_names[idx], val) for idx, val in zip(top_indices, top_values)
            ]
        else:
            top_features[class_idx] = [
                (idx, val) for idx, val in zip(top_indices, top_values)
            ]

    return top_features
