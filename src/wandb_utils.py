"""Utilities for Weights & Biases logging."""

from typing import Any
import numpy as np


def log_probe_metrics(
    wandb: Any,
    feature_name: str,
    metrics: dict[str, Any],
) -> None:
    """Log probe evaluation metrics to W&B.

    Args:
        wandb: W&B module
        feature_name: Name of feature being probed
        metrics: Dictionary of metrics
    """
    wandb.log(
        {
            f"{feature_name}/accuracy": metrics["accuracy"],
            f"{feature_name}/n_samples": metrics.get("n_samples", 0),
        }
    )


def log_confusion_matrix(
    wandb: Any,
    feature_name: str,
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> None:
    """Log confusion matrix to W&B.

    Args:
        wandb: W&B module
        feature_name: Name of feature
        confusion_matrix: Confusion matrix array
        class_names: List of class names
    """
    wandb.log(
        {
            f"{feature_name}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=None,
                preds=None,
                class_names=class_names,
            )
        }
    )


def log_layer_sweep_results(
    wandb: Any,
    layer_results: list[dict[str, Any]],
) -> None:
    """Log results from layer sweep.

    Args:
        wandb: W&B module
        layer_results: List of results per layer
    """
    for result in layer_results:
        layer_idx = result["layer_idx"]
        accuracy = result["accuracy"]

        wandb.log(
            {
                "layer_idx": layer_idx,
                "layer_accuracy": accuracy,
            }
        )


def create_wandb_config(
    model_name: str,
    layer_idx: int,
    num_personas: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create W&B config dictionary.

    Args:
        model_name: Name of model
        layer_idx: Layer being probed
        num_personas: Number of personas
        **kwargs: Additional config params

    Returns:
        Config dictionary
    """
    config = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "num_personas": num_personas,
    }
    config.update(kwargs)
    return config


def log_activation_stats(
    wandb: Any,
    stats: dict[str, float],
    prefix: str = "activations",
) -> None:
    """Log activation statistics to W&B.

    Args:
        wandb: W&B module
        stats: Dictionary of statistics
        prefix: Prefix for metric names
    """
    wandb.log({f"{prefix}/{k}": v for k, v in stats.items()})
