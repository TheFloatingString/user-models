"""Utilities for extracting and processing activations."""

from typing import Any
import torch
import numpy as np


def extract_layer_activations(
    model: Any,
    text: str,
    layer_idx: int,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract activations from a specific layer.

    Args:
        model: HookedTransformer model
        text: Input text
        layer_idx: Layer index to extract from
        pooling: Pooling strategy ('mean', 'max', 'last')

    Returns:
        Pooled activation array
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(text)
        layer_acts = cache[f"blocks.{layer_idx}.hook_resid_post"]

        if pooling == "mean":
            pooled = layer_acts.mean(dim=1)
        elif pooling == "max":
            pooled = layer_acts.max(dim=1)[0]
        elif pooling == "last":
            pooled = layer_acts[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return pooled.cpu().numpy()[0]


def extract_multi_layer_activations(
    model: Any,
    text: str,
    layer_indices: list[int],
    pooling: str = "mean",
) -> dict[int, np.ndarray]:
    """Extract activations from multiple layers.

    Args:
        model: HookedTransformer model
        text: Input text
        layer_indices: List of layer indices
        pooling: Pooling strategy

    Returns:
        Dict mapping layer idx to activations
    """
    activations = {}

    with torch.no_grad():
        _, cache = model.run_with_cache(text)

        for layer_idx in layer_indices:
            layer_acts = cache[f"blocks.{layer_idx}.hook_resid_post"]

            if pooling == "mean":
                pooled = layer_acts.mean(dim=1)
            elif pooling == "max":
                pooled = layer_acts.max(dim=1)[0]
            elif pooling == "last":
                pooled = layer_acts[:, -1, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            activations[layer_idx] = pooled.cpu().numpy()[0]

    return activations


def normalize_activations(
    activations: np.ndarray,
    method: str = "standard",
) -> np.ndarray:
    """Normalize activation vectors.

    Args:
        activations: Array of shape (n_samples, n_features)
        method: Normalization method ('standard', 'l2', 'none')

    Returns:
        Normalized activations
    """
    if method == "standard":
        mean = activations.mean(axis=0)
        std = activations.std(axis=0)
        return (activations - mean) / (std + 1e-8)
    elif method == "l2":
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        return activations / (norms + 1e-8)
    elif method == "none":
        return activations
    else:
        raise ValueError(f"Unknown normalization: {method}")


def compute_activation_statistics(
    activations: np.ndarray,
) -> dict[str, float]:
    """Compute statistics for activations.

    Args:
        activations: Array of activations

    Returns:
        Dictionary with statistics
    """
    return {
        "mean": float(activations.mean()),
        "std": float(activations.std()),
        "min": float(activations.min()),
        "max": float(activations.max()),
        "l2_norm": float(np.linalg.norm(activations)),
    }
