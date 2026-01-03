"""Activation steering utilities for reducing refusals."""

from typing import Any
import torch
import numpy as np


def compute_steering_vector(
    model: Any,
    helpful_texts: list[str],
    refusing_texts: list[str],
    layer_idx: int,
) -> np.ndarray:
    """Compute steering vector from helpful vs refusing examples.

    Args:
        model: HookedTransformer model
        helpful_texts: Examples of helpful responses
        refusing_texts: Examples of refusing responses
        layer_idx: Layer to compute steering vector for

    Returns:
        Steering vector as numpy array
    """
    helpful_acts = []
    refusing_acts = []

    with torch.no_grad():
        for text in helpful_texts:
            _, cache = model.run_with_cache(text)
            acts = cache[f"blocks.{layer_idx}.hook_resid_post"]
            pooled = acts.mean(dim=1).cpu().numpy()[0]
            helpful_acts.append(pooled)

        for text in refusing_texts:
            _, cache = model.run_with_cache(text)
            acts = cache[f"blocks.{layer_idx}.hook_resid_post"]
            pooled = acts.mean(dim=1).cpu().numpy()[0]
            refusing_acts.append(pooled)

    helpful_mean = np.mean(helpful_acts, axis=0)
    refusing_mean = np.mean(refusing_acts, axis=0)

    steering_vector = helpful_mean - refusing_mean

    norm = np.linalg.norm(steering_vector)
    if norm > 1e-8:
        steering_vector = steering_vector / norm

    return steering_vector


def apply_steering_hook(
    steering_vector: torch.Tensor, coefficient: float = 1.0
) -> callable:
    """Create hook function for activation steering.

    Args:
        steering_vector: Steering vector tensor
        coefficient: Scaling coefficient

    Returns:
        Hook function
    """

    def hook_fn(activations: torch.Tensor, hook: Any) -> torch.Tensor:
        """Apply steering to activations."""
        if len(activations.shape) == 3:
            steering_shaped = steering_vector.unsqueeze(0).unsqueeze(0)
            return activations + coefficient * steering_shaped
        return activations

    return hook_fn


def generate_with_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    steering_vector: np.ndarray,
    layer_idx: int,
    coefficient: float = 1.0,
    max_tokens: int = 200,
) -> str:
    """Generate text with activation steering applied.

    Args:
        model: HookedTransformer model
        tokenizer: Model tokenizer
        prompt: Input prompt
        steering_vector: Steering vector
        layer_idx: Layer to apply steering
        coefficient: Steering strength
        max_tokens: Max generation tokens

    Returns:
        Generated text
    """
    steering_tensor = torch.tensor(
        steering_vector, dtype=model.cfg.dtype, device=model.cfg.device
    )

    hook_fn = apply_steering_hook(steering_tensor, coefficient)
    hook_name = f"blocks.{layer_idx}.hook_resid_post"

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        tokens = tokens.to(model.cfg.device)

        output = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated
