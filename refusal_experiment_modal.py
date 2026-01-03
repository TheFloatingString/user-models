"""Modal script for refusal rate experiment with activation steering."""

import modal
import os
import json


def download_model() -> None:
    """Download and cache Gemma model during image build."""
    from huggingface_hub import snapshot_download

    print("Downloading Gemma-2-9b model...")
    snapshot_download(
        repo_id="google/gemma-2-9b",
        ignore_patterns=["*.safetensors"],
    )
    print("Model cached!")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformer-lens>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "wandb>=0.16.0",
        "huggingface-hub>=0.20.0",
        "accelerate>=0.25.0",
    )
    .pip_install("hf_transfer")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .run_function(
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App("refusal-experiment")
volume = modal.Volume.from_name("refusal-results", create_if_missing=True)
VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def compute_steering_vectors(
    layer_indices: list[int],
) -> dict[int, list[float]]:
    """Compute steering vectors for helpful vs refusing responses.

    Args:
        layer_indices: Layers to compute vectors for

    Returns:
        Dict mapping layer_idx to steering vector
    """
    import torch
    from transformer_lens import HookedTransformer
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading Gemma model...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-9b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    helpful_examples = [
        "To register to vote, you can visit your local election office.",
        "You can access healthcare through community health centers.",
        "Filing taxes is straightforward with free online services.",
        "Most hospitals provide emergency care regardless of payment.",
        "Voter registration can be done online in most states.",
    ]

    refusing_examples = [
        "I cannot provide advice on voter registration matters.",
        "I'm unable to assist with healthcare-related questions.",
        "I cannot help with tax-related inquiries.",
        "I'm not able to provide information about that topic.",
        "I cannot assist with that type of question.",
    ]

    from src.activation_steering import compute_steering_vector

    steering_vectors = {}

    for layer_idx in layer_indices:
        print(f"Computing steering vector for layer {layer_idx}...")
        vector = compute_steering_vector(
            model, helpful_examples, refusing_examples, layer_idx
        )
        steering_vectors[layer_idx] = vector.tolist()

    return steering_vectors


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_with_steering_batch(
    prompts: list[str],
    steering_vector: list[float],
    layer_idx: int,
    coefficient: float = 1.0,
) -> list[str]:
    """Generate responses with activation steering.

    Args:
        prompts: List of prompts
        steering_vector: Steering vector
        layer_idx: Layer to apply steering
        coefficient: Steering strength

    Returns:
        List of generated responses
    """
    import torch
    import numpy as np
    from transformer_lens import HookedTransformer
    from huggingface_hub import login
    from transformers import AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading model and tokenizer...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-9b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    from src.activation_steering import generate_with_steering

    vector_np = np.array(steering_vector)
    responses = []

    for prompt in prompts:
        print(f"Generating for: {prompt[:50]}...")
        response = generate_with_steering(
            model,
            tokenizer,
            prompt,
            vector_np,
            layer_idx,
            coefficient,
        )
        responses.append(response)

    return responses


@app.local_entrypoint()
def main(
    layer_indices: str = "20,25,30",
    coefficient: float = 1.5,
    output_file: str = "data/steering_vectors.json",
) -> None:
    """Compute steering vectors for experiment.

    Args:
        layer_indices: Comma-separated layer indices
        coefficient: Steering coefficient
        output_file: Output path
    """
    layers = [int(x.strip()) for x in layer_indices.split(",")]

    print(f"Computing steering vectors for layers: {layers}")

    steering_vectors = compute_steering_vectors.remote(layers)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "layer_indices": layers,
                "coefficient": coefficient,
                "vectors": steering_vectors,
            },
            f,
            indent=2,
        )

    print(f"Saved steering vectors to {output_file}")
