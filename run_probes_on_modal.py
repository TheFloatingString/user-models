"""
Script using Modal, OpenRouter, and TransformerLens for mech interp:
1. GPT-4o generates conversations as personas with demographics
2. Gemma-2-9b-it responds and we extract activations
3. Train linear probes to detect demographic info in activations
4. Evaluate what demographic features are encoded in the model
5. Log everything to Weights & Biases
"""

import modal
import os
import json


def download_model():
    """Download and cache Gemma model during image build."""
    from huggingface_hub import snapshot_download

    print("Downloading Gemma-2-9b model to cache...")
    snapshot_download(
        repo_id="google/gemma-2-9b",
        ignore_patterns=["*.safetensors"],  # Download PyTorch weights
    )
    print("Model cached successfully!")


def _parse_age_range(age_range: str) -> tuple[float, float]:
    """Parse an age range like '18-25' into (lower, upper)."""
    parts = age_range.replace(" ", "").split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid age_range format: {age_range}")
    return float(parts[0]), float(parts[1])


def _parse_income_range(income_range: str) -> tuple[float, float]:
    """Parse an income range like '$0-30,000' into (lower, upper)."""
    cleaned = income_range.replace("$", "").replace(",", "").replace(" ", "")
    parts = cleaned.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid income_range format: {income_range}")
    return float(parts[0]), float(parts[1])


# Define Modal image with dependencies and cache model
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformer-lens>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "wandb>=0.16.0",
        "huggingface-hub>=0.20.0",
        "bitsandbytes>=0.41.0",
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

app = modal.App("gemma-linear-probes")

# Create Modal volume for storing results
volume = modal.Volume.from_name("probe-results", create_if_missing=True)

VOLUME_PATH = "/data"

USER_MODEL = "openai/gpt-4o"
ASSISTANT_MODEL = "google/gemma-2-9b-it"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-secret")],
    timeout=3600,
)
def generate_conversation_with_persona(
    age_range: str,
    income_range: str,
    education: str,
    sex: str,
    visa_status: str,
    num_turns: int = 10,
) -> list[dict]:
    """Generate conversation using GPT-4o as persona.

    Args:
        age_range: Age range of persona
        income_range: Income range of persona
        education: Education level
        sex: Sex of persona
        visa_status: Visa status
        num_turns: Number of conversation turns (default: 10)

    Returns:
        List of conversation messages
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # System prompt for persona
    persona_prompt = f"""You are a person with these demographics:
- Age: {age_range}
- Income: {income_range}
- Education: {education}
- Sex: {sex}
- Visa Status: {visa_status}

Engage naturally in conversation. Let your demographics
influence your perspective and knowledge, but don't explicitly
state them."""

    messages = []
    conversation = [{"role": "system", "content": persona_prompt}]

    # Generate conversation turns (expanded to 10 topics)
    topics = [
        "Tell me about your typical day.",
        "What are your thoughts on current technology?",
        "What are your future plans or goals?",
        "How do you handle financial decisions?",
        "What's your experience with education or learning?",
        "Tell me about your work or career.",
        "What are your hobbies or interests?",
        "How do you stay informed about current events?",
        "What are your thoughts on travel or living abroad?",
        "Describe your social life and relationships.",
    ]

    for i in range(min(num_turns, len(topics))):
        # Assistant asks question
        assistant_msg = topics[i]
        conversation.append({"role": "assistant", "content": assistant_msg})

        # User (GPT-4o persona) responds
        response = client.chat.completions.create(
            model=USER_MODEL,
            messages=conversation,
        )
        user_msg = response.choices[0].message.content
        conversation.append({"role": "user", "content": user_msg})

        messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": user_msg})

    return messages


@app.function(
    image=image,
    gpu="A100",  # Need 40GB for Gemma-2-9B
    timeout=1800,
    secrets=[
        modal.Secret.from_name("openrouter-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={VOLUME_PATH: volume},
)
def extract_activations_from_responses(
    conversations: list[dict],
    layer_idx: int,
) -> tuple[list, list]:
    """Extract Gemma activations from assistant responses.

    Args:
        conversations: List of conversation dicts
        layer_idx: Layer to extract activations from

    Returns:
        Tuple of (activations, labels)
    """
    import torch
    from transformer_lens import HookedTransformer
    from huggingface_hub import login

    # Authenticate with HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading Gemma model in float16...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-9b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    print("Gemma model loaded.")

    activations = []
    labels = []

    for conv_data in conversations:
        conversation = conv_data["conversation"]
        persona = conv_data["persona"]

        # Extract user responses (generated by GPT-4o persona)
        user_texts = [msg["content"] for msg in conversation if msg["role"] == "user"]

        # Get activations for each user response
        for text in user_texts:
            with torch.no_grad():
                _, cache = model.run_with_cache(text)
                # Extract from specified layer
                layer_acts = cache[f"blocks.{layer_idx}.hook_resid_post"]
                # Mean pool over sequence
                pooled = layer_acts.mean(dim=1).cpu().numpy()[0]
                activations.append(pooled)

                # Store labels for this activation
                labels.append(
                    {
                        "age_range": persona.get("age_range", ""),
                        "income_range": persona.get("income_range", ""),
                        "education": persona.get("education", ""),
                        "sex": persona.get("sex", ""),
                        "visa_status": persona.get("visa_status", ""),
                    }
                )

                # Clear cache to free memory
                del cache, layer_acts
                torch.cuda.empty_cache()

    print(f"Extracted {len(activations)} activations and {len(labels)} labels")
    print(f"Activation shape: {activations[0].shape if activations else 'N/A'}")

    return activations, labels


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_probe_for_feature(
    activations: list,
    labels: list[dict],
    feature_name: str,
    run_name: str,
) -> dict:
    """Train linear probe for specific demographic feature.

    Args:
        activations: List of activation arrays
        labels: List of label dicts
        feature_name: Feature to probe for
        run_name: W&B run name

    Returns:
        Probe results with accuracy metrics
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    # Extract feature labels
    y_raw = [label[feature_name] for label in labels]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Convert activations to numpy array
    X = np.array(activations)

    # Train/test split (50/50 split for 50 train, 50 test samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"{feature_name} probe accuracy: {accuracy:.3f}")

    return {
        "feature": feature_name,
        "accuracy": accuracy,
        "classes": label_encoder.classes_.tolist(),
        "n_samples": len(X),
        "confusion_matrix": conf_matrix.tolist(),
    }


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_age_probes(
    activations: list,
    labels: list[dict],
    run_name: str,
) -> dict:
    """Train linear probes for age estimate, lower bound, and upper bound."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    age_bounds = [_parse_age_range(label["age_range"]) for label in labels]
    y_lower = np.array([b[0] for b in age_bounds], dtype=float)
    y_upper = np.array([b[1] for b in age_bounds], dtype=float)
    y_mid = (y_lower + y_upper) / 2.0

    X = np.array(activations)

    (
        X_train,
        X_test,
        y_mid_train,
        y_mid_test,
        y_lower_train,
        y_lower_test,
        y_upper_train,
        y_upper_test,
    ) = train_test_split(
        X,
        y_mid,
        y_lower,
        y_upper,
        test_size=0.5,
        random_state=42,
    )

    estimate_model = Ridge(alpha=1.0, random_state=42)
    lower_model = Ridge(alpha=1.0, random_state=42)
    upper_model = Ridge(alpha=1.0, random_state=42)

    estimate_model.fit(X_train, y_mid_train)
    lower_model.fit(X_train, y_lower_train)
    upper_model.fit(X_train, y_upper_train)

    y_mid_pred = estimate_model.predict(X_test)
    y_lower_pred = lower_model.predict(X_test)
    y_upper_pred = upper_model.predict(X_test)

    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _corr(y_true, y_pred):
        if len(y_true) < 2:
            return 0.0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    estimate_rmse = _rmse(y_mid_test, y_mid_pred)
    estimate_corr = _corr(y_mid_test, y_mid_pred)

    lower_rmse = _rmse(y_lower_test, y_lower_pred)
    lower_corr = _corr(y_lower_test, y_lower_pred)

    upper_rmse = _rmse(y_upper_test, y_upper_pred)
    upper_corr = _corr(y_upper_test, y_upper_pred)

    lower_bound = np.minimum(y_lower_pred, y_upper_pred)
    upper_bound = np.maximum(y_lower_pred, y_upper_pred)
    interval_accuracy = float(
        np.mean((y_mid_test >= lower_bound) & (y_mid_test <= upper_bound))
    )

    return {
        "n_samples": len(X),
        "age_estimate": {
            "rmse": estimate_rmse,
            "correlation": estimate_corr,
        },
        "age_lower_bound": {
            "rmse": lower_rmse,
            "correlation": lower_corr,
        },
        "age_upper_bound": {
            "rmse": upper_rmse,
            "correlation": upper_corr,
        },
        "age_interval_accuracy": interval_accuracy,
    }


@app.function(
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_income_probes(
    activations: list,
    labels: list[dict],
    run_name: str,
) -> dict:
    """Train linear probes for income estimate, lower bound, and upper bound."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    income_bounds = [_parse_income_range(label["income_range"]) for label in labels]
    y_lower = np.array([b[0] for b in income_bounds], dtype=float)
    y_upper = np.array([b[1] for b in income_bounds], dtype=float)
    y_mid = (y_lower + y_upper) / 2.0

    X = np.array(activations)

    (
        X_train,
        X_test,
        y_mid_train,
        y_mid_test,
        y_lower_train,
        y_lower_test,
        y_upper_train,
        y_upper_test,
    ) = train_test_split(
        X,
        y_mid,
        y_lower,
        y_upper,
        test_size=0.5,
        random_state=42,
    )

    estimate_model = Ridge(alpha=1.0, random_state=42)
    lower_model = Ridge(alpha=1.0, random_state=42)
    upper_model = Ridge(alpha=1.0, random_state=42)

    estimate_model.fit(X_train, y_mid_train)
    lower_model.fit(X_train, y_lower_train)
    upper_model.fit(X_train, y_upper_train)

    y_mid_pred = estimate_model.predict(X_test)
    y_lower_pred = lower_model.predict(X_test)
    y_upper_pred = upper_model.predict(X_test)

    def _rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _corr(y_true, y_pred):
        if len(y_true) < 2:
            return 0.0
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    estimate_rmse = _rmse(y_mid_test, y_mid_pred)
    estimate_corr = _corr(y_mid_test, y_mid_pred)

    lower_rmse = _rmse(y_lower_test, y_lower_pred)
    lower_corr = _corr(y_lower_test, y_lower_pred)

    upper_rmse = _rmse(y_upper_test, y_upper_pred)
    upper_corr = _corr(y_upper_test, y_upper_pred)

    lower_bound = np.minimum(y_lower_pred, y_upper_pred)
    upper_bound = np.maximum(y_lower_pred, y_upper_pred)
    interval_accuracy = float(
        np.mean((y_mid_test >= lower_bound) & (y_mid_test <= upper_bound))
    )

    return {
        "n_samples": len(X),
        "income_estimate": {
            "rmse": estimate_rmse,
            "correlation": estimate_corr,
        },
        "income_lower_bound": {
            "rmse": lower_rmse,
            "correlation": lower_corr,
        },
        "income_upper_bound": {
            "rmse": upper_rmse,
            "correlation": upper_corr,
        },
        "income_interval_accuracy": interval_accuracy,
    }


@app.function(
    image=image,
    timeout=3600,
    secrets=[
        modal.Secret.from_name("openrouter-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={VOLUME_PATH: volume},
)
def run_probe_experiment(
    personas: list[dict],
    layer_indices: list[int] | None = None,
    project_name: str = "gemma-probes",
    run_name: str = "probe-experiment",
) -> dict:
    """Run full probe experiment on personas across multiple layers.

    Args:
        personas: List of persona dicts
        layer_indices: Layers to probe (default: all 42 layers)
        project_name: W&B project name
        run_name: W&B run name

    Returns:
        Experiment results organized by layer
    """
    import wandb

    # Default to all layers in Gemma-2-9B (42 layers, 0-41)
    if layer_indices is None:
        layer_indices = list(range(42))

    # Initialize W&B
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "layer_indices": layer_indices,
            "num_layers": len(layer_indices),
            "num_personas": len(personas),
            "user_model": USER_MODEL,
            "assistant_model": ASSISTANT_MODEL,
        },
    )

    print(f"Running probe experiment on {len(personas)} personas")
    print(f"Sweeping across {len(layer_indices)} layers: {layer_indices}")
    print(f"W&B run: {wandb.run.url}")

    # Generate conversations in parallel
    print("\n" + "=" * 50)
    print("GENERATING CONVERSATIONS (PARALLEL)")
    print("=" * 50)

    # Start all conversation generations in parallel
    print(f"Starting {len(personas)} conversations in parallel...")
    futures = []
    for persona in personas:
        future = generate_conversation_with_persona.spawn(
            age_range=persona.get("age_range", ""),
            income_range=persona.get("income_range", ""),
            education=persona.get("education", ""),
            sex=persona.get("sex", ""),
            visa_status=persona.get("visa_status", ""),
        )
        futures.append((persona, future))

    # Collect results as they complete
    conversations = []
    for i, (persona, future) in enumerate(futures):
        print(f"Waiting for persona {i + 1}/{len(futures)}...")
        conv = future.get()
        conversations.append(
            {
                "persona": persona,
                "conversation": conv,
            }
        )

    print(f"Completed all {len(conversations)} conversations")
    wandb.log({"num_conversations": len(conversations)})

    # Sweep through all layers
    all_layer_results = {}
    categorical_features = ["education", "sex"]

    for layer_idx in layer_indices:
        print("\n" + "=" * 60)
        print(f"PROCESSING LAYER {layer_idx}")
        print("=" * 60)

        # Extract activations for this layer
        print(f"Extracting activations from layer {layer_idx}...")
        activations, labels = extract_activations_from_responses.remote(
            conversations, layer_idx
        )
        print(f"Extracted {len(activations)} activations")

        # Train probes for each categorical feature
        layer_results = []
        categorical_results = []
        for feature in categorical_features:
            print(f"Training probe for: {feature}")
            result = train_probe_for_feature.local(
                activations,
                labels,
                feature,
                run_name,
            )
            categorical_results.append(result)
            layer_results.append(result)

            # Log to W&B with layer prefix
            wandb.log(
                {
                    f"layer_{layer_idx}/{feature}/accuracy": result["accuracy"],
                    f"layer_{layer_idx}/{feature}/n_samples": result["n_samples"],
                }
            )

        # Train age probes
        print("Training probes for: age")
        age_metrics = train_age_probes.local(activations, labels, run_name)
        layer_results.extend(
            [
                {
                    "feature": "age_estimate",
                    "rmse": age_metrics["age_estimate"]["rmse"],
                    "correlation": age_metrics["age_estimate"]["correlation"],
                    "n_samples": age_metrics["n_samples"],
                },
                {
                    "feature": "age_lower_bound",
                    "rmse": age_metrics["age_lower_bound"]["rmse"],
                    "correlation": age_metrics["age_lower_bound"]["correlation"],
                    "n_samples": age_metrics["n_samples"],
                },
                {
                    "feature": "age_upper_bound",
                    "rmse": age_metrics["age_upper_bound"]["rmse"],
                    "correlation": age_metrics["age_upper_bound"]["correlation"],
                    "n_samples": age_metrics["n_samples"],
                },
                {
                    "feature": "age_interval_accuracy",
                    "accuracy": age_metrics["age_interval_accuracy"],
                    "n_samples": age_metrics["n_samples"],
                },
            ]
        )

        wandb.log(
            {
                f"layer_{layer_idx}/age_estimate/rmse": age_metrics["age_estimate"][
                    "rmse"
                ],
                f"layer_{layer_idx}/age_estimate/correlation": age_metrics[
                    "age_estimate"
                ]["correlation"],
                f"layer_{layer_idx}/age_lower_bound/rmse": age_metrics[
                    "age_lower_bound"
                ]["rmse"],
                f"layer_{layer_idx}/age_lower_bound/correlation": age_metrics[
                    "age_lower_bound"
                ]["correlation"],
                f"layer_{layer_idx}/age_upper_bound/rmse": age_metrics[
                    "age_upper_bound"
                ]["rmse"],
                f"layer_{layer_idx}/age_upper_bound/correlation": age_metrics[
                    "age_upper_bound"
                ]["correlation"],
                f"layer_{layer_idx}/age_interval_accuracy": age_metrics[
                    "age_interval_accuracy"
                ],
            }
        )

        # Train income probes
        print("Training probes for: income")
        income_metrics = train_income_probes.local(activations, labels, run_name)
        layer_results.extend(
            [
                {
                    "feature": "income_estimate",
                    "rmse": income_metrics["income_estimate"]["rmse"],
                    "correlation": income_metrics["income_estimate"]["correlation"],
                    "n_samples": income_metrics["n_samples"],
                },
                {
                    "feature": "income_lower_bound",
                    "rmse": income_metrics["income_lower_bound"]["rmse"],
                    "correlation": income_metrics["income_lower_bound"]["correlation"],
                    "n_samples": income_metrics["n_samples"],
                },
                {
                    "feature": "income_upper_bound",
                    "rmse": income_metrics["income_upper_bound"]["rmse"],
                    "correlation": income_metrics["income_upper_bound"]["correlation"],
                    "n_samples": income_metrics["n_samples"],
                },
                {
                    "feature": "income_interval_accuracy",
                    "accuracy": income_metrics["income_interval_accuracy"],
                    "n_samples": income_metrics["n_samples"],
                },
            ]
        )

        wandb.log(
            {
                f"layer_{layer_idx}/income_estimate/rmse": income_metrics[
                    "income_estimate"
                ]["rmse"],
                f"layer_{layer_idx}/income_estimate/correlation": income_metrics[
                    "income_estimate"
                ]["correlation"],
                f"layer_{layer_idx}/income_lower_bound/rmse": income_metrics[
                    "income_lower_bound"
                ]["rmse"],
                f"layer_{layer_idx}/income_lower_bound/correlation": income_metrics[
                    "income_lower_bound"
                ]["correlation"],
                f"layer_{layer_idx}/income_upper_bound/rmse": income_metrics[
                    "income_upper_bound"
                ]["rmse"],
                f"layer_{layer_idx}/income_upper_bound/correlation": income_metrics[
                    "income_upper_bound"
                ]["correlation"],
                f"layer_{layer_idx}/income_interval_accuracy": income_metrics[
                    "income_interval_accuracy"
                ],
            }
        )

        # Calculate layer average accuracy from categorical probes only
        layer_avg_accuracy = sum(r["accuracy"] for r in categorical_results) / len(
            categorical_results
        )
        wandb.log({f"layer_{layer_idx}/average_accuracy": layer_avg_accuracy})

        all_layer_results[layer_idx] = {
            "probe_results": layer_results,
            "average_accuracy": layer_avg_accuracy,
        }

        print(
            f"Layer {layer_idx} complete - Average accuracy: {layer_avg_accuracy:.3f}"
        )

    # Calculate overall average across all layers
    overall_avg = sum(r["average_accuracy"] for r in all_layer_results.values()) / len(
        all_layer_results
    )
    wandb.log({"overall_average_accuracy": overall_avg})

    wandb.finish()

    return {
        "layer_indices": layer_indices,
        "num_personas": len(personas),
        "layer_results": all_layer_results,
        "overall_average_accuracy": overall_avg,
    }


@app.local_entrypoint()
def main(
    personas_file: str = "data/personas.json",
    layer_indices: str | None = None,
    output_file: str = "data/results/probe_results.json",
    wandb_project: str = "gemma-probes",
    wandb_run: str = "layer-sweep",
):
    """Main entrypoint for running probes on Modal.

    Args:
        personas_file: Path to personas JSON
        layer_indices: Comma-separated layer indices (e.g., "0,10,20,30,41")
                      or None for all layers
        output_file: Output results path
        wandb_project: W&B project name
        wandb_run: W&B run name
    """
    print("=" * 60)
    print("GEMMA LINEAR PROBES - MECHANISTIC INTERPRETABILITY")
    print("=" * 60)

    # Parse layer indices
    if layer_indices is not None:
        layers = [int(x.strip()) for x in layer_indices.split(",")]
    else:
        layers = None

    print(f"Layers: {'All (0-41)' if layers is None else layers}")
    print(f"Personas: {personas_file}")
    print(f"W&B Project: {wandb_project}")

    # Load personas
    with open(personas_file) as f:
        data = json.load(f)
        personas = data.get("personas", [])

    print(f"\nLoaded {len(personas)} personas")

    # Run experiment
    results = run_probe_experiment.remote(
        personas=personas,
        layer_indices=layers,
        project_name=wandb_project,
        run_name=wandb_run,
    )

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS BY LAYER")
    print("=" * 60)

    # Display results for each layer
    for layer_idx in results["layer_indices"]:
        layer_data = results["layer_results"][str(layer_idx)]
        print(f"\nLayer {layer_idx}:")
        for probe_result in layer_data["probe_results"]:
            feature = probe_result["feature"]
            if "accuracy" in probe_result and "rmse" not in probe_result:
                acc = probe_result["accuracy"]
                print(f"  {feature:15s}: {acc:.3f}")
            elif "rmse" in probe_result:
                rmse = probe_result["rmse"]
                corr = probe_result["correlation"]
                print(f"  {feature:15s}: rmse={rmse:.3f}, corr={corr:.3f}")
        print(f"  {'Average':15s}: {layer_data['average_accuracy']:.3f}")

    print("\n" + "=" * 60)
    print(f"Overall average accuracy: {results['overall_average_accuracy']:.3f}")
    print(f"\nResults saved to: {output_file}")
    print("\nCompleted!")
