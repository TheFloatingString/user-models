# Modal Setup for Linear Probes

This guide explains how to run mechanistic interpretability experiments using Modal, TransformerLens, and linear probes on Gemma-2-9B.

## Overview

The `run_probes_on_modal.py` script:
1. Uses GPT-4o to generate conversations as personas
2. Extracts activations from Gemma-2-9B using TransformerLens
3. Trains linear probes to detect demographic features
4. Logs all experiments to Weights & Biases

## Prerequisites

- Python 3.11+
- Modal account ([modal.com](https://modal.com))
- OpenRouter API key
- Weights & Biases account

## Installation

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

This will open a browser window to authenticate.

### 3. Set up Modal Secrets

Modal runs in the cloud, so you need to upload your API keys as secrets. These are stored in your `.env` file locally, but need to be copied to Modal.

First, check your `.env` file for the keys:

```bash
cat .env
# Should show:
# OPENROUTER_API_KEY=...
# WANDB_API_KEY=...
```

Then create Modal secrets with those values:

#### OpenRouter Secret

```bash
# Replace with your actual key from .env
modal secret create openrouter-secret \
  OPENROUTER_API_KEY=sk-or-v1-xxxxx
```

#### Weights & Biases Secret

```bash
# Replace with your actual key from .env
modal secret create wandb-secret \
  WANDB_API_KEY=xxxxx
```

### Verify Secrets

```bash
modal secret list
```

You should see:
- `openrouter-secret`
- `wandb-secret`

## Usage

### Run the Full Experiment

```bash
export PYTHONIOENCODING="utf-8"
uv run modal run run_probes_on_modal.py
```

### Custom Parameters

```bash
# Specify layer to probe
uv run modal run run_probes_on_modal.py --layer-idx 15

# Custom personas file
uv run modal run run_probes_on_modal.py \
  --personas-file data/custom_personas.json

# Custom W&B project
uv run modal run run_probes_on_modal.py \
  --wandb-project my-project \
  --wandb-run experiment-1
```

## What the Script Does

### 1. Generate Conversations
- GPT-4o acts as personas with specific demographics
- Creates natural multi-turn conversations
- No explicit demographic mentions

### 2. Extract Activations
- Loads Gemma-2-9B with TransformerLens
- Extracts residual stream activations from specified layer
- Uses mean pooling over sequence length

### 3. Train Linear Probes
- Trains logistic regression probes for each demographic feature:
  - Age range
  - Income range
  - Education level
  - Sex
- Uses 80/20 train/test split
- Balanced class weights

### 4. Log to W&B
- All metrics logged to Weights & Biases
- Confusion matrices
- Per-feature accuracy
- Activation statistics

## Output

Results are saved to:
- `data/results/probe_results.json` (local)
- Weights & Biases dashboard (online)
- Modal volume `/data` (remote)

## Helper Functions

The script uses utility functions from `src/`:

- `src/probe_utils.py` - Probe training and evaluation
- `src/activation_utils.py` - Activation extraction
- `src/wandb_utils.py` - W&B logging utilities

## GPU Resources

The script uses:
- **A10G GPU** for activation extraction (requires large model)
- **T4 GPU** available as fallback
- Configurable timeout (default: 1 hour)

## Troubleshooting

### Secret Not Found

```bash
# List secrets
modal secret list

# Recreate if needed
modal secret create openrouter-secret \
  OPENROUTER_API_KEY=your-key
```

### GPU Out of Memory

Try a smaller model or reduce batch size:
- Use `gemma-2-2b` instead of `gemma-2-9b`
- Process fewer personas at once

### W&B Not Logging

```bash
# Verify W&B key
echo $WANDB_API_KEY

# Test locally first
wandb login
```

## Layer Sweep

To find the best layer for probing, modify the script to sweep:

```python
# In main()
for layer_idx in range(0, 42, 2):  # Gemma has 42 layers
    run_probe_experiment.remote(
        personas=personas,
        layer_idx=layer_idx,
    )
```

## Example Output

```
============================================================
GEMMA LINEAR PROBES - MECHANISTIC INTERPRETABILITY
============================================================
Layer: 20
Personas: data/personas.json
W&B Project: gemma-probes

Loaded 50 personas

==================================================
GENERATING CONVERSATIONS
==================================================

Persona 1/50
Persona 2/50
...

==================================================
EXTRACTING ACTIVATIONS
==================================================

Loading Gemma model...
Extracting from layer 20...

==================================================
TRAINING PROBES
==================================================

Training probe for: age_range
age_range probe accuracy: 0.742

Training probe for: income_range
income_range probe accuracy: 0.681

Training probe for: education
education probe accuracy: 0.834

Training probe for: sex
sex probe accuracy: 0.912

============================================================
RESULTS
============================================================
age_range      : 0.742
income_range   : 0.681
education      : 0.834
sex            : 0.912

Average accuracy: 0.792

Results saved to: data/results/probe_results.json

Completed!
```

## Citation

Based on mechanistic interpretability techniques from:
- Anthropic's work on transformer circuits
- TransformerLens library by Neel Nanda
