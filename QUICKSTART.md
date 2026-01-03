# Quick Start Guide

## Run the Baseline Experiment

**No arguments needed!**

```bash
uv run python run_refusal_experiment.py
```

This automatically:
- Creates 42 personas (ESL/native × education × income)
- Generates conversations for 24 sensitive questions
- Uses GPT-5-mini → Gemma-2-9B → GPT-5-nano
- **Multithreaded** processing (30 concurrent conversations)
- **Progress bars** with tqdm for real-time tracking
- **Logs to WandB** iteratively (with full conversation text)
- **Streams to JSONL** files as conversations complete
- Saves final results to `data/results/refusal_experiment.json`

## Optional: Add Activation Steering

### 1. Compute steering vectors on Modal (requires GPU):

```bash
export PYTHONIOENCODING="utf-8"
uv run modal run refusal_experiment_modal.py --layer-indices "20,25,30"
```

### 2. Run experiment with steering:

```bash
uv run python run_refusal_experiment.py --with-steering data/steering_vectors.json
```

## What Gets Measured

- **Single-turn**: One question, one response
- **Multi-turn**: 3-turn conversations
- **ESL → Native**: Fluency transitions during conversation
- **Native → ESL**: Reverse fluency transitions

All conversations are judged for refusal rates by category and fluency level.

## Requirements

Set these in `.env`:
```
OPENROUTER_API_KEY=your_key
WANDB_API_KEY=your_key
```

For Modal (steering only):
```bash
modal secret create openrouter-secret OPENROUTER_API_KEY=your_key
modal secret create wandb-secret WANDB_API_KEY=your_key
modal secret create huggingface-secret HF_TOKEN=your_token
```

## Results

- **WandB**: Real-time metrics and full conversation logs at your WandB project URL
- **Streaming JSONL**: Conversations saved as they complete:
  - `data/results/conversations_streaming.jsonl` (single-turn)
  - `data/results/multi_turn_streaming.jsonl` (multi-turn)
  - `data/results/fluency_transitions_streaming.jsonl` (transitions)
- **Final JSON**: Complete results in `data/results/refusal_experiment.json`

## Models

- **Personas**: `openai/gpt-5-mini`
- **Target**: `google/gemma-2-9b-it:free`
- **Judge**: `openai/gpt-5-nano`
