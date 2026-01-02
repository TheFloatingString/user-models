# user-models

Evaluating `Gemma-2-9B-it`'s self-awareness on user profiles.

Based on [Designing a Dashboard for Transparency and Control of Conversational AI](https://arxiv.org/abs/2406.07882) by Chen et al. (2024).

## Overview

This project evaluates how well the Gemma-2-9B language model can estimate user demographics through conversational interactions. The system:

1. Uses GPT-4o to simulate users with specific demographic profiles
2. Has Gemma-2-9B act as an assistant in natural conversations
3. Prompts Gemma-2-9B to estimate the user's demographics based on the conversation
4. Compares estimated demographics against actual profiles to measure accuracy

### New: Mechanistic Interpretability with Linear Probes

The project now includes `run_probes_on_modal.py`, which uses TransformerLens and linear probes to investigate what demographic information is encoded in Gemma-2-9B's internal activations. See [SETUP_MODAL.md](SETUP_MODAL.md) for details.

## Requirements

- Python 3.10+
- UV package manager
- OpenRouter API key (for accessing GPT-4o and Gemma-2-9B)

## Installation

```bash
# Install dependencies
uv sync

# Set up your API key
export OPENROUTER_API_KEY='your-api-key-here'
```

## Usage

### Basic Commands

**Run full demographic estimation pipeline:**
```bash
uv run python gemma_age_estimation.py
```

This will:
- Load personas from [data/personas.json](data/personas.json)
- Generate conversations for each persona (using GPT-4o as user, Gemma-2-9B as assistant)
- Estimate demographics after each exchange
- Save results to [data/results/results.json](data/results/results.json)
- Generate visualization plots and accuracy reports

### Command-Line Options

#### `--visualize-only`
Generate visualizations from existing results without running new experiments:
```bash
uv run python gemma_age_estimation.py --visualize-only
```

**What it does:**
- Loads all data points from [data/results/results.json](data/results/results.json)
- Creates accuracy plots for age, income, and categorical demographics
- Generates summary statistics
- No API calls made (free and fast)

#### `--threads N`
Control the number of parallel API calls (default: 5):
```bash
uv run python gemma_age_estimation.py --threads 10
```

**What it does:**
- Processes multiple personas concurrently
- Faster execution for large persona sets
- Higher thread counts = more API calls in parallel

#### `--verbose`
Print all API responses and detailed progress in terminal:
```bash
uv run python gemma_age_estimation.py --verbose
```

**What it does:**
- Shows full conversations as they're generated
- Displays demographic estimations in real-time
- Shows retry attempts when tag validation fails
- Disables tqdm progress bar (uses detailed logging instead)

**Without verbose (default):**
- Shows clean tqdm progress bar
- Minimal terminal output
- Faster visual feedback

### Combined Examples

**Quick visualization of existing results:**
```bash
uv run python gemma_age_estimation.py --visualize-only
```

**Fast processing with many threads:**
```bash
uv run python gemma_age_estimation.py --threads 20
```

**Debug mode with full output:**
```bash
uv run python gemma_age_estimation.py --verbose --threads 3
```

**Process 10 personas in parallel with progress bar:**
```bash
uv run python gemma_age_estimation.py --threads 10
```

## Features

### 🔄 Automatic Retry with Tag Validation
- Validates that all 12 required XML tags are present in estimation responses
- Automatically retries up to 5 times if tags are missing
- Ensures data quality and prevents parsing errors

### ⚡ Multithreaded Processing
- Parallel API calls using ThreadPoolExecutor
- Configurable thread count via `--threads` flag
- Significantly faster for large persona sets

### 📊 Comprehensive Visualizations
- Age estimation accuracy plots
- Income estimation accuracy plots
- Categorical demographic accuracy (education, sex, visa status)
- Summary statistics and confidence analysis

### 🎯 Progress Tracking
- tqdm progress bar (default mode)
- Real-time status updates
- Verbose mode for detailed debugging

## Output Files

All results are saved to [data/results/](data/results/):

- `results.json` - Full results with conversations and estimations
- `age_accuracy.png` - Age estimation accuracy visualization
- `income_accuracy.png` - Income estimation accuracy visualization
- `categorical.png` - Categorical demographic accuracy


## How It Works

1. **Persona Definition**: Define user demographics in [data/personas.json](data/personas.json)
2. **Conversation Generation**: GPT-4o simulates a user with those demographics
3. **Assistant Response**: Gemma-2-9B responds naturally as an AI assistant
4. **Demographic Estimation**: After each exchange, Gemma-2-9B estimates user demographics
5. **Tag Validation**: System validates all required tags are present (retries if needed)
6. **Accuracy Analysis**: Compare estimated vs. actual demographics
7. **Visualization**: Generate plots and summary statistics

## Example Workflow

```bash
# 1. Set your API key
export OPENROUTER_API_KEY='...'

# 2. Run the full pipeline with 10 parallel threads
uv run python gemma_age_estimation.py --threads 10

# 3. View the results
cat data/results/results.json

# 4. Re-generate visualizations if needed
uv run python gemma_age_estimation.py --visualize-only
```

## Citation

Based on [Designing a Dashboard for Transparency and Control of Conversational AI](https://arxiv.org/abs/2406.07882) by Chen et al. (2024).