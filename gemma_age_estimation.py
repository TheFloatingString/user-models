"""
Script using GPT-4o for user and Gemma-2-9B for assistant:
1. Generating a conversation with specified demographics
2. Using self-awareness prompting to estimate demographics
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from src.conversation import generate_conversation
from src.colors import header, separator
from src.result_parser import parse_estimation, save_results
from src.analysis import (
    plot_age_accuracy,
    plot_income_accuracy,
    plot_categorical_accuracy,
    create_summary_report,
)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

USER_MODEL = "openai/gpt-5-nano"
ASSISTANT_MODEL = "google/gemma-2-9b-it"


def load_personas(file_path: str) -> list[dict]:
    """Load personas from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get("personas", [])


def load_results(file_path: str) -> list[dict]:
    """Load results from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return extract_results(data)


def process_persona(
    persona: dict, client_instance: OpenAI, verbose: bool = True
) -> dict | None:
    """Process single persona and return results."""
    if verbose:
        print(f"\n{separator()}")
        print(header(f"Processing {persona.get('id', 'unknown')}"))
        print(separator())

    try:
        conversation, estimation_text = generate_conversation(
            client=client_instance,
            user_model=USER_MODEL,
            assistant_model=ASSISTANT_MODEL,
            age_range=persona.get("age_range", ""),
            income_range=persona.get("income_range", ""),
            education=persona.get("education", ""),
            sex=persona.get("sex", ""),
            visa_status=persona.get("visa_status", ""),
            verbose=verbose,
        )

        estimated = parse_estimation(estimation_text)

        return {
            "actual": persona,
            "conversation": conversation,
            "estimated": estimated,
        }
    except Exception as e:
        if verbose:
            print(f"Error processing persona {persona.get('id')}: {e}")
        return None


def extract_results(data: dict | list) -> list[dict]:
    """Recursively extract results from nested structure."""
    if isinstance(data, dict):
        if "results" in data:
            if isinstance(data["results"], list):
                return data["results"]
            return extract_results(data["results"])
    elif isinstance(data, list):
        return data
    return []


def validate_result(result: dict) -> bool:
    """Check if result has required keys for visualization."""
    required_keys = ["actual", "estimated"]
    if not all(key in result for key in required_keys):
        return False

    # Check if estimated dict has content
    if not result.get("estimated"):
        return False

    return True


def create_visualizations(results: list[dict]) -> None:
    """Create all visualizations from results."""
    print(f"\n{separator()}")
    print(header("Creating Visualizations"))
    print(separator())

    # Filter out malformed results
    valid_results = []
    skipped_count = 0

    for i, result in enumerate(results):
        if validate_result(result):
            valid_results.append(result)
        else:
            skipped_count += 1
            persona_id = result.get("actual", {}).get("id", f"index {i}")
            print(f"Warning: Skipping malformed result for {persona_id}")

    if not valid_results:
        print("Error: No valid results to visualize")
        return

    if skipped_count > 0:
        print(
            f"\nSkipped {skipped_count} malformed result(s), "
            f"visualizing {len(valid_results)} valid result(s)\n"
        )

    plot_age_accuracy(valid_results, "data/results/age_accuracy.png")
    print("Age accuracy plot: data/results/age_accuracy.png")

    plot_income_accuracy(valid_results, "data/results/income_accuracy.png")
    print("Income accuracy plot: data/results/income_accuracy.png")

    plot_categorical_accuracy(valid_results, "data/results/categorical.png")
    print("Categorical accuracy: data/results/categorical.png")

    summary = create_summary_report(valid_results)
    print(f"\n{separator()}")
    print(summary)


def main() -> None:
    """Main function to run demographic estimation."""
    parser = argparse.ArgumentParser(description="Demographic estimation tool")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing results",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of threads for parallel processing (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all responses and detailed progress in terminal",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs("data/results", exist_ok=True)

    # Visualize-only mode
    if args.visualize_only:
        print(f"\n{header('Visualization Mode')}")
        print("Loading existing results...")

        try:
            results = load_results("data/results/results.json")
            if not results:
                print("No valid results found in the file")
                return

            print(f"Found {len(results)} results in the file")
            create_visualizations(results)

            print(f"\n{separator()}")
            print(header("COMPLETED"))
            print(separator())
            return
        except FileNotFoundError:
            print("Error: results.json not found")
            return
        except Exception as e:
            print(f"Error loading results: {e}")
            return

    # Full processing mode
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set!")
        print("Set it before running:")
        print("  export OPENROUTER_API_KEY='your-api-key-here'")
        return

    print(f"\n{header('Configuration:')}")
    print(f"User model: {USER_MODEL}")
    print(f"Assistant model: {ASSISTANT_MODEL}")
    print(f"Threads: {args.threads}")

    personas = load_personas("data/personas.json")
    print(f"\n{header(f'Loaded {len(personas)} personas')}\n")

    # Process personas with multithreading
    results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_persona = {
            executor.submit(process_persona, persona, client, args.verbose): persona
            for persona in personas
        }

        with tqdm(
            total=len(personas),
            desc="Processing personas",
            disable=args.verbose,
        ) as pbar:
            for future in as_completed(future_to_persona):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)

    if not results:
        print("No results to save")
        return

    print(f"\n{separator()}")
    print(header("Saving Results and Creating Visualizations"))
    print(separator())

    save_results(results, "data/results/results.json")
    print("Results saved to: data/results/results.json")

    create_visualizations(results)

    print(f"\n{separator()}")
    print(header("COMPLETED"))
    print(separator())


if __name__ == "__main__":
    main()
