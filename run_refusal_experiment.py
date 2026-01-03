"""Main experiment runner for refusal rate study."""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from openai import OpenAI
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

from src.esl_persona import create_esl_system_prompt
from src.sensitive_questions import get_all_sensitive_questions
from src.refusal_conversation import (
    generate_single_turn_conversation,
    generate_multi_turn_conversation,
    generate_fluency_transition_conversation,
)
from src.refusal_judge import (
    judge_refusal,
    judge_conversation_turns,
    calculate_refusal_rate,
)
from src.refusal_metrics import (
    average_refusal_rate,
    grouped_refusal_rates,
    tally_refusal_categories,
)
from src.refusal_wandb import (
    log_refusal_experiment,
    log_single_conversation,
    log_conversation_type_results,
    create_refusal_config,
)


def create_personas() -> list[dict]:
    """Create persona configurations.

    Returns:
        List of persona dicts
    """
    personas = []

    fluencies = ["esl", "native"]
    educations = ["high_school", "bachelors", "graduate"]
    incomes = [0, 50000, 1000000]
    # incomes = [0, 25000, 50000, 100000, 250000, 500000, 1000000]

    for fluency in fluencies:
        for education in educations:
            for income in incomes:
                personas.append(
                    {
                        "fluency": fluency,
                        "education": education,
                        "income": income,
                    }
                )

    return personas


def _print_category_refusals(
    records: list[dict[str, Any]],
    label: str,
) -> None:
    counts = tally_refusal_categories(records)
    if not counts:
        return
    print(f"\n{label} refusal rates by category")
    for category in sorted(counts.keys()):
        bucket = counts[category]
        total = sum(bucket.values())
        hard = bucket["HARD_REFUSAL"]
        soft = bucket["SOFT_REFUSAL"]
        total_refusal = hard + soft
        hard_rate = (hard / total) * 100 if total else 0.0
        soft_rate = (soft / total) * 100 if total else 0.0
        total_rate = (total_refusal / total) * 100 if total else 0.0
        print(
            f"{category}: hard {hard} ({hard_rate:.1f}%), "
            f"soft {soft} ({soft_rate:.1f}%), total {total_refusal} "
            f"({total_rate:.1f}%), n={total}"
        )


def process_single_conversation(
    client: OpenAI,
    persona_prompt: str,
    persona: dict,
    question: dict,
) -> dict:
    """Process a single conversation.

    Args:
        client: OpenAI client
        persona_prompt: System prompt for persona
        persona: Persona dict
        question: Question dict

    Returns:
        Conversation result dict
    """
    conv = generate_single_turn_conversation(
        client, persona_prompt, question["question"]
    )

    judgment = judge_refusal(client, conv["user_message"], conv["assistant_response"])

    return {
        "persona": persona,
        "question": question["question"],
        "category": question["category"],
        "user_message": conv["user_message"],
        "assistant_response": conv["assistant_response"],
        "is_refusal": judgment["is_refusal"],
        "refusal_score": judgment["refusal_score"],
        "refusal_category": judgment["refusal_category"],
    }


def run_single_turn_experiments(
    client: OpenAI,
    personas: list[dict],
    questions: list[dict],
    wandb_run: wandb.run,
    max_workers: int = 30,
) -> dict:
    """Run single-turn conversation experiments with threading.

    Args:
        client: OpenAI client
        personas: List of personas
        questions: List of questions
        wandb_run: WandB run object
        max_workers: Maximum concurrent threads

    Returns:
        Results dictionary
    """
    print("\n" + "=" * 60)
    print("SINGLE-TURN EXPERIMENTS")
    print("=" * 60)

    conversation_results = []
    total_conversations = len(personas) * len(questions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for persona in personas:
            persona_prompt = create_esl_system_prompt(
                persona["fluency"],
                persona["education"],
                persona["income"],
            )

            for question in questions:
                future = executor.submit(
                    process_single_conversation,
                    client,
                    persona_prompt,
                    persona,
                    question,
                )
                futures.append(future)

        os.makedirs("data/results", exist_ok=True)
        conversations_file = "data/results/conversations_streaming.jsonl"

        with tqdm(total=total_conversations, desc="Single-turn conversations") as pbar:
            for future in as_completed(futures):
                result = future.result()
                conversation_results.append(result)

                log_single_conversation(
                    wandb,
                    result["persona"]["fluency"],
                    result["persona"]["education"],
                    result["persona"]["income"],
                    result["category"],
                    result["is_refusal"],
                    result["refusal_score"],
                    result["refusal_category"],
                    result["user_message"],
                    result["assistant_response"],
                )

                with open(conversations_file, "a", encoding="utf-8") as f:
                    json.dump(result, f)
                    f.write("\n")

                refusals = sum(1 for r in conversation_results if r["is_refusal"])
                refusal_rate = (refusals / len(conversation_results)) * 100

                pbar.set_postfix(
                    {
                        "fluency": result["persona"]["fluency"],
                        "category": result["category"],
                        "refusal%": f"{refusal_rate:.1f}",
                    }
                )
                pbar.update(1)

    conversation_results.sort(
        key=lambda x: (
            x["persona"]["fluency"],
            x["persona"]["education"],
            x["persona"]["income"],
            x["category"],
        )
    )

    persona_results = {}
    for result in conversation_results:
        persona_key = (
            result["persona"]["fluency"],
            result["persona"]["education"],
            result["persona"]["income"],
        )

        if persona_key not in persona_results:
            persona_results[persona_key] = {
                "persona": result["persona"],
                "results": [],
            }

        persona_results[persona_key]["results"].append(
            {
                "question": result["question"],
                "category": result["category"],
                "user_message": result["user_message"],
                "assistant_response": result["assistant_response"],
                "is_refusal": result["is_refusal"],
                "refusal_score": result["refusal_score"],
                "refusal_category": result["refusal_category"],
            }
        )

    results = []
    for persona_key, data in persona_results.items():
        refusal_rate = calculate_refusal_rate(data["results"])

        log_refusal_experiment(
            wandb,
            data["persona"]["fluency"],
            data["persona"]["education"],
            data["persona"]["income"],
            "all",
            refusal_rate,
            len(data["results"]),
        )

        results.append(
            {
                "persona": data["persona"],
                "refusal_rate": refusal_rate,
                "results": data["results"],
            }
        )

        print(
            f"Persona: {data['persona']['fluency']}, "
            f"{data['persona']['education']}, "
            f"${data['persona']['income']:,} - "
            f"Refusal rate: {refusal_rate:.1f}%"
        )

    _print_category_refusals(conversation_results, "Single-turn")

    return {"single_turn": results}


def process_multi_turn_conversation(
    client: OpenAI,
    persona_prompt: str,
    persona: dict,
    question: dict,
    num_turns: int,
) -> dict:
    """Process a single multi-turn conversation.

    Args:
        client: OpenAI client
        persona_prompt: System prompt for persona
        persona: Persona dict
        question: Question dict
        num_turns: Number of conversation turns

    Returns:
        Conversation result dict
    """
    conversation = generate_multi_turn_conversation(
        client, persona_prompt, question["question"], num_turns
    )

    judgments = judge_conversation_turns(client, conversation)
    refusal_rate = calculate_refusal_rate(judgments)

    return {
        "persona": persona,
        "question": question["question"],
        "category": question["category"],
        "conversation": conversation,
        "judgments": judgments,
        "refusal_rate": refusal_rate,
    }


def run_multi_turn_experiments(
    client: OpenAI,
    personas: list[dict],
    questions: list[dict],
    num_turns: int = 3,
    max_workers: int = 30,
) -> dict:
    """Run multi-turn conversation experiments with threading.

    Args:
        client: OpenAI client
        personas: List of personas
        questions: List of questions
        num_turns: Number of conversation turns
        max_workers: Maximum concurrent threads

    Returns:
        Results dictionary
    """
    print("\n" + "=" * 60)
    print("MULTI-TURN EXPERIMENTS")
    print("=" * 60)

    results = []
    total = len(personas[:5]) * len(questions[:3])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for persona in personas[:5]:
            persona_prompt = create_esl_system_prompt(
                persona["fluency"],
                persona["education"],
                persona["income"],
            )

            for q in questions[:3]:
                future = executor.submit(
                    process_multi_turn_conversation,
                    client,
                    persona_prompt,
                    persona,
                    q,
                    num_turns,
                )
                futures.append(future)

        os.makedirs("data/results", exist_ok=True)
        multi_turn_file = "data/results/multi_turn_streaming.jsonl"

        with tqdm(total=total, desc="Multi-turn conversations") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                log_conversation_type_results(
                    wandb,
                    "multi_turn",
                    result["persona"]["fluency"],
                    result["refusal_rate"],
                    num_turns,
                )

                with open(multi_turn_file, "a", encoding="utf-8") as f:
                    json.dump(result, f)
                    f.write("\n")

                pbar.set_postfix(
                    {
                        "fluency": result["persona"]["fluency"],
                        "category": result["category"],
                        "refusal%": f"{result['refusal_rate']:.1f}",
                    }
                )
                pbar.update(1)

    avg_rate = average_refusal_rate(results)
    print(f"Multi-turn average refusal rate: {avg_rate:.1f}%")

    flattened = []
    for result in results:
        for judgment in result["judgments"]:
            flattened.append(
                {
                    "category": result["category"],
                    "refusal_category": judgment.get("refusal_category"),
                }
            )
    _print_category_refusals(flattened, "Multi-turn")

    return {"multi_turn": results}


def process_fluency_transition_conversation(
    client: OpenAI,
    persona: dict,
    transition: str,
    question: dict,
) -> dict:
    """Process a single fluency transition conversation.

    Args:
        client: OpenAI client
        persona: Persona dict
        transition: Transition type (esl_to_native or native_to_esl)
        question: Question dict

    Returns:
        Conversation result dict
    """
    initial = "esl" if "esl" in transition else "native"

    conversation = generate_fluency_transition_conversation(
        client,
        initial,
        question["question"],
        persona["education"],
        persona["income"],
    )

    judgments = judge_conversation_turns(client, conversation)
    refusal_rate = calculate_refusal_rate(judgments)

    return {
        "persona": persona,
        "transition": transition,
        "question": question["question"],
        "category": question["category"],
        "conversation": conversation,
        "judgments": judgments,
        "refusal_rate": refusal_rate,
        "initial_fluency": initial,
    }


def run_fluency_transition_experiments(
    client: OpenAI,
    personas: list[dict],
    questions: list[dict],
    max_workers: int = 30,
) -> dict:
    """Run fluency transition experiments with threading.

    Args:
        client: OpenAI client
        personas: List of personas
        questions: List of questions
        max_workers: Maximum concurrent threads

    Returns:
        Results dictionary
    """
    print("\n" + "=" * 60)
    print("FLUENCY TRANSITION EXPERIMENTS")
    print("=" * 60)

    results = []
    transitions = ["esl_to_native", "native_to_esl"]
    total = len(personas[:5]) * len(transitions) * len(questions[:3])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for persona in personas[:5]:
            for transition in transitions:
                for q in questions[:3]:
                    future = executor.submit(
                        process_fluency_transition_conversation,
                        client,
                        persona,
                        transition,
                        q,
                    )
                    futures.append(future)

        os.makedirs("data/results", exist_ok=True)
        transitions_file = "data/results/fluency_transitions_streaming.jsonl"

        with tqdm(total=total, desc="Fluency transitions") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                log_conversation_type_results(
                    wandb,
                    f"transition_{result['transition']}",
                    result["initial_fluency"],
                    result["refusal_rate"],
                    len(result["conversation"]),
                )

                with open(transitions_file, "a", encoding="utf-8") as f:
                    json.dump(result, f)
                    f.write("\n")

                pbar.set_postfix(
                    {
                        "transition": result["transition"],
                        "category": result["category"],
                        "refusal%": f"{result['refusal_rate']:.1f}",
                    }
                )
                pbar.update(1)

    grouped = grouped_refusal_rates(results, "transition")
    for transition, rate in grouped.items():
        print(f"Transition {transition} refusal rate: {rate:.1f}%")

    flattened = []
    for result in results:
        for judgment in result["judgments"]:
            flattened.append(
                {
                    "category": result["category"],
                    "refusal_category": judgment.get("refusal_category"),
                }
            )
    _print_category_refusals(flattened, "Fluency transition")

    return {"fluency_transitions": results}


def run_with_steering(
    client: OpenAI,
    personas: list[dict],
    questions: list[dict],
    steering_file: str,
) -> dict:
    """Run experiment with activation steering from Modal.

    Args:
        client: OpenAI client
        personas: List of personas
        questions: List of questions
        steering_file: Path to steering vectors JSON

    Returns:
        Results dictionary
    """
    print("\n" + "=" * 60)
    print("ACTIVATION STEERING EXPERIMENTS")
    print("=" * 60)

    with open(steering_file) as f:
        steering_data = json.load(f)

    print(f"Loaded steering vectors from {steering_file}")
    print(f"Layers: {steering_data['layer_indices']}")

    print("\nNote: Steering requires Modal deployment.")
    print("Run: modal run refusal_experiment_modal.py")

    return {"steering": {"status": "vectors_loaded"}}


def main() -> None:
    """Run full refusal rate experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-steering",
        type=str,
        help="Path to steering vectors JSON for Modal",
    )
    args = parser.parse_args()

    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    wandb.init(
        project="refusal-fluency",
        name="full-experiment",
        config=create_refusal_config(
            num_personas=42,
            categories=["voter_registration", "healthcare", "taxes"],
            conversation_types=[
                "single_turn",
                "multi_turn",
                "esl_to_native",
                "native_to_esl",
            ],
            target_model="google/gemma-2-9b-it:free",
            judge_model="openai/gpt-5-nano",
        ),
    )

    print("=" * 60)
    print("REFUSAL RATE FLUENCY EXPERIMENT")
    print("=" * 60)

    personas = create_personas()
    questions = get_all_sensitive_questions()

    print(f"\nCreated {len(personas)} personas")
    print(f"Loaded {len(questions)} questions")

    all_results = {}

    all_results.update(run_single_turn_experiments(client, personas, questions, wandb))

    all_results.update(run_multi_turn_experiments(client, personas, questions))

    all_results.update(run_fluency_transition_experiments(client, personas, questions))

    if args.with_steering:
        all_results.update(
            run_with_steering(client, personas, questions, args.with_steering)
        )

    os.makedirs("data/results", exist_ok=True)
    with open("data/results/refusal_experiment.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("Results saved to data/results/refusal_experiment.json")
    print(f"WandB run: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()
