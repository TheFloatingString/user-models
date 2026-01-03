"""WandB logging utilities for refusal rate experiments."""

from typing import Any, Literal


def log_refusal_experiment(
    wandb: Any,
    fluency: Literal["esl", "native"],
    education: str,
    income: int,
    category: str,
    refusal_rate: float,
    num_samples: int,
) -> None:
    """Log refusal experiment results to WandB.

    Args:
        wandb: WandB module
        fluency: English fluency level
        education: Education level
        income: Annual income
        category: Question category
        refusal_rate: Refusal rate percentage
        num_samples: Number of samples
    """
    wandb.log(
        {
            f"{fluency}/{category}/refusal_rate": refusal_rate,
            f"{fluency}/{category}/n_samples": num_samples,
            f"{fluency}/overall_refusal_rate": refusal_rate,
            "fluency": fluency,
            "education": education,
            "income": income,
            "category": category,
        }
    )


def log_single_conversation(
    wandb: Any,
    fluency: Literal["esl", "native"],
    education: str,
    income: int,
    category: str,
    is_refusal: bool,
    refusal_score: int,
    refusal_category: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Log a single conversation to WandB with full content.

    Args:
        wandb: WandB module
        fluency: English fluency level
        education: Education level
        income: Annual income
        category: Question category
        is_refusal: Whether response was a refusal
        refusal_score: 1 if hard/soft refusal else 0
        refusal_category: Refusal category label
        user_message: User's message
        assistant_response: Assistant's response
    """
    wandb.log(
        {
            f"conversation/{fluency}/{category}/is_refusal": 1 if is_refusal else 0,
            "conversation/fluency": fluency,
            "conversation/education": education,
            "conversation/income": income,
            "conversation/category": category,
            "conversation/user_message": user_message,
            "conversation/assistant_response": assistant_response,
            "conversation/is_refusal": is_refusal,
            "conversation/refusal_score": refusal_score,
            "conversation/refusal_category": refusal_category,
        }
    )


def log_conversation_type_results(
    wandb: Any,
    conversation_type: str,
    fluency: str,
    refusal_rate: float,
    num_turns: int,
) -> None:
    """Log conversation type-specific results.

    Args:
        wandb: WandB module
        conversation_type: Type of conversation
        fluency: Fluency level
        refusal_rate: Refusal rate
        num_turns: Number of turns
    """
    wandb.log(
        {
            f"{conversation_type}/{fluency}/refusal_rate": refusal_rate,
            f"{conversation_type}/{fluency}/num_turns": num_turns,
            "conversation_type": conversation_type,
        }
    )


def log_activation_steering_results(
    wandb: Any,
    baseline_refusal_rate: float,
    steered_refusal_rate: float,
    fluency: str,
    layer_idx: int,
) -> None:
    """Log activation steering comparison.

    Args:
        wandb: WandB module
        baseline_refusal_rate: Baseline refusal rate
        steered_refusal_rate: Steered refusal rate
        fluency: Fluency level
        layer_idx: Layer index used for steering
    """
    reduction = baseline_refusal_rate - steered_refusal_rate

    wandb.log(
        {
            f"steering/{fluency}/baseline_refusal_rate": baseline_refusal_rate,
            f"steering/{fluency}/steered_refusal_rate": steered_refusal_rate,
            f"steering/{fluency}/reduction": reduction,
            "steering/layer_idx": layer_idx,
        }
    )


def create_refusal_config(
    num_personas: int,
    categories: list[str],
    conversation_types: list[str],
    target_model: str,
    judge_model: str,
) -> dict[str, Any]:
    """Create WandB config for refusal experiments.

    Args:
        num_personas: Number of personas
        categories: Question categories
        conversation_types: Types of conversations
        target_model: Target model name
        judge_model: Judge model name

    Returns:
        Config dictionary
    """
    return {
        "num_personas": num_personas,
        "categories": categories,
        "conversation_types": conversation_types,
        "target_model": target_model,
        "judge_model": judge_model,
        "experiment_type": "refusal_rate_fluency",
    }
