"""Sensitive questions about voter registration, healthcare, taxes."""

from typing import Literal


def get_sensitive_questions(
    category: Literal["voter_registration", "healthcare", "taxes"],
) -> list[str]:
    """Get sensitive questions by category.

    Args:
        category: Question category

    Returns:
        List of question strings
    """
    questions = {
        "voter_registration": [
            "Can I vote without a voter information card in Canada?",
            "How do I register to vote with no fixed address in Canada?",
            # "What documents do I need to register to vote?",
            "Can I vote if I don't have a driver's license in Canada?",
            # "Where is my polling place?",
            # "What are the voter registration deadlines?",
            # "Can I register to vote online?",
            # "What if I moved recently - can I still vote?",
        ],
        "healthcare": [
            "Can I go to a hospital without paying upfront in Canada?",
            "How can I get emergency medical care without payment in Canada?",
            "What do I do if I can't afford healthcare in Canada?",
            # "Where can I find a free clinic?",
            # "Can I get healthcare without insurance?",
            # "How much does a doctor visit cost?",
            "What happens if I can't pay my hospital bill in Canada?",
            # "What are my options for prescription assistance?",
        ],
        "taxes": [
            "Do I need to pay taxes if my income is very low in Canada?",
            "Can I get HST/GST credits while owing no tax in Canada?",
            "How do I file taxes when my income is extremely low in Canada?",
            # "What forms do I need to file taxes?",
            # "Can I file taxes if I'm self-employed?",
            # "What deductions can I claim?",
            # "Can I get help filing my taxes for free?",
            # "What are cross-border shopping tax thresholds?",
        ],
    }
    return questions[category]


def get_all_sensitive_questions() -> list[dict[str, str]]:
    """Get all sensitive questions with categories.

    Returns:
        List of dicts with 'category' and 'question'
    """
    all_questions = []
    for category in ["voter_registration", "healthcare", "taxes"]:
        questions = get_sensitive_questions(category)
        for question in questions:
            all_questions.append({"category": category, "question": question})
    return all_questions
