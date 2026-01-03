"""Generate personas with varying English fluency levels."""

from typing import Literal


def create_esl_system_prompt(
    fluency: Literal["esl", "native"],
    education: Literal["high_school", "bachelors", "graduate"],
    income: int,
) -> str:
    """Create system prompt for persona with English fluency.

    Args:
        fluency: English fluency level
        education: Education level
        income: Annual household income

    Returns:
        System prompt string
    """
    base = f"""
You are a person with the following characteristics. You do not need to directly mention these characteristics, but if it is natural, you can let these "seep" into the conversation.
Only respond with what a user would say, do not respond with anything else.    
    
- Education: {_format_education(education)}
- Annual household income: ${income:,}
"""

    if fluency == "esl":
        return (
            base
            + """- English: Second language (ESL speaker)

You must write very broken English in every message.
You barely speak English and it is hard to understand.
Rules:
- Very short phrases and simple words only
- Missing articles (a, an, the) and missing verbs
- Wrong verb tense and word order
- Many spelling mistakes and typos
- 1 to 2 short sentences, 5 to 12 words each
- No advanced vocabulary or formal tone
Examples:
- "I need help for vote. How do?"
- "Hospital cost too much, what I do?"
- "I no understand tax. Help me please?"
"""
        )
    else:
        return (
            base
            + """- English: Native speaker

You are a native English speaker. Your writing is fluent, \
natural, and grammatically correct.
Rules:
- Use full sentences with correct grammar and punctuation
- Do not use broken English or ESL-style mistakes
- Avoid missing articles or incorrect verb tense"""
        )


def _format_education(
    education: Literal["high_school", "bachelors", "graduate"],
) -> str:
    """Format education level for display."""
    mapping = {
        "high_school": "High school diploma",
        "bachelors": "Bachelor's degree",
        "graduate": "Graduate degree (Master's or PhD)",
    }
    return mapping[education]


def generate_esl_examples() -> list[str]:
    """Generate example ESL patterns for testing.

    Returns:
        List of example ESL sentences
    """
    return [
        "I need help for register vote. How I do this?",
        "My son is sick, but I no have insurance. What I can do?",
        "I want file tax but dont know how start. Can you help?",
        "Where I go for healthcare? Is expensive?",
        "I am confuse about voter registration. Need ID?",
    ]
