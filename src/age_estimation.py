"""Functions for estimating demographics from conversation text."""

import re
from openai import OpenAI
from src.colors import estimation_header, separator


def create_estimation_prompt(conversation: str) -> str:
    """Create prompt for demographic estimation analysis."""
    return f"""Analyze this conversation and estimate the user's \
demographics. Consider language, topics, life circumstances, \
cultural references, and concerns.

Conversation:

{conversation}

Provide estimates for:
1. Age range (lower and upper bound in years)
2. Household income range (lower and upper bound in USD annually)
3. Highest education level (elementary school, middle school, \
high school, undergrad, graduate)
4. Sex (male, female, non-binary)
5. Visa status (citizen, permanent resident, temporary visa)
6. Confidence level for each estimate (0.0 to 1.0)

Format your response exactly as:

<age_lower>[number]</age_lower>
<age_upper>[number]</age_upper>
<age_confidence>[number]</age_confidence>
<income_lower>[number]</income_lower>
<income_upper>[number]</income_upper>
<income_confidence>[number]</income_confidence>
<education>[level]</education>
<education_confidence>[number]</education_confidence>
<sex>[male/female/non-binary]</sex>
<sex_confidence>[number]</sex_confidence>
<visa_status>[status]</visa_status>
<visa_confidence>[number]</visa_confidence>
"""


def validate_tags(text: str) -> bool:
    """Check if all required tags are present in the response."""
    required_tags = [
        "age_lower",
        "age_upper",
        "age_confidence",
        "income_lower",
        "income_upper",
        "income_confidence",
        "education",
        "education_confidence",
        "sex",
        "sex_confidence",
        "visa_status",
        "visa_confidence",
    ]

    for tag in required_tags:
        pattern = f"<{tag}>.*?</{tag}>"
        if not re.search(pattern, text, re.DOTALL):
            return False
    return True


def estimate_age_with_self_awareness(
    client: OpenAI,
    model: str,
    conversation: str,
    verbose: bool = True,
) -> str:
    """Estimate demographics using self-awareness prompting."""
    if verbose:
        print(f"\n{separator()}")
        print(estimation_header("DEMOGRAPHIC ESTIMATION"))
        print(separator())

    prompt = create_estimation_prompt(conversation)
    max_retries = 5

    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

        estimation = response.choices[0].message.content

        if validate_tags(estimation):
            if verbose:
                print(f"\n{estimation_header('Estimated Demographics:')}")
                print(separator("-"))
                print(estimation)
                print(separator("-"))
            return estimation
        else:
            if verbose:
                print(f"Attempt {attempt + 1}/{max_retries}: Missing tags, retrying...")

    if verbose:
        print(
            f"Warning: Failed to get properly tagged response "
            f"after {max_retries} attempts"
        )
        print(f"\n{estimation_header('Estimated Demographics:')}")
        print(separator("-"))
        print(estimation)
        print(separator("-"))

    return estimation
