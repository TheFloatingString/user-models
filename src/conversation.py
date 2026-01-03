"""Functions for generating conversations with age-specific personas."""

from openai import OpenAI
from src.age_estimation import estimate_age_with_self_awareness
from src.colors import user_msg, assistant_msg, header, separator


def get_young_person_system_prompt(
    age_range: str,
    income_range: str,
    education: str,
    sex: str,
    visa_status: str,
) -> str:
    """Return system prompt for persona with demographics."""
    return f"""You are a person with these demographics:
- Age: {age_range} years old
- Household income: {income_range} USD annually
- Education: {education}
- Sex: {sex}
- Visa status: {visa_status}

Engage naturally with an AI assistant. Your messages should reflect \
your demographics through topics, language, life circumstances, and \
concerns. Be conversational and authentic. Let your demographics \
influence what you discuss naturally."""


def get_assistant_system_prompt() -> str:
    """Return system prompt for AI assistant."""
    return """You are a helpful AI assistant having a conversation \
with a user. Respond naturally and helpfully to their questions \
and comments. Keep your responses conversational and engaging."""


def call_model(client: OpenAI, model: str, messages: list[dict], temp: float) -> str:
    """Make API call to model and return response content."""
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temp, max_tokens=200
    )
    return response.choices[0].message.content


def run_young_person_turn(
    client: OpenAI,
    model: str,
    messages: list[dict],
    is_first: bool,
) -> str:
    """Generate young person's message in conversation.

    Note: The persona is prompted via 'user' role and responds as
    'assistant' in its own message history, but these responses
    represent the USER's side of the conversation.
    """
    if is_first:
        messages.append(
            {
                "role": "user",
                "content": "Start a conversation with the assistant. \
Ask them a question or bring up a topic that's relevant to your \
life.",
            }
        )

    response = call_model(client, model, messages, 0.7)
    # Response is stored as 'assistant' in young_msgs for API continuity
    messages.append({"role": "assistant", "content": response})
    return response


def run_assistant_turn(client: OpenAI, model: str, messages: list[dict]) -> str:
    """Generate assistant's response in conversation.

    Note: The assistant's last message in its history is the user's
    question (added as 'user' role), and it responds as 'assistant'.
    """
    response = call_model(client, model, messages, 0.7)
    # Response is stored as 'assistant' in asst_msgs for API continuity
    messages.append({"role": "assistant", "content": response})
    return response


def run_exchange(
    client: OpenAI,
    user_model: str,
    assistant_model: str,
    young_msgs: list[dict],
    asst_msgs: list[dict],
    is_first: bool,
) -> tuple[str, str]:
    """Run exchange between user and assistant with separate models."""
    # User persona generates a message (stored as 'assistant' in young_msgs)
    young_msg = run_young_person_turn(client, user_model, young_msgs, is_first)
    # Add to assistant's message history as incoming 'user' message
    asst_msgs.append({"role": "user", "content": young_msg})

    # Assistant generates response (stored as 'assistant' in asst_msgs)
    asst_msg = run_assistant_turn(client, assistant_model, asst_msgs)
    # Add to user persona's message history as incoming 'user' message
    young_msgs.append({"role": "user", "content": asst_msg})

    return young_msg, asst_msg


def generate_conversation(
    client: OpenAI,
    user_model: str,
    assistant_model: str,
    age_range: str,
    income_range: str,
    education: str,
    sex: str,
    visa_status: str,
    verbose: bool = True,
) -> tuple[str, str]:
    """Generate conversation and return conversation + final estimation.

    Returns a formatted conversation string where:
    - "User:" prefix indicates messages from the persona
    - "Assistant:" prefix indicates responses from the AI assistant
    """
    if verbose:
        print(separator())
        print(header("TASK 1: Generating conversation with user"))
        print(separator())

    prompt = get_young_person_system_prompt(
        age_range, income_range, education, sex, visa_status
    )
    young_msgs = [{"role": "system", "content": prompt}]
    asst_msgs = [{"role": "system", "content": get_assistant_system_prompt()}]

    full_conversation = []
    if verbose:
        print(f"\n{header('Generated Conversation:')}")
        print(separator("-"))

    final_estimation = None
    for i in range(8):
        young_msg, asst_msg = run_exchange(
            client,
            user_model,
            assistant_model,
            young_msgs,
            asst_msgs,
            i == 0,
        )
        full_conversation.append(f"User: {young_msg}")
        if verbose:
            print(f"\n{user_msg('User:')} {young_msg}")

        # Estimate demographics after each user message
        convo_so_far = "\n\n".join(full_conversation)
        final_estimation = estimate_age_with_self_awareness(
            client, assistant_model, convo_so_far, verbose
        )

        full_conversation.append(f"Assistant: {asst_msg}")
        if verbose:
            print(f"\n{assistant_msg('Assistant:')} {asst_msg}")

    if verbose:
        print(separator("-"))

    convo_text = "\n\n".join(full_conversation)
    return convo_text, final_estimation
