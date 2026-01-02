"""Terminal color formatting using termcolor."""

from termcolor import colored


def user_msg(text: str) -> str:
    """Format user message in cyan."""
    return colored(text, "cyan")


def assistant_msg(text: str) -> str:
    """Format assistant message in green."""
    return colored(text, "green")


def header(text: str) -> str:
    """Format header in bold yellow."""
    return colored(text, "yellow", attrs=["bold"])


def estimation_header(text: str) -> str:
    """Format estimation header in magenta."""
    return colored(text, "magenta", attrs=["bold"])


def separator(char: str = "=", length: int = 80) -> str:
    """Format separator in yellow."""
    return colored(char * length, "yellow")
