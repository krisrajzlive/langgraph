"""A small set of local tools used by the tool-calling agent demo.

These are intentionally deterministic/offline so the demo runs without any
external API keys - only the LLM itself (Ollama) needs to be reachable.
"""
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 1)'."""
    allowed = set("0123456789.+-*/() ")
    if not set(expression) <= allowed:
        return "Error: expression contains unsupported characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city (mock/offline data for demo purposes)."""
    mock_data = {
        "bengaluru": "22C, light rain",
        "mumbai": "31C, humid and sunny",
        "delhi": "38C, hazy sunshine",
        "new york": "18C, partly cloudy",
        "london": "15C, overcast",
    }
    return mock_data.get(city.strip().lower(), f"No weather data available for '{city}'.")


@tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    return str(len(text.split()))


ALL_TOOLS = [calculator, get_weather, word_count]
