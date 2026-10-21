"""Manage terminal interaction and temporary conversation history.

Each run_chat invocation owns its history and never saves it to disk.
Model communication is delegated to ollama.py.
"""

import sys
from importlib.metadata import version

from wow_intelligence.config import OllamaConfig
from wow_intelligence.instructions import AssistantInstructions
from wow_intelligence.language import EnglishLanguagePolicy
from wow_intelligence.ollama import Message, OllamaError, generate


def run_chat(config: OllamaConfig | None = None) -> None:
    """Read English questions and retain turns with an accepted English response.

    Each request includes previous messages and the new question. Build a new
    list to preserve confirmed history if Ollama fails. The initial system
    message carries the assistant's instructions and stays first on every turn.
    Language checks run before contacting Ollama and before saving its response.
    Slash commands run locally and never enter the conversation history.
    /exit, Ctrl+C, and end of input discard the session history.
    """
    config = config if config is not None else OllamaConfig()
    history: list[Message] = [AssistantInstructions().system_message()]
    language_policy = EnglishLanguagePolicy()
    print("WoW Intelligence — experimental terminal chat")
    print("Conversation context lasts for this session only; nothing is saved to disk.")
    print("Context is limited. Answers are not verified against sources.")
    print("Type /help for commands, /exit to leave, or press Ctrl+C.")
    print("English only. Ambiguous input may require a complete English sentence.")

    try:
        while True:
            prompt = input("You: ").strip()
            if prompt == "/exit":
                break
            if not prompt:
                continue
            if prompt == "/help":
                print("Ask questions in English about World of Warcraft: Forever.")
                print("/help — Show this help.")
                print("/version — Show the application version.")
                print("/exit — End the session and discard its history.")
                print("History stays in memory for this session only. Context is limited.")
                print("Answers are not verified against sources.")
                continue
            if prompt == "/version":
                print(f"WoW Intelligence {version('wow-intelligence')}")
                continue
            if prompt.startswith("/"):
                print("Unknown command. Type /help for available commands.")
                continue
            if not language_policy.is_english(prompt):
                print(language_policy.input_rejection)
                continue
            messages: list[Message] = [*history, {"role": "user", "content": prompt}]
            try:
                answer = generate(
                    messages, model=config.model,
                    base_url=config.base_url, timeout=config.timeout,
                )
            except OllamaError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                continue
            if not language_policy.is_english(answer):
                print(language_policy.output_rejection)
                continue
            history = [*messages, {"role": "assistant", "content": answer}]
            print(f"Assistant: {answer}")
    except (EOFError, KeyboardInterrupt):
        print()

    print("Goodbye.")
