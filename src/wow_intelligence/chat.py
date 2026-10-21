"""Manage terminal interaction and temporary conversation history.

Each run_chat invocation owns its history and never saves it to disk.
Model communication is delegated to ollama.py.
"""

import sys

from wow_intelligence.instructions import AssistantInstructions
from wow_intelligence.language import EnglishLanguagePolicy
from wow_intelligence.ollama import Message, OllamaError, generate


def run_chat() -> None:
    """Read English questions and retain turns with an accepted English response.

    Each request includes previous messages and the new question. Build a new
    list to preserve confirmed history if Ollama fails. The initial system
    message carries the assistant's instructions and stays first on every turn.
    Language checks run before contacting Ollama and before saving its response.
    /exit, Ctrl+C, and end of input discard the session history.
    """
    history: list[Message] = [AssistantInstructions().system_message()]
    language_policy = EnglishLanguagePolicy()
    print("WoW Intelligence — experimental terminal chat")
    print("Conversation context lasts for this session only; nothing is saved to disk.")
    print("Context is limited. Answers are not verified against sources.")
    print("Type /exit to leave, or press Ctrl+C.")
    print("English only. Ambiguous input may require a complete English sentence.")

    try:
        while True:
            prompt = input("You: ").strip()
            if prompt == "/exit":
                break
            if not prompt:
                continue
            if not language_policy.is_english(prompt):
                print(language_policy.input_rejection)
                continue
            messages: list[Message] = [*history, {"role": "user", "content": prompt}]
            try:
                answer = generate(messages)
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
