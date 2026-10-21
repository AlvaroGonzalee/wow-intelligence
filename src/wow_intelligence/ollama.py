"""Handle HTTP communication with Ollama and validate model responses.

Session management and WoW policies belong to the chat interface and the
future knowledge layer, respectively.
"""

import json
from typing import Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(Exception):
    """Report a connection failure, invalid response, or context budget overflow."""


class Message(TypedDict):
    """Describe the author and content of a message sent to the model.

    TypedDict provides type hints for a dictionary. It does not maintain state
    or automatically validate values at runtime.
    """

    role: Literal["system", "user", "assistant"]
    content: str


def generate(
    messages: list[Message],
    *,
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120,
) -> str:
    """Send the full conversation to /api/chat and return the assistant's text.

    Leave the supplied list unchanged. Reject oversized histories before
    connecting rather than dropping earlier messages. The UTF-8 byte budget
    plus per-message overhead is a conservative approximation for Qwen3 with
    a 4096-token context, not an exact tokenizer. It reserves room for the chat
    template and 512 output tokens. Revisit this budget if the model or context
    size changes.

    Raise ValueError for invalid arguments and OllamaError for request failures,
    histories exceeding the budget, or unusable responses.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a non-empty list.")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("Each message must have a valid role and non-empty text.")
    if timeout <= 0:
        raise ValueError("Timeout must be greater than zero.")

    # Bound input without trimming history. Count multibyte characters by their
    # UTF-8 size and allow overhead for the chat template on each message.
    input_budget = sum(len(message["content"].encode("utf-8")) + 64 for message in messages)
    if input_budget > 3000:
        raise OllamaError(
            "The conversation is too long for the current context budget. "
            "No messages were removed. Exit and start a new session."
        )

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"num_ctx": 4096, "num_predict": 512},
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.code}. Check the server and model name."
        ) from exc
    except TimeoutError as exc:
        raise OllamaError("The request to Ollama timed out.") from exc
    except URLError as exc:
        raise OllamaError(
            "Could not connect to Ollama. Check that the local service is running."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaError("Ollama did not return valid JSON.") from exc

    if not isinstance(result, dict) or "error" in result:
        raise OllamaError("Ollama returned an invalid response or reported an error.")
    if result.get("done") is not True:
        raise OllamaError("Ollama did not finish generating the response.")
    reply = result.get("message")
    if not isinstance(reply, dict) or reply.get("role") != "assistant":
        raise OllamaError("Ollama returned an invalid assistant message.")
    text = reply.get("content")
    if not isinstance(text, str) or not text.strip():
        raise OllamaError("Ollama returned no usable text.")
    return text.strip()
