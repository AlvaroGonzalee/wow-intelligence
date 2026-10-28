from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.ollama.messages import INPUT_BUDGET, Message, message_size


def generate(
    messages: list[Message],
    *,
    model: str = "qwen3:8b",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120,
    response_schema: dict | None = None,
) -> str:
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
    if response_schema is not None and (not isinstance(response_schema, dict) or not response_schema):
        raise ValueError("Response schema must be a non-empty JSON schema object.")


    if message_size(messages) > INPUT_BUDGET:
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
    if response_schema is not None:
        payload["format"] = response_schema
        payload["options"]["temperature"] = 0
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
