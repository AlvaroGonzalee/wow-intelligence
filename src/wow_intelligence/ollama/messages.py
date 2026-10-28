from typing import Literal, TypedDict


INPUT_BUDGET = 3000


class Message(TypedDict):
    # Describe the author and content of a message sent to the model.
    role: Literal["system", "user", "assistant"]
    content: str


def message_size(messages: list["Message"]) -> int:
    return sum(len(message["content"].encode("utf-8")) + 64 for message in messages)
