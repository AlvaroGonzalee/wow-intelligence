import re


class ConversationPolicy:
    @staticmethod
    # Recognize exact social phrases and unresolved references conservatively.
    def local_reply(text: str) -> str | None:
        normalized = text.strip().rstrip(".!?").casefold()
        if normalized in {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}:
            return "Hello! What would you like to know about World of Warcraft: Forever?"
        if normalized in {"thanks", "thank you", "thank you very much"}:
            return "You're welcome!"
        if normalized in {"ok", "okay", "yes", "no"}:
            return "Understood. What would you like to ask about Forever?"
        if normalized in {"how are you", "how are you doing"}:
            return "I'm ready to help with your Forever questions."
        if normalized in {"what can you do", "how can you help", "what do you know"}:
            return (
                "I can answer Forever questions using the evidence loaded for this session. "
                "Use /coverage to see what is available and /sources to see answer references."
            )
        return None

    @staticmethod
    def needs_context(text: str) -> bool:
        normalized = text.strip().rstrip(".!?").casefold()
        return normalized in {"which is better", "why", "how much", "what about that"} or bool(
            re.fullmatch(
                r"(?:where|how|what|when|why)\b[^.!?]*\b(?:it|them|that item|this item|that quest|this quest)\b[^.!?]*",
                normalized,
            )
        )
