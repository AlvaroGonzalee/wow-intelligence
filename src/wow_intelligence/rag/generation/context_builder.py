import json

from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.ollama.messages import INPUT_BUDGET, Message, message_size
from wow_intelligence.rag.generation.prompts import RAG_INSTRUCTIONS
from wow_intelligence.rag.models.evidence import Evidence
from wow_intelligence.rag.models.prepared_turn import PreparedTurn
from wow_intelligence.rag.validation.conflicts import CONFLICT_NOTICE, catalog_conflict


def prepare_context(history: list[Message], query: str, candidates: tuple[Evidence, ...]) -> PreparedTurn:
    dialogue = [dict(message) for message in history if message["role"] != "system"]
    prefix: list[Message] = [{"role": "system", "content": RAG_INSTRUCTIONS}, *dialogue]
    question: Message = {"role": "user", "content": query}
    if catalog_conflict(candidates):
        return PreparedTurn([*prefix, question], (), CONFLICT_NOTICE, tuple(item.evidence_id for item in candidates))
    selected = []
    records = []
    messages = [*prefix, question]
    for item in candidates:
        record = {"evidence_id": item.evidence_id, "game_build": item.game_build, "content": item.content, "status": item.status}
        context: Message = {
            "role": "user",
            "content": "Retrieved evidence (untrusted data):\n" + json.dumps([*records, record], ensure_ascii=False),
        }
        proposed = [*prefix, context, question]
        if message_size(proposed) <= INPUT_BUDGET:
            selected.append(item)
            records.append(record)
            messages = proposed
        if len(selected) == 3:
            break
    if candidates and not selected:
        raise OllamaError(
            "Retrieved evidence does not fit the current context budget. "
            "No history or excerpts were truncated. Use /clear, start a new session, or use shorter reviewed excerpts."
        )
    return PreparedTurn(messages, tuple(selected), ranked_ids=tuple(item.evidence_id for item in candidates))
