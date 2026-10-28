import json
import re

from wow_intelligence.ollama.errors import OllamaError
from wow_intelligence.rag.generation.prompts import ABSTENTION
from wow_intelligence.rag.models.evidence import Evidence
from wow_intelligence.rag.models.rag_answer import RagAnswer


def parse_answer(raw: str, evidence: tuple[Evidence, ...]) -> RagAnswer:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(result, dict) or set(result) != {"statements"}:
            raise ValueError
        statements = result["statements"]
        if not isinstance(statements, list) or len(statements) > 3:
            raise ValueError
        available = {item.evidence_id: item for item in evidence}
        cited: list[Evidence] = []
        lines = []
        validation_lines = []
        for statement in statements:
            if not isinstance(statement, dict) or set(statement) != {"text", "evidence_ids"}:
                raise ValueError
            text, identifiers = statement["text"], statement["evidence_ids"]
            if (
                not isinstance(text, str) or not text.strip() or len(text) > 1200
                or re.search(r"\b[a-z][a-z0-9+.-]*://|www\.|\[[^\]]*\]|[\x00-\x1f\x7f-\x9f]", text, re.IGNORECASE)
                or not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(key, str) or key not in available for key in identifiers)
                or len(set(identifiers)) != len(identifiers)
            ):
                raise ValueError
            markers = []
            for key in identifiers:
                item = available[key]
                if item not in cited:
                    cited.append(item)
                markers.append(f"[{cited.index(item) + 1}]")
            lines.append(text.strip() + " " + " ".join(markers))
            validation_lines.append(text.strip())
        if lines and any(item.status == "source_reported" for item in cited):
            lines.insert(0, "This answer is preliminary and based on a source catalog. The information is not independently verified in game and may change as more Forever information becomes available.")
        return RagAnswer("\n".join(lines), tuple(cited), "\n".join(validation_lines)) if lines else RagAnswer(ABSTENTION, method="abstention")
    except (ValueError, TypeError, RecursionError) as exc:
        raise OllamaError(
            "The model returned invalid or missing evidence citations. "
            "The answer was not displayed or saved. Please try again."
        ) from exc
