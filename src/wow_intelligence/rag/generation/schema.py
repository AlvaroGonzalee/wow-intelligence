from wow_intelligence.rag.models.evidence import Evidence


def answer_schema(evidence: tuple[Evidence, ...]) -> dict:
    return {
        "type": "object", "required": ["statements"], "additionalProperties": False,
        "properties": {"statements": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object", "required": ["text", "evidence_ids"], "additionalProperties": False,
                "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": 1200},
                    "evidence_ids": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "enum": [item.evidence_id for item in evidence]},
                    },
                },
            },
        }},
    }
