from pathlib import Path
import json

from wow_intelligence.rag.storage.json_loader import _unique_object


def load_questions(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Evaluation questions exceed 1 MiB.")
    suite = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(suite, dict) or set(suite) != {"schema_version", "name", "game_build", "limitations", "cases"}:
        raise ValueError("Invalid evaluation suite fields.")
    if type(suite["schema_version"]) is not int or suite["schema_version"] != 1:
        raise ValueError("Unsupported evaluation schema version.")
    if any(not isinstance(suite[key], str) or not suite[key].strip() for key in ("name", "game_build", "limitations")):
        raise ValueError("Evaluation metadata must contain nonempty text.")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("Evaluation requires at least one case.")
    seen = set()
    for case in suite["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "category", "question", "expected", "references", "answer_terms"}:
            raise ValueError("Invalid evaluation case fields.")
        if any(not isinstance(case[key], str) or not case[key].strip() for key in ("id", "category", "question")):
            raise ValueError("Cases require nonempty identifiers, categories and questions.")
        if case["id"] in seen:
            raise ValueError("Duplicate evaluation case ID.")
        seen.add(case["id"])
        if case["expected"] not in ("answer", "abstain") or not isinstance(case["references"], dict):
            raise ValueError("Cases require answer/abstain and a reference mapping.")
        if (case["expected"] == "answer") != bool(case["references"]):
            raise ValueError("Answer cases need references; abstention cases must have none.")
        if any(not isinstance(key, str) or not key or not isinstance(fragments, list) or not fragments
               or any(not isinstance(fragment, str) or not fragment for fragment in fragments)
               for key, fragments in case["references"].items()):
            raise ValueError("Each reference requires an evidence ID and nonempty source fragments.")
        groups = case["answer_terms"]
        if not isinstance(groups, list) or (case["expected"] == "answer" and not groups) or any(not isinstance(group, list) or not group
                                               or any(not isinstance(term, str) or not term for term in group)
                                               for group in groups):
            raise ValueError("Answer terms must be lists of nonempty alternative strings.")
    return suite
