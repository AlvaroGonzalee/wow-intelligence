from dataclasses import replace
import json
import re


_INTENTS = {
    "item level": r"\bitem levels?\b",
    "required level": r"\brequired (?:character )?levels?\b|\bcharacter levels?\b|\brequires?\b.*\blevels?\b|\blevels?\b.*\bequip\b",
    "quality": r"\bquality\b|\brarity\b|\bhow rare\b",
    "subclass": r"\bsubclass\b",
    "tuned level": r"\btuned levels?\b",
    "entry level": r"\bentry levels?\b",
    "players": r"\bplayers?\b|\bparty size\b|\bgroup size\b|\bhow big\b.*\bparty\b",
    "lockout": r"\blockout\b|\breset\b",
    "entrance zone": r"\bentrance zone\b|\bwhich zone\b|\bwhere\b.*\b(?:entrance|located)\b|\bwhere is\b",
}


def catalog_fields(item) -> dict:
    header = re.match(r".+? (?:items|instances) catalog reports ", item.content)
    if not header:
        return {}
    content = item.content[header.end():].split(". Missing fields", 1)[0].split(". Boss lists", 1)[0]
    fields = dict(re.findall(r"(?:^|; )([a-z ]+): ([^;]+)", content))
    if "name" not in fields:
        return {}
    fields["name"] = fields["name"].removesuffix(" (new)").strip()
    return fields


def catalog_names(evidence) -> tuple[str, ...]:
    names = set()
    for item in evidence:
        fields = catalog_fields(item)
        for key in ("name", "entrance zone"):
            value = fields.get(key)
            if value and value != "unknown" and len(value) <= 160:
                names.add(value)
    return tuple(sorted(names, key=lambda name: (-len(name), name)))


def catalog_answer(query: str, evidence):
    from .context import parse_answer
    matches = []
    for item in evidence:
        fields = catalog_fields(item)
        if not fields:
            continue
        pattern = re.escape(fields["name"]).replace(r"\-", r"[-\s]")
        for match in re.finditer(r"(?<!\w)" + pattern + r"(?!\w)", query, re.IGNORECASE):
            matches.append((match.start(), match.end(), item, fields))
    matches = [entry for entry in matches if not any(
        other[0] <= entry[0] and other[1] >= entry[1] and other[1] - other[0] > entry[1] - entry[0]
        for other in matches
    )]
    records = {}
    masked = list(query)
    for start, end, item, fields in matches:
        records.setdefault(item.evidence_id, (item, fields))
        masked[start:end] = " " * (end - start)
    if not records or len(records) > 3:
        return None
    question = "".join(masked)
    requested = [field for field, pattern in _INTENTS.items() if re.search(pattern, question, re.IGNORECASE)]
    if not requested:
        return None
    statements = []
    for item, fields in records.values():
        facts = []
        for field in requested:
            if field not in fields:
                continue
            value = fields[field].removesuffix(".")
            if field == "lockout":
                value = re.sub(r"^(\d+) d$", r"\1 days", value)
            if field == "subclass":
                value = re.sub(r" \([\d,]+\)$", "", value)
            facts.append(f"{field}: {value}")
        if facts:
            text = f"According to the catalog, {fields['name']} has " + "; ".join(facts) + "."
            if re.search(r"\bdrops?\b|\bdrop rate\b", question, re.IGNORECASE):
                text += " The cited catalog does not establish drop sources or rates."
            statements.append({"text": text, "evidence_ids": [item.evidence_id]})
    if not statements:
        return None
    answer = parse_answer(json.dumps({"statements": statements}), evidence)
    return replace(answer, method="catalog")
