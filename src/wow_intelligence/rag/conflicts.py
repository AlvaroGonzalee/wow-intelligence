import re
from urllib.parse import urlsplit

from wow_intelligence.sources import Evidence


CONFLICT_NOTICE = "The retrieved catalog sources disagree on this information. I cannot give a reliable answer until the conflicting records are reviewed."
_FIELDS = {"quality", "item level", "required level", "tuned level", "entry level",
           "players", "lockout", "entrance zone", "type"}


def catalog_conflict(evidence: tuple[Evidence, ...]) -> bool:
    observed = {}
    for item in evidence:
        header = re.match(r".+? (items|instances) catalog reports ", item.content)
        entity = re.match(r"/(items|wow-forever/dungeons)/(\d+)(?:-|$)", urlsplit(item.url).path)
        if not header or not entity:
            continue
        if (header[1] == "items") != (entity[1] == "items"):
            continue
        for field, value in re.findall(r"(?:^|; )([a-z ]+): ([^;]+)(?=;)", item.content[header.end():]):
            value = value.strip().casefold()
            if field not in _FIELDS or value.casefold() in {"unknown", "-"}:
                continue
            key = (entity[1], entity[2], item.game_build, field)
            if key in observed and observed[key] != value:
                return True
            observed[key] = value
    return False
