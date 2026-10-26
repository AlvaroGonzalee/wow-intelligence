import math
import re
from collections import Counter

from wow_intelligence.sources import Evidence
from .corpus import Corpus


_STOPWORDS = frozenset(
    "a an the and or to of for in on at by with from is are was were be been "
    "i me my you your we our it its this that these those what which where when "
    "how why who do does did can could would should will please tell about "
    "world warcraft wow forever".split()
)


def _tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in _STOPWORDS]


class Retriever:
    # Index only eligible records for one exact build before scoring relevance.
    def __init__(self, corpus: Corpus, game_build: str, *, include_catalog: bool = False) -> None:
        if not isinstance(game_build, str) or not game_build.strip() or game_build.strip().casefold() == "unknown":
            raise ValueError("RAG requires an explicit known game build.")
        self.evidence = tuple(item for item in corpus.evidence if item.is_usable_for("forever", game_build, include_catalog=include_catalog))
        self._counts = tuple(Counter(_tokens(item.content)) for item in self.evidence)
        self._lengths = tuple(sum(counts.values()) for counts in self._counts)
        self._average = sum(self._lengths) / len(self._lengths) if self._lengths else 1
        self._frequency = Counter(term for counts in self._counts for term in counts)

    def search(self, query: str, limit: int = 12) -> tuple[Evidence, ...]:
        if not isinstance(query, str) or type(limit) is not int or limit < 1:
            raise ValueError("Search requires text and a positive integer limit.")
        terms = set(_tokens(query))
        ranked = []
        for item, counts, length in zip(self.evidence, self._counts, self._lengths):
            score = 0.0
            for term in terms & counts.keys():
                frequency = counts[term]
                inverse = math.log(1 + (len(self.evidence) - self._frequency[term] + 0.5) / (self._frequency[term] + 0.5))
                score += inverse * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / (self._average or 1)))
            if score > 0:
                ranked.append((score, item))
        ranked.sort(key=lambda result: (-result[0], result[1].evidence_id))
        return tuple(item for _, item in ranked[:limit])
