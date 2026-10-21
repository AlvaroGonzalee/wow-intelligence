"""Check chat input and generated output locally with Lingua."""

from lingua import Language, LanguageDetectorBuilder


class EnglishLanguagePolicy:
    """Accept detected English and a small set of common English acknowledgments.

    Language identification is statistical, not proof of language. Unknown text
    is rejected. Isolated game names and short ambiguous phrases may require
    a complete English sentence. Mixed-language segmentation is experimental
    in Lingua and cannot guarantee detection of every foreign word.
    """

    input_rejection = "Please ask your question in English using a complete sentence."
    output_rejection = (
        "The model's response could not be confirmed as English and was not displayed. "
        "This turn was not saved. Please try again."
    )

    def __init__(self) -> None:
        """Build one reusable offline detector for the lifetime of a chat session."""
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_minimum_relative_distance(0.05)
            .build()
        )

    def is_english(self, text: str) -> bool:
        """Accept English text; reject unknown or detected non-English segments.

        Exact greetings avoid unreliable classification of single words such
        as 'hello'. Otherwise check the complete text and recheck any foreign
        segments. Lingua can label English fragments such as "I don't" as French
        during segmentation; only independently confirmed foreign segments block
        a text already classified as English. This tradeoff can miss brief mixing.
        The relative-distance threshold is a heuristic, not a confidence percent.
        No translation or network request is performed here.
        """
        normalized = text.strip(" \t\r\n.!?").casefold()
        if normalized in {"hello", "hi", "thanks", "thank you", "goodbye", "yes", "ok", "okay"}:
            return True
        if self._detector.detect_language_of(text) != Language.ENGLISH:
            return False
        segments = self._detector.detect_multiple_languages_of(text)
        for segment in segments:
            if segment.language != Language.ENGLISH:
                language = self._detector.detect_language_of(
                    text[segment.start_index:segment.end_index]
                )
                if language not in (None, Language.ENGLISH):
                    return False
        return True
