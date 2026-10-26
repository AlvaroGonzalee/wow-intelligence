from lingua import Language, LanguageDetectorBuilder


class EnglishLanguagePolicy:
    # Accept detected English and a small set of common English acknowledgments.
    input_rejection = "Please ask your question in English using a complete sentence."
    output_rejection = (
        "The model's response could not be confirmed as English and was not displayed. "
        "This turn was not saved. Please try again."
    )

    def __init__(self) -> None:
        self._detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_minimum_relative_distance(0.05)
            .build()
        )

    def is_english(self, text: str) -> bool:
        normalized = text.strip(" \t\r\n.!?").casefold()
        if normalized in {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "thanks", "thank you", "goodbye", "yes", "no", "ok", "okay",
        }:
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
