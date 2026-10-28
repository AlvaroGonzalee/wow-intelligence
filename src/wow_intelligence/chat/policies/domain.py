import re


class WoWDomainPolicy:
    # Reject explicit unrelated topics while leaving uncertain input to the model.
    input_rejection = "I can only help with World of Warcraft: Forever."

    _game_context = re.compile(
        r"\b(world of warcraft|wow|azeroth|forever|rogue|warrior|mage|priest|"
        r"warlock|paladin|druid|shaman|hunter|quest|quests|dungeon|dungeons|"
        r"raid|raids|talents|guild|auction house|gold|silver|copper|"
        r"thousand needles|orgrimmar|stormwind)\b", re.IGNORECASE,
    )
    _unrelated = re.compile(
        r"\b(weather forecast|weather in|capital of|stock price|"
        r"write (?:me )?(?:a )?(?:python|javascript|java) (?:script|program)|"
        r"recipe for|cook (?:a |some )?(?:pasta|rice|chicken)|"
        r"minecraft|fortnite|league of legends)\b", re.IGNORECASE,
    )
    _calculation = re.compile(
        r"\b(square root|calculate|compute|what is \d|what's \d)\b|"
        r"\d\s*[+*/×÷-]\s*\d", re.IGNORECASE,
    )

    def is_clearly_unrelated(self, text: str, previous_question: str = "") -> bool:
        if self._game_context.search(text):
            return False
        if self._unrelated.search(text):
            return True
        return bool(
            self._calculation.search(text)
            and not self._game_context.search(previous_question)
        )
