"""Exercise the installed offline detector with representative chat messages."""

import unittest

from wow_intelligence.language import EnglishLanguagePolicy


class EnglishLanguagePolicyTests(unittest.TestCase):
    """Check English, non-English, mixed, and ambiguous inputs with real Lingua."""

    @classmethod
    def setUpClass(cls):
        """Reuse one detector for the entire language test suite."""
        cls.policy = EnglishLanguagePolicy()

    def test_accepts_english_with_game_names_and_numbers(self):
        for text in (
            "My character is a level 27 Rogue.",
            "What class and level did I just tell you?",
            "What about my Rogue in Thousand Needles?",
            "You told me you are a level 27 Rogue.",
            "I don't know based on the verified WoW Forever sources currently available to me.",
            "I am ready.",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.policy.is_english(text))

    def test_accepts_common_short_greetings_and_acknowledgments(self):
        for text in ("Hello!", "Hi", "Thanks!", "Yes", "OK", "Thank you."):
            with self.subTest(text=text):
                self.assertTrue(self.policy.is_english(text))

    def test_rejects_non_english(self):
        for text in (
            "Hola, soy un Rogue de nivel 27. Confírmalo brevemente.",
            "Hola. Eres un Rogue de nivel 27.",
            "Bonjour, je voudrais des informations sur mon personnage.",
            "Ich möchte mehr über meinen Charakter erfahren.",
            "我的角色是一名盗贼。",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.policy.is_english(text))

    def test_rejects_representative_mixed_language_messages(self):
        for text in (
            "My character is a Rogue y quiero saber sus habilidades.",
            "I am a Rogue. Quiero saber donde conseguir una daga.",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.policy.is_english(text))

    def test_rejects_unknown_or_ambiguous_input(self):
        for text in ("", "   ", "27", "???", "Rogue", "Hola"):
            with self.subTest(text=text):
                self.assertFalse(self.policy.is_english(text))


if __name__ == "__main__":
    unittest.main()
