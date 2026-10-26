import unittest

from wow_intelligence.domain import WoWDomainPolicy


class WoWDomainPolicyTests(unittest.TestCase):
    # Cover clear rejections, game calculations, and uncertain follow-ups.
    def setUp(self):
        self.policy = WoWDomainPolicy()

    def test_rejects_representative_unrelated_requests(self):
        for text in (
            "What is the weather in Madrid?", "What is the capital of France?",
            "Give me a recipe for pancakes.", "Write a Python script to rename files.",
            "How do I play Minecraft?", "What is the square root of 144?",
            "Calculate 25 times 4.", "What is 2 + 2?",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.policy.is_clearly_unrelated(text))

    def test_leaves_game_questions_and_calculations_to_model(self):
        for text in (
            "What gear should my level 27 Rogue use?",
            "What is the lore of Thousand Needles?",
            "Calculate the cost of 25 items at 4 gold each.",
            "What is the square root of 144 for this WoW calculation?",
            "Which recipe for alchemy should I learn in WoW?",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.policy.is_clearly_unrelated(text))

    def test_leaves_unrecognized_names_and_ambiguous_followups_to_model(self):
        for text in ("Hello", "Which is better?", "What level am I?", "Where is Mankrik?"):
            with self.subTest(text=text):
                self.assertFalse(self.policy.is_clearly_unrelated(text))

    def test_allows_calculation_followup_with_game_context(self):
        self.assertFalse(self.policy.is_clearly_unrelated(
            "Calculate 25 times 4.", "These items cost 4 gold each.",
        ))

    def test_rejects_calculation_without_game_context(self):
        self.assertTrue(self.policy.is_clearly_unrelated("Calculate 25 times 4.", "Hello"))

    def test_previous_game_context_does_not_exempt_explicit_unrelated_topics(self):
        self.assertTrue(self.policy.is_clearly_unrelated(
            "What is the weather in Madrid?", "I play a Rogue in WoW.",
        ))


if __name__ == "__main__":
    unittest.main()
