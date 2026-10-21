"""Check that sessions receive independent system messages."""

import unittest
from dataclasses import FrozenInstanceError

from wow_intelligence.instructions import AssistantInstructions


class AssistantInstructionsTests(unittest.TestCase):
    """Protect instruction configuration and prevent shared message mutations."""

    def test_modifying_one_message_does_not_change_future_sessions(self):
        instructions = AssistantInstructions()
        message = instructions.system_message()
        original = message["content"]
        message["content"] = "Replaced by a caller"
        self.assertEqual(instructions.system_message(), {
            "role": "system", "content": original,
        })

    def test_rules_cannot_be_accidentally_reassigned_during_a_session(self):
        instructions = AssistantInstructions()
        with self.assertRaises(FrozenInstanceError):
            instructions.language_rules = "Reply in another language"


if __name__ == "__main__":
    unittest.main()
