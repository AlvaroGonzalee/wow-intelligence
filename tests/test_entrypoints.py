from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EntryPointTests(unittest.TestCase):
    # Verify application startup and the separate maintenance commands.
    def run_module(self, module, argument):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-m", module, argument],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_primary_module_and_explicit_chat_command_report_the_same_version(self):
        expected = f"WoW Intelligence {version('wow-intelligence')}\n"
        for module in ("wow_intelligence", "wow_intelligence.cli.chat"):
            with self.subTest(module=module):
                self.assertEqual(self.run_module(module, "--version"), expected)

    def test_maintenance_commands_expose_their_arguments(self):
        for command, flag in (
            ("wow_intelligence.cli.evaluate", "--questions"),
            ("wow_intelligence.cli.build_index", "--embedding-model"),
            ("wow_intelligence.cli.import_catalog", "--retrieved-at"),
            ("wow_intelligence.cli.import_item", "--item"),
        ):
            with self.subTest(command=command):
                self.assertIn(flag, self.run_module(command, "--help"))


if __name__ == "__main__":
    unittest.main()
