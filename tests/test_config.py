"""Test environment settings and startup failures without contacting Ollama."""

import io
import os
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from wow_intelligence.__main__ import main
from wow_intelligence.config import OllamaConfig


class OllamaConfigTests(unittest.TestCase):
    """Verify defaults, explicit overrides, and invalid configuration rejection."""

    def test_absent_variables_use_defaults(self):
        self.assertEqual(OllamaConfig.from_env({}), OllamaConfig())

    def test_reads_supported_variables_without_changing_mapping(self):
        values = {
            "OLLAMA_BASE_URL": "https://localhost:11435/",
            "OLLAMA_MODEL": "other:8b",
            "OLLAMA_TIMEOUT": "12.5",
        }
        original = values.copy()
        self.assertEqual(OllamaConfig.from_env(values), OllamaConfig(
            base_url="https://localhost:11435/", model="other:8b", timeout=12.5,
        ))
        self.assertEqual(values, original)

    @patch.dict(os.environ, {"OLLAMA_MODEL": "environment-model"}, clear=True)
    def test_reads_process_environment_only_when_no_mapping_is_supplied(self):
        self.assertEqual(OllamaConfig.from_env().model, "environment-model")
        self.assertEqual(OllamaConfig.from_env({}).model, "qwen3:8b")

    def test_partial_override_retains_other_defaults(self):
        self.assertEqual(OllamaConfig.from_env({"OLLAMA_MODEL": "other:8b"}),
                         OllamaConfig(model="other:8b"))

    def test_rejects_invalid_urls(self):
        for value in (
            "", "localhost:11434", "ftp://localhost", "http://", "http://[bad",
            "http://localhost:bad", "http://localhost:65536", "http://localhost:0",
            "http://user:pass@localhost", "http://localhost?key=value",
            "http://localhost#fragment", "http://local host", "\nhttp://localhost",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "OLLAMA_BASE_URL"):
                    OllamaConfig.from_env({"OLLAMA_BASE_URL": value})

    def test_rejects_empty_or_whitespace_model_names(self):
        for value in ("", " ", "some model", "qwen3:8b\n"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "OLLAMA_MODEL"):
                    OllamaConfig.from_env({"OLLAMA_MODEL": value})

    def test_rejects_invalid_timeouts(self):
        for value in ("", "no", "0", "-1", "nan", "inf", "-inf", "1e999"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "OLLAMA_TIMEOUT"):
                    OllamaConfig.from_env({"OLLAMA_TIMEOUT": value})

    @patch("wow_intelligence.__main__.run_chat")
    @patch("sys.argv", ["wow-intelligence"])
    @patch.dict(os.environ, {"OLLAMA_MODEL": "startup-model"}, clear=True)
    def test_startup_passes_configuration_to_chat(self, run_chat):
        main()
        run_chat.assert_called_once_with(OllamaConfig(model="startup-model"))

    @patch("wow_intelligence.__main__.run_chat")
    @patch("sys.argv", ["wow-intelligence"])
    @patch.dict(os.environ, {"OLLAMA_TIMEOUT": "invalid"}, clear=True)
    def test_invalid_startup_settings_exit_before_chat(self, run_chat):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("OLLAMA_TIMEOUT", errors.getvalue())
        run_chat.assert_not_called()

    @patch("wow_intelligence.__main__.run_chat")
    @patch.dict(os.environ, {"OLLAMA_TIMEOUT": "invalid"}, clear=True)
    def test_help_and_version_work_with_invalid_configuration(self, run_chat):
        for argument in ("--help", "--version"):
            with self.subTest(argument=argument), patch("sys.argv", ["wow-intelligence", argument]):
                with patch("sys.stdout", new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 0)
        run_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
