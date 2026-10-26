import io
import unittest
from pathlib import Path
from unittest.mock import patch

from wow_intelligence.__main__ import main


FIXTURE = str(Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json")


class RagCliTests(unittest.TestCase):
    # Verify explicit mode selection and useful startup errors.
    def setUp(self):
        self.chat = self.enterContext(patch("wow_intelligence.__main__.run_chat"))
        self.output = self.enterContext(patch("sys.stdout", new_callable=io.StringIO))
        self.errors = self.enterContext(patch("sys.stderr", new_callable=io.StringIO))

    def test_corpus_and_build_enable_rag(self):
        with patch("sys.argv", ["wow-intelligence", "--corpus", FIXTURE, "--game-build", "test-build"]):
            main()
        rag = self.chat.call_args.kwargs["rag"]
        self.assertEqual(rag.game_build, "test-build")
        self.assertEqual(len(rag.retriever.evidence), 2)

    def test_check_corpus_exits_without_loading_ollama_configuration_or_chat(self):
        with patch("sys.argv", ["wow-intelligence", "--corpus", FIXTURE, "--game-build", "test-build", "--check-corpus"]):
            with patch("wow_intelligence.__main__.OllamaConfig.from_env") as config:
                main()
                config.assert_not_called()
        self.chat.assert_not_called()
        self.assertIn("Eligible evidence for build test-build: 2", self.output.getvalue())

    def test_rejects_unpaired_flags_unknown_build_and_missing_corpus(self):
        for arguments in (
            ["--corpus", FIXTURE], ["--game-build", "test-build"], ["--check-corpus"],
            ["--corpus", FIXTURE, "--game-build", "unknown"],
            ["--corpus", "/does-not-exist/corpus.json", "--game-build", "test-build"],
        ):
            with self.subTest(arguments=arguments), patch("sys.argv", ["wow-intelligence", *arguments]):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
        self.chat.assert_not_called()
