from unittest.mock import patch
from urllib.error import HTTPError, URLError
import io
import json
import unittest

from wow_intelligence.ollama.client import generate
from wow_intelligence.ollama.errors import OllamaError


class GenerateTests(unittest.TestCase):
    # Verify provider requests and error handling without a running server.
    def reply(self, payload: object) -> io.BytesIO:
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_sends_local_request_and_returns_response(self, open_url):
        open_url.return_value = self.reply({"done": True, "message": {"role": "assistant", "content": " Ready. "}})

        self.assertEqual(generate([{"role": "user", "content": "Hello"}]), "Ready.")

        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertGreater(open_url.call_args.kwargs["timeout"], 0)

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_allows_connection_settings(self, open_url):
        open_url.return_value = self.reply({"done": True, "message": {"role": "assistant", "content": "Ready."}})

        generate([{"role": "user", "content": "Hello"}], model="test-model", base_url="http://localhost:11435/", timeout=5)

        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11435/api/chat")
        self.assertEqual(json.loads(request.data)["model"], "test-model")
        self.assertEqual(open_url.call_args.kwargs["timeout"], 5)

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_rejects_empty_history_before_connecting(self, open_url):
        with self.assertRaises(ValueError):
            generate([])
        open_url.assert_not_called()

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_rejects_nonpositive_timeout_before_connecting(self, open_url):
        with self.assertRaises(ValueError):
            generate([{"role": "user", "content": "Hello"}], timeout=0)
        open_url.assert_not_called()

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_reports_connection_failure(self, open_url):
        open_url.side_effect = URLError("Connection refused")
        with self.assertRaisesRegex(OllamaError, "connect"):
            generate([{"role": "user", "content": "Hello"}])

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_preserves_full_history_in_request(self, open_url):
        messages = [
            {"role": "system", "content": "Answer only in English."},
            {"role": "user", "content": "I am level 27."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What level am I?"},
        ]
        open_url.return_value = self.reply({
            "done": True, "message": {"role": "assistant", "content": "Level 27."},
        })
        self.assertEqual(generate(messages), "Level 27.")
        self.assertEqual(json.loads(open_url.call_args.args[0].data)["messages"], messages)
        self.assertEqual(len(messages), 4)

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_rejects_invalid_messages_before_connecting(self, open_url):
        for message in (
            {}, {"role": "user", "content": " "},
            {"role": "unknown", "content": "Hello"},
            {"role": "user", "content": 123},
        ):
            with self.subTest(message=message):
                with self.assertRaises(ValueError):
                    generate([message])
        open_url.assert_not_called()

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_rejects_oversized_history_without_silently_trimming(self, open_url):
        with self.assertRaisesRegex(OllamaError, "too long"):
            generate([{"role": "user", "content": "x" * 4000}])
        open_url.assert_not_called()

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_history_budget_counts_utf8_and_message_overhead(self, open_url):
        for messages in (
            [{"role": "user", "content": "🐉" * 800}],
            [{"role": "user", "content": "Hi"}] * 50,
        ):
            with self.subTest(messages=messages):
                with self.assertRaises(OllamaError):
                    generate(messages)
        open_url.assert_not_called()

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_reports_timeout(self, open_url):
        open_url.side_effect = TimeoutError("Timed out")
        with self.assertRaisesRegex(OllamaError, "timed out"):
            generate([{"role": "user", "content": "Hello"}])

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_reports_http_error(self, open_url):
        open_url.side_effect = HTTPError("http://localhost", 404, "Not found", {}, None)
        self.addCleanup(open_url.side_effect.close)
        with self.assertRaisesRegex(OllamaError, "404"):
            generate([{"role": "user", "content": "Hello"}])

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_reports_unreadable_response(self, open_url):
        for raw in (b"not JSON", b"\xff"):
            with self.subTest(raw=raw):
                open_url.return_value = io.BytesIO(raw)
                with self.assertRaisesRegex(OllamaError, "JSON"):
                    generate([{"role": "user", "content": "Hello"}])

    @patch('wow_intelligence.ollama.client.urlopen')
    def test_rejects_invalid_or_incomplete_response(self, open_url):
        for payload in (
            [],
            {"error": "model not found"},
            {"done": False, "response": "Partial"},
            {"done": True},
            {"done": True, "message": {"role": "assistant", "content": "   "}},
            {"done": True, "message": {"role": "assistant", "content": 123}},
            {"done": True, "message": "invalid"},
            {"done": True, "message": {"role": "user", "content": "Wrong author"}},
        ):
            with self.subTest(payload=payload):
                open_url.return_value = self.reply(payload)
                with self.assertRaises(OllamaError):
                    generate([{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
