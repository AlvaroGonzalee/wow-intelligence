from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import tempfile
import unittest

from wow_intelligence.rag.ingestion.ahledger.common import ImportError
from wow_intelligence.rag.ingestion.ahledger.items import item_corpus, write_review_bundle
from wow_intelligence.rag.pipeline import RagPipeline
from wow_intelligence.rag.storage.json_loader import load_knowledge_base


class AhledgerTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 21, tzinfo=timezone.utc)
        self.payload = {
            "id": 123, "name": "Synthetic test token", "quality": "COMMON",
            "itemClass": "Trade Goods", "itemSubclass": "Synthetic", "level": 5,
            "requiredLevel": None, "vendorSell": 0,
            "url": "https://ahledger.com/items/123?utm_source=test",
            "attribution": {"text": "Data from AHledger", "url": "https://ahledger.com/?utm_source=test"},
        }
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "bundle"

    def test_import_preserves_raw_data_attribution_and_unknowns(self):
        raw = json.dumps(self.payload).encode()
        path = write_review_bundle(raw, self.output, self.now, expected_id=123)
        knowledge_base = load_knowledge_base(path)
        item = knowledge_base.evidence[0]
        self.assertEqual(item.status, "unverified_for_forever")
        self.assertIsNone(item.game_build)
        self.assertIsNone(item.verified_at)
        self.assertIn("required level: unknown", item.content)
        self.assertIn("vendor sell value in copper: 0", item.content)
        self.assertEqual(item.source.url, self.payload["attribution"]["url"])
        self.assertEqual((self.output / "raw.json").read_bytes(), raw)
        review = json.loads((self.output / "review.json").read_text())
        self.assertEqual(review["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(RagPipeline(knowledge_base, "test-build").retriever.evidence, ())

    def test_remote_fields_cannot_approve_evidence_or_supply_a_build(self):
        self.payload.update(game_build="123", status="verified_for_forever", verified_at=self.now.isoformat())
        item = item_corpus(self.payload, self.now)["evidence"][0]
        self.assertEqual(item["status"], "unverified_for_forever")
        self.assertIsNone(item["game_build"])

    def test_existing_bundle_is_never_overwritten(self):
        raw = json.dumps(self.payload).encode()
        write_review_bundle(raw, self.output, self.now)
        with self.assertRaises(ImportError):
            write_review_bundle(b"{}", self.output, self.now)
        self.assertEqual((self.output / "raw.json").read_bytes(), raw)

    def test_mismatched_item_or_invalid_payload_does_not_publish_a_bundle(self):
        for raw, expected in ((b"{", None), (b"[]", None), (json.dumps(self.payload).encode(), 999)):
            with self.subTest(raw=raw), self.assertRaises(ImportError):
                write_review_bundle(raw, self.output, self.now, expected_id=expected)
            self.assertFalse(self.output.exists())

    def test_attribution_links_and_numeric_types_are_validated(self):
        for key, value in (("level", True), ("vendorSell", -1), ("name", ""), ("url", "https://other.invalid/item"), ("attribution", None)):
            payload = {**self.payload, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                item_corpus(payload, self.now)

    def test_terminal_controls_fail_before_publication(self):
        self.payload["name"] = "bad\u001b[31m"
        with self.assertRaises(ValueError):
            write_review_bundle(json.dumps(self.payload).encode(), self.output, self.now)
        self.assertFalse(self.output.exists())

    def test_size_limit_is_enforced(self):
        with patch('wow_intelligence.rag.ingestion.ahledger.items.MAX_RESPONSE_BYTES', 2), self.assertRaises(ImportError):
            write_review_bundle(b"{} ", self.output, self.now)
