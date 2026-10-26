import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wow_intelligence.rag import CorpusError, load_corpus


FIXTURE = Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json"


class CorpusTests(unittest.TestCase):
    # Keep malformed metadata and implicit approvals out of the corpus.
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "corpus.json"
        self.data = json.loads(FIXTURE.read_text())

    def load(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return load_corpus(self.path)

    def test_loads_dates_and_resolves_shared_source_objects(self):
        corpus = self.load(self.data)
        self.assertEqual(len(corpus.evidence), 3)
        self.assertIs(corpus.evidence[0].source, corpus.sources[0])
        self.assertIsNotNone(corpus.evidence[0].verified_at.utcoffset())

    def test_defaults_do_not_approve_sources_or_verify_evidence(self):
        source = self.data["sources"][0]
        for key in ("permission_status", "permission_reference", "enabled"):
            del source[key]
        self.data["evidence"][0].pop("status")
        corpus = self.load(self.data)
        self.assertEqual(corpus.sources[0].permission_status, "pending_review")
        self.assertFalse(corpus.sources[0].enabled)
        self.assertEqual(corpus.evidence[0].status, "unknown")
        self.assertFalse(corpus.evidence[1].is_usable_for("forever", "test-build"))

    def test_rejects_unknown_fields_duplicate_ids_and_broken_source_links(self):
        for mutation in (
            lambda d: d.update(extra=True),
            lambda d: d["sources"][0].update(typo=True),
            lambda d: d["evidence"][0].update(typo=True),
            lambda d: d["sources"].append(d["sources"][0]),
            lambda d: d["evidence"].append(d["evidence"][0]),
            lambda d: d["evidence"][0].update(source_id="missing"),
            lambda d: d["evidence"][0].update(evidence_id="bad [id]"),
        ):
            data = copy.deepcopy(self.data)
            mutation(data)
            with self.subTest(data=data), self.assertRaises(CorpusError):
                self.load(data)

    def test_reuses_source_and_evidence_validation(self):
        for record, field, value in (
            ("sources", "permission_status", "pending_review"),
            ("sources", "enabled", "true"),
            ("sources", "permission_reference", None),
            ("evidence", "game_build", "unknown"),
            ("evidence", "game_version", "classic"),
            ("evidence", "retrieved_at", "2026-01-01T00:00:00"),
            ("evidence", "verified_at", None),
            ("evidence", "content", "\u001b[31mred"),
            ("evidence", "url", "file:///tmp/test"),
        ):
            data = copy.deepcopy(self.data)
            data[record][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(CorpusError):
                self.load(data)

    def test_rejects_invalid_root_schema_and_arrays(self):
        for data in ([], {}, {**self.data, "schema_version": True},
                     {**self.data, "schema_version": 2}, {**self.data, "sources": {}}):
            with self.subTest(data=data), self.assertRaises(CorpusError):
                self.load(data)

    def test_rejects_malformed_json_duplicate_keys_and_invalid_encoding(self):
        for raw in (b"{", b"\xff", b'{"schema_version":1,"schema_version":1}'):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(CorpusError):
                load_corpus(self.path)

    def test_reports_missing_file_and_limits_read_size(self):
        with self.assertRaises(CorpusError):
            load_corpus(self.path)
        self.path.write_bytes(b" " * 11)
        with patch("wow_intelligence.rag.corpus.MAX_CORPUS_BYTES", 10), self.assertRaises(CorpusError):
            load_corpus(self.path)

    def test_accepts_empty_corpus(self):
        self.assertEqual(self.load({"schema_version": 1, "sources": [], "evidence": []}).evidence, ())
