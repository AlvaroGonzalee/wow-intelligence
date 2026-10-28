import io
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from wow_intelligence.ollama import OllamaError
from wow_intelligence.rag import RagPipeline, load_corpus
from wow_intelligence.rag.embedder import OllamaEmbedder, normalized_vector
from wow_intelligence.rag.vectorstore import VectorIndex, build_index


FIXTURE = Path(__file__).parent / "fixtures" / "rag_test_data_fictional.json"


class FakeEmbedder:
    # Supply deterministic vectors to test semantic plumbing without a model.
    model = "fictional-embedding-model"
    identity = {"model": "fictional-embedding-model:latest", "digest": "fictional-digest"}

    def embed(self, texts):
        return tuple((0.0, 1.0) if "violet" in text.casefold() else (1.0, 0.0) for text in texts)


class EmbedderTests(unittest.TestCase):
    # Validate transport, model identity and numeric safety without network.
    @patch("wow_intelligence.rag.embedder.urlopen")
    def test_batches_use_no_truncation_and_normalize_vectors(self, request):
        request.side_effect = [
            io.BytesIO(json.dumps({"models": [{"name": "test:latest", "digest": "abc", "capabilities": ["embedding"]}]}).encode()),
            io.BytesIO(json.dumps({"embeddings": [[3, 4], [0, 2]]}).encode()),
        ]
        embedder = OllamaEmbedder("test")
        self.assertEqual(embedder.embed(["first", "second"]), ((0.6, 0.8), (0.0, 1.0)))
        payload = json.loads(request.call_args.args[0].data)
        self.assertFalse(payload["truncate"])
        self.assertEqual(payload["input"], ["first", "second"])
        self.assertEqual(embedder.identity["digest"], "abc")
        self.assertEqual(request.call_count, 2)

    @patch("wow_intelligence.rag.embedder.urlopen")
    def test_missing_or_chat_only_models_fail_without_downloads(self, request):
        for models in ([], [{"name": "test", "digest": "abc", "capabilities": ["completion"]}]):
            request.reset_mock()
            request.return_value = io.BytesIO(json.dumps({"models": models}).encode())
            with self.subTest(models=models), self.assertRaises(OllamaError):
                OllamaEmbedder("test").embed(["query"])
            self.assertEqual(request.call_count, 1)
            self.assertTrue(request.call_args.args[0].full_url.endswith("/api/tags"))

    def test_rejects_invalid_vectors_and_batches(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], [True], ["1"]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                normalized_vector(vector)
        with self.assertRaises(ValueError):
            normalized_vector([1, 2], 3)
        for texts in ([], [""], ["x"] * 33):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                OllamaEmbedder("test").embed(texts)

    @patch("wow_intelligence.rag.embedder.urlopen")
    def test_rejects_wrong_embedding_count(self, request):
        request.return_value = io.BytesIO(b'{"embeddings":[]}')
        embedder = OllamaEmbedder("test")
        embedder._identity = {"model": "test", "digest": "abc"}
        with self.assertRaises(OllamaError):
            embedder.embed(["query"])


class VectorIndexTests(unittest.TestCase):
    # Prevent stale indexes, partial publication and semantic version leakage.
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self.folder) / "index.sqlite3"
        self.corpus = load_corpus(FIXTURE)
        self.evidence = RagPipeline(self.corpus, "test-build").retriever.evidence
        self.embedder = FakeEmbedder()

    def test_round_trip_and_hybrid_retrieval_without_keyword_overlap(self):
        build_index(self.path, self.evidence, self.embedder)
        index = VectorIndex(self.path, self.evidence, self.embedder.identity)
        self.assertEqual(index.search([1, 0]), ("amber-token",))
        pipeline = RagPipeline(self.corpus, "test-build", index_path=self.path, embedder=self.embedder)
        self.assertEqual(pipeline.retrieval_mode, "hybrid")
        self.assertEqual(pipeline.retriever.search("azure repository")[0].evidence_id, "amber-token")
        self.assertNotIn("amber-other-build", index.vectors)
        self.assertIn("hybrid", pipeline.coverage_text())

    def test_content_permission_and_model_changes_reject_stale_index(self):
        build_index(self.path, self.evidence, self.embedder)
        for evidence, identity in (
            ((replace(self.evidence[0], content="Changed source fact"), self.evidence[1]), self.embedder.identity),
            ((replace(self.evidence[0], source=replace(self.evidence[0].source, enabled=False)), self.evidence[1]), self.embedder.identity),
            (self.evidence, {**self.embedder.identity, "digest": "changed"}),
            (self.evidence[:1], self.embedder.identity),
        ):
            with self.subTest(identity=identity, count=len(evidence)), self.assertRaises(ValueError):
                VectorIndex(self.path, evidence, identity)

    def test_failures_never_publish_partial_index_or_replace_existing_one(self):
        with patch.object(self.embedder, "embed", side_effect=OllamaError("Unavailable")):
            with self.assertRaises(OllamaError):
                build_index(self.path, self.evidence, self.embedder)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.folder).iterdir()), [])
        build_index(self.path, self.evidence, self.embedder)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            build_index(self.path, self.evidence, self.embedder)
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_vectors_and_non_database_files_are_rejected(self):
        self.path.write_text("not a database")
        with self.assertRaises(ValueError):
            VectorIndex(self.path, self.evidence, self.embedder.identity)
        other = Path(self.folder) / "valid.sqlite3"
        build_index(other, self.evidence, self.embedder)
        with closing(sqlite3.connect(other)) as connection, connection:
            connection.execute("UPDATE vectors SET vector = ?", (struct.pack("<2f", float("nan"), 1.0),))
        with self.assertRaises(ValueError):
            VectorIndex(other, self.evidence, self.embedder.identity)

    def test_empty_or_incomplete_semantic_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_index(self.path, (), self.embedder)
        with self.assertRaises(ValueError):
            RagPipeline(self.corpus, "test-build", embedder=self.embedder)
        with self.assertRaises(ValueError):
            RagPipeline(self.corpus, "test-build", index_path=self.path)

    def test_index_command_selects_only_eligible_evidence(self):
        from wow_intelligence.rag.index import main
        with patch("sys.argv", ["index", "--corpus", str(FIXTURE), "--game-build", "test-build",
                                "--embedding-model", "fictional-model", "--output", str(self.path)]):
            with patch("wow_intelligence.rag.index.OllamaEmbedder", return_value=self.embedder):
                with patch("sys.stdout", new_callable=io.StringIO):
                    main()
        index = VectorIndex(self.path, self.evidence, self.embedder.identity)
        self.assertEqual(set(index.vectors), {"amber-token", "violet-token"})
