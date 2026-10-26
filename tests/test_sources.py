import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

from wow_intelligence.sources import Evidence, Source


class SourceTests(unittest.TestCase):
    # Keep source registration separate from permission to use its content.
    def setUp(self):
        self.source = Source(
            source_id="sample-source", name="Fictional test source",
            url="https://example.invalid/source", game_version="forever",
        )

    def test_new_source_is_pending_and_disabled(self):
        self.assertEqual(self.source.permission_status, "pending_review")
        self.assertFalse(self.source.enabled)

    def test_enabling_requires_explicit_approval_and_permission_reference(self):
        for status in ("pending_review", "rejected"):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    replace(self.source, permission_status=status, enabled=True)
        with self.assertRaises(ValueError):
            replace(self.source, permission_status="approved")
        approved = replace(
            self.source, permission_status="approved",
            permission_reference="Fictional permission record for tests", enabled=True,
        )
        self.assertTrue(approved.enabled)

    def test_rejects_missing_source_identity_and_scope(self):
        for field in ("source_id", "name", "game_version"):
            for value in ("", "  ", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.source, **{field: value})

    def test_requires_absolute_http_url_without_credentials(self):
        for url in ("", "   ", None, "example.invalid", "ftp://example.invalid",
                    "https:///path", "https://user:password@example.invalid",
                    "https://example.invalid:bad", "https://example.invalid/a b"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    replace(self.source, url=url)
        self.assertEqual(replace(self.source, url="http://example.invalid/page").url,
                         "http://example.invalid/page")

    def test_rejects_unknown_permission_status_and_non_boolean_enabled(self):
        for changes in ({"permission_status": "automatic"}, {"enabled": "false"},
                        {"permission_status": "approved", "permission_reference": " "}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.source, **changes)

    def test_source_cannot_be_modified_in_place(self):
        with self.assertRaises(FrozenInstanceError):
            self.source.enabled = True


class EvidenceTests(unittest.TestCase):
    # Require traceability, explicit verification, and matching game context.
    def setUp(self):
        self.timestamp = datetime(2026, 10, 21, tzinfo=timezone.utc)
        self.source = Source(
            source_id="sample-source", name="Fictional test source",
            url="https://example.invalid/source", game_version="forever",
            permission_status="approved", enabled=True,
            permission_reference="Fictional permission record for tests",
        )
        self.evidence = Evidence(
            evidence_id="sample-evidence", source=self.source,
            url="https://example.invalid/source/record",
            content="Fictional text for testing; not a game fact.",
            game_version="forever", game_build="test-build-1",
            retrieved_at=self.timestamp, status="verified_for_forever",
            verified_at=self.timestamp,
        )

    def test_only_matching_version_and_build_are_eligible(self):
        self.assertTrue(self.evidence.is_usable_for("forever", "test-build-1"))
        self.assertFalse(self.evidence.is_usable_for("classic", "test-build-1"))
        self.assertFalse(self.evidence.is_usable_for("forever", "test-build-2"))

    def test_unapproved_or_disabled_source_is_not_eligible(self):
        for status in ("pending_review", "rejected", "approved"):
            with self.subTest(status=status):
                source = replace(self.source, permission_status=status, enabled=False)
                evidence = replace(self.evidence, source=source)
                self.assertFalse(evidence.is_usable_for("forever", "test-build-1"))

    def test_non_verified_states_are_not_eligible(self):
        for status in ("unknown", "unverified_for_forever", "superseded", "withheld"):
            with self.subTest(status=status):
                evidence = replace(self.evidence, status=status, verified_at=None)
                self.assertFalse(evidence.is_usable_for("forever", "test-build-1"))

    def test_unknown_build_is_recordable_but_not_verified(self):
        pending = replace(self.evidence, status="unknown", game_build=None, verified_at=None)
        self.assertIsNone(pending.game_build)
        self.assertFalse(pending.is_usable_for("forever", "test-build-1"))
        for build in (None, "", " ", "unknown"):
            with self.subTest(build=build):
                with self.assertRaises(ValueError):
                    replace(self.evidence, game_build=build)

    def test_verification_requires_date_and_forever_scope(self):
        with self.assertRaises(ValueError):
            replace(self.evidence, verified_at=None)
        classic_source = replace(self.source, game_version="classic")
        with self.assertRaises(ValueError):
            replace(self.evidence, source=classic_source, game_version="classic")
        classic = replace(
            self.evidence, source=classic_source, game_version="classic", status="unknown",
        )
        self.assertFalse(classic.is_usable_for("classic", "test-build-1"))

    def test_requires_source_and_matching_scope(self):
        for source in (None, "sample-source"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    replace(self.evidence, source=source)
        with self.assertRaises(ValueError):
            replace(self.evidence, game_version="classic")

    def test_rejects_missing_evidence_identity_content_or_scope(self):
        for field in ("evidence_id", "content", "game_version"):
            for value in ("", " ", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.evidence, **{field: value})

    def test_requires_citable_url_and_known_status(self):
        with self.assertRaises(ValueError):
            replace(self.evidence, url="relative/path")
        with self.assertRaises(ValueError):
            replace(self.evidence, status="probably_true")

    def test_timestamps_must_include_timezone(self):
        for field in ("retrieved_at", "verified_at"):
            for value in ("2026-10-21", datetime(2026, 10, 21)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.evidence, **{field: value})
        with self.assertRaises(ValueError):
            replace(self.evidence, retrieved_at=None)

    def test_evidence_cannot_be_modified_in_place(self):
        with self.assertRaises(FrozenInstanceError):
            self.evidence.status = "unknown"


if __name__ == "__main__":
    unittest.main()
