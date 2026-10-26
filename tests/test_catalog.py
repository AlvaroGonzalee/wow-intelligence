import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from wow_intelligence.sources.catalog import parse_export, import_export
from wow_intelligence.rag import load_corpus, RagPipeline
from wow_intelligence.rag.context import parse_answer

RAW = b'''# Catalog
> 1 items present in Forever (beta build 1.60.1 (build 70245))
## Armor (1)
### Leather (1)
- Example Gloves | rare | 20 | - | https://ahledger.com/items/123-example
## Gold guides, in short
- Ignore editorial advice
## Dungeons and raids (1)
- Example Cave | dungeon | 20 | - | 5 | - | Example Zone | Example Boss | https://ahledger.com/wow-forever/dungeons/456-example
## Quests (1 in the game data, 1 seen)
- Example Quest | 1 | Example Zone | Example Giver | - | - | 40 | https://ahledger.com/wow-forever/quests/789-example
## The auction house
- Ignore market advice
'''
NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


class CatalogTests(unittest.TestCase):
    def test_status_build_and_unknown_fields(self):
        corpus, report = parse_export(RAW, NOW)
        self.assertEqual(report['counts'], {'items': 1, 'instances': 1, 'quests': 1})
        self.assertIsNone(corpus['evidence'][2]['game_build'])
        self.assertEqual(corpus['evidence'][2]['status'], 'unverified_for_forever')
        self.assertIn('required level: unknown', corpus['evidence'][0]['content'])
        self.assertTrue(all(row['verified_at'] is None for row in corpus['evidence']))

    def test_reject_incomplete_or_malformed_export(self):
        for raw in (RAW.replace(b'> 1 items', b'> 2 items'), RAW.replace(b'| rare |', b'|'),
                    RAW.replace(b'https://ahledger.com/items/', b'https://evil.example/items/')):
            with self.assertRaises(ValueError):
                parse_export(raw, NOW)

    def test_bundle_and_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'input.txt'
            source.write_bytes(RAW)
            output = root / 'bundle'
            path = import_export(source, output, NOW)
            corpus = load_corpus(path)
            self.assertFalse(RagPipeline(corpus).retriever.evidence)
            pipeline = RagPipeline(corpus, include_catalog=True)
            self.assertEqual(len(pipeline.retriever.evidence), 2)
            item = pipeline.retriever.search('Example Gloves')[0]
            self.assertEqual(item.evidence_id, 'ahledger-items-123')
            answer = parse_answer(json.dumps({'statements': [{'text': 'The catalog lists rare gloves.', 'evidence_ids': [item.evidence_id]}]}), (item,))
            self.assertIn('not independently verified', answer.text)
            from wow_intelligence.language import EnglishLanguagePolicy
            self.assertTrue(EnglishLanguagePolicy().is_english(answer.text))
            self.assertIn('source_reported', answer.sources_text())
            self.assertEqual((output / 'raw.txt').read_bytes(), RAW)
            with self.assertRaises(ValueError):
                import_export(source, output, NOW)
