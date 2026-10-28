from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

from wow_intelligence.rag.ingestion.ahledger.common import ImportError, _ahledger_url
from wow_intelligence.rag.storage.json_loader import load_knowledge_base
from wow_intelligence.rag.validation.metadata import _require_timestamp


MAX_EXPORT_BYTES = 3 * 1024 * 1024


EXPORT_URL = 'https://ahledger.com/llms-full.txt'


def parse_export(raw: bytes, retrieved_at: datetime) -> tuple[dict, dict]:
    _require_timestamp(retrieved_at, 'retrieved_at')
    if len(raw) > MAX_EXPORT_BYTES:
        raise ImportError('Export exceeds 3 MiB.')
    text = raw.decode('utf-8')
    match = re.search(r'> ([\d,]+) items present in Forever \(beta build (\d+\.\d+\.\d+) \(build (\d+)\)\)', text)
    if not match:
        raise ImportError('Missing supported Forever catalog build header.')
    build = f'{match[2]}.{match[3]}'
    source = dict(source_id='ahledger-export', name='AHledger', url='https://ahledger.com',
                  game_version='forever', permission_status='approved', enabled=True,
                  permission_reference='https://ahledger.com/data ; https://ahledger.com/terms ; offered text export with attribution reviewed 2026-10-21')
    records, counts = [], Counter()
    section, subclass, mode = '', '', 'items'
    for line in text.splitlines():
        if line.startswith('## '):
            section = line[3:]
            if section.startswith('Gold guides'):
                mode = 'skip'
            elif section.startswith('Dungeons and raids ('):
                mode = 'instances'
            elif section.startswith('Quests ('):
                mode = 'quests'
            elif mode != 'items':
                mode = 'skip'
            subclass = ''
        elif line.startswith('### '):
            subclass = line[4:]
        elif line.startswith('- ') and mode != 'skip':
            values = line[2:].split(' | ')
            labels = {
                'items': ('name', 'quality', 'item level', 'required level'),
                'instances': ('name', 'type', 'tuned level', 'entry level', 'players', 'lockout', 'entrance zone', 'listed bosses'),
                'quests': ('title', 'level', 'zone observed', 'giver', 'rewards', 'money', 'experience'),
            }[mode]
            if len(values) != len(labels) + 1:
                raise ImportError(f'Unexpected {mode} row format.')
            url = _ahledger_url(values[-1])
            path = {'items': 'items', 'instances': 'wow-forever/dungeons', 'quests': 'wow-forever/quests'}[mode]
            identifier = re.fullmatch(r'https://ahledger.com/' + path + r'/(\d+)-[^\s]+', url)
            if not identifier:
                raise ImportError('Unexpected catalog record URL.')
            facts = '; '.join(f"{key}: {value if value != '-' else 'unknown'}" for key, value in zip(labels, values))
            if mode == 'items':
                facts += f'; class: {section}; subclass: {subclass}'
            content = f'AHledger {mode} catalog reports {facts}.'
            if mode == 'instances':
                content += ' Boss lists may contain duplicate or variant entries; order and encounter availability are not confirmed.'
            content += ' Missing fields are unknown. Catalog presence does not establish availability, drops or tactics.'
            records.append(dict(evidence_id=f'ahledger-{mode}-{identifier[1]}', source_id=source['source_id'],
                                url=url, content=content, game_version='forever', retrieved_at=retrieved_at.isoformat(),
                                game_build=None if mode == 'quests' else build,
                                status='unverified_for_forever' if mode == 'quests' else 'source_reported', verified_at=None))
            counts[mode] += 1
    if counts['items'] != int(match[1].replace(',', '')):
        raise ImportError('Item count differs from export header.')
    for kind, header in (('instances', r'## Dungeons and raids \((\d+)\)'), ('quests', r'## Quests \([\d,]+ in the game data, ([\d,]+) seen\)')):
        expected = re.search(header, text)
        if not expected or counts[kind] != int(expected[1].replace(',', '')):
            raise ImportError(f'Missing or incomplete {kind} section.')
    report = dict(source_url=EXPORT_URL, retrieved_at=retrieved_at.isoformat(), sha256=hashlib.sha256(raw).hexdigest(),
                  catalog_build=build, counts=dict(counts), attribution='Data from AHledger (https://ahledger.com).',
                  limitations=['Source-reported beta catalog, not independently verified in game.',
                               'Quest observations have no per-record build and remain ineligible.',
                               'No talent mechanics, rotations, encounter tactics or complete recipes.',
                               'No automatic refresh. Unknown fields and boss-list anomalies are preserved.'])
    return dict(schema_version=1, sources=[source], evidence=records), report


def import_export(input_path: Path, output: Path, retrieved_at: datetime) -> Path:
    if output.exists():
        raise ImportError('Output already exists; choose a new snapshot directory.')
    with input_path.open('rb') as stream:
        raw = stream.read(MAX_EXPORT_BYTES + 1)
    corpus, report = parse_export(raw, retrieved_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.catalog-', dir=output.parent) as temporary:
        folder = Path(temporary)
        (folder / 'raw.txt').write_bytes(raw)
        (folder / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
        (folder / 'coverage.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        load_knowledge_base(folder / 'corpus.json')
        if output.exists():
            raise ImportError('Output already exists.')
        os.rename(folder, output)
    return output / 'corpus.json'
