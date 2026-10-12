#!/usr/bin/env python3
"""Fail closed on lost observations, broken foreign keys, or corrupted artifacts."""
import hashlib,json,sqlite3
from pathlib import Path
root=Path(__file__).resolve().parents[2]
lit=root/'literature'
c=json.loads((lit/'coverage.json').read_text())
manifest=json.loads((lit/'inputs/source_manifest.json').read_text())
con=sqlite3.connect(lit/'data/papers.sqlite')
assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
assert con.execute('PRAGMA foreign_key_check').fetchall()==[]
assert c['unmapped_observations']==[]
assert c['observation_count']==c['mapped_observation_count']
assert con.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==c['observation_count']
assert con.execute('SELECT COUNT(*) FROM publications').fetchone()[0]==c['publication_count']
for s in manifest['sources']:
    assert c['source_counts'][s['source_id']]==s['extracted_records'],s['source_id']
if (root/'research/ttt-efficient-corpus.json').exists():
    old=json.loads((root/'research/ttt-efficient-corpus.json').read_text())
    assert c['source_counts']['repository-ttt-corpus']==len(old)
assert c['latest_week_observation_count']==c['latest_week_mapped_count']==17
assert con.execute('SELECT COUNT(*) FROM publications WHERE arxiv_id IS NOT NULL').fetchone()[0]==con.execute('SELECT COUNT(DISTINCT arxiv_id) FROM publications').fetchone()[0]
for line in (lit/'data/SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((lit/'data'/name).read_bytes()).hexdigest()==expected,name
for path, in con.execute('SELECT path FROM assets'):
    assert (root/path).is_file(),path
con.close()
print('PASS: integrity, foreign keys, unique arXiv IDs, all source rows, latest 17 entries, source corpus, asset paths and hashes')
print(json.dumps({k:c[k] for k in ['paper_count','task_paper_count','other_material_count','observation_count','mapped_observation_count','status_counts','pdf_asset_count','correction_count']},ensure_ascii=False,indent=2))
