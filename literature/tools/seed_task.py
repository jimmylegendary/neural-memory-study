#!/usr/bin/env python3
"""Generate normalized observations from preserved bibliographic inputs.

Only public bibliographic metadata is copied. Original private report body and
project implementation details are deliberately excluded. Historical values are
claims of the source record, not automatically verified metadata.
"""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'literature/inputs'
REPORT = ROOT / 'deliverables/neural-memory/reports/2026-10-06-sleep-time-compute/source/sources/references.json'
rows = []

def add(source, source_date, title, alias='', published='', updated='', arxiv='', url='', topics='', kind='paper', note='', canonical=''):
    rows.append(dict(source_id=source, source_date=source_date, title=title,
                     alias=alias, reported_published=published, reported_updated=updated,
                     reported_arxiv=arxiv, reported_url=url, topics=topics,
                     kind=kind, note=note, canonical_arxiv=canonical))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    early = (OUT / 'early_bibliography.txt').read_text(encoding='utf-8')
    parts = re.findall(r'^\[(\d+)\]\s+(.*)$', early, re.M)
    if len(parts) != 43:
        raise SystemExit('Expected 43 early bibliography entries')
    for num, text in parts:
        a = re.search(r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)', text)
        urls = re.findall(r'https?://\S+', text)
        date = re.search(r'\((\d{4}(?:-\d{2}(?:-\d{2})?)?)\)', text)
        title = text[:a.start()].strip().rstrip('.') if a else text.split('https://')[0].strip().rstrip('.')
        note = text.split(' — ', 1)[1] if ' — ' in text else ''
        kind = 'paper' if a else ('community' if 'Reddit' in text else 'official')
        add('initial-report-2026-08-16', '2026-08-16', title,
            published=date.group(1) if date else '', arxiv=a.group(1) if a else '',
            url=urls[0] if urls else '', kind=kind, note=note)
        if int(num) <= 6:
            add('initial-outlook-2026-08-11', '2026-08-11', title,
                arxiv=a.group(1), url=urls[0],
                note='8월11일 최초 회수 PDF의 참고문헌; 발행일은 이 행에서 추정하지 않음.')

    snapshot = OUT / 'prior_report_references.json'
    if not snapshot.exists():
        snapshot.write_bytes(REPORT.read_bytes())
    refs = json.loads(snapshot.read_text(encoding='utf-8'))
    if len(refs) != 55:
        raise SystemExit('Expected original report snapshot of 55 references')
    for r in refs:
        add('report-2026-10-06', '2026-10-06', r['title'], r['key'], r['date'],
            r['updated'], r['arxiv'] or '', r['url'], '', r['type'], r['note'])

    weekly = (OUT / 'weekly_observations.psv').read_text(encoding='utf-8')
    week_lines = [x for x in weekly.splitlines() if x.strip()]
    if len(week_lines) != 105:
        raise SystemExit('Expected 105 individually enumerated weekly entries')
    for line in week_lines:
        sd, pd, title, alias, arxiv, tags = line.split('|')
        add('weekly-' + sd, sd, title, alias, pd, arxiv=arxiv,
            url='https://arxiv.org/abs/' + arxiv if arxiv else '', topics=tags,
            note='주간 요약의 논문 항목. 날짜/제목/식별자는 당시 주장으로 보존하며 원문 재검증 상태와 분리.')

    add('precursor-2026-08-07', '2026-08-07', 'FlashAccel', 'FlashAccel',
        arxiv='2607.10186', url='https://arxiv.org/abs/2607.10186', topics='hbf,systems',
        note='회수된 선행 HOPE/HBF 대화의 참조. 이 날짜를 정기 task 생성일로 단정하지 않음.')
    oldids = {'Hindsight Memory-PRM':'2608.31254', 'Invalidation Contracts':'2609.00243',
              'PlanFence':'2609.03340', 'MemForest':'2609.07997', 'MeClear':'2609.07964',
              'HBFSim':'2609.09902', 'UNISON':'2609.09643'}
    for alias, arxiv in oldids.items():
        candidates = [r for r in rows if r['alias'] == alias and r['source_id'].startswith('weekly')]
        title = candidates[0]['title'] if candidates else alias
        add('legacy-index-2026-10-06', '2026-10-06', title, alias, arxiv=arxiv,
            url='https://arxiv.org/abs/' + arxiv,
            note='이전 Markdown 색인의 식별자 주장; 정정판과 대조할 원본값.')
    updates = {'RoMeRL':'2026-08-10 v2', 'RPMem':'2026-09-22 v2', 'HBFSim':'',
               'HBFlex':'', 'PMPA':'2026-09-15', 'DRSR':'2026-09-24',
               'KV-streams':'2026-09-29 v2', 'EfficientAgent':'2026-09-29',
               'ActiveSaddler':'2026-10-02', 'Trillion-Parameter MoE in a Box':'2026-09-15'}
    for r in rows:
        if r['source_id'].startswith('weekly') and r['alias'] in updates:
            r['reported_updated'] = updates[r['alias']]
    for idx, row in enumerate(rows, 1):
        row['observation_id'] = f'obs-{idx:04d}'
    if len(rows) != 217:
        raise SystemExit('Observation census changed unexpectedly')
    columns = ['observation_id'] + [k for k in rows[0] if k != 'observation_id']
    with (OUT / 'task_bibliography.tsv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    sources = []
    for sid in dict.fromkeys(r['source_id'] for r in rows):
        rr = [r for r in rows if r['source_id'] == sid]
        sources.append(dict(source_id=sid, source_date=rr[0]['source_date'],
                            kind='task_history' if 'weekly' in sid or 'precursor' in sid else 'bibliography',
                            extracted_records=len(rr),
                            extraction='Bibliographic entries only; no private project body republished.'))
    manifest = dict(snapshot_date='2026-10-12', earliest_recovered_task_document='2026-08-11',
                    precursor_reference='2026-08-07', sources=sources)
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Preserved {len(rows)} observations from {len(sources)} sources')

if __name__ == '__main__':
    main()
