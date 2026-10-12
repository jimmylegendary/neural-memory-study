#!/usr/bin/env python3
"""Final metadata audit migration; preserve all original observations.

HMA metadata: Oxford University Research Archive, reviewed 2026-10-12.
Orabona title change: current arXiv page explicitly states Changed title.
Two source-labelled technical blogs are not counted as papers.
"""
from pathlib import Path
import json
L=Path(__file__).resolve().parents[1]
p=L/'tools/build_catalog.py'
s=p.read_text(encoding='utf-8')
old="is_match=m.get('status')=='ok' and (not informative or any(same_title(t,m.get('title','')) for t in title_inputs) or alias_in_abstract)"
new="""accepted_old_titles={norm(t) for t in resolutions.get('historical_title_aliases',{}).get(a,[])}
        explicit_title_history=bool(accepted_old_titles.intersection(norm(t) for t in title_inputs))
        is_match=m.get('status')=='ok' and (not informative or any(same_title(t,m.get('title','')) for t in title_inputs) or alias_in_abstract or explicit_title_history)"""
if old in s:s=s.replace(old,new,1)
elif 'explicit_title_history' not in s:raise SystemExit('Cannot locate audited identity rule')
old="corrections=resolve_rows(imp.rows,resolutions,refs)"
new="""for row in imp.rows:
        assigned=resolutions.get('type_overrides',{}).get(norm(row['title']))
        if assigned:
            row['original_kind']=row['kind']
            row['kind']=assigned
    corrections=resolve_rows(imp.rows,resolutions,refs)"""
if old in s and 'type_overrides' not in s:s=s.replace(old,new,1)
s=s.replace("VERSION = '1.1.0'", "VERSION = '1.2.0'")
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
rpath=L/'inputs/resolutions.json';r=json.loads(rpath.read_text(encoding='utf-8'))
title='Hardware-Managed Heterogeneous High-Bandwidth Memory and Flash in LLM Inference Systems'
doi='10.1109/LCA.2026.3723326'
url='https://ora.ox.ac.uk/objects/uuid%3Aa9f80041-ff2e-4714-9bc4-51c513604d36'
r.setdefault('dois',{})[title]=doi
r.setdefault('publisher_metadata',{})['doi:'+doi.lower()]={
 'title':title,'doi':doi,'authors':['H. Atassi','N. Zilberman','A. Awad'],
 'publication_date':'2026-08-12','date_precision':'day','date_basis':'journal_publication',
 'url':url,'checked_at':'2026-10-12',
 'publisher_evidence':{'source_url':url,'journal':'IEEE Computer Architecture Letters','publication_status':'Published','peer_review_status':'Peer reviewed','acceptance_date':'2026-08-04','repository_deposit_date':'2026-08-11','date_caveat':'Journal publication date is not substituted for the separate archive deposit date.','review_scope':'bibliographic metadata only; no reproduction of results'}
}
r.setdefault('historical_title_aliases',{})['1912.13213']=['A Modern Introduction to Online Learning']
r.setdefault('resolution_evidence',{})['1912.13213-title-history']={
 'source_url':'https://arxiv.org/abs/1912.13213','checked_at':'2026-10-12',
 'current_title':'Online Learning: A Modern Introduction Using Convex Optimization',
 'old_title':'A Modern Introduction to Online Learning',
 'evidence':'Same author and persistent arXiv identifier; v10 comment explicitly states Changed title. v1 submitted 2019-12-31; current v10 revised 2026-06-21. Historical observations retain the old title.'
}
r.setdefault('type_overrides',{}).update({
 'flashdecodingforlongcontextinference':'official',
 'muonanoptimizerforthehiddenlayersofneuralnetworks':'official'
})
rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
readme=L/'README.md';text=readme.read_text(encoding='utf-8')
if 'TASK_CHRONOLOGICAL.md' not in text:
    text=text.replace('| 발행일순 목록 | [CHRONOLOGICAL.md](CHRONOLOGICAL.md) |','| 전체 발행일순 목록 | [CHRONOLOGICAL.md](CHRONOLOGICAL.md) |\n| 이 task의 논문만 | [TASK_CHRONOLOGICAL.md](TASK_CHRONOLOGICAL.md) |')
    readme.write_text(text,encoding='utf-8')
print('v1.2: HMA publisher DOI verified, title history preserved, two technical blogs separated')
