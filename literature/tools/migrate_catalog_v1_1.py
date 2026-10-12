#!/usr/bin/env python3
"""One-time, explicit migration following inspection of generated catalogue.

Changes are guarded/idempotent. No source observation is overwritten. This file
is retained as the migration/audit recipe; normal builds use the resulting code.
"""
from pathlib import Path
import json

LIT=Path(__file__).resolve().parents[1]
p=LIT/'tools/build_catalog.py'
s=p.read_text(encoding='utf-8')
old="blocks=re.split(r'\\n\\s*\\n',text)"
new="blocks=re.split(r'\\n\\s*\\n|\\n(?=\\s*[-+]\\s+)',text)"
if old in s:s=s.replace(old,new,1)
elif new not in s:raise SystemExit('Cannot locate bibliography split rule')
old="titled=re.findall(r'(?<!\\*)\\*([^*\\n]{8,})\\*(?!\\*)',block)"
new="titled=[' '.join(t.split()) for t in re.findall(r'(?<!\\*)\\*([^*]{8,})\\*(?!\\*)',block)]"
if old in s:s=s.replace(old,new,1)
elif new not in s:raise SystemExit('Cannot locate bibliography title rule')
# Preserve source-metadata summary so short method aliases absent from the title
# can be checked against the actual primary abstract, not a secondary mirror.
s=s.replace("'title':text('a:title'),'authors'", "'title':text('a:title'),'summary':text('a:summary'),'authors'")
s=s.replace("or cache.get(a,{}).get('method')=='arxiv_search_metadata']", "or cache.get(a,{}).get('method')=='arxiv_search_metadata' or 'summary' not in cache.get(a,{})]")
old="is_match=m.get('status')=='ok' and (not informative or any(same_title(t,m.get('title','')) for t in title_inputs))"
new="""alias_in_abstract=any(len(alias)>=4 and len(alias)<=24 and re.search(r'(?<![\\w-])'+re.escape(alias)+r'(?![\\w-])',m.get('summary',''),re.I) for alias in aliases if alias not in ('Memory','Agent','Sleep','Graph'))
        is_match=m.get('status')=='ok' and (not informative or any(same_title(t,m.get('title','')) for t in title_inputs) or alias_in_abstract)"""
if old in s:s=s.replace(old,new,1)
elif 'alias_in_abstract' not in s:raise SystemExit('Cannot locate identity rule')
old="(lit/'CHRONOLOGICAL.md').write_text(header+'## 이 task에서 회수한 논문\\n\\n'+table(task)+'\\n\\n## 선행 저장소 문헌 (위 task 목록과 중복 제거)\\n\\n'+table([p for p in paper if 'task' not in p['scopes']])+'\\n',encoding='utf-8')"
new="""(lit/'CHRONOLOGICAL.md').write_text(header+'## 전체 논문 — 발행일순\\n\\n'+table(paper)+'\\n',encoding='utf-8')
    (lit/'TASK_CHRONOLOGICAL.md').write_text(header+'## 이 task의 회수 문헌만 — 발행일순\\n\\n'+table(task)+'\\n',encoding='utf-8')"""
if old in s:s=s.replace(old,new,1)
elif 'TASK_CHRONOLOGICAL.md' not in s:raise SystemExit('Cannot locate chronological export')
s=s.replace("VERSION = '1.0.0'", "VERSION = '1.1.0'")
compile(s,str(p),'exec')
p.write_text(s,encoding='utf-8')

rpath=LIT/'inputs/resolutions.json'
r=json.loads(rpath.read_text(encoding='utf-8'))
r['aliases'].update({'PReCache':'2609.34054','AREX-2':'2609.38288','PACE':'2610.01349'})
r.setdefault('resolution_evidence',{}).update({
 'PReCache':{'url':'https://arxiv.org/abs/2609.34054','checked_at':'2026-10-12','basis':'primary arXiv title/abstract; old 2609.35680 resolves to a different field'},
 'AREX-2':{'url':'https://arxiv.org/abs/2609.38288','checked_at':'2026-10-12','basis':'primary arXiv title/abstract; old 2609.36887 is WEFT, a different work'},
 'PACE':{'url':'https://arxiv.org/abs/2610.01349','checked_at':'2026-10-12','basis':'primary arXiv title/abstract; old 2609.39185 is a different work'}
})
rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('v1.1 migration: bullet and multiline parsing, abstract alias checks, global chronology, historical identifier corrections')
