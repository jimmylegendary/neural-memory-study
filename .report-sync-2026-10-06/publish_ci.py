"""Reconstruct the approved report; abort unless the PDF is byte-identical.
Only writes the dated report directory and appends a link to the root README.
No credentials are read by this script. Git publication is a separate CI step.
"""
from pathlib import Path
import csv, hashlib, json, shutil, subprocess, sys, zipfile
import fitz
from PIL import Image, ImageDraw, ImageFont

REPO = Path.cwd()
PAYLOAD = REPO / '.report-sync-2026-10-06'
REL = Path('deliverables/neural-memory/reports/2026-10-06-sleep-time-compute')
OUT = REPO / REL
SRC = OUT / 'source'
PDF_NAME = 'Neural_Memory_Report_2026-10-06_KO.pdf'
PREVIEW_NAME = 'Neural_Memory_Report_Preview.jpg'
ZIP_NAME = 'Neural_Memory_Report_Source_Package.zip'
EXPECTED_PDF = '249d1748b51d8d795de1e395f60b7d0ca6cf9a20e436c2b8207c1015a656b27f'
HASHES = {
 'build/build_report.py':'67f951a179886f140f8b113ee9a3f5364a76904fb7ac72465fc43a40506d62bb',
 'build/report_engine.py':'98c2523933605036cd719e62688c019fd6afe3698610156295aeb1edff46c955',
 'build/report_style.css':'e02d1c6527d5063c6e80e51044646194554624d99e8e6af057cbf75437328b43',
 'sources/references.json':'ab1cc10afb96f35c7b76898c9da7c2e60382da2c18f34e41de976aabd76a8403',
 'SOURCE_AUDIT.md':'7c81029e62e0fa3f65b194e278e42bcfb297b28b3dd0ac5ad38219aec7fa4816',
 'README.md':'c227d236030058cc3a0d01bb25daff132820b2359aac25edcefd6a8d93347d8d'
}
def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def check(path, expected):
 actual = sha(path)
 if actual != expected:
  raise RuntimeError(f'Integrity mismatch: {path.name}: {actual} != {expected}')
 print('VERIFIED',path.relative_to(REPO),actual,flush=True)

def write_csv(path, heads, rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f); writer.writerow(heads); writer.writerows(rows)

if OUT.exists():
 raise RuntimeError('Publication directory already exists; refusing to overwrite')
SRC.mkdir(parents=True)
for name,expected in HASHES.items():
 parts=sorted((PAYLOAD/'chunks').glob(name.replace('/','__')+'.part*'))
 if not parts: raise RuntimeError('Missing payload: '+name)
 raw=b''.join(p.read_bytes() for p in parts)
 if name=='build/build_report.py' and hashlib.sha256(raw).hexdigest()!=expected:
  # Repair one transport transcription; the complete approved-source hash is mandatory.
  raw=raw.replace(b"sw=1.7,dash='5 5'",b"sw=1.7,dash='5 4'")
 target=SRC/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 check(target,expected)

(SRC/'requirements.txt').write_text('weasyprint==68.0\nPyMuPDF==1.26.7\nPillow==12.3.0\n',encoding='utf-8')
refs=json.loads((SRC/'sources/references.json').read_text())
with (SRC/'sources/references.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(refs[0]));writer.writeheader();writer.writerows(refs)
check(SRC/'sources/references.csv','ccfc2bbc773f4565991066e569c08adc588d9b1c282e36c1853ba20a369c763e')
rows=[]
for cat,val in [('ALFWorld',16.2),('WebShop',16.3),('tau2-bench',3.9)]:
 rows.append(['JitMem gains',14,'gain over strongest reported baseline',cat,val,'percentage_points','Author reported; not independent reproduction; different benchmarks'])
for cat,val in [('Lost ACK',0.5),('Still in-flight',56),('Redelivery',74)]:
 rows.append(['LIMBO duplicate writes',26,'frontier-model duplicate write rate',cat,val,'percent','Author reported; three fault conditions; not inference latency'])
write_csv(SRC/'sources/chart_data.csv',['figure','source_id','series','category','value','unit','interpretation'],rows)
check(SRC/'sources/chart_data.csv','7a114f602ab5d8e50d9b1b39ad8b91194300056c1ad7da02bddbd8716a23b31a')
write_csv(SRC/'sources/illustrative_cost_data.csv',['N_reuses','saved_query_cost','build_refresh_cost','kind'],[[n,2*n,100,'ILLUSTRATIVE ONLY — NOT BENCHMARK DATA'] for n in range(0,101,10)])
check(SRC/'sources/illustrative_cost_data.csv','769a369ef64f7e0d7b29cfe97a46865927561b9e65fc9f7aaf4e74649bde0639')
subprocess.run([sys.executable,'build_report.py'],cwd=SRC/'build',check=True)
pdf=SRC/'deliverables'/PDF_NAME
check(pdf,EXPECTED_PDF)
doc=fitz.open(pdf)
if len(doc)!=49: raise RuntimeError(f'Unexpected page count: {len(doc)}')
media=json.loads((SRC/'sources/media_manifest.json').read_text());tables=json.loads((SRC/'sources/table_manifest.json').read_text())
if (len(media),len(tables),len(refs)) != (31,21,55):raise RuntimeError('Unexpected source/media counts')
index='# Figure / Table index\n\n## Figures\n\n| 번호 | PDF 쪽 | 형식 | 설명 | 원본 |\n|---|---:|---|---|---|\n'
for m in media:index+=f"| {m['number']:02d} | {m['page']} | {m['type']} | {m['title']} | [SVG](assets/{m['file']}) |\n"
index+='\n## Tables\n\n| 번호 | PDF 쪽 | 제목 |\n|---|---:|---|\n'
for t in tables:index+=f"| {t['number']:02d} | {t['page']} | {t['title']} |\n"
(SRC/'FIGURE_TABLE_INDEX.md').write_text(index,encoding='utf-8')
check(SRC/'FIGURE_TABLE_INDEX.md','9c0c1db33799b08b772f9d60b2c124c7a0f605e1474a2bae8266714420a22764')
(OUT/'FIGURE_TABLE_INDEX.md').write_text(index.replace('](assets/','](source/assets/'),encoding='utf-8')
# A newly rendered preview, not represented as the original JPEG bytes.
selected=[0,7,11,26]
w,h,gap,caption=650,920,24,34
sheet=Image.new('RGB',(w*2+gap*3,(h+caption)*2+gap*3),'#e6edf1')
draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',20)
for i,n in enumerate(selected):
 pix=doc[n].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
 image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);image.thumbnail((w,h))
 x=gap+(i%2)*(w+gap)+(w-image.width)//2;y=gap+(i//2)*(h+caption+gap)
 sheet.paste(image,(x,y));draw.text((gap+(i%2)*(w+gap),y+h+7),f'PDF p.{n+1:02d}',font=font,fill='#23465a')
preview=SRC/'deliverables'/PREVIEW_NAME;sheet.save(preview,quality=92,subsampling=0)
for name in [PDF_NAME,PREVIEW_NAME]:shutil.copyfile(SRC/'deliverables'/name,OUT/name)
shutil.copyfile(SRC/'SOURCE_AUDIT.md',OUT/'SOURCE_AUDIT.md')
# Keep original executable source; exclude caches and font binaries.
for cache in SRC.rglob('__pycache__'):shutil.rmtree(cache)
(SRC/'build/layout.json').unlink(missing_ok=True)
record={
 'publication_date':'2026-10-06','repository':'jimmylegendary/neural-memory-study','destination':REL.as_posix(),
 'original_pdf_sha256':EXPECTED_PDF,'published_pdf_sha256':sha(OUT/PDF_NAME),'pdf_byte_identical':True,
 'pages':49,'numbered_figures':31,'tables':21,'reference_records':55,
 'preview_regenerated':True,'source_zip_repacked':True,
 'preserved_source_sha256':HASHES,
 'verification_scope':'File integrity and deterministic rendering, not new literature validation or experimental replication.',
 'original_preview_sha256':'4b52ef699b2be9f988be097fc2c1cd8631b8456a5f4d092b5b42ce452c66f547',
 'original_source_zip_sha256':'a64fa13c2e08f9997dbfa834ff14d6c4a23fec8b80b60da0408b91292fe8bb00'
}
(OUT/'PUBLICATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'README.md').write_text(f'''# Sleep-time Compute & Neural Memory — 2026-10-06

49쪽 한국어 PDF와 재현 가능한 편집 원본을 보존하는 별도 연구 리포트다.
기존 Neural Memory 모노그래프 및 Sleep-Time Compute v2를 대체하지 않는다.

| 자료 | 링크 |
|---|---|
| 한국어 PDF | [리포트]({PDF_NAME}) |
| 대표 페이지 미리보기 | [JPG]({PREVIEW_NAME}) |
| 편집 원본 전체 ZIP | [소스 패키지]({ZIP_NAME}) |
| 편집 가능한 소스 | [source/](source/) |
| 그림·표 색인 | [FIGURE_TABLE_INDEX.md](FIGURE_TABLE_INDEX.md) |
| 제작 단계의 출처 대조 기록 | [SOURCE_AUDIT.md](SOURCE_AUDIT.md) |
| 게시·무결성 기록 | [PUBLICATION.json](PUBLICATION.json) · [manifest.json](manifest.json) · [SHA256SUMS](SHA256SUMS) |

![대표 페이지]({PREVIEW_NAME})

## 보존 및 검증

PDF는 앞서 완성한 파일을 동일 소스로 재빌드했으며 SHA-256이 원본과 정확히 일치한다.
미리보기는 그 PDF에서 새로 렌더링했고, ZIP은 보존한 편집 소스와 생성 산출물을 다시 묶었다.
따라서 PDF의 동일성은 검증했지만 JPG·ZIP이 이전 파일과 byte-identical하다고 주장하지 않는다.
원본 build code, CSS, reference JSON/CSV, chart data와 source audit은 SHA-256으로 대조했다.

이번 동기화는 문헌 재조사나 실험 재현이 아니다. `SOURCE_AUDIT.md`의 검토 기록은 원본 제작 단계의 기록이다.
PDF 내부의 제작 당시 표기와 연구 내용은 그대로 보존했다. 현재 정식 보관 위치는 이 저장소다.
글꼴 파일·원문 논문 PDF는 이 신규 패키지에 재배포하지 않는다.

```bash
sha256sum -c SHA256SUMS
```

재편집 방법과 근거의 한계는 [source/README.md](source/README.md)에 있다.
''',encoding='utf-8')
with zipfile.ZipFile(OUT/ZIP_NAME,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for file in sorted(SRC.rglob('*')):
  if not file.is_file():continue
  if file.suffix.lower() in {'.ttf','.otf','.woff','.woff2'}:raise RuntimeError('Font redistribution prohibited')
  info=zipfile.ZipInfo('neural-memory-report-v2/'+file.relative_to(SRC).as_posix(),date_time=(2026,10,6,0,0,0))
  info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
  z.writestr(info,file.read_bytes())
files=[]
for file in sorted(OUT.rglob('*')):
 if file.is_file():files.append({'path':file.relative_to(OUT).as_posix(),'bytes':file.stat().st_size,'sha256':sha(file)})
manifest={'report_date':'2026-10-06','pdf_byte_identical':True,'files':files}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files.append({'path':'manifest.json','bytes':(OUT/'manifest.json').stat().st_size,'sha256':sha(OUT/'manifest.json')})
(OUT/'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['path']}\n" for x in files),encoding='utf-8')
subprocess.run(['sha256sum','-c','SHA256SUMS'],cwd=OUT,check=True)
readme=REPO/'README.md'
link=REL.as_posix()+'/README.md'
text=readme.read_text(encoding='utf-8')
if link not in text:
 text+=f'\n\n## 2026-10-06 동향 통합 리포트\n\n[Sleep-time Compute & Neural Memory — 49쪽 한국어 PDF 및 편집 원본]({link})\n\n기존 학습서·Sleep-Time Compute v2와 구분되는 별도 동향/설계 분석 보고서다. PDF·도식·출처·재현 소스를 함께 보존한다.\n'
 readme.write_text(text,encoding='utf-8')
print('PUBLICATION_READY',json.dumps({'pdf_sha256':EXPECTED_PDF,'pages':len(doc),'files':len(files)},ensure_ascii=False),flush=True)
