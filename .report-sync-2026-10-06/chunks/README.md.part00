# Sleep-time Compute & Neural Memory — Korean publication package

**Version 2.0 · 2026-10-06**

## 결과물

- `deliverables/Neural_Memory_Report_2026-10-06_KO.pdf`: 49쪽 한국어 PDF
- 본문: 31개 numbered vector figure, 21개 표
- 55개 주석형 출처; 클릭 가능한 목차·인용·원문 링크
- 기존 조사 기록의 식별자 및 해석 정정 포함

독립 문헌 종합 및 설계 제안입니다. 학술지 심사·공식 기관 승인·독립 실험 재현을 주장하지 않습니다.

## 편집 구조

```text
build/build_report.py         # 내용과 원본 SVG 도식 작성
build/report_engine.py        # 문단·표·그림 helper
build/report_style.css        # 한국어 A4 편집 스타일
build/report.html             # 생성된 전체 HTML
assets/                      # 31개 SVG + 표지 SVG
sources/references.json       # 날짜·버전·원문·검토범위·핵심주석
sources/references.csv
sources/media_manifest.json
sources/table_manifest.json
sources/chart_data.csv        # 확인된 논문 수치만
sources/illustrative_cost_data.csv # 가상 비용 예시, 별도 격리
SOURCE_AUDIT.md                # 정정/제한사항
FIGURE_TABLE_INDEX.md
```

## 재생성

Python 3.11 이상, WeasyPrint의 system dependency와 Nanum 글꼴이 필요합니다. 글꼴 파일 자체는 이 패키지에 포함하지 않습니다. Linux에서 NanumBarunGothic 및 NanumMyeongjo를 설치하거나 `report_style.css`의 @font-face 경로를 사용 환경에 맞게 바꾸세요.

```bash
pip install -r requirements.txt
cd build
python build_report.py
```

build는 상위 패키지 디렉터리를 기준으로 assets/sources/deliverables를 사용합니다. `make_sources.py`는 서지 초기 작성 기록이므로 이미 수정된 JSON을 덮어쓸 수 있습니다. 일반적인 편집에서는 references.json을 수정하고 build_report.py만 실행하세요.

수치를 바꾸면 source registry, figure caption, chart_data.csv를 함께 갱신하세요. 수정 후 PDF를 실제 이미지로 렌더링하여 overflow·겹침·글꼴·목차를 점검해야 합니다.

## 근거와 권리

원문 figure 파일을 재배포하지 않습니다. 본문 도식은 원문 메커니즘을 독립적으로 설명한 SVG 또는 본 보고서의 설계 가설입니다. 원저작물의 권리는 각 저자/발행처에 있습니다. 연구 수치는 저자 보고값이며 독립 재현하지 않았습니다.

이 PDF/package의 생성은 GitHub 기존 Markdown을 자동으로 갱신하지 않습니다.
