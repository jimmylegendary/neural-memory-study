# Sleep-Time Compute 쉬운 설명본

학회 제출 수준의 `SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf`를 training 비전공자가 section별로 따라 읽도록 12권으로 다시 구성한 companion이다. 새로운 주장이나 수치를 만들지 않고, 원 Study의 claim ID와 저자 작성 figure ID를 그대로 사용한다. Training 용어는 `TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf`의 named destination으로 연결된다.

## 읽기 경로

- 빠른 의사결정: E01 정의 → E02 문제 → E08 promisingness → E10 infra → E11 device
- 연구 설계: E03 계보 → E04 대안 → E05 mechanism → E06 정면 비교 → E09 scaling → E12 falsifier
- 전체 통합본: [`SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf`](../../build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf)

각 booklet은 질문, 세 문장, 직관, 예·반례, canonical figure, strongest alternative, unknown, 원문·Background link 순서를 지킨다. `manifest.json`이 Study section, claim, figure, Background concept의 정확한 mapping을 고정한다.

## 재현

```bash
pytest -q easy/sleep-time-compute/tests
python3 easy/sleep-time-compute/build.py --all --combined
python3 easy/sleep-time-compute/qa.py --all --combined --strict --render-all-pages
python3 build/overflow_gate.py build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf
sha256sum -c easy/sleep-time-compute/reports/SHA256SUMS
```

최종 QA는 12개 individual PDF와 1개 combined PDF의 manifest consistency, unresolved placeholder, source figure 권리, missing glyph, PDF link annotation, 전 페이지 right-margin overflow를 검사한다. 결과는 [`qa-report.json`](reports/qa-report.json), build hash와 page count는 [`build-manifest.json`](reports/build-manifest.json)과 [`SHA256SUMS`](reports/SHA256SUMS)에 기록된다.
