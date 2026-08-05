# 연구 산출물 배포 패키지

reader-facing 파일을 프로그램별로 분리한 canonical index다. 원본 생성 경로는 재현 빌드를 위해
유지되며, 각 프로젝트의 `manifest.json`이 package file과 source file의 대응·byte size·SHA-256·
권리 metadata를 기록한다.

- [`neural-memory/`](neural-memory/README.md) — Neural Memory artifact 31개

Sleep-Time Compute 패키지는 v1을 2026-08-06에 폐기했고 v2 재구축 완료 시점에 다시 생성한다.
사유와 유지 범위는 [저장소 README](../README.md) §Sleep-Time Compute v2 참조.

재생성:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables
```

검증:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables --check
```
