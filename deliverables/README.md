# 연구 산출물 배포 패키지

두 연구 프로그램의 reader-facing 파일을 동일한 계층에 분리했다.

- [`neural-memory/`](neural-memory/README.md): Neural Memory artifact 31개
- [`sleep-time-compute/`](sleep-time-compute/README.md): Sleep-Time Compute artifact 46개

이 디렉터리는 배포·열람용 canonical index다. 원본 생성 경로는 유지되며 각 프로젝트의
`manifest.json`이 package file과 source file의 대응 관계를 기록한다.

재생성:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables
```

검증:

```bash
python3 scripts/package_deliverables.py --root . --output deliverables --check
```
