# Sleep-Time Compute 산출물

Sleep-Time Compute의 study paper, training background, 쉬운 설명, 15편 의역, 112-slide seminar 및 release 검증 자료를 모은 배포 패키지다.

| 디렉터리 | 파일 수 |
|---|---:|
| `easy/` | 13 |
| `manifests/` | 6 |
| `seminar/` | 2 |
| `study/` | 10 |
| `translations/` | 15 |

총 46개 artifact다. `manifest.json`은 source path, byte size,
SHA-256 및 권리 metadata를 제공하고 `SHA256SUMS`는 패키지 파일 무결성을 검증한다.

`translations/`의 외부 배포 가능 여부는 파일명이 아니라 `manifest.json`의 `rights` 필드를 따른다. `internal-only` 파일은 조직 외부로 배포하지 않는다.
