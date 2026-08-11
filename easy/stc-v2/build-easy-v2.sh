#!/usr/bin/env bash
# Sleep-Time Compute v2 쉬운 설명본 12권을 PDF로 빌드한다.
# Neural Memory 판(easy/README.md)의 레시피를 그대로 쓴다 — 템플릿·필터 공용.
# 게이트 둘: build/overflow_gate.py PASS + Missing character 0.
set -u
cd /home/jimmy/repos/neural-memory-study || exit 1

OUT=easy/pdf-stc-v2
mkdir -p "$OUT"
ok=0; fail=0; gate_fail=0; glyph_fail=0

for md in easy/stc-v2/E*.md; do
  id=$(basename "$md" .md)
  pdf="$OUT/$id.pdf"
  err="$OUT/$id.err.txt"
  printf '%-34s ' "$id"

  # breakable.lua 필수 — 본문 빌드가 쓰는 필터다. 없으면 중점·슬래시로 이어진
  # arXiv ID·식별자 런이 줄바꿈 없이 오른쪽 여백을 넘는다(build/METHOD.md §1).
  pandoc "$md" -o "$pdf" \
    --template=easy/build/template-easy.tex \
    --lua-filter=build/breakable.lua \
    --lua-filter=easy/build/callouts.lua \
    --lua-filter=easy/build/mathfit.lua \
    --pdf-engine=lualatex \
    --resource-path=.:easy:figures \
    2> "$err"

  if [ ! -s "$pdf" ]; then
    echo "BUILD FAIL  (see $err)"; fail=$((fail+1)); continue
  fi

  pages=$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages/{print $2}')
  # grep -c 는 0건일 때 exit 1 이라 || echo 0 을 붙이면 "0\n0" 이 된다. tr -d 로 정규화한다.
  glyphs=$(grep -c 'Missing character' "$err" 2>/dev/null | head -1 | tr -d '[:space:]')
  glyphs=${glyphs:-0}
  gate=$(python3 build/overflow_gate.py "$pdf" 2>&1 | tail -1)

  case "$gate" in
    PASS*) g=PASS ;;
    *)     g=FAIL; gate_fail=$((gate_fail+1)) ;;
  esac
  [ "$glyphs" != "0" ] && glyph_fail=$((glyph_fail+1))

  printf '%3s쪽  glyph=%-3s  overflow=%s\n' "$pages" "$glyphs" "$g"
  ok=$((ok+1))
done

echo
echo "=== 빌드 $ok / 실패 $fail | overflow 실패 $gate_fail | 글리프 잔존 $glyph_fail ==="
[ $fail -eq 0 ] && [ $gate_fail -eq 0 ] && [ $glyph_fail -eq 0 ] && echo "ALL GREEN" || echo "확인 필요"
