#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/../.." && pwd)"
output_dir="$repo_root/build/publications"
style_path="$script_dir/publication.css"

mkdir -p "$output_dir"

render_document() {
  local source_path="$1"
  local stem="$2"
  local title="$3"
  local subtitle="$4"
  local date_value="$5"

  local html_path="$output_dir/$stem.html"
  local pdf_path="$output_dir/$stem.pdf"
  local docx_path="$output_dir/$stem.docx"

  tail -n +5 "$source_path" |
    pandoc - \
      --from=markdown+tex_math_single_backslash \
      --standalone \
      --toc \
      --toc-depth=2 \
      --mathml \
      --embed-resources \
      --css="$style_path" \
      --metadata "title=$title" \
      --metadata "subtitle=$subtitle" \
      --metadata "date=$date_value" \
      -o "$html_path"

  google-chrome \
    --headless \
    --disable-gpu \
    --no-sandbox \
    --no-pdf-header-footer \
    --print-to-pdf="$pdf_path" \
    "file://$html_path"

  pandoc "$source_path" \
    --from=markdown+tex_math_single_backslash \
    --standalone \
    --toc \
    --toc-depth=2 \
    -o "$docx_path"
}

render_document \
  "$script_dir/SLEEP-TIME-COMPUTE-MONOGRAPH-KR.md" \
  "SLEEP-TIME-COMPUTE-MONOGRAPH-KR" \
  "모델이 잠들 때 무엇이 일어나는가?" \
  "Sleep-Time Compute, 지속학습, 장기기억을 하나의 생애주기로 보는 연구 모노그래프" \
  "2026-07-25"

render_document \
  "$repo_root/paper-en/sleep-time-compute/PAPER.md" \
  "SLEEP-TIME-COMPUTE-POSITION-PAPER-EN" \
  "Sleep-Time Compute as a Governed Memory Lifecycle" \
  "A position and review paper on delayed learning, memory destinations, capacity limits, and wake–sleep infrastructure" \
  "25 July 2026"

(
  cd "$output_dir"
  sha256sum \
    SLEEP-TIME-COMPUTE-MONOGRAPH-KR.html \
    SLEEP-TIME-COMPUTE-MONOGRAPH-KR.pdf \
    SLEEP-TIME-COMPUTE-MONOGRAPH-KR.docx \
    SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.html \
    SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.pdf \
    SLEEP-TIME-COMPUTE-POSITION-PAPER-EN.docx \
    > SHA256SUMS
)
