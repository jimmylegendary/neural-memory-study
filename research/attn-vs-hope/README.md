# Attn vs HOPE native live-Sheet archive

This package preserves the user-reviewed operational model exported from the native one-tab Google Sheet. It is a read-only reference snapshot; do not overwrite it with the generated E5 workbook.

## Source and export identity

| Field | Value |
|---|---|
| Google Sheet ID | `1BZLzsGgdE43GXAMtuhrg7btPRq94crQWjo5mb0VWNWk` |
| Source | <https://docs.google.com/spreadsheets/d/1BZLzsGgdE43GXAMtuhrg7btPRq94crQWjo5mb0VWNWk> |
| Source updated | `2026-07-27T01:45:55.568Z` |
| Exported | `2026-07-27T10:46:51.477084070+09:00` |
| Size | `29,321` bytes |
| SHA-256 | `0c04b122a214b9e2543abf6925260f307ddbdda7bc9cedca7e3d1f92bb7b8267` |

Files: [native XLSX](Attn-vs-HOPE.xlsx) · [export manifest](EXPORT-MANIFEST.json) · [standard-library verifier](verify_snapshot.py)

## Snapshot contract

- One visible worksheet, `시트1`, with serialized range `A1:P89`.
- 38 defined names, 872 formula cells, and 117 Korean notes/comments.
- Zero cached formula errors and zero external workbook links in the portable XLSX package.
- The separate live-Sheet validation recorded 872 formulas and zero effective formula errors at the source update above. Portable OOXML inspection checks cached values but does not recalculate Excel formulas.
- Corrected contracts: current `H1` note; CMS-level bound gates in `P24:P33` and `P65:P74`; Titans/update bounds independent of CMS in `P41:P46` and `P82:P87`.

Key cached scenario outputs, rounded to three decimals:

| Scenario | Cell | ms |
|---|---:|---:|
| Full Attention TTFT | `O18` | 3204.916 |
| HOPE forward TTFT | `O46` | 552.123 |
| HOPE including online work TTFT | `O48` | 6757.768 |
| Full Attention ITL | `O59` | 10.888 |
| HOPE forward ITL | `O87` | 0.604 |
| HOPE including online work ITL | `O89` | 18.890 |

## Verify

From the repository root:

```bash
python3 research/attn-vs-hope/verify_snapshot.py --self-test
python3 research/attn-vs-hope/verify_snapshot.py
```

Both commands must print the manifest checksum. The self-test first proves that an in-memory checksum corruption and an injected external-workbook relationship are rejected, then validates the real XLSX.

## Artifact boundary

- [Native one-tab XLSX](Attn-vs-HOPE.xlsx): user-reviewed operational model and reference snapshot.
- [Generated 11-tab XLSX](../../experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx): reproducible analytical companion built from the E5 engine; it must not replace this snapshot.
- [E5 engine and reproduction guide](../../experiments/E5-attn-vs-hope/README.md).
- [Google meeting package](../google-meeting/README.md), which uses both artifacts as distinct evidence sources.
