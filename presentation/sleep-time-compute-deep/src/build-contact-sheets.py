#!/usr/bin/env python3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path.cwd()
SLIDE_DIR = REPO / "presentation/sleep-time-compute-deep/build/slides"
OUT_DIR = REPO / "presentation/sleep-time-compute-deep/build/contact-sheets"
THUMB = (320, 180)
LABEL_H = 24
COLS = 4
ROWS = 4


def main() -> None:
    slides = sorted(SLIDE_DIR.glob("slide-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
    if len(slides) != 112:
        raise SystemExit(f"expected 112 rendered slides, found {len(slides)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    for sheet_index in range(0, len(slides), COLS * ROWS):
        group = slides[sheet_index : sheet_index + COLS * ROWS]
        canvas = Image.new("RGB", (COLS * THUMB[0], ROWS * (THUMB[1] + LABEL_H)), "#d8d8d8")
        draw = ImageDraw.Draw(canvas)
        for local_index, slide_path in enumerate(group):
            row, col = divmod(local_index, COLS)
            x = col * THUMB[0]
            y = row * (THUMB[1] + LABEL_H)
            image = Image.open(slide_path).convert("RGB").resize(THUMB, Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            slide_number = sheet_index + local_index + 1
            draw.rectangle((x, y + THUMB[1], x + THUMB[0], y + THUMB[1] + LABEL_H), fill="#f4f1ea")
            draw.text((x + 8, y + THUMB[1] + 4), f"SLIDE {slide_number:03d}", fill="#222222", font=font)
        start = sheet_index + 1
        end = sheet_index + len(group)
        canvas.save(OUT_DIR / f"contact-{start:03d}-{end:03d}.png")
    print(f"contact-sheets={len(list(OUT_DIR.glob('contact-*.png')))} slides={len(slides)}")


if __name__ == "__main__":
    main()
