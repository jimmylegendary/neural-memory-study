#!/usr/bin/env python3
"""Build a labeled 4×7 visual-QA contact sheet from 28 rendered slides."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


EXPECTED_SLIDES = 28
COLUMNS = 4
ROWS = 7
THUMBNAIL_SIZE = (320, 180)
LABEL_HEIGHT = 30
GAP = 16
BACKGROUND = "#E9E5DC"
FRAME = "#B8B1A4"
INK = "#17191C"
SLIDE_PATTERN = re.compile(r"^slide[-_ ]?(\d+)\.png$", re.IGNORECASE)


class ContactSheetError(ValueError):
    """Raised when rendered slide inputs do not satisfy the QA contract."""


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Create a labeled 4x7 contact sheet from slide PNG renders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "build" / "renders",
        help="Directory containing slide-01.png through slide-28.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "build" / "sleep-time-compute-contact-sheet.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def discover_slides(input_directory: Path) -> list[tuple[int, Path]]:
    if not input_directory.is_dir():
        raise ContactSheetError(f"render directory does not exist: {input_directory}")

    indexed: dict[int, Path] = {}
    for candidate in input_directory.iterdir():
        if not candidate.is_file():
            continue
        match = SLIDE_PATTERN.fullmatch(candidate.name)
        if not match:
            continue
        slide_number = int(match.group(1))
        if slide_number in indexed:
            raise ContactSheetError(
                f"duplicate render for slide {slide_number}: "
                f"{indexed[slide_number].name}, {candidate.name}"
            )
        indexed[slide_number] = candidate

    required = set(range(1, EXPECTED_SLIDES + 1))
    actual = set(indexed)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if len(indexed) != EXPECTED_SLIDES or missing or unexpected:
        raise ContactSheetError(
            f"expected exactly 28 slide renders numbered 1..28; "
            f"found {len(indexed)}, missing={missing}, unexpected={unexpected}"
        )
    return sorted(indexed.items())


def load_label_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), 17)
    return ImageFont.load_default()


def build_contact_sheet(slides: list[tuple[int, Path]]) -> Image.Image:
    cell_width = THUMBNAIL_SIZE[0]
    cell_height = THUMBNAIL_SIZE[1] + LABEL_HEIGHT
    canvas_width = GAP + COLUMNS * (cell_width + GAP)
    canvas_height = GAP + ROWS * (cell_height + GAP)
    sheet = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = load_label_font()

    for position, (slide_number, slide_path) in enumerate(slides):
        row, column = divmod(position, COLUMNS)
        x = GAP + column * (cell_width + GAP)
        y = GAP + row * (cell_height + GAP)
        with Image.open(slide_path) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGB"),
                THUMBNAIL_SIZE,
                method=Image.Resampling.LANCZOS,
            )
        frame = Image.new("RGB", THUMBNAIL_SIZE, "white")
        paste_x = (THUMBNAIL_SIZE[0] - thumbnail.width) // 2
        paste_y = (THUMBNAIL_SIZE[1] - thumbnail.height) // 2
        frame.paste(thumbnail, (paste_x, paste_y))
        sheet.paste(frame, (x, y))
        draw.rectangle(
            (x, y, x + THUMBNAIL_SIZE[0] - 1, y + THUMBNAIL_SIZE[1] - 1),
            outline=FRAME,
            width=1,
        )
        label = f"SLIDE {slide_number:02d}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (
                x + (THUMBNAIL_SIZE[0] - label_width) // 2,
                y + THUMBNAIL_SIZE[1] + 5,
            ),
            label,
            fill=INK,
            font=font,
        )
    return sheet


def main() -> int:
    arguments = parse_arguments()
    try:
        slides = discover_slides(arguments.input_dir)
        sheet = build_contact_sheet(slides)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(arguments.output, format="PNG", optimize=True)
    except (ContactSheetError, OSError) as error:
        print(f"contact-sheet error: {error}", file=sys.stderr)
        return 1

    print(
        f"Contact sheet written: {arguments.output} "
        f"({COLUMNS}x{ROWS}, {EXPECTED_SLIDES} slides)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
