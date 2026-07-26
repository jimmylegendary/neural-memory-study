from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "make-contact-sheet.py"


class ContactSheetTest(unittest.TestCase):
    def test_builds_a_labeled_four_by_seven_sheet_from_28_slides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory)
            renders = work / "renders"
            renders.mkdir()
            output = work / "contact-sheet.png"
            for index in range(1, 29):
                Image.new("RGB", (160, 90), (index * 5, 40, 70)).save(
                    renders / f"slide-{index:02d}.png"
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(renders),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            with Image.open(output) as sheet:
                self.assertGreater(sheet.width, sheet.height * 0.75)
                self.assertLess(sheet.width, sheet.height * 1.25)

    def test_rejects_an_incomplete_slide_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory)
            renders = work / "renders"
            renders.mkdir()
            output = work / "contact-sheet.png"
            Image.new("RGB", (160, 90), "white").save(renders / "slide-01.png")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(renders),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())
            self.assertIn("28", completed.stderr)


if __name__ == "__main__":
    unittest.main()
