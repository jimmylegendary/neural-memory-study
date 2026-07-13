#!/usr/bin/env python3
"""Deterministic full-document right-margin overflow gate for the built PDF.

Method note: in a luatexja CJK document, long unbreakable Latin/code runs overflow the right
margin WITHOUT producing an "Overfull \\hbox" warning, so the LaTeX log is not a valid gate.
This rasterizes EVERY page and flags ink in the right-margin band. Sampling pages (as an earlier
QA pass did) misses most of them; this scans all of them. Exit 1 if any page overflows.

Usage: python3 overflow_gate.py [BOOK.pdf]
"""
import subprocess, sys, os, glob, re, tempfile
from PIL import Image
import numpy as np

PDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "BOOK.pdf")
DPI = 150
TEXT_RIGHT_CM = 18.5      # a4 21cm - 2.5cm geometry margin
BAND_START_CM = 18.8      # tolerance past the text edge (ignore glyphs that legitimately touch it)
INK_THRESH = 110          # < this = ink
MIN_INK_PX = 40           # ignore stray antialias specks

def main():
    ppc = DPI / 2.54
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), PDF, os.path.join(tmp, "p")], check=True)
        pages = sorted(glob.glob(os.path.join(tmp, "p-*.png")),
                       key=lambda f: int(re.findall(r"-(\d+)\.png", f)[0]))
        x_band = int(BAND_START_CM * ppc)
        hits = []
        for f in pages:
            n = int(re.findall(r"-(\d+)\.png", f)[0])
            im = np.asarray(Image.open(f).convert("L"))
            H, W = im.shape
            band = im[:, x_band:W] < INK_THRESH
            band[:int(0.05 * H)] = False       # ignore header
            band[int(0.96 * H):] = False       # ignore folio
            ink = int(band.sum())
            if ink > MIN_INK_PX:
                cols = np.where(band.any(axis=0))[0]
                hits.append((n, ink, round((x_band + cols.max()) / ppc, 2)))
        print(f"overflow gate: {len(pages)} pages scanned, text_right={TEXT_RIGHT_CM}cm band>{BAND_START_CM}cm")
        if hits:
            print(f"FAIL: {len(hits)} page(s) overflow the right margin:")
            for n, ink, far in hits:
                print(f"  p{n}: ink={ink}px far_right={far}cm")
            sys.exit(1)
        print("PASS: no right-margin overflow on any page.")
        sys.exit(0)

if __name__ == "__main__":
    main()
