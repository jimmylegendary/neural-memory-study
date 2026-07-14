#!/usr/bin/env python3
"""LaTeX → tight transparent PNG, content-hash cached. Gives true LaTeX math in the pptx deck.

eq(latex) compiles a `standalone` document with pdflatex, converts to a 300-dpi transparent PNG
(pdftocairo), trims to the glyph bbox, and caches by md5(latex|color|pt) under assets/eq/.
Returns (png_path, width_in, height_in) at 300 dpi so the caller can place it 1:1 (crispest).

Korean/annotations do NOT go inside the LaTeX (pdflatex has no CJK here) — keep those as pptx text
next to the image. LaTeX body should be pure math (minimal \\text{...} in Latin is fine).
"""
import os, hashlib, subprocess, tempfile, shutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "assets", "eq")
DPI = 300

_TEX = r"""\documentclass[border=1pt]{standalone}
\usepackage{amsmath,amssymb,xcolor}
\begin{document}
{\color[HTML]{%s}\fontsize{%d}{%d}\selectfont $\displaystyle %s$}
\end{document}
"""


def eq(latex, color="22252B", pt=15):
    """Render a LaTeX math string to a cached transparent PNG. Returns (path, w_in, h_in)."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{latex}|{color}|{pt}|{DPI}".encode()).hexdigest()[:16]
    png = os.path.join(CACHE, key + ".png")
    if not os.path.exists(png):
        d = tempfile.mkdtemp(prefix="eq_")
        try:
            tex = os.path.join(d, "e.tex")
            with open(tex, "w") as f:
                f.write(_TEX % (color, pt, int(pt * 1.2), latex))
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "-output-directory", d, tex],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(["pdftocairo", "-png", "-r", str(DPI), "-transp", "-singlefile",
                            os.path.join(d, "e.pdf"), png[:-4]],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # trim fully-transparent border to a tight bbox
            im = Image.open(png).convert("RGBA")
            bbox = im.getbbox()
            if bbox:
                im.crop(bbox).save(png)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    w, h = Image.open(png).size
    return png, w / DPI, h / DPI


if __name__ == "__main__":
    p, w, h = eq(r"\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}")
    print(p, f"{w:.2f}in x {h:.2f}in")
