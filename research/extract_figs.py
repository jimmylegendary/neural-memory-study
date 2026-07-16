#!/usr/bin/env python3
"""Extract figures from paper PDFs (PyMuPDF), mimicking the reffigs pymupdf-clip pipeline.
Figure region = union of image + vector-drawing rects sitting above a 'Figure N' caption in the
same column. Saves reffigs/<label>/figures/figN.png + figures_manifest.json.  Tables are NOT
extracted (reproduced natively in LaTeX instead)."""
import fitz, os, re, json, sys

CAP_RE = re.compile(r'^(Figure|Fig\.?)\s*([0-9]+)\b', re.I)


def block_text(b):
    return " ".join(s["text"] for l in b.get("lines", []) for s in l.get("spans", [])).strip()


def extract(pdf_path, outdir, min_dim=46, zoom=3.0):
    doc = fitz.open(pdf_path)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    figs = []
    for pno in range(len(doc)):
        page = doc[pno]
        W, H = page.rect.width, page.rect.height
        midx = W / 2.0
        d = page.get_text("dict")
        blocks = d["blocks"]
        img_rects = [fitz.Rect(b["bbox"]) for b in blocks if b.get("type") == 1]
        draw_rects = [fitz.Rect(dr["rect"]) for dr in page.get_drawings()]
        draw_rects = [r for r in draw_rects if r.width > 6 and r.height > 6 and r.width < W and r.height < H]
        # captions
        caps = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            t = block_text(b)
            m = CAP_RE.match(t)
            if m:
                caps.append((m.group(2), t[:400], fitz.Rect(b["bbox"])))
        for num, cap, cr in caps:
            cx = (cr.x0 + cr.x1) / 2
            # two-column heuristic: does page use columns?
            two_col = any((r.x1 < midx - 4) for r in img_rects + draw_rects) and \
                      any((r.x0 > midx + 4) for r in img_rects + draw_rects) or W > 500
            if cx < midx:
                cx0, cx1 = 0, (midx if two_col else W)
            else:
                cx0, cx1 = (midx if two_col else 0), W

            def incol(r):
                c = (r.x0 + r.x1) / 2
                return cx0 - 6 <= c <= cx1 + 6
            # graphics above the caption, in column, within 0.8H
            g = [r for r in img_rects + draw_rects
                 if incol(r) and r.y1 <= cr.y0 + 3 and r.y0 > cr.y0 - 0.82 * H]
            if not g:
                continue
            gx0 = min(r.x0 for r in g); gx1 = max(r.x1 for r in g)
            gy0 = min(r.y0 for r in g); gy1 = max(r.y1 for r in g)
            clip = fitz.Rect(min(gx0, cr.x0) - 4, gy0 - 3, max(gx1, cr.x1) + 4, cr.y1 + 4) & page.rect
            if clip.width < min_dim or clip.height < min_dim:
                continue
            pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom))
            fid = f"fig{num}"
            # if dup figure number across pages, suffix page
            fn = f"{fid}.png"
            if os.path.exists(os.path.join(outdir, "figures", fn)):
                fid = f"fig{num}_p{pno+1}"; fn = f"{fid}.png"
            pix.save(os.path.join(outdir, "figures", fn))
            figs.append({"figure_id": fid, "figure_number": num, "caption": cap,
                         "image": f"figures/{fn}", "page": pno + 1,
                         "clip": [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)]})
    man = {"source_pdf": pdf_path, "engine": "pymupdf-clip", "figure_count": len(figs), "figures": figs}
    json.dump(man, open(os.path.join(outdir, "figures_manifest.json"), "w"), ensure_ascii=False, indent=1)
    return len(figs)


if __name__ == "__main__":
    REPO = "/home/jimmy/repos/neural-memory-study"
    targets = sys.argv[1:] or [
        "2505.23884", "2605.28053", "2604.06169", "2506.05233",
        "2512.23675", "2501.12352", "2411.19379", "2504.03624",
    ]
    for pid in targets:
        pdf = os.path.join(REPO, "papers", "external", f"{pid}.pdf")
        if not os.path.exists(pdf):
            pdf = os.path.join(REPO, "papers", f"{pid}.pdf")
        if not os.path.exists(pdf):
            print(f"  MISSING pdf {pid}"); continue
        out = os.path.join(REPO, "reffigs", f"ext-{pid}")
        n = extract(pdf, out)
        print(f"  {pid}: {n} figures -> reffigs/ext-{pid}/figures/")
