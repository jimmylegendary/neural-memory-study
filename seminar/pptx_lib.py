#!/usr/bin/env python3
"""Reusable python-pptx shape helpers for the TTT/neural-memory seminar (draw diagrams directly).

Design goals: 16:9, CJK-safe (Noto Sans CJK KR), no text/figure overlap. Every diagram is built from
box() + arrow() + label(); coordinates in inches (slide is 13.333 x 7.5).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

EMU = 914400
SW, SH = 13.333, 7.5
FONT = "Noto Sans CJK KR"

# palette
INK   = RGBColor(0x22, 0x25, 0x2b)
MUTE  = RGBColor(0x5a, 0x5f, 0x68)
BLUE  = RGBColor(0x2a, 0x6f, 0x97)   # compute / forward
BLUEB = RGBColor(0xdb, 0xe7, 0xf3)
RED   = RGBColor(0xc1, 0x12, 0x1f)   # memory-bound / loss / problem
REDB  = RGBColor(0xf6, 0xdd, 0xdd)
GOLD  = RGBColor(0xc9, 0xa2, 0x27)   # state / update
GOLDB = RGBColor(0xf6, 0xec, 0xc7)
GREEN = RGBColor(0x2a, 0x9d, 0x8f)   # result / key
GREENB= RGBColor(0xda, 0xf0, 0xed)
GREY  = RGBColor(0x8a, 0x8f, 0x98)
GREYB = RGBColor(0xec, 0xee, 0xf1)
WHITE = RGBColor(0xff, 0xff, 0xff)


def deck():
    p = Presentation()
    p.slide_width = Emu(int(SW * EMU)); p.slide_height = Emu(int(SH * EMU))
    return p


def _blank(p):
    return p.slides.add_slide(p.slide_layouts[6])


def _set_font(run, size, color=INK, bold=False, italic=False, mono=False):
    run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Noto Sans Mono CJK KR" if mono else FONT


def text(slide, x, y, w, h, lines, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, mono=False, wrap=True, line_spacing=1.2, space_after=4):
    """lines: str or list of (str, opts) where opts overrides size/color/bold/italic/mono/align/
    line_spacing/space_after. line_spacing is a multiple (1.2 = 120%)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = Pt(3); tf.margin_right = Pt(3); tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        opts = {}
        if isinstance(ln, tuple):
            ln, opts = ln
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = opts.get("align", align)
        ls = opts.get("line_spacing", line_spacing)
        if ls: para.line_spacing = ls
        sa = opts.get("space_after", space_after)
        if sa is not None: para.space_after = Pt(sa); para.space_before = Pt(0)
        r = para.add_run(); r.text = ln
        _set_font(r, opts.get("size", size), opts.get("color", color),
                  opts.get("bold", bold), opts.get("italic", italic), opts.get("mono", mono))
    return tb


def picture(slide, path, x, y, w=None, h=None):
    """Embed an image AS-IS. Give w or h (inches); the other is inferred to keep aspect ratio."""
    kw = {}
    if w is not None: kw["width"] = Inches(w)
    if h is not None: kw["height"] = Inches(h)
    return slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def fit_image(slide, path, x, y, maxw, maxh, frame=False):
    """Place an image fit (aspect-preserving) inside the box (x,y,maxw,maxh), centered.
    Returns (px, py, w, h) actual placement in inches. If frame, draw a thin grey border box."""
    iw, ih = Image.open(path).size
    ar = iw / ih
    if maxw / maxh > ar:
        h = maxh; w = h * ar
    else:
        w = maxw; h = w / ar
    px = x + (maxw - w) / 2; py = y + (maxh - h) / 2
    if frame:
        fr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(px - 0.04), Inches(py - 0.04),
                                    Inches(w + 0.08), Inches(h + 0.08))
        fr.fill.background(); fr.line.color.rgb = GREY; fr.line.width = Pt(0.75); fr.shadow.inherit = False
    slide.shapes.add_picture(path, Inches(px), Inches(py), Inches(w), Inches(h))
    return px, py, w, h


def box(slide, x, y, w, h, lines, fc=BLUEB, ec=BLUE, size=11, tcolor=INK, bold=False,
        rounded=True, align=PP_ALIGN.CENTER, mono=False, lw=1.0, line_spacing=1.18,
        space_after=3, anchor=MSO_ANCHOR.MIDDLE):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                                 Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fc
    shp.line.color.rgb = ec; shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = Pt(6); tf.margin_right = Pt(6); tf.margin_top = Pt(3); tf.margin_bottom = Pt(3)
    if isinstance(lines, str):
        lines = [lines]
    for i, ln in enumerate(lines):
        opts = {}
        if isinstance(ln, tuple):
            ln, opts = ln
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = opts.get("align", align)
        ls = opts.get("line_spacing", line_spacing)
        if ls: para.line_spacing = ls
        sa = opts.get("space_after", space_after)
        if sa is not None: para.space_after = Pt(sa); para.space_before = Pt(0)
        r = para.add_run(); r.text = ln
        _set_font(r, opts.get("size", size), opts.get("color", tcolor), opts.get("bold", bold),
                  opts.get("italic", False), opts.get("mono", mono))
    return shp


def arrow(slide, x1, y1, x2, y2, color=MUTE, lw=1.5, dashed=False):
    cxn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cxn.line.color.rgb = color; cxn.line.width = Pt(lw)
    le = cxn.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    tail = le.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'h': 'med'})
    le.append(tail)
    if dashed:
        d = le.makeelement(qn('a:prstDash'), {'val': 'dash'}); le.insert(0, d)
    return cxn


def title_bar(slide, kicker, title, section_color=BLUE):
    """top title band: small kicker + big title."""
    box(slide, 0.0, 0.0, SW, 0.06, "", fc=section_color, ec=section_color, rounded=False)
    text(slide, 0.55, 0.16, SW - 1.1, 0.35, kicker, size=13.5, color=section_color, bold=True, space_after=0)
    text(slide, 0.55, 0.5, SW - 1.1, 0.62, title, size=24, color=INK, bold=True, space_after=0)


def section_divider(p, label, title, color=BLUE):
    s = _blank(p)
    box(s, 0, 0, SW, SH, "", fc=color, ec=color, rounded=False)
    text(s, 0.9, 2.7, SW - 1.8, 1.0, label, size=20, color=WHITE, bold=True)
    text(s, 0.9, 3.4, SW - 1.8, 1.6, title, size=34, color=WHITE, bold=True)
    return s


def title_slide(p, title, subtitle, presenter):
    s = _blank(p)
    box(s, 0, 0, SW, SH, "", fc=RGBColor(0x10, 0x1a, 0x2b), ec=RGBColor(0x10, 0x1a, 0x2b), rounded=False)
    box(s, 0, 5.7, SW, 0.05, "", fc=GREEN, ec=GREEN, rounded=False)
    text(s, 0.9, 2.2, SW - 1.8, 1.6, title, size=34, color=WHITE, bold=True)
    text(s, 0.9, 3.9, SW - 1.8, 1.2, subtitle, size=16, color=RGBColor(0xb8, 0xc4, 0xd4))
    text(s, 0.9, 6.0, SW - 1.8, 0.8, presenter, size=13, color=RGBColor(0x8a, 0x9a, 0xac))
    return s


def slide(p, kicker, title, color=BLUE):
    s = _blank(p); title_bar(s, kicker, title, color); return s
