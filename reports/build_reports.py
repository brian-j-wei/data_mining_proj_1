"""Build the Part 2 and Part 3 reports: Markdown -> formatted Word documents.

    python reports/build_reports.py          # build both .docx files and print word counts
    python reports/build_reports.py --pdf    # also export PDFs through Microsoft Word (macOS)

Requires pandoc on PATH and python-docx (see reports/requirements.txt). The
figures and tables the reports cite are produced by the scripts in scripts/;
run those first (see run_all.sh).

Formatting follows the assignment: US Letter, 1-inch margins, Calibri 11 pt,
1.15 line spacing, bold section headers, numbered and captioned tables/figures.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
REFERENCE_DOCX = HERE / "templates" / "reference.docx"

# (source markdown, output name, prose word-count target)
REPORTS = [
    ("part2_client_report.md", "part2_client_report.docx", (1000, 1500)),
    ("part3_technical_appendix.md", "part3_technical_appendix.docx", (1500, 2200)),
]

FONT = "Calibri"
MONO = "Consolas"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
INK_2 = RGBColor(0x40, 0x40, 0x40)
RULE = "1F3A5F"
HAIRLINE = "D0D0D0"
HEADER_FILL = "E9EEF5"
TABLE_PT = 11      # assignment: 11 or 12 pt everywhere, tables included
LINE = 1.15        # assignment: 1.15 line spacing


# --------------------------------------------------------------------------
# Reference document (page setup and styles pandoc copies into each report)
# --------------------------------------------------------------------------

def set_font(style, size=None, bold=None, italic=None, color=None, name=FONT):
    font = style.font
    font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)
    for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        rfonts.attrib.pop(qn(theme_attr), None)
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if italic is not None:
        font.italic = italic
    if color is not None:
        font.color.rgb = color


def set_spacing(style, before=0, after=6, line=1.15, keep_next=None, align=None):
    fmt = style.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if keep_next is not None:
        fmt.keep_with_next = keep_next
    if align is not None:
        fmt.alignment = align


def add_page_number(section):
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
        if tag:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), tag)
            run._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = text
            run._r.append(instr)
    run.font.size = Pt(11)
    run.font.color.rgb = INK_2


def make_reference_docx(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"],
                   check=True)
    doc = Document(path)

    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(1))
    section.footer_distance = Inches(0.5)
    add_page_number(section)

    styles = doc.styles
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text"):
        set_font(styles[name], size=11)
    set_spacing(styles["Normal"], after=6)
    set_spacing(styles["Body Text"], after=6)
    set_spacing(styles["First Paragraph"], after=6)
    set_spacing(styles["Compact"], after=3)

    set_font(styles["Title"], size=20, bold=True, color=ACCENT)
    set_spacing(styles["Title"], before=0, after=4, line=LINE)
    set_font(styles["Subtitle"], size=12, bold=False, italic=False, color=INK_2)
    set_spacing(styles["Subtitle"], after=4, line=LINE)
    for name in ("Title", "Subtitle"):
        # pandoc's default title styles carry expanded character spacing
        rpr = styles[name].element.get_or_add_rPr()
        for spacing in rpr.findall(qn("w:spacing")):
            rpr.remove(spacing)
    for name in ("Author", "Date"):
        set_font(styles[name], size=11, color=INK_2)
        set_spacing(styles[name], after=2, line=LINE)

    for level, size, before in [(1, 14, 14), (2, 12, 10), (3, 11, 8)]:
        style = styles[f"Heading {level}"]
        set_font(style, size=size, bold=True, italic=False, color=ACCENT)
        set_spacing(style, before=before, after=4, line=LINE, keep_next=True)

    for name in ("Table Caption", "Image Caption"):
        set_font(styles[name], size=11, italic=False, color=INK_2)
    set_spacing(styles["Table Caption"], before=8, after=4, line=LINE, keep_next=True)
    set_spacing(styles["Image Caption"], before=2, after=10, line=LINE)
    # figure paragraphs hold only an image; multiple spacing would pad it with blank space
    for name in ("Figure", "Captioned Figure"):
        set_spacing(styles[name], before=6, after=0, line=1.0, keep_next=True,
                    align=WD_ALIGN_PARAGRAPH.CENTER)

    if "Source Code" not in [s.name for s in styles]:
        styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
    code = styles["Source Code"]
    set_font(code, size=11, name=MONO)
    set_spacing(code, before=4, after=8, line=LINE)
    ppr = code.element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F4F5F7")
    ppr.append(shd)
    set_font(styles["Verbatim Char"], size=11, name=MONO)

    doc.save(path)
    return path


# --------------------------------------------------------------------------
# Post-processing of pandoc's output
# --------------------------------------------------------------------------

TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
               "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
               "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
               "tblCaption", "tblDescription"]
TCPR_ORDER = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
              "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark"]


def put_child(parent, child, order):
    """Replace parent's <child> and keep the schema's element order."""
    tag = child.tag.split("}")[1]
    old = parent.find(qn(f"w:{tag}"))
    if old is not None:
        parent.remove(old)
    rank = order.index(tag)
    for i, existing in enumerate(parent):
        name = existing.tag.split("}")[1]
        if name in order and order.index(name) > rank:
            parent.insert(i, child)
            return
    parent.append(child)


def border_set(tag, edges):
    borders = OxmlElement(f"w:{tag}")
    for edge, spec in edges.items():
        el = OxmlElement(f"w:{edge}")
        if spec is None:
            el.set(qn("w:val"), "nil")
        else:
            size, color = spec
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(size))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), color)
        borders.append(el)
    return borders


TEXT_WIDTH = 9360    # 6.5 in, in twentieths of a point
TWIPS_PER_CHAR = 120  # average Calibri glyph at 11 pt
CELL_PADDING = 160    # left + right cell margins
MIN_COLUMN = 700


MATH_WIDTH = 1.5     # rendered math is wider per character than body text
CODE_WIDTH = 1.05    # Consolas at 11 pt is about as wide as the estimate for prose
MIN_WRAPPED = 1700   # narrowest a wrapping prose column may get


def cell_runs(cell):
    """(text, is_code) for each text run in a cell; inline math is not a w:r."""
    for run in cell._tc.iter(qn("w:r")):
        text = "".join(t.text or "" for t in run.iter(qn("w:t")))
        style = run.find(f"{qn('w:rPr')}/{qn('w:rStyle')}")
        yield text, style is not None and style.get(qn("w:val")) == "VerbatimChar"


def math_chars(cell):
    return sum(len(t.text or "") for t in cell._tc.iter(qn("m:t")))


def cell_chars(cell):
    """Effective character width of a cell, counting inline math (which python-docx skips)."""
    text = sum(len(t) * (CODE_WIDTH if code else 1.0) for t, code in cell_runs(cell))
    return text + MATH_WIDTH * math_chars(cell)


def cell_unbreakable(cell):
    """Widest piece of a cell that must stay on one line: a code span, a word, or the math."""
    pieces = [0.0, MATH_WIDTH * math_chars(cell)]
    for text, code in cell_runs(cell):
        if code:
            pieces.append(len(text.strip(" ,")) * CODE_WIDTH)
        else:
            pieces += [len(word) for word in text.split()]
    return max(pieces)


def cell_text(cell):
    return "".join(t.text or "" for t in cell._tc.iter(qn("w:t"), qn("m:t")))


def fit_columns(table):
    """Size columns to their content instead of pandoc's equal widths.

    A header may wrap at word boundaries, so it only needs its longest word;
    body cells should not wrap. Narrow tables stay narrow (and centered).
    When a table is too wide, prose columns wrap first so that numbers, math
    and code keep their single line.
    """
    header_cells = table.rows[0].cells
    columns = []
    for j, head in enumerate(header_cells):
        header = cell_text(head)
        body_cells = [row.cells[j] for row in table.rows[1:]]
        body = max((cell_chars(c) for c in body_cells), default=0)
        word = max((len(w) for w in header.split()), default=0)
        floor = max([word] + [cell_unbreakable(c) for c in body_cells])
        # a column can wrap if some long cell has a break point (between words or code spans)
        can_wrap = any(cell_chars(c) > 12 and cell_chars(c) > cell_unbreakable(c) + 3
                       for c in body_cells)
        columns.append((header, body, word, floor, can_wrap))
    floors = [max(MIN_WRAPPED, c[3] * TWIPS_PER_CHAR + CELL_PADDING) for c in columns]

    # headers wrap onto two lines, or three if the table is still too wide
    for header_share in (0.45, 0.3):
        needs = [max(MIN_COLUMN, max(body, word, header_share * len(header))
                     * TWIPS_PER_CHAR + CELL_PADDING)
                 for header, body, word, _, _ in columns]
        if sum(needs) <= TEXT_WIDTH:
            break

    natural = sum(needs)
    if natural <= TEXT_WIDTH:
        total = min(TEXT_WIDTH, natural * 1.1)
        widths = [n * total / natural for n in needs]
    else:
        # wrappable columns give way first, down to their longest unbreakable piece
        widths = list(needs)
        flexible = [j for j, c in enumerate(columns) if c[4]]
        slack = {j: max(0.0, widths[j] - floors[j]) for j in flexible}
        excess = natural - TEXT_WIDTH
        room = sum(slack.values())
        for j in flexible:
            widths[j] -= slack[j] * min(1.0, excess / room) if room else 0
        if sum(widths) > TEXT_WIDTH:
            widths = [w * TEXT_WIDTH / sum(widths) for w in widths]
    widths = [int(w) for w in widths]

    tbl = table._tbl
    tblW = tbl.tblPr.find(qn("w:tblW"))
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(sum(widths)))
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    put_child(tbl.tblPr, layout, TBLPR_ORDER)

    for col, width in zip(tbl.tblGrid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            tcW = OxmlElement("w:tcW")
            tcW.set(qn("w:w"), str(width))
            tcW.set(qn("w:type"), "dxa")
            put_child(cell._tc.get_or_add_tcPr(), tcW, TCPR_ORDER)


def style_table(table):
    """Booktabs-style table: rules above/below, hairlines between rows, shaded header."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    fit_columns(table)
    tblPr = table._tbl.tblPr
    put_child(tblPr, border_set("tblBorders", {
        "top": (8, RULE), "left": None, "bottom": (8, RULE), "right": None,
        "insideH": (2, HAIRLINE), "insideV": None,
    }), TBLPR_ORDER)

    margins = OxmlElement("w:tblCellMar")
    for edge, width in (("top", 30), ("left", 80), ("bottom", 30), ("right", 80)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(width))
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    put_child(tblPr, margins, TBLPR_ORDER)

    for r, row in enumerate(table.rows):
        for cell in row.cells:
            for para in cell.paragraphs:
                fmt = para.paragraph_format
                fmt.space_before, fmt.space_after, fmt.line_spacing = Pt(0), Pt(0), LINE
                for run in para.runs:
                    run.font.size = Pt(TABLE_PT)
                    if r == 0:
                        run.font.bold = True
            # inline math in cells carries its own run properties
            for math_run in cell._tc.iter(qn("m:r")):
                rpr = math_run.find(qn("w:rPr"))
                if rpr is None:
                    rpr = OxmlElement("w:rPr")
                    math_run.insert(1 if math_run.find(qn("m:rPr")) is not None else 0, rpr)
                for tag in ("w:sz", "w:szCs"):
                    el = rpr.find(qn(tag))
                    if el is None:
                        el = OxmlElement(tag)
                        rpr.append(el)
                    el.set(qn("w:val"), str(int(TABLE_PT * 2)))
            if r == 0:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), HEADER_FILL)
                put_child(tcPr, shd, TCPR_ORDER)
                put_child(tcPr, border_set("tcBorders", {"bottom": (6, RULE)}), TCPR_ORDER)
                cell.vertical_alignment = 3  # bottom

        # short tables stay on one page; long ones keep the header with two body rows
        keep_rows = len(table.rows) - 1 if len(table.rows) <= 11 else 3
        if r < keep_rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True

        # rows never break across pages; the header row repeats instead
        trPr = row._tr.get_or_add_trPr()
        if trPr.find(qn("w:cantSplit")) is None:
            trPr.insert(0, OxmlElement("w:cantSplit"))
        if r == 0 and trPr.find(qn("w:tblHeader")) is None:
            trPr.append(OxmlElement("w:tblHeader"))


def polish(docx_path):
    doc = Document(docx_path)
    for table in doc.tables:
        style_table(table)
        # body text directly after a table needs air below the bottom rule
        following = table._tbl.getnext()
        if following is not None and following.tag == qn("w:p"):
            Paragraph(following, doc).paragraph_format.space_before = Pt(10)
    # captions directly follow figures; keep the pair on one page
    for para in doc.paragraphs:
        if para.style.name in ("Captioned Figure", "Figure"):
            para.paragraph_format.keep_with_next = True
    doc.save(docx_path)


# --------------------------------------------------------------------------
# Word counts (prose only: tables, captions, figures, code and display math excluded)
# --------------------------------------------------------------------------

def prose_word_count(markdown):
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    kept = [line for line in text.splitlines()
            if not line.lstrip().startswith(("|", "Table:", "![", "<!--"))]
    text = re.sub(r"\$[^$\n]+\$", " x ", "\n".join(kept))
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return sum(1 for token in text.split() if re.search(r"[A-Za-z0-9]", token))


# --------------------------------------------------------------------------

def build(source, output, reference):
    subprocess.run(
        ["pandoc", source, "--from", "markdown", "--to", "docx",
         "--reference-doc", str(reference), "--output", str(output)],
        cwd=HERE, check=True,
    )
    polish(output)
    return output


WORD_SANDBOX = Path.home() / "Library/Containers/com.microsoft.Word/Data/tmp"


def export_pdf(docx_path):
    """Save a PDF copy through Microsoft Word (macOS only).

    Word is sandboxed, so the file is staged in Word's own container to avoid
    file-access prompts, exported there, and the PDF copied back.
    """
    WORD_SANDBOX.mkdir(parents=True, exist_ok=True)
    staged = WORD_SANDBOX / docx_path.name
    staged_pdf = staged.with_suffix(".pdf")
    shutil.copy2(docx_path, staged)
    staged_pdf.unlink(missing_ok=True)

    script = f'''
    tell application "Microsoft Word"
        open POSIX file "{staged}"
        set theDoc to active document
        save as theDoc file name "{staged_pdf}" file format format PDF
        close theDoc saving no
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True, timeout=300)
    pdf_path = docx_path.with_suffix(".pdf")
    shutil.move(staged_pdf, pdf_path)
    staged.unlink(missing_ok=True)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", action="store_true", help="also export PDFs via Microsoft Word")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    reference = make_reference_docx(REFERENCE_DOCX)

    for source, name, (low, high) in REPORTS:
        if not (HERE / source).exists():
            print(f"skip   {source} (not found)")
            continue
        output = build(source, OUTPUT_DIR / name, reference)
        words = prose_word_count((HERE / source).read_text(encoding="utf-8"))
        status = "ok" if low <= words <= high else f"OUTSIDE {low}-{high}"
        print(f"built  {output.relative_to(HERE.parent)}  ({words} prose words, {status})")
        if args.pdf:
            print(f"pdf    {export_pdf(output).relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
