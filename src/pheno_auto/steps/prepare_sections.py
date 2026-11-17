#!/usr/bin/env python3
"""Step 2. Split each downloaded article into sections for extraction.

For every article folder written by fetch_phenopacket_literature.py it reads the
JATS XML and writes one plain-text file per unit. Each unit has a `kind`:

  Sections. Administrative ones (SKIP list: acknowledgements, funding, conflicts,
  statements, consent, author contributions, ethics, abbreviations, ...) are not
  turned into units and are listed in skipped.tsv. The rest are labelled with
  SECTION_RULES (case, suppl, materials, results, discussion, conclusion, intro),
  applied to the sec-type first and, only if none matches, to the heading;
  anything else -> other:
    body sections        each top-level <sec> of the body (subsections inherit its label)
    back-matter sections each <sec>/<notes> of the back matter (after all other units)
  Text of the body outside any section ("Main text") -> main, whether or not the
  article has sections (it can be a whole short report or letter, or an introduction
  before the first heading). `other` is kept for real sections, titled or untitled,
  that no rule matches; `case` only for sections that the rules label as case.
  A section headed exactly "Main text" (e.g. AJHG reports) is also main.
  Appendices (each <app>) -> appendix, whatever their heading (the skip list still
  applies, so an administrative appendix such as "Abbreviations" is skipped; an
  appendix with nothing but its heading/label, e.g. only supplementary-file links, is
  skipped too, reason "heading only"); the heading is the first line of the unit's text.

  Fixed kinds, set by where the text comes from:
    abstract      title and abstract
    table         each table of the article, as tab-separated rows with its caption
    figure        each figure legend (after all other units)
    suppl_table   with --supplementary: tables from supplementary files
                  (for table and suppl_table: columns empty in every row are dropped; a table
                  with a chunk still longer than --max-table-chars (default --max-chars), because
                  a header or row cannot be split, is skipped whole: skipped.tsv, reason
                  "table over --max-chars")
                  (.xlsx .xls .csv .tsv .txt .docx .pdf .html, and these inside .zip archives;
                  .xls files that are really .xlsx or HTML are read as such)
                  PDF tables are repaired: a cell wrapped over several lines is put on one line,
                  a word cut after its drop capital ("G" | "lycine") is joined, two neighbouring
                  columns never filled in the same row are one column, and a table cut by a page
                  break is joined back (the first table of page p+1 continues the last of page p
                  when it has as many columns and repeats the header or starts with values); its
                  title then reads "page 40-41 table 2". Tables without ruling lines, which
                  pdfplumber does not detect, are found from the words' positions (rows of 3 or more
                  aligned pieces) and taken out of the page text; a "Table N" caption above goes in
                  the title. PDF prose is written one paragraph per
                  line: line breaks inside a paragraph are joined and words cut by a hyphen at a
                  line end are rejoined ("dis-" / "ability" -> "disability")
    suppl_text    with --supplementary: prose of supplementary .docx/.pdf/.html/.txt files.
                  Tables are kept out of it (Word tables; PDF table regions; HTML <table>s),
                  and headings are marked "## " on their own line (Word heading styles,
                  HTML <h1>-<h6>); PDF and .txt headings are not detectable. Prose longer than
                  --max-chars is cut only at those headings (heading blocks packed up to the
                  limit; a block longer than the limit is split and its heading repeated)

Figure legends and back-matter sections are numbered after all other units, so
adding them does not rename sections processed by an earlier run.

Use the kinds to run extraction on some units only and to compare them in the
evaluation; pass every kind to process everything.

Output
  <out>/PMID_<pmid>/<nn>_<kind>_<slug>.txt
  <out>/index.tsv     pmid, section_id, kind, part, title, n_chars, path
                      part = where the unit comes from: front (abstract), body, back,
                      floats (<floats-group> tables/figures), suppl_file (supplementary files).
                      Units inside sub-articles (peer review) are not written (skipped.tsv)
  <out>/truncated.tsv supplementary tables/texts cut by --max-supp-chunks: pmid, title, type,
                      kept, total (01b_curation_source.py flags such papers as not checkable)
  <out>/skipped.tsv   units not written, with the reason: administrative sections (SKIP
                      list) and anything inside a <sub-article> (peer-review reports, author
                      replies), treated as administrative

Usage
  pheno-auto prepare-sections --articles literature/articles --out work/sections
  pheno-auto prepare-sections --articles literature/articles --out work/sections --supplementary
  pheno-auto prepare-sections --articles literature/articles --out work/paragraphs --per-paragraph

Standard library only for the article, .csv/.tsv/.txt, .docx and .zip. Other supplementary
formats need: .xlsx `pip install openpyxl`, .xls `pip install xlrd`, .pdf `pip install pdfplumber`.
Old Word .doc and .rtf files are converted to .docx with LibreOffice when it is installed
(soffice on PATH, or the usual Windows/macOS install folder); without it they are reported as
".doc (needs LibreOffice)". A .doc that is really a .docx or a web page is read as such.
The run ends with a count of supplementary files read and not read, by format.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter

# XML elements left out of a section's running text. Tables, figures and supplementary
# files are turned into units of their own; references, display equations and graphics
# are dropped. Inline formulas (often lab values such as "Mg 0.5 mmol/L") and body
# footnotes are kept; the LaTeX copy of a formula (<tex-math>) is dropped so only its
# MathML text remains.
SKIP_TAGS = {"table-wrap", "fig", "fig-group", "supplementary-material", "disp-formula", "tex-math",
             "object-id", "ref-list", "graphic", "inline-graphic", "media", "table-wrap-foot"}
BLOCK_TAGS = {"p", "title", "list-item", "disp-quote", "boxed-text", "def-item", "sec", "caption", "label"}

# Sections that are not turned into units at all (administrative text). Checked
# against "<sec-type> <heading>" of body sections, appendices and back-matter
# sections before they are labelled. Skipped sections are listed in skipped.tsv.
SKIP = re.compile(r"acknowledg|funding|financial|grant|competing|conflict|disclosure|disclaimer|statement|consent|"
                  r"author|contribution|contributor|declaration|availability|abbreviation|pre-?publication|"
                  r"copyright|licen[cs]e|publisher|ethic|footnote", re.I)

# Section labels. One rule list, SECTION_RULES, applied to the XML sec-type first,
# when there is one; only if no rule matches it, to the heading. Within each, the
# first matching rule wins, so the order decides overlaps ("Patients and methods"
# -> case, "Results and discussion" -> results, "Discussion and conclusions" ->
# discussion). A section that matches nothing is `other` (the residual).
SECTION_RULES = [
    ("case", r"\bcase|patient|subject|famil(y|ies)|report|presentation|population"
             r"|investigation|diagnosis|treatment"),
    ("suppl", r"supplem|additional|more information"),
    ("materials", r"material|method|design|model|\bdata\b|statistic"),
    ("results", r"result|analys|outcome|finding|(genetic|molecular|functional) stud|\btest|identif|variant|mutation"
                r"|correlation|evaluation"),
    ("discussion", r"discussion|comment|relevance|limitation"),
    ("conclusion", r"\bconclu|endnote|future"),
    ("intro", r"background|intro"),
]


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


# Part of the article a unit comes from: the top-level child of <article> that holds it.
# Supplementary files are part "suppl_file". Units inside sub-articles (peer-review reports,
# author replies) are part "sub-article" and are not written: like administrative sections
# they go to skipped.tsv and keep their number.
TOP_PART = {"front": "front", "body": "body", "back": "back", "floats-group": "floats",
            "sub-article": "sub-article", "response": "sub-article"}


def element_parts(root: ET.Element) -> dict:
    """Element -> part (front, body, back, floats, sub-article, or the tag of another top-level child)."""
    parts = {}
    for top in root:
        p = TOP_PART.get(local(top.tag), local(top.tag))
        for e in top.iter():
            parts[e] = p
    return parts


def clean(text: str) -> str:
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def text_of(elem: ET.Element, skip: set[str] = SKIP_TAGS) -> str:
    """Readable text of an element: block elements become paragraphs, skipped tags are dropped."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        tag = local(e.tag)
        if tag in skip:
            if e.tail:
                parts.append(e.tail)
            return
        block = tag in BLOCK_TAGS
        if block:
            parts.append("\n")
        if e.text:
            parts.append(e.text)
        for child in e:
            walk(child)
        if block:
            parts.append("\n")
        if e.tail:
            parts.append(e.tail)

    walk(elem)
    lines = [clean(line) for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def is_skipped(title: str, sec_type: str = "") -> bool:
    """True for administrative sections (SKIP), which are not turned into units."""
    return bool(SKIP.search(f"{sec_type or ''} {title or ''}"))


def guess_kind(title: str, sec_type: str = "") -> str:
    """Label of a top-level section: from its sec-type when a rule matches it, else from its heading."""
    st, hd = (sec_type or "").lower(), (title or "").lower()
    if hd.strip() == "main text":             # a section headed exactly "Main text" holds the whole article
        return "main"
    for text in (st, hd):                     # sec-type first, then heading
        if text.strip():
            for kind, pat in SECTION_RULES:
                if re.search(pat, text):
                    return kind
    return "other"


def split_long(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries so that no part exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n"):
        while len(para) > max_chars:                      # a single huge paragraph
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(para[:cut])
            para = para[cut:].lstrip()
        if cur and len(cur) + len(para) + 1 > max_chars:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


PER_PARAGRAPH = False          # set by --per-paragraph
TABLE_CHARS = 0                # set by --max-table-chars; 0 = use --max-chars for tables too


def table_limit(max_chars: int) -> int:
    return TABLE_CHARS or max_chars


HEADING_LINE = re.compile(r"^## ", re.M)


def split_at_headings(text: str, max_chars: int) -> list[str]:
    """Supplementary prose with "## " heading lines (Word heading styles, HTML <h1>-<h6>):
    cut only at headings, packing consecutive heading blocks into parts of at most max_chars.
    A block that is longer than max_chars on its own is split by split_long, and every piece
    after the first starts again with the block's heading, so each part says where it comes
    from. Text without headings, or short enough, goes to units() unchanged."""
    if PER_PARAGRAPH or len(text) <= max_chars or not HEADING_LINE.search(text):
        return units(text, max_chars)
    starts = [m.start() for m in HEADING_LINE.finditer(text)]
    if starts[0] != 0:
        starts.insert(0, 0)                               # text before the first heading
    blocks = [text[a:b].strip("\n") for a, b in zip(starts, starts[1:] + [len(text)])]
    pieces = []
    for blk in (b for b in blocks if b.strip()):
        if len(blk) <= max_chars:
            pieces.append(blk)
            continue
        first, _, rest = blk.partition("\n")
        heading = first if first.startswith("## ") else ""
        body = rest if heading else blk
        room = max_chars - len(heading) - 1 if heading else max_chars
        pieces += [f"{heading}\n{p}" if heading else p for p in split_long(body, room)]
    parts, cur = [], ""
    for pc in pieces:                                     # pack whole pieces; break only between them
        if cur and len(cur) + len(pc) + 1 > max_chars:
            parts.append(cur)
            cur = pc
        else:
            cur = f"{cur}\n{pc}" if cur else pc
    if cur:
        parts.append(cur)
    return parts


def split_paragraphs(text: str, max_chars: int, min_chars: int = 120) -> list[str]:
    """One unit per paragraph. Headings and very short lines are joined to the next
    paragraph, and the section heading is repeated at the top of every unit so the
    model knows where the paragraph comes from."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    heading = lines[0] if len(lines[0]) < min_chars else ""
    body = lines[1:] if heading else lines
    paras, cur = [], ""
    for ln in body:
        cur = f"{cur}\n{ln}" if cur else ln
        if len(ln) >= min_chars:                          # a real paragraph ends the unit
            paras.append(cur)
            cur = ""
    if cur:
        if paras:
            paras[-1] = f"{paras[-1]}\n{cur}"
        else:
            paras.append(cur)
    out = []
    for para in paras:
        unit = f"{heading}\n{para}" if heading else para
        out.extend(split_long(unit, max_chars))           # a giant paragraph is still split
    return out


def units(text: str, max_chars: int) -> list[str]:
    return split_paragraphs(text, max_chars) if PER_PARAGRAPH else split_long(text, max_chars)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
FOOTMARK = re.compile(r"[a-zA-Z*†‡§¶#‖]{1,2}(\s*,\s*[a-zA-Z*†‡§¶#‖]{1,2})*")


def cell_text(cell: ET.Element) -> str:
    """A table cell's text: a line break (<break/>) is a space ("Control<break/>chromosomes"), and a
    footnote mark stays apart from the word it is set on, in brackets: "DEM4322[b]", "Profound[c]",
    "2.78[f]" (a footnote <xref>, or a superscript of one or two letters or symbols; superscript
    numbers, as in 10<sup>3</sup>, are left as they are)."""
    parts: list[str] = []

    def walk(e: ET.Element, in_mark: bool = False) -> None:
        tag = local(e.tag)
        if tag in SKIP_TAGS:
            parts.append(e.tail or "")
            return
        if tag == "break":
            parts.append(" " + (e.tail or ""))
            return
        inner = clean("".join(e.itertext()))
        mark = not in_mark and ((tag == "xref" and e.get("ref-type") in ("table-fn", "fn")) or
                                (tag == "sup" and FOOTMARK.fullmatch(inner or "") is not None))
        if mark and inner:
            tail = e.tail or ""
            parts.append(f"[{inner}]" + (" " if tail[:1].isalnum() else "") + tail)
            return
        parts.append(e.text or "")
        for ch in e:
            walk(ch, in_mark)
        parts.append(e.tail or "")

    parts.append(cell.text or "")
    for ch in cell:
        walk(ch)
    return clean(re.sub(r"[ \t\r\n]+", " ", "".join(parts)))


def merge_header_rows(head: list[list[str]]) -> list[str]:
    """Several header rows as one: per column, its header texts from top to bottom, each once
    ("Family ID" spanning three rows -> "Family ID"; "Mutations of GIPC3" over "mRNA NM_133261.2"
    -> "Mutations of GIPC3 mRNA NM_133261.2"), so the first row names every column."""
    width = max(len(r) for r in head)
    out = []
    for j in range(width):
        seen: list[str] = []
        for r in head:
            t = r[j].strip() if j < len(r) else ""
            if t and t not in seen:
                seen.append(t)
        out.append(" ".join(seen))
    return out


def table_rows(table: ET.Element) -> list[list[str]]:
    """Expand a JATS/HTML table into a rectangular grid, honouring rowspan and colspan."""
    grid: list[list[str]] = []
    carry: dict[int, tuple[int, str]] = {}                # column -> (rows left, text)
    for tr in table.iter():
        if local(tr.tag) != "tr":
            continue
        row: list[str] = []
        col = 0
        cells = [c for c in tr if local(c.tag) in ("td", "th")]
        ci = 0
        while ci < len(cells) or col in carry:
            if col in carry:
                left, txt = carry[col]
                row.append(txt)
                if left > 1:
                    carry[col] = (left - 1, txt)
                else:
                    del carry[col]
                col += 1
                continue
            cell = cells[ci]
            ci += 1
            txt = cell_text(cell)
            try:
                cs, rs = int(cell.get("colspan", 1) or 1), int(cell.get("rowspan", 1) or 1)
            except ValueError:
                cs, rs = 1, 1
            for _ in range(max(cs, 1)):
                row.append(txt)
                if rs > 1:
                    carry[col] = (rs - 1, txt)
                col += 1
        if any(row):
            grid.append(row)
    return grid


def drop_empty_columns(rows: list[list[str]]) -> list[list[str]]:
    """Pad rows to one width and drop columns empty in every row (e.g. Excel sheets whose used
    range runs to the last column)."""
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j].strip() for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def rows_to_chunks(header_lines: list[str], rows: list[list[str]], n_header: int, max_chars: int) -> list[str]:
    """Serialise rows as TSV, repeating the header rows in every chunk. Rows are never cut, so a
    chunk can still exceed max_chars (a huge header or row); callers skip such tables."""
    rows = drop_empty_columns(rows)
    lines = ["\t".join(r) for r in rows]
    head, body = lines[:n_header], lines[n_header:]
    prefix = "\n".join(header_lines + head)
    chunks, cur = [], []
    size = len(prefix)
    for line in body:
        if cur and size + len(line) + 1 > max_chars:
            chunks.append("\n".join([prefix] + cur))
            cur, size = [], len(prefix)
        cur.append(line)
        size += len(line) + 1
    chunks.append("\n".join([prefix] + cur))
    return chunks


def table_sections(root: ET.Element, max_chars: int) -> list[tuple[str, str, str, str]]:
    out = []
    parts = element_parts(root)
    for tw in root.iter():
        if local(tw.tag) != "table-wrap":
            continue
        label = clean(" ".join(text_of(e, skip=set()) for e in tw if local(e.tag) == "label"))
        caption = clean(" ".join(text_of(e, skip=set()).replace("\n", " ") for e in tw if local(e.tag) == "caption"))
        foot = clean(" ".join(text_of(e, skip=set()).replace("\n", " ") for e in tw.iter() if local(e.tag) == "table-wrap-foot"))
        rows: list[list[str]] = []
        n_head = 0
        for t in tw.iter():
            if local(t.tag) == "table":
                thead = [e for e in t if local(e.tag) == "thead"]
                n_head = max(n_head, sum(1 for h in thead for tr in h.iter() if local(tr.tag) == "tr"))
                rows.extend(table_rows(t))
        if not rows:
            continue                                      # table published only as an image
        header = [". ".join(x for x in (label.rstrip(". "), caption) if x)]
        if foot:
            header.append(f"Footnotes: {foot}")
        merged = n_head > 1 and len(rows) > n_head
        if merged:                                        # header rows (spanning cells) merged into one
            rows = [merge_header_rows(rows[:n_head])] + rows[n_head:]
            n_head = 1
        header.append("Rows are tab-separated; the first row is the column header." if merged else
                      "Rows are tab-separated; the first row(s) are column headers.")
        title = label or "Table"
        limit = table_limit(max_chars)
        chunks = rows_to_chunks(header, rows, max(n_head, 1), limit)
        kind = "oversized" if any(len(c) > limit for c in chunks) else "table"   # skipped whole
        for i, chunk in enumerate(chunks):
            out.append((kind, title if i == 0 else f"{title} (part {i + 1})", chunk, parts.get(tw, "")))
    return out


# --------------------------------------------------------------------------- #
# Article
# --------------------------------------------------------------------------- #
def parse_xml(path: str) -> ET.Element | None:
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    for attempt in (raw, re.sub(r"<!DOCTYPE[^>]*(\[[^\]]*\])?\s*>", "", raw, count=1)):
        try:
            return ET.fromstring(attempt)
        except ET.ParseError:
            continue
    return None


def article_sections(root: ET.Element, max_chars: int) -> list[tuple[str, str, str, str]]:
    """Return [(kind, title, text, part)] for one article."""
    out: list[tuple[str, str, str, str]] = []
    find = lambda tag: next((e for e in root.iter() if local(e.tag) == tag), None)  # noqa: E731
    parts = element_parts(root)

    title_el = find("article-title")
    title = clean(text_of(title_el, skip=set()).replace("\n", " ")) if title_el is not None else ""
    abstracts = [text_of(e) for e in root.iter() if local(e.tag) == "abstract" and parts.get(e) == "front"]
    front = "\n".join(x for x in [title] + abstracts if x)
    if front:
        out.append(("abstract", "Title and abstract", front, "front"))

    body = next((e for e in root if local(e.tag) == "body"), None)      # the article's own body, not a sub-article's
    if body is not None:
        top_secs = [e for e in body if local(e.tag) == "sec"]
        loose = ET.Element("body")
        loose.extend([e for e in body if local(e.tag) != "sec"])
        loose_text = text_of(loose)
        if loose_text:                                    # text outside any section (common in case reports)
            for i, part in enumerate(units(loose_text, max_chars)):
                out.append(("main", "Main text" + (f" (part {i + 1})" if i else ""), part, "body"))
        for sec in top_secs:
            t_el = next((e for e in sec if local(e.tag) == "title"), None)
            sec_title = clean(text_of(t_el, skip=set()).replace("\n", " ")) if t_el is not None else ""
            text = text_of(sec)
            if not text or text == sec_title:
                continue
            st = sec.get("sec-type", "")
            kind = "skip" if is_skipped(sec_title, st) else guess_kind(sec_title, st)
            for i, part in enumerate(units(text, max_chars)):
                out.append((kind, (sec_title or "Untitled section") + (f" (part {i + 1})" if i else ""), part, "body"))

    for app in (e for e in root.iter() if local(e.tag) == "app"):
        text = text_of(app)
        if text:
            t_el = next((e for e in app if local(e.tag) == "title"), None)
            app_title = clean(text_of(t_el, skip=set()).replace("\n", " ")) if t_el is not None else ""
            st = app.get("sec-type", "")
            kind = "skip" if is_skipped(app_title, st) else "appendix"   # own kind; administrative ones skipped
            if kind == "appendix" and not text_of(app, skip=SKIP_TAGS | {"title", "label"}):
                kind = "empty"            # only a label/heading left (e.g. links to supplementary files)
            for i, part in enumerate(units(text, max_chars)):
                out.append((kind, "Appendix" + (f" (part {i + 1})" if i else ""), part,
                            parts.get(app, "")))                       # name kept stable; part: where it sits

    out.extend(table_sections(root, max_chars))
    return out


def late_sections(root: ET.Element, max_chars: int) -> list[tuple[str, str, str, str]]:
    """Figure legends and back-matter sections.

    They are numbered after every other unit (including supplementary ones), so adding
    them does not rename the sections an earlier run already processed.
    """
    out: list[tuple[str, str, str, str]] = []
    parts = element_parts(root)
    for fig in (e for e in root.iter() if local(e.tag) == "fig"):
        label = clean(" ".join(text_of(e, skip=set()) for e in fig if local(e.tag) == "label"))
        caption = "\n".join(text_of(e, skip=set()) for e in fig if local(e.tag) == "caption")
        if not caption.strip():
            continue                                      # an image without a legend
        title = label or "Figure"
        text = f"Figure legend: {title}\n{caption}" if label else f"Figure legend\n{caption}"
        for i, part in enumerate(split_long(text, max_chars)):
            out.append(("figure", title + (f" (part {i + 1})" if i else ""), part, parts.get(fig, "")))
    back = next((e for e in root if local(e.tag) == "back"), None)      # the article's own back matter
    if back is not None:
        # Skipped or labelled with the same rules as body sections. References, author
        # footnotes (<fn-group>) and <ack> elements are not sections and are not read;
        # appendices are read with the body.
        for sec in (e for e in back if local(e.tag) in ("sec", "notes")):
            t_el = next((e for e in sec if local(e.tag) == "title"), None)
            sec_title = clean(text_of(t_el, skip=set()).replace("\n", " ")) if t_el is not None else ""
            text = text_of(sec)
            if not text or text == sec_title:
                continue
            st = sec.get("sec-type", "") or sec.get("notes-type", "")
            kind = "skip" if is_skipped(sec_title, st) else guess_kind(sec_title, st)
            name = "Back matter: " + (sec_title or "Untitled section")
            for i, part in enumerate(units(text, max_chars)):
                out.append((kind, name + (f" (part {i + 1})" if i else ""), part, "back"))
    return out


# --------------------------------------------------------------------------- #
# Supplementary files
# --------------------------------------------------------------------------- #
TRUNCATED: list[list] = []              # (title, table/text, kept, total) of the current article (--max-supp-chunks)
SUPP_SKIPPED: dict[str, int] = {}        # extension -> files that could not be read (reported at the end)
SUPP_READ: dict[str, int] = {}
NO_TABLE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".eps", ".svg", ".mp4", ".avi", ".mov",
                ".wmv", ".mpg", ".mpeg", ".mp3", ".wav", ".pdb", ".fasta", ".fa", ".gz", ".bam", ".vcf"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def supplementary_captions(root: ET.Element | None) -> dict[str, str]:
    """File name -> "label. caption" from the article's <supplementary-material> elements."""
    out: dict[str, str] = {}
    if root is None:
        return out
    for sm in root.iter():
        if local(sm.tag) != "supplementary-material":
            continue
        label = clean(" ".join(text_of(e, skip=set()) for e in sm if local(e.tag) == "label"))
        cap = clean(" ".join(text_of(e, skip=set()).replace("\n", " ") for e in sm if local(e.tag) == "caption"))
        text = ". ".join(x for x in (label.rstrip(". "), cap) if x)
        if not text:
            continue
        for e in sm.iter():
            for k, v in e.attrib.items():
                if k.endswith("href") and v:
                    out[os.path.basename(v).lower()] = text
    return out


def _docx_parts(data: bytes) -> list[tuple[str, object]]:
    """Body of a .docx in order: ("h", heading text), ("p", text) and ("tbl", rows). Standard library only.

    A paragraph is a heading when its style is a heading style (style name "heading 1".."heading 9",
    "Title" or "Subtitle", or a style with an outline level; looked up in word/styles.xml, so
    localised style ids such as "berschrift1" work) or when it has an outline level of its own."""
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        part = "word/document.xml"
        if part not in names and "_rels/.rels" in names:  # main part named differently: follow _rels/.rels
            rels = ET.fromstring(z.read("_rels/.rels"))
            part = next((r.get("Target", "").lstrip("/") for r in rels
                         if r.get("Type", "").endswith("/officeDocument")), part)
        if part not in names:                             # no index: any XML part holding a Word body
            part = next((n for n in names if n.lower().endswith(".xml") and b"<w:body" in z.read(n)), None)
            if part is None:
                raise ValueError("no Word document part in this .docx (not a Word file?)")
        body = ET.fromstring(z.read(part)).find(f"{W}body")
        heading_ids = set()
        if "word/styles.xml" in names:
            try:
                for st in ET.fromstring(z.read("word/styles.xml")).iter(f"{W}style"):
                    sid = st.get(f"{W}styleId") or ""
                    nm = st.find(f"{W}name")
                    name = (nm.get(f"{W}val") if nm is not None else "") or ""
                    ppr = st.find(f"{W}pPr")
                    lvl = ppr.find(f"{W}outlineLvl") if ppr is not None else None
                    if re.match(r"(?i)(heading\s*\d|title$|subtitle$)", name) or (
                            lvl is not None and (lvl.get(f"{W}val") or "9") < "9"):
                        heading_ids.add(sid)
            except ET.ParseError:
                pass

    def is_heading(p) -> bool:
        ppr = p.find(f"{W}pPr")
        if ppr is None:
            return False
        ps, lvl = ppr.find(f"{W}pStyle"), ppr.find(f"{W}outlineLvl")
        sid = (ps.get(f"{W}val") if ps is not None else "") or ""
        return (sid in heading_ids or bool(re.match(r"(?i)(heading\s*\d|title$|subtitle$)", sid))
                or (lvl is not None and (lvl.get(f"{W}val") or "9") < "9"))

    def ptext(p) -> str:
        bits = []
        for e in p.iter():
            if e.tag == f"{W}t" and e.text:
                bits.append(e.text)
            elif e.tag in (f"{W}tab",):
                bits.append(" ")
            elif e.tag in (f"{W}br", f"{W}cr"):
                bits.append(" ")
        return clean("".join(bits))

    def table(tbl) -> list[list[str]]:
        rows, above = [], {}
        for tr in tbl.findall(f"{W}tr"):
            row, col = [], 0
            for tc in tr.findall(f"{W}tc"):
                pr = tc.find(f"{W}tcPr")
                span, merge = 1, None
                if pr is not None:
                    gs = pr.find(f"{W}gridSpan")
                    if gs is not None:
                        span = int(gs.get(f"{W}val", "1") or 1)
                    vm = pr.find(f"{W}vMerge")
                    if vm is not None:
                        merge = vm.get(f"{W}val", "continue")
                text = clean(" ".join(ptext(p) for p in tc.iter(f"{W}p")))
                if merge == "continue":
                    text = above.get(col, "")
                for _ in range(span):
                    row.append(text)
                    above[col] = text
                    col += 1
            rows.append(row)
        return rows

    parts: list[tuple[str, object]] = []
    for child in body:
        if child.tag == f"{W}p":
            t = ptext(child)
            if t:
                parts.append(("h" if is_heading(child) else "p", t))
        elif child.tag == f"{W}tbl":
            parts.append(("tbl", table(child)))
    return parts


def _html_tables(data: bytes) -> list[list[list[str]]]:
    """Tables of an HTML file as rows of cell texts. Standard library only."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tables, self.row, self.cell, self.depth = [], None, None, 0

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.depth += 1
                if self.depth == 1:
                    self.tables.append([])
            elif tag == "tr" and self.depth:
                self.row = []
            elif tag in ("td", "th") and self.row is not None:
                self.cell = []
            elif tag == "br" and self.cell is not None:
                self.cell.append(" ")

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.cell is not None and self.row is not None:
                self.row.append(clean("".join(self.cell)))
                self.cell = None
            elif tag == "tr" and self.row is not None and self.tables:
                self.tables[-1].append(self.row)
                self.row = None
            elif tag == "table" and self.depth:
                self.depth -= 1

        def handle_data(self, d):
            if self.cell is not None:
                self.cell.append(d)

    p = P()
    p.feed(data.decode("utf-8", errors="replace"))
    return [t for t in p.tables if t]


def _html_text(data: bytes, drop_tables: bool = False) -> str:
    """Text of an HTML file. Headings <h1>-<h6> are marked "## " on their own line; with
    drop_tables, <table> elements are left out (they are read separately as tables)."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", data.decode("utf-8", errors="replace"))
    if drop_tables:
        prev = None
        while prev != text:                               # innermost tables first, then the outer ones
            prev = text
            text = re.sub(r"(?is)<table\b(?:(?!<table\b).)*?</table\s*>", "\n", text)
    text = re.sub(r"(?i)<h[1-6]\b[^>]*>", "\n## ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", text)
    import html as _html
    return _html.unescape(re.sub(r"<[^>]+>", " ", text))


PEER_REVIEW = re.compile(r"peer review file|reviewer comments|reviewers'? comments|response to (?:the )?reviewers?|"
                         r"point-by-point response|rebuttal letter", re.I)


PDF_MARKS = {"+", "-", "−", "±", "+/-", "-/+", "x", "yes", "no", "y", "n", "nd", "n.d.", "na", "n/a", "n.a.", "?", "<1"}
PDF_NUMBER = re.compile(r"^[<>≤≥~]?\s*-?\d+([.,]\d+)?(\s*[-–/±]\s*\d+([.,]\d+)?)*\s*-?%?$")


def clean_cell(text: str) -> str:
    """A PDF cell on one line: pdfplumber keeps the line breaks of a cell wrapped over several lines
    ("558-IV-1\n(U)"), which would break the table's rows in the section file."""
    return clean(text.replace("\r", " ").replace("\n", " "))


def tidy_pdf_rows(rows: list[list[str]]) -> list[list[str]]:
    """pdfplumber's table cells, repaired:
    - a word cut after its drop capital ("G" | "lycine" -> "Glycine" | "");
    - pairs of neighbouring columns that are never both filled in one row joined into one: the
      phantom column pdfplumber puts beside text that is not aligned from row to row
      ("\tValine\t\t256" under "Leucine\t\t165")."""
    width = max((len(r) for r in rows), default=0)
    rows = [list(r) + [""] * (width - len(r)) for r in rows]
    for r in rows:
        for j in range(width - 1):
            if re.fullmatch(r"[A-Z]", r[j]) and r[j + 1][:1].islower():
                r[j], r[j + 1] = r[j] + r[j + 1], ""
    j = 0
    while j < len(rows[0]) - 1 if rows else False:
        a = [r[j].strip() != "" for r in rows]
        b = [r[j + 1].strip() != "" for r in rows]
        if any(a) and any(b) and not any(x and y for x, y in zip(a, b)):
            for r in rows:
                r[j] = r[j] or r[j + 1]
                del r[j + 1]
        else:
            j += 1
    return rows


def is_value_row(row: list[str]) -> bool:
    """A row of values (numbers, ranges, +/- marks), at most one label: a table's body, not a header."""
    cells = [c.strip() for c in row if c.strip()]
    vals = sum(bool(PDF_NUMBER.match(c)) or c.lower() in PDF_MARKS for c in cells)
    return vals >= 2 and len(cells) - vals <= 1


def join_pdf_page_breaks(found: list[list]) -> list[tuple[str, int, list[list[str]]]]:
    """Tables cut by a page break joined: the first table of page p+1 continues the last table of
    page p when it has as many columns (filled somewhere) and either repeats that table's header
    row (dropped) or starts with a row of values instead of a header; or when the first runs to the
    bottom of its page and the second starts at the top of the next page without a caption, with as
    many columns (after leaving out a column that is empty throughout, ruled on that page only).
    Returns (pages, k, rows), the
    pages written "40" or "40-41"."""
    def width(rows):
        return sum(any(r[j].strip() for r in rows if j < len(r)) for j in range(max(len(r) for r in rows)))

    out: list[dict] = []
    for pno, k, n_on_page, rows, *rest in found:
        cap = rest[0] if rest else ""
        cont = rest[1] if len(rest) > 1 else False        # carried on by caption_tables with the same columns
        notes = rest[2] if len(rest) > 2 else ""
        span = rest[3] if len(rest) > 3 else None         # (top, bottom) on the page, as fractions of its height
        legend = rest[4] if len(rest) > 4 else ""         # "Supplementary Table 1" written under the table
        if out:
            prev = out[-1]
            same_header = [c.strip() for c in rows[0]] == [c.strip() for c in prev["rows"][0]]
            # a table running to the bottom of its page, then one at the top of the next page without a
            # caption: the same table, its header not repeated (its first row is a label row, "Other | VSD")
            runs_on = (span and prev["span"] and prev["span"][1] > 0.6 and span[0] < 0.3 and not cap)
            body = rows
            if runs_on and max(map(len, rows)) != max(map(len, prev["rows"])):
                body = drop_empty_columns(rows)               # a column ruled on one page only, empty
            if (pno == prev["last"] + 1 and k == 1 and prev["last_k"] == prev["n_last"]
                    and (cont and len(rows[0]) == len(prev["rows"][0]) or
                         width(rows) == width(prev["rows"]) and (same_header or is_value_row(rows[0])) or
                         runs_on and max(map(len, body)) == max(map(len, prev["rows"])))):
                prev["rows"] = prev["rows"] + (body[1:] if same_header else body)
                prev.update(last=pno, last_k=k, n_last=n_on_page, span=span, notes=" ".join(x for x in (prev["notes"], notes) if x),
                            cap=prev["cap"] or legend)
                continue
        out.append({"first": pno, "last": pno, "k": k, "last_k": k, "n_last": n_on_page, "rows": rows, "cap": cap or legend,
                    "notes": notes, "span": span})
    return [(str(x["first"]) if x["first"] == x["last"] else f"{x['first']}-{x['last']}", x["k"], x["rows"], x["cap"], x["notes"])
            for x in out]


LIST_START = re.compile(r"^([•▪◦*\-–]\s|\(?\d{1,2}[.)]\s|\(?[a-z][.)]\s)")
HYPHEN_CUT = re.compile(r"([A-Za-z]+)-$")


def pdf_paragraphs(text: str) -> str:
    """PDF prose as one paragraph per line. pdfplumber breaks every printed line; a break inside a
    paragraph becomes a space, and a word cut by a hyphen at the line end is rejoined ("intellectual
    dis-" / "ability" -> "disability"; kept as "follow-up" when the document writes that word with a
    hyphen elsewhere, or when the part before the hyphen is a capital or a single letter, "X-linked").
    A break is kept at a blank line, before a list item, after a short line ending a sentence (a
    paragraph's last line), and after a short line without end punctuation followed by a capital (a
    heading). A full line is the 80th percentile of the line lengths."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lens = sorted(len(ln.strip()) for ln in lines if len(ln.strip()) > 20)
    full = lens[min(len(lens) - 1, int(len(lens) * 0.8))] if lens else 80
    flat = " ".join(lines).lower()
    out, cur = [], ""
    for i, ln in enumerate(lines):
        a = ln.strip()
        if not a:
            if cur:
                out.append(cur)
                cur = ""
            continue
        if not cur:
            cur = a
        else:
            prev = cur.split("\n")[-1]
            m = HYPHEN_CUT.search(prev)
            if m and a[:1].islower():
                left, right = m.group(1), re.match(r"[A-Za-z]+", a)
                right = right.group(0) if right else ""
                keep = (len(left) == 1 or left[:1].isupper() and len(left) <= 3
                        or f"{left}-{right}".lower() in flat.replace(f"{left}-\n", "") and
                        f"{left}{right}".lower() not in flat)
                cur = cur + a if keep else cur[:-1] + a
            else:
                cur = cur + " " + a
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not nxt:
            continue
        ends_par = (re.search(r"[.!?:][\"')\]]?$", a) and len(a) < 0.75 * full)
        heading = (len(a) < 0.5 * full and not re.search(r"[,;\-–(.!?]$", a) and nxt[:1].isupper()
                   and len(a.split()) <= 8)
        if ends_par or heading or LIST_START.match(nxt) or nxt.startswith("## "):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return "\n".join(out)


TABLE_CAPTION = re.compile(r"^\s*(supplementa(l|ry)\s+)?(table|tab\.)\s*[A-Z]?\d+", re.I)

TABLE_FOOT = re.compile(r"^\s*([*†‡§¶#‖]|[a-z]\s|abbreviations?\b|notes?\b|data are\b|values are\b|key\b)", re.I)


def _pdf_lines(page, exclude=()):
    """The page's words (outside `exclude`) as lines sorted top-down, each line's words left to right;
    plus the median word height and the median gap between words."""
    words = [w for w in page.extract_words(keep_blank_chars=False, use_text_flow=False)
             if not any(x0 <= (w["x0"]+w["x1"])/2 <= x1 and t <= (w["top"]+w["bottom"])/2 <= b for x0, t, x1, b in exclude)]
    lines, cur = [], []
    for w in sorted(words, key=lambda w: w["top"]):
        if cur and abs(w["top"] - cur[-1]["top"]) > 2.5:
            lines.append(sorted(cur, key=lambda w: w["x0"])); cur = []
        cur.append(w)
    if cur:
        lines.append(sorted(cur, key=lambda w: w["x0"]))
    gaps = [b["x0"] - a["x1"] for ln in lines for a, b in zip(ln, ln[1:]) if b["x0"] > a["x1"]]
    size = statistics.median(w["bottom"] - w["top"] for w in words) if words else 10.0
    small = [g for g in gaps if g < 0.8 * size]          # spaces between words (on a table page most gaps are columns)
    space = statistics.median(small) if small else 0.25 * size
    return lines, size, space


def _pieces(ln, wide):
    out, cur = [], [ln[0]]
    for a, b in zip(ln, ln[1:]):
        if b["x0"] - a["x1"] > wide:
            out.append(cur); cur = []
        cur.append(b)
    out.append(cur)
    return out


def _prose_line(ln, bounds, space):
    """A line of running text below a table (a footnote, a paragraph): words at normal spacing that
    run across two or more column boundaries, which a table cell never does."""
    run = [ln[0]]
    for a, b in zip(ln, ln[1:]):
        if b["x0"] - a["x1"] <= 1.6 * space:
            run.append(b)
        else:
            run = [b]
        if sum(run[0]["x0"] < x <= run[-1]["x0"] for x in bounds) >= 2:
            return True
    return False


def _data_bounds(body, bounds, wide, size):
    """Column edges from where the cells start: the left edges of the pieces of every line (header
    and body) right of the row labels, grouped when within 0.8 x the text height, kept when 3 or
    more lines start a piece there. Used instead of the header's pieces when it finds at least as
    many columns: header labels set close together ("Patient 1-1 Patient 2-1") read as one piece,
    and a header centred over its column is not where the cells start."""
    second = sorted(ps[1][0]["x0"] for ps in (_pieces(ln, wide) for ln in body) if len(ps) >= 3)
    left = min(bounds[0], second[len(second) // 10] - 0.8 * size) if second else bounds[0]   # where the first data column starts
    xs = sorted(p[0]["x0"] for ln in body for p in _pieces(ln, wide) if p[0]["x0"] >= left)
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 0.8 * size:
            groups[-1].append(x)
        else:
            groups.append([x])
    starts = [g[0] for g in groups if len(g) >= 3]
    if len(starts) < len(bounds):
        return bounds
    return [x - 0.3 * size for x in starts]


class TableRows(list):
    """A table's rows, with the footnotes printed under it (.notes)."""
    notes = ""


def table_notes(page, page_tables):
    """The footnotes printed under each table on a PDF page: the lines just below it that start
    with a footnote mark used in the table (†, *, ¶, §, •, °, a, b ...), with a word of the table
    ("(-), (+) indicate ..."), with "Abbreviations", "Note" ..., or that list abbreviations
    ("Ht: height; Wt: weight"), each with the lines it wraps
    onto. They stop at a caption, at a line of another kind, or at a wide gap (4 x the text height).
    A "Table N" line just under a table without a caption is its legend: taken as the footnotes,
    and its "Table N" as the caption. Returns {index of the table: (text, caption, bbox)}."""
    lines, size, space = _pdf_lines(page)
    out = {}
    for n, (bb, cap, rows, cont) in enumerate(page_tables):
        flat = " ".join(c for r in rows for c in r)
        marks = {ch for ch in flat if not ch.isalnum() and not ch.isspace() and ch not in ",.;:-–()/'\"[]%+=<>"}
        tokens = {t.strip(",.;:") for t in flat.split()}
        nxt = min([b[1] for b, *_ in page_tables if b[1] > bb[3] + 1] + [page.height])
        got, prev_b, prev_note, below = [], bb[3], False, ""
        for ln in lines:
            top = ln[0]["top"]
            if top <= bb[3] - 1 or top >= nxt:
                continue
            text = " ".join(w["text"] for w in ln)
            if top - prev_b > 4 * size:
                break
            if TABLE_CAPTION.match(text):
                if got or cap:
                    break                             # the next table
                below = TABLE_CAPTION.match(text).group(0)   # a legend under a table without a caption:
                got.append(ln); prev_b = max(w["bottom"] for w in ln); prev_note = True   # "Supplementary Table 1: Abbreviations: ..."
                continue
            first = text.split()[0]
            is_note = (TABLE_FOOT.match(text) or text[0] in marks
                       or re.match(r"\S{1,10}\s?[:=]\s", text) and ";" in text      # "Ht: height; Wt: weight; ..."
                       or (first.strip(",.;:") in tokens and not first[0].isalnum()))
            wrapped = prev_note and top - prev_b <= 1.5 * size and ln[0]["x0"] >= got[-1][0]["x0"] - 2
            if not (is_note or wrapped):
                break
            got.append(ln); prev_b = max(w["bottom"] for w in ln); prev_note = True
        if got:
            words = [w for ln in got for w in ln]
            out[n] = (" ".join(" ".join(w["text"] for w in ln) for ln in got), below,
                      (min(w["x0"] for w in words) - 1, min(w["top"] for w in words) - 1,
                       max(w["x1"] for w in words) + 1, max(w["bottom"] for w in words) + 1))
    return out


def _group_owner(page, body, words, wide, size):
    """For a group row (body[0]) above the column header (body[1]): the group name of each column,
    from the drawing. Columns between two vertical rules (drawn beside the group row) belong to the
    name between the same rules;
    without rules, a name set at a column's left edge runs to the next name, and centred names
    (most often) take the columns nearest to their centre. None when it cannot be told."""
    if len(body) < 2:
        return None
    gap = 1.6 * statistics.median([b["x0"] - a["x1"] for ln in body for a, b in zip(ln, ln[1:]) if 0 < b["x0"] - a["x1"] < 0.8 * size] or [0.25 * size])
    labels = [{"x0": p[0]["x0"], "x1": p[-1]["x1"], "text": " ".join(w["text"] for w in p)} for p in _pieces(body[0], gap)]
    col = {id(w): c for w, c in words}
    span = {}
    for w in body[1]:
        c = col.get(id(w))
        if c:
            a, b = span.get(c, (w["x0"], w["x1"]))
            span[c] = (min(a, w["x0"]), max(b, w["x1"]))
    if not labels or not span:
        return None
    top, bottom = min(w["top"] for w in body[0]), max(w["bottom"] for w in body[0])
    rules = sorted({round((r["x0"] + r["x1"]) / 2, 1) for r in list(page.lines) + list(page.rects)
                    if r["x1"] - r["x0"] < 3 and r["bottom"] - r["top"] > 2 and r["top"] <= top + 1 and r["bottom"] >= bottom - 1})
    owner = {}
    for c, (a, b) in span.items():
        cx = (a + b) / 2
        if rules:
            lo = max([x for x in rules if x <= cx], default=-1e9); hi = min([x for x in rules if x > cx], default=1e9)
            inside = [l for l in labels if lo < (l["x0"] + l["x1"]) / 2 < hi]
            owner[c] = inside[0]["text"] if len(inside) == 1 else None
        elif all(any(abs(l["x0"] - a2) <= 0.8 * size for a2, _ in span.values()) for l in labels):
            left = [l for l in labels if l["x0"] <= a + 0.8 * size]
            owner[c] = left[-1]["text"] if left else None
        else:
            owner[c] = min(labels, key=lambda l: abs((l["x0"] + l["x1"]) / 2 - cx))["text"]
    return owner


def _group_header(rows, owner=None):
    """A group row above the column header (family names over their members: "TAA288 ... MS239")
    folded into it: each group name goes on the header cells of its columns ("TAA288 III:2", ...,
    "MS239 III:16"), as the patients are named elsewhere. `owner` (from _group_owner) says which
    columns are under which name; without it a name runs from its column to the next name's."""
    if len(rows) < 3 or rows[0][0].strip():
        return rows
    top, head = rows[0], rows[1]
    filled = [j for j, c in enumerate(top) if c.strip()]
    n_head = sum(1 for c in head[1:] if c.strip())
    if not filled or len(filled) > n_head / 2 or any(not head[j].strip() for j in filled):
        return rows
    group, new = "", list(head)
    for j in range(1, len(head)):
        if top[j].strip():
            group = top[j].strip()
        if owner:
            group = owner.get(j) or ""
        if group and head[j].strip() and not head[j].startswith(group):
            new[j] = f"{group} {head[j].strip()}"
    return [new] + rows[2:]


def _row_rules(page, words):
    """Row bands from the PDF's drawing: the edges of shaded row backgrounds (filled rectangles) and
    horizontal rules that span at least half the table's width. [] when there are fewer than 3
    such edges inside the table (then rows come from the white space)."""
    x0 = min(w["x0"] for w, _ in words); x1 = max(w["x1"] for w, _ in words)
    top = min(w["top"] for w, _ in words); bottom = max(w["bottom"] for w, _ in words)
    ys = []
    for r in list(page.rects) + list(page.lines):
        if min(r["x1"], x1) - max(r["x0"], x0) < 0.5 * (x1 - x0):
            continue
        if r.get("object_type") == "line" and abs(r["bottom"] - r["top"]) > 1:
            continue                                  # a vertical line
        for y in ((r["top"], r["bottom"]) if r["bottom"] - r["top"] > 1.5 else ((r["top"] + r["bottom"]) / 2,)):
            if top - 1 < y < bottom + 1:
                ys.append(y)
    ys = sorted(ys)
    edges = []
    for y in ys:
        if not edges or y - edges[-1] > 1.5:
            edges.append(y)
    if len(edges) < 3:
        return []
    edges = [top - 0.5] + edges + [bottom + 0.5]
    mids = [(w["top"] + w["bottom"]) / 2 for w, _ in words]
    return [(a, b) for a, b in zip(edges, edges[1:]) if any(a <= m < b for m in mids)]


def caption_tables(page, exclude=(), carry=None):
    """Tables found from their caption ("Table 2", "Supplemental Table 2. ..."), for borderless tables
    whose cells wrap over several lines (text_tables cut those into fragments, and merged cells
    whose columns are only a few points apart). The caption is the cue for the start; the column
    header is the first line below it with 3 or more pieces, and its pieces give the columns
    (a word goes to the column whose band holds its left edge; the bands split the gaps between
    header pieces; a word right after a normal space stays with the word before it in the first
    cell, e.g. a long row label or a trailing "-"). Rows are the bands of lines separated by more white space than the
    line spacing inside a cell (cells centred vertically interleave their lines, so lines are not
    rows). The table ends at a footnote line, the next caption, a wide vertical gap, or a line that
    starts left of the table. A table reaching the page bottom continues at the top of the next page
    with the same columns: pass the returned `carry`.
    Returns ([(bbox, caption, rows, continued)], carry)."""
    lines, size, space = _pdf_lines(page, exclude)
    if len(lines) < 3:
        return [], None
    wide = max(2.5 * space, 0.6 * size)
    tol = 0.8 * size
    out, i, new_carry, out_end = [], 0, None, -1
    starts = []                                       # (line index where the body starts, caption, bounds, left, continued)
    if carry:
        top_lines = [k for k, ln in enumerate(lines) if ln[0]["top"] < page.height * 0.3]
        for k in top_lines:                           # skip a running head: start at the first line with 2+ pieces
            if TABLE_CAPTION.match(" ".join(w["text"] for w in lines[k])):
                break                                 # a new table starts here instead
            if len(_pieces(lines[k], wide)) >= 2:
                starts.append((k, carry["cap"], carry["bounds"], carry["left"], True)); break
    for k, ln in enumerate(lines):
        text = " ".join(w["text"] for w in ln)
        if not TABLE_CAPTION.match(text):
            continue
        near = [q for q in range(k + 1, min(k + 6, len(lines))) if len(_pieces(lines[q], wide)) >= 3]
        if not near:
            continue
        h = max(near, key=lambda q: (len(_pieces(lines[q], wide)), -q))   # the column header: the line with most pieces
        cap_end = k + 1
        while cap_end < h and len(_pieces(lines[cap_end], wide)) == 1 and lines[cap_end][0]["top"] - lines[cap_end - 1][-1]["bottom"] < 0.8 * size:
            cap_end += 1                              # the caption wrapped onto more lines
        cap = " ".join(" ".join(w["text"] for w in lines[q]) for q in range(k, cap_end))
        hp = _pieces(lines[h], wide)
        bounds = [hp[1][0]["x0"] - tol] + [(hp[q - 1][-1]["x1"] + hp[q][0]["x0"]) / 2 for q in range(2, len(hp))]
        starts.append((cap_end, cap, bounds, min(hp[0][0]["x0"], ln[0]["x0"]), False))
    starts.sort()
    for n, (b0, cap, bounds, left, cont) in enumerate(starts):
        if starts and n and b0 <= out_end:
            continue
        body = []
        for q in range(b0, len(lines)):
            ln = lines[q]
            text = " ".join(w["text"] for w in ln)
            if body:
                if TABLE_CAPTION.match(text) or ln[0]["top"] - max(w["bottom"] for w in lines[q - 1]) > 2.5 * size:
                    break
                if len(_pieces(ln, wide)) == 1 and TABLE_FOOT.match(text) and ln[0]["x0"] <= left + tol:
                    break
            if ln[0]["x0"] < left - 2 * tol:
                break
            if body and _prose_line(ln, bounds, space):
                break
            body.append(ln)
        out_end = b0 + len(body)
        if len(body) < 2:
            continue
        if not cont:
            bounds = _data_bounds(body, bounds, wide, size)
        ncol = len(bounds) + 1
        words = []                                    # (word, column)
        for ln in body:
            prev = None
            for w in ln:
                c = sum(w["x0"] >= b for b in bounds)
                if prev and c == prev[1] + 1 and w["x0"] - prev[0]["x1"] <= 1.6 * space and w["x0"] - bounds[c - 1] > 0.6 * size:
                    c = prev[1]                       # one space after the word before: the same cell running past its band
                words.append((w, c)); prev = (w, c)
        # line spacing inside a cell: the step between successive lines of the same column
        steps = []
        for c in range(ncol):
            tops = sorted({round(w["top"], 1) for w, cc in words if cc == c})
            steps += [b - a for a, b in zip(tops, tops[1:]) if b - a > 2.5]
        # the smallest step seen 3+ times: wrapped lines of a cell (steps between rows are larger)
        step = next((x for x in sorted(steps) if sum(abs(y - x) <= 0.5 for y in steps) >= 3), 1.2 * size)
        lead = max(0.0, step - size)                  # white space between the lines of one cell
        single = steps and sum(abs(y - step) <= 0.5 for y in steps) >= 0.8 * len(steps)   # no cell wraps: a row per line
        iv = sorted((w["top"], w["bottom"]) for w, _ in words)
        bands, (t, b) = [], iv[0]
        for t2, b2 in iv[1:]:
            if t2 - b > lead + 0.3 * size:
                bands.append((t, b)); t, b = t2, b2
            else:
                b = max(b, b2)
        bands.append((t, b))
        ruled = _row_rules(page, words)               # row shading or rules drawn in the PDF: the real row edges
        if ruled:
            bands, single = ruled, False
        rows = []
        for t, b in bands:
            cell = [[] for _ in range(ncol)]
            for w, c in words:
                m = (w["top"] + w["bottom"]) / 2
                if t - 0.1 <= m <= b + 0.1 if not ruled else t <= m < b:
                    cell[c].append(w)
            rows.append([" ".join(x["text"] for x in sorted(ws, key=lambda w: (round(w["top"]), w["x0"]))) for ws in cell])
        if single or len(bands) <= 2 and len(body) > 4:   # a row per line (a line without a label continues the row above)
            rows = []
            for ln in body:
                cells = [""] * ncol
                for w in ln:
                    c = next(cc for ww, cc in words if ww is w)
                    cells[c] = (cells[c] + " " + w["text"]).strip()
                if rows and (not cells[0] or not any(cells[1:]) and any(rows[-1][1:])):
                    rows[-1] = [(a + " " + x).strip() for a, x in zip(rows[-1], cells)]   # a cell or label wrapped onto this line
                else:
                    rows.append(cells)
        rows = _group_header(rows, _group_owner(page, body, words, wide, size))
        allw = [w for ln in body for w in ln]
        bbox = (min(w["x0"] for w in allw), min(w["top"] for w in allw), max(w["x1"] for w in allw), max(w["bottom"] for w in allw))
        if not cont and b0 > 0:                       # include the caption lines in the region
            capk = next(q for q in range(b0 - 1, -1, -1) if TABLE_CAPTION.match(" ".join(w["text"] for w in lines[q])))
            bbox = (min(bbox[0], lines[capk][0]["x0"]), lines[capk][0]["top"], bbox[2], bbox[3])
        out.append((bbox, cap, rows, cont))
        if out_end >= len(lines) and bbox[3] > page.height * 0.85:   # the table runs to the page bottom
            new_carry = {"cap": cap, "bounds": bounds, "left": left}
        else:
            new_carry = None
    return out, new_carry


def text_tables(page, exclude=()):
    """Tables without ruling lines (pdfplumber's find_tables only sees ruled tables; a borderless table
    stayed in the page text as run-on lines). Found from the words' positions: a run of at least 4
    lines that each break into 3 or more pieces at wide gaps (2.5 x the page's usual word gap, at
    least 0.6 x the text height), whose pieces start at the same x in at least 3 columns. A line with
    one piece between such lines, or just below, is a label wrapped onto the next line. A "Table N" /
    "Supplementary Table N" line just above is the caption. Two-column prose (2 pieces per line) and
    ordinary text are not taken. Returns [(bbox, caption, rows)]."""
    words = [w for w in page.extract_words(keep_blank_chars=False, use_text_flow=False)
             if not any(x0 <= (w["x0"]+w["x1"])/2 <= x1 and t <= (w["top"]+w["bottom"])/2 <= b for x0, t, x1, b in exclude)]
    if len(words) < 12:
        return []
    words.sort(key=lambda w: (round(w["top"]), w["x0"]))
    lines, cur = [], []
    for w in sorted(words, key=lambda w: w["top"]):
        if cur and abs(w["top"] - cur[-1]["top"]) > 2.5:
            lines.append(sorted(cur, key=lambda w: w["x0"])); cur = []
        cur.append(w)
    if cur:
        lines.append(sorted(cur, key=lambda w: w["x0"]))
    gaps = [b["x0"] - a["x1"] for ln in lines for a, b in zip(ln, ln[1:]) if b["x0"] > a["x1"]]
    if not gaps:
        return []
    space = statistics.median(gaps)
    size = statistics.median(w["bottom"] - w["top"] for w in words)
    wide = max(2.5 * space, 0.6 * size)
    def chunks(ln):
        out, cur = [], [ln[0]]
        for a, b in zip(ln, ln[1:]):
            if b["x0"] - a["x1"] > wide:
                out.append(cur); cur = []
            cur.append(b)
        out.append(cur)
        return [{"x0": c[0]["x0"], "x1": c[-1]["x1"], "top": min(w["top"] for w in c),
                 "bottom": max(w["bottom"] for w in c), "text": " ".join(w["text"] for w in c)} for c in out]
    L = [chunks(ln) for ln in lines]
    tables, i = [], 0
    while i < len(L):
        if len(L[i]) < 3:
            i += 1; continue
        j = i
        while j + 1 < len(L) and (len(L[j + 1]) >= 3 or (len(L[j + 1]) <= 2 and j + 2 < len(L) and len(L[j + 2]) >= 3
                                  and L[j + 1][0]["top"] - L[j][0]["bottom"] < 1.5 * size)):
            j += 1
        if (j + 1 < len(L) and len(L[j + 1]) == 1 and L[j + 1][0]["top"] - L[j][0]["bottom"] < 1.5 * size
                and abs(L[j + 1][0]["x0"] - L[i][0]["x0"]) <= 0.8 * size):
            j += 1                                      # the last row's label wrapped onto one more line
        block = L[i:j + 1]
        if sum(len(c) >= 3 for c in block) >= 4:
            starts = sorted(c["x0"] for ln in block if len(ln) >= 3 for c in ln)
            cols = []
            for x in starts:
                if cols and x - cols[-1][-1] <= 0.8 * size:
                    cols[-1].append(x)
                else:
                    cols.append([x])
            need = max(2, 0.3 * sum(len(c) >= 3 for c in block))
            cols = [statistics.median(c) for c in cols if len(c) >= need]
            if len(cols) >= 3:
                rows = []
                for ln in block:
                    row = [""] * len(cols)
                    for c in ln:
                        k = min(range(len(cols)), key=lambda k: abs(cols[k] - c["x0"]) if cols[k] <= c["x0"] + 0.8 * size else 1e9)
                        row[k] = (row[k] + " " + c["text"]).strip()
                    if len(ln) <= 2 and rows and not any(row[1:]):
                        rows[-1][0] = (rows[-1][0] + " " + row[0]).strip()   # a label wrapped onto the next line
                    else:
                        rows.append(row)
                top = min(c["top"] for ln in block for c in ln); bottom = max(c["bottom"] for ln in block for c in ln)
                x0 = min(c["x0"] for ln in block for c in ln); x1 = max(c["x1"] for ln in block for c in ln)
                cap = ""
                for back in range(i - 1, max(-1, i - 4), -1):           # a "Table N" caption just above
                    t = " ".join(c["text"] for c in L[back])
                    if TABLE_CAPTION.match(t):
                        cap = " ".join(" ".join(c["text"] for c in L[q]) for q in range(back, i))
                        top = min(top, L[back][0]["top"])
                        break
                tables.append(((x0, top, x1, bottom), cap, rows))
        i = j + 1
    return tables


def is_real_table(rows: list[list[str]]) -> bool:
    """A PDF table worth keeping: at least 2 rows with 2 or more filled cells, and short cells.
    pdfplumber also returns framed text boxes (e.g. a reviewer comment in a box) as one-column
    'tables' of long paragraphs; those are left to the page text."""
    filled = [[c for c in r if c.strip()] for r in rows]
    multi = [r for r in filled if len(r) >= 2]
    if len(multi) < 2:
        return False
    cells = sorted(len(c) for r in filled for c in r)
    return cells[len(cells) // 2] <= 100                  # median cell length: data, not prose


def read_supplementary(name: str, data: bytes, depth: int = 0) -> list[tuple[str, str, object]]:
    """One supplementary file -> [("table", title, rows) | ("text", title, text)]."""
    import io
    ext = os.path.splitext(name)[1].lower()
    out: list[tuple[str, str, object]] = []
    if ext in (".csv", ".tsv", ".txt", ".tab"):
        text = data.decode("utf-8", errors="replace").replace("\x00", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")      # old Mac / mixed line endings
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # very long cells
        sample = text[:4096]
        if ext in (".txt", ".tab") and "\t" not in sample:
            out.append(("text", name, text))              # prose
        else:
            delim = "\t" if sample.count("\t") >= sample.count(",") else ","
            out.append(("table", name, [[clean(c) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delim)]))
    elif ext in (".xlsx", ".xlsm"):
        import openpyxl                                   # ImportError is reported by the caller
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = [[clean(str(c)) if c is not None else "" for c in r] for r in ws.iter_rows(values_only=True)]
            out.append(("table", f"{name} / {ws.title}", rows))
    elif ext == ".xls" and data[:2] == b"PK":             # really an .xlsx with the wrong extension
        out += read_supplementary(name[:-4] + ".xlsx", data, depth)
    elif ext == ".xls" and data.lstrip()[:1] == b"<":     # HTML or XML table saved as .xls
        out += [("table", f"{name} / table {i}", rows) for i, rows in enumerate(_html_tables(data), 1)]
    elif ext == ".xls":
        import xlrd                                       # pip install xlrd
        wb = xlrd.open_workbook(file_contents=data)
        for ws in wb.sheets():
            rows = [[clean(str(ws.cell_value(r, c))) for c in range(ws.ncols)] for r in range(ws.nrows)]
            out.append(("table", f"{name} / {ws.name}", rows))
    elif ext in (".html", ".htm"):
        tables = _html_tables(data)
        out += [("table", f"{name} / table {i}", rows) for i, rows in enumerate(tables, 1)]
        prose = _html_text(data, drop_tables=True)        # the prose around the tables, tables left out
        if re.search(r"[A-Za-z]{3,}.*\s.*[A-Za-z]{3,}", prose):
            out.append(("text", name, prose))
    elif ext == ".docx":
        parts = _docx_parts(data)
        n_tab, text = 0, []
        for i, (kind, val) in enumerate(parts):
            if kind == "tbl":
                n_tab += 1
                prev = parts[i - 1][1] if i and parts[i - 1][0] in ("p", "h") else ""
                cap = prev if isinstance(prev, str) and len(prev) < 400 else ""
                out.append(("table", f"{name} / table {n_tab}" + (f": {cap}" if cap else ""), val))
            elif kind == "h":
                text.append(f"## {val}")                  # Word heading style, marked for later splitting
            else:
                text.append(val)
        if text:
            out.append(("text", name, "\n".join(text)))
    elif ext == ".pdf":
        import pdfplumber                                 # pip install pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            first = " ".join((p.extract_text() or "") for p in pdf.pages[:2])
            if PEER_REVIEW.search(first):                 # reviewer comments and responses, not data
                SUPP_SKIPPED[".pdf (peer review file)"] = SUPP_SKIPPED.get(".pdf (peer review file)", 0) + 1
                return []
            text, found = [], []                      # found: [page, k, number of tables on the page, rows, caption, continued]
            carry = None                                  # a captioned table running on to the next page
            for pno, page in enumerate(pdf.pages, 1):
                boxes = []                                # regions of the real tables on this page
                page_tables = []
                for tb in page.find_tables() or []:
                    rows = tidy_pdf_rows([[clean_cell(c or "") for c in r] for r in tb.extract()])
                    if is_real_table(rows):               # boxed paragraphs are detected as tables too
                        page_tables.append((tb.bbox, "", rows))
                        boxes.append(tb.bbox)
                got, carry = caption_tables(page, exclude=list(boxes), carry=carry)   # captioned, cells over several lines
                for bbox, cap, rows, cont in got:
                    rows = tidy_pdf_rows([[clean_cell(c) for c in r] for r in rows])
                    if is_real_table(rows):
                        page_tables.append((bbox, cap, rows, cont))
                        boxes.append(bbox)
                for bbox, cap, rows in text_tables(page, exclude=list(boxes)):   # tables without ruling lines
                    rows = tidy_pdf_rows([[clean_cell(c) for c in r] for r in rows])
                    if is_real_table(rows):
                        page_tables.append((bbox, cap, rows, False))
                        boxes.append(bbox)
                page_tables = [t if len(t) == 4 else (*t, False) for t in page_tables]
                page_tables.sort(key=lambda t: (t[0][1], t[0][0]))   # in reading order on the page
                notes = table_notes(page, page_tables) if page_tables else {}
                for k, (bb, cap, rows, cont) in enumerate(page_tables, 1):
                    nt = notes.get(k - 1, ("", "", None))
                    found.append([pno, k, len(page_tables), rows, cap, cont, nt[0],
                                  (bb[1] / page.height, bb[3] / page.height), nt[1]])   # where the table starts and ends on its page
                    if nt[2]:
                        boxes.append(nt[2])               # the footnotes go with the table, not the page text
                if boxes:                                 # page text without the tables' characters

                    def outside(obj, boxes=boxes) -> bool:
                        if obj.get("object_type") != "char":
                            return True
                        cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
                        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)
                    text.append(page.filter(outside).extract_text() or "")
                else:
                    text.append(page.extract_text() or "")
            for title_pages, k, rows, cap, notes in join_pdf_page_breaks(found):
                rows = TableRows(rows); rows.notes = notes
                out.append(("table", f"{name} / page {title_pages} table {k}" + (f": {cap[:300]}" if cap else ""), rows))
            if any(t.strip() for t in text):
                out.append(("text", name, pdf_paragraphs("\n".join(text))))
    elif ext in (".doc", ".rtf"):                         # old Word / RTF: may really be .docx or HTML
        if data[:2] == b"PK":
            out += read_supplementary(os.path.splitext(name)[0] + ".docx", data, depth)
        elif data.lstrip()[:1] == b"<":                   # a web page saved with a .doc name
            out += read_supplementary(os.path.splitext(name)[0] + ".html", data, depth)
        else:                                             # binary Word 97-2003 or RTF: convert with LibreOffice
            converted = convert_to_docx(name, data)
            if converted is None:
                raise NotImplementedError(f"{ext} (needs LibreOffice)")
            out += [(k, t.replace(os.path.splitext(name)[0] + ".docx", name), v)
                    for k, t, v in read_supplementary(os.path.splitext(name)[0] + ".docx", converted, depth)]
    elif ext == ".zip" and depth < 2:
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.infolist():
                base = os.path.basename(m.filename)
                if m.is_dir() or base.startswith(("._", "~$")) or "__MACOSX" in m.filename:
                    continue
                out += read_supplementary_safe(f"{name}/{base}", z.read(m), depth + 1)
    else:
        raise NotImplementedError(ext)
    return out


def find_soffice() -> str | None:
    """LibreOffice's command-line program, if installed (PATH, or the usual Windows / macOS place)."""
    import shutil
    for cand in ("soffice", "libreoffice"):
        if shutil.which(cand):
            return shutil.which(cand)
    for path in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                 "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if os.path.exists(path):
            return path
    return None


def convert_to_docx(name: str, data: bytes) -> bytes | None:
    """A .doc/.rtf converted to .docx by LibreOffice in a temporary folder (the original is not
    touched); None if LibreOffice is not installed or the conversion fails."""
    import pathlib
    import subprocess
    import tempfile
    soffice = find_soffice()
    if not soffice:
        return None
    with tempfile.TemporaryDirectory(prefix="doc2docx_") as tmp:
        src = os.path.join(tmp, "in" + os.path.splitext(name)[1].lower())
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            subprocess.run([soffice, "--headless", "-env:UserInstallation=" + pathlib.Path(tmp, "profile").as_uri(),
                            "--convert-to", "docx", "--outdir", tmp, src],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        out = os.path.join(tmp, "in.docx")
        if not os.path.exists(out):
            return None
        with open(out, "rb") as fh:
            return fh.read()


def read_supplementary_safe(name: str, data: bytes, depth: int = 0) -> list[tuple[str, str, object]]:
    ext = os.path.splitext(name)[1].lower() or "(none)"
    if ext in NO_TABLE_EXT:
        return []
    try:
        got = read_supplementary(name, data, depth)
        SUPP_READ[ext] = SUPP_READ.get(ext, 0) + 1
        return got
    except ImportError as e:
        SUPP_SKIPPED[f"{ext} (needs: pip install {e.name})"] = SUPP_SKIPPED.get(f"{ext} (needs: pip install {e.name})", 0) + 1
    except NotImplementedError as e:
        key = str(e) if str(e).startswith(ext) else ext   # e.g. ".doc (needs LibreOffice)"
        SUPP_SKIPPED[key] = SUPP_SKIPPED.get(key, 0) + 1
    except Exception as e:                                # noqa: BLE001 - a bad file must not stop the run
        print(f"  warning: could not read {name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        SUPP_SKIPPED[f"{ext} (unreadable)"] = SUPP_SKIPPED.get(f"{ext} (unreadable)", 0) + 1
    return []


def supplementary_sections(folder: str, max_chars: int, max_chunks: int,
                           captions: dict[str, str] | None = None) -> list[tuple[str, str, str, str]]:
    """Units from the article's supplementary files: tables as suppl_table, prose as suppl_text."""
    out: list[tuple[str, str, str, str]] = []
    captions = captions or {}
    max_chunks = max_chunks if max_chunks > 0 else 10 ** 9   # 0 = no cap
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            items = read_supplementary_safe(name, fh.read())
        cap = captions.get(name.lower(), "")
        for kind, title, val in items:
            head = f"Supplementary file: {title}" + (f" ({cap})" if cap else "")
            if kind == "table":
                rows = [r for r in val if any(c.strip() for c in r)]  # type: ignore[union-attr]
                if len(rows) < 2:
                    continue
                notes = getattr(val, "notes", "")
                header = [head] + ([f"Footnotes: {notes}"] if notes else []) + ["Rows are tab-separated; the first row is the column header."]
                limit = table_limit(max_chars)
                chunks = rows_to_chunks(header, rows, 1, limit)
                if len(chunks) > max_chunks:
                    print(f"  note: {title} truncated to {max_chunks} of {len(chunks)} chunks", file=sys.stderr)
                    TRUNCATED.append([title, "table", max_chunks, len(chunks)])
                kind = "oversized" if any(len(c) > limit for c in chunks) else "suppl_table"  # skipped whole
                for i, chunk in enumerate(chunks[:max_chunks]):
                    out.append((kind, title + (f" (part {i + 1})" if i else ""), chunk, "suppl_file"))
            else:
                parts = split_at_headings(clean_lines(str(val)), max_chars)   # at "## " headings when long
                if len(parts) > max_chunks:
                    print(f"  note: {title} truncated to {max_chunks} of {len(parts)} parts", file=sys.stderr)
                    TRUNCATED.append([title, "text", max_chunks, len(parts)])
                for i, part in enumerate(parts[:max_chunks]):
                    out.append(("suppl_text", title + (f" (part {i + 1})" if i else ""), f"{head}\n{part}", "suppl_file"))
    return out


def clean_lines(text: str) -> str:
    return "\n".join(c for c in (clean(ln) for ln in text.splitlines()) if c)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "section"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--articles", required=True, help="the articles/ folder from the download script")
    ap.add_argument("--out", required=True, help="output folder for section text files")
    ap.add_argument("--max-chars", type=int, default=12000, help="split units longer than this (default 12000)")
    ap.add_argument("--max-table-chars", type=int, default=0,
                    help="size limit for table and suppl_table chunks (default: the same as --max-chars); "
                         "a table with a header or row longer than this is skipped")
    ap.add_argument("--per-paragraph", action="store_true",
                    help="one unit per paragraph instead of per section (tables are unchanged)")
    ap.add_argument("--supplementary", action="store_true", help="also convert supplementary files (tables and prose)")
    ap.add_argument("--max-supp-chunks", type=int, default=200,
                    help="cap on chunks per supplementary sheet or file (default 200; 0 = no cap; "
                         "a note is printed when it applies)")
    ap.add_argument("--no-figures-back", action="store_true",
                    help="leave out figure legends and back-matter sections (the behaviour of earlier versions)")
    args = ap.parse_args()

    global PER_PARAGRAPH, TABLE_CHARS
    PER_PARAGRAPH = args.per_paragraph
    TABLE_CHARS = args.max_table_chars
    os.makedirs(args.out, exist_ok=True)
    index, n_articles, failed, skipped, truncated = [], 0, [], [], []
    for adir in sorted(glob.glob(os.path.join(args.articles, "PMID_*"))):
        pmid = os.path.basename(adir).split("_", 1)[1]
        xmls = sorted(glob.glob(os.path.join(adir, "fulltext", "*.xml")))
        if not xmls:
            failed.append((pmid, "no XML"))
            continue
        root = parse_xml(xmls[-1])
        if root is None:
            failed.append((pmid, "XML could not be parsed"))
            continue
        units = article_sections(root, args.max_chars)
        if args.supplementary:
            TRUNCATED.clear()
            units += supplementary_sections(os.path.join(adir, "supplementary"), args.max_chars, args.max_supp_chunks,
                                            supplementary_captions(root))
            truncated += [[pmid] + t for t in TRUNCATED]
        if not args.no_figures_back:
            units += late_sections(root, args.max_chars)
        if not units:
            failed.append((pmid, "no text found"))
            continue
        odir = os.path.join(args.out, f"PMID_{pmid}")
        os.makedirs(odir, exist_ok=True)
        for old in glob.glob(os.path.join(odir, "*.txt")):
            os.remove(old)
        for i, (kind, title, text, part) in enumerate(units, 1):
            if kind in ("skip", "empty", "oversized") or part == "sub-article":   # keeps its number
                skipped.append([pmid, f"{i:02d}", title, len(text),
                                "sub-article (peer review)" if part == "sub-article" else
                                "heading only" if kind == "empty" else
                                "table over --max-chars" if kind == "oversized" else "SKIP list"])
                continue
            sid = f"{i:02d}_{kind}_{slug(title)}"
            path = os.path.join(odir, sid + ".txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            index.append([pmid, sid, kind, part, title, len(text), os.path.relpath(path, args.out)])
        n_articles += 1

    with open(os.path.join(args.out, "index.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pmid", "section_id", "kind", "part", "title", "n_chars", "path"])
        w.writerows(index)
    with open(os.path.join(args.out, "skipped.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pmid", "number", "title", "n_chars", "reason"])
        w.writerows(skipped)
    with open(os.path.join(args.out, "truncated.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")              # supplementary files cut by --max-supp-chunks
        w.writerow(["pmid", "title", "type", "kept", "total"])
        w.writerows(truncated)
    kinds: dict[str, int] = {}
    for r in index:
        kinds[r[2]] = kinds.get(r[2], 0) + 1
    print(f"{n_articles} articles, {len(index)} units: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    parts: dict[str, int] = {}
    for r in index:
        parts[r[3]] = parts.get(r[3], 0) + 1
    print("units by part: " + ", ".join(f"{k} {v}" for k, v in sorted(parts.items())))
    why = Counter(r[4] for r in skipped)
    print(f"{why['SKIP list']} administrative sections skipped (SKIP list)"
          + (f", {why['heading only']} appendices with only a heading (no text)" if why["heading only"] else "")
          + (f", {why['table over --max-chars']} chunks of tables with a header or row longer than --max-chars"
             if why["table over --max-chars"] else "")
          + (f", {why['sub-article (peer review)']} units inside sub-articles (peer review)"
             if why["sub-article (peer review)"] else "") + "; see skipped.tsv")
    for pmid, why in failed:
        print(f"  skipped PMID {pmid}: {why}")
    if args.supplementary:
        print("Supplementary files read: " + (", ".join(f"{k} {v}" for k, v in sorted(SUPP_READ.items())) or "none"))
        if SUPP_SKIPPED:
            print("Supplementary files NOT read: " + ", ".join(f"{k} {v}" for k, v in sorted(SUPP_SKIPPED.items())))
    print(f"Wrote {os.path.join(args.out, 'index.tsv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
