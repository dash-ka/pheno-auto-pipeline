#!/usr/bin/env python3
"""Step 3 for tables (companion of extract_findings.py, which reads the running text). One model
call per table: does the table record clinical phenotypes of individual patients, and if so, which,
each with the cell that records it.

The model first judges the table: which rows or columns report one individual patient (a row or
column labelled with a patient identifier, "P3", "II-1", "Patient 2", "Case 1"; or the paper's own
single patient without one, "This study", "our case", when the counts show one patient). Counts or
percentages over several patients, patients cited from other publications, and diseases or genotypes
compared are not individual patients: such a table gives no findings. Then, for each patient, one
finding per clinical phenotype:

  {"reason": "Phenotypic features of SMS: two cohorts in percent, plus two patients of this study",
   "patient_labels": ["SMS324", "SMS335"],
   "legend": [{"symbol": "+", "meaning": "present"}, {"symbol": "-", "meaning": "absent"}],
   "findings": [{"patient": "SMS324",
                 "evidence": {"row_label": "Hypotonia", "column_label": "SMS324", "cell_value": "+"},
                 "phenotype_quote": "Hypotonia", "phenotype_status": "present",
                 "standard_term": "hypotonia"}]}

A cell describing several phenotypes ("OD: cataract OS: phthisis OU: RD") gives one finding per
phenotype, each with the same evidence. The quote comes from the row label, the column label or the
cell value: the label in a +/- table, the words in a description cell.

Tables: the sections of kind table and suppl_table (--kinds). The chunks of one long table
("Table 2", "Table 2 (part 2)") are joined and sent together; a table longer than --max-chars is
sent in blocks of rows, each with the table's caption, footnotes and header.

Repairs done in code before the checks: a PDF table's rows are rebuilt as a grid (cells broken over
lines joined; a word cut after its drop capital, "G" | "lycine", joined; when a row is not found, the
row is also tried without its empty cells, for PDFs that put an empty cell between all cells); a short
column label is found as whole words in a longer header ("2M" in "Patient 2M"); a patient label keeps
its identifier only ("1435-IV-5 (U) (14 y)" -> "1435-IV-5", "5 5 8 -III-1" -> "558-III-1"; the full
label goes in other_names); the HPO lookup also tries the term without a closing note in brackets
("Eosinophilia (high eosinophils)" -> "Eosinophilia"). phenotype_status "normal" is written as absent
(status excluded).

Checks done in code, written as columns:
  evidence_found     verified   the cell value is in the patient's row (patients in rows) or in the
                                feature's row (patients in columns), in the column the label names
                     shifted    it is in that row, but in a neighbouring column
                     in_row     it is in that row (or the lines continuing it in a PDF table); its
                                column could not be checked
                     anywhere   only found elsewhere in the table
                     no         not found in the table
  quote_in_evidence  yes / parts / no: the quote within the row label, column label or cell value
  check              what went wrong: "patient not listed", "patient not a label of its cell", ...
The finding is placed on its cell (on the quote within it when the quote is in the cell), in the
section (chunk) that holds it; its context, given to BeLink and shown in the app, reads like a
sentence: "SMS324 · Hypotonia: +".

Output (in --out), as 03f writes it, so relink.py, 03d, 03e, 04/05 and the app read it
unchanged (and a table run and a prose run can be concatenated before merging):
  PMID_<pmid>/<section_id>.json   per table: the text sent, the chunks it came from, the answer
  predictions.tsv                 one row per finding; evidence = "row_label | column_label | cell_value";
                                  plus table_reason, patients_in
  failures.tsv                    tables without a usable answer (rerun to retry them)

Usage (single line)
  python extract_table_findings.py --sections data/sections --hpo data/hp.obo --out runs/tables_gptoss120b --model gpt-oss-120b --base-url http://localhost:8000/v1 --pmids data/pmids.txt --workers 4 --temperature 0
  Combined with a prose run before relinking:
  python concat_predictions.py runs/findings_gptoss120b/predictions.tsv runs/tables_gptoss120b/predictions.tsv runs/both/predictions.tsv

--print-prompt shows the instructions; --prompt FILE replaces them. --collect-only rebuilds
predictions.tsv from the saved answers. Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote as urlquote

DEFAULT_KINDS = ("abstract,case,results,materials,main,appendix,other,suppl,suppl_text,figure,intro,"
                 "discussion,conclusion")
ALL_KINDS = ("abstract,intro,materials,case,results,discussion,conclusion,suppl,main,appendix,other,"
             "table,suppl_table,suppl_text,figure")
DEFAULT_PARTS = "front,body,back,floats,suppl_file"
ALL_PARTS = "front,body,back,floats,sub-article,suppl_file"
FORMATS = ["json_schema", "json_object", "none"]
PLACEHOLDERS = {"", "none", "unknown", "na", "n/a", "nil", "null", "not reported", "none reported", "not stated",
                "not specified", "not mentioned", "not available", "not applicable", "none stated", "none mentioned",
                "none explicitly reported", "none explicitly stated", "no abnormalities", "not provided", "unspecified"}


DEFAULT_PROMPT = """\
You are an expert clinician. Your task is to study the provided article table and extract the clinical phenotypes (i.e. clinical abnormalities) it records for individual patients, if any.

Thoroughly read the table, its caption and its footnotes, and analyse its content. Try to identify if there is any column or row in this table that refers to the article's own patient case ("This study", "present case", "our patient").
If there are rows or columns in this table that report an association between a clinical phenotype and an individual, including the article's own patient case, 
extract individual patients and the associated abnormal clinical phenotypes the table reports for each patient.
Return your answer as the following JSON object:
{"reason": <one sentence: what the table records and which rows or columns are individual patients>,
"patient_labels": [<the labels of the individual patients reported in association with individual clinical phenotypes in the provided table. copy the patient labels verbatim from the table>],
"legend": [{"symbol": <a mark used in the cells>, "meaning": <its meaning, from the footnotes or the table>}],
"findings": # a list of abnormal clinical phenotypes reported in association with an individual patient from this table (one entry per patient's clinical phenotype), structured as follows
[{
	"patient": <an individual patient (one person) whose own clinical phenotype is reported in the provided table. This must be one of the labels in "patient_labels">,
	"evidence": {"row_label": <the row label of the row/column from the table reporting the association between this individual patient and an abnormal clinical phenotype>, "column_label": <the column label of the row/column from the table reporting the association between this individual patient and an abnormal clinical phenotype>, "cell_value": <the cell value from the row/column of the table reporting the association between this individual patient and an abnormal clinical phenotype>},
    "phenotype_quote": <an abnormal clinical phenotype the table reports for this patient, such as a sign, symptom, or an abnormal clinical finding (a clinical finding is abnormal if its value is outside the reference range stated in the table). Use the shortest run of words within the evidence that states the abnormal clinical phenotype of this patient, copied verbatim from the table. Do not rephrase, normalise or expand abbreviations. One finding per clinical phenotype rule: if the quote lists several clinical phenotypes, split the listed phenotypes into separate findings (one finding for each clinical phenotype), each one with the same evidence.>,
	"phenotype_status": <"present" if the this patient presents this clinical abnormality, or "normal" if the table states this clinical phenotype is absent or normal in this patient.>,
	"standard_term": <the standard medical term for this clinical phenotype. For the abnormal lab value, name the abnormality with its direction.>}
]}

N.B. A rows or a column reporting an aggregate over several patients (>1), or patients cited from other publications are not individual patients and should be ignored. An aggregate means MORE THAN ONE patient. Check for each column what are the values it is reporting (ask yourself, do the values refer to one or more than one patients).
If the table has no individual patients, answer with an empty "patient_labels" list and an empty list of findings.
If a cell describes several abnormal clinical phenotypes, split the clinical phenotypes into separate findings (one per clinical phenotype), each with the same evidence. Cells that are unknown, not evaluated or not available (N, N/A, nd, ?) must be ignored.
Ignore columns and rows that report sex, ages and ages at onset, genotypes and variants, treatments and procedures, references, molecular or cellular assays. Clinical phenotypes should not be inferred from lab measurements. 

Table:
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reason", "patient_labels", "legend", "findings"],
    "properties": {
        "reason": {"type": "string"},
        "patient_labels": {"type": "array", "items": {"type": "string"}},
        "legend": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["symbol", "meaning"],
            "properties": {"symbol": {"type": "string"}, "meaning": {"type": "string"}}}},
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["patient", "evidence", "phenotype_quote", "phenotype_status", "standard_term"],
            "properties": {
                "patient": {"type": "string"},
                "evidence": {"type": "object", "additionalProperties": False,
                             "required": ["row_label", "column_label", "cell_value"],
                             "properties": {"row_label": {"type": "string"}, "column_label": {"type": "string"},
                                            "cell_value": {"type": "string"}}},
                "phenotype_quote": {"type": "string"},
                "phenotype_status": {"type": "string", "enum": ["present", "normal"]},
                "standard_term": {"type": "string"}}}},
    },
}

COLS = ["pmid", "section_id", "kind", "patient_id", "hpo_id", "label", "status", "grounded", "span",
        "ontogpt_id", "chosen_span", "located_spans", "unlocated", "context_source", "context",
        "context_start", "context_end",
        "other_names", "sex", "vital_status", "age_at_last_examination",
        "section_position", "section_count", "section_part", "section_title", "first_pass",
        "evidence", "evidence_found", "quote_in_evidence", "check", "table_reason", "patients_in"]

TABLE_KINDS = "table,suppl_table"


# ---------------------------------------------------------------- shared with extract_findings.py

def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


class HPOLookup:
    """A term's HP id when it equals (after normalisation) an HPO label or exact synonym."""

    def __init__(self, path: str, synonyms: str = "exact"):
        scopes = {"EXACT"} if synonyms == "exact" else {"EXACT", "RELATED", "BROAD", "NARROW"}
        self.by_label, self.by_syn, self.label = {}, {}, {}
        term = None

        def flush():
            if term and term.get("id", "").startswith("HP:") and not term.get("obsolete"):
                self.label[term["id"]] = term.get("name", "")
                if term.get("name"):
                    self.by_label.setdefault(norm_key(term["name"]), term["id"])
                for syn in term.get("syn", []):
                    self.by_syn.setdefault(norm_key(syn), set()).add(term["id"])

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("["):
                    flush()
                    term = {} if line == "[Term]" else None
                elif term is None:
                    continue
                elif line.startswith("id: "):
                    term["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    term["name"] = line[6:].strip()
                elif line.startswith("is_obsolete: true"):
                    term["obsolete"] = True
                elif line.startswith("synonym: "):
                    m = re.match(r'synonym: "((?:[^"\\]|\\.)*)" (\w+)', line)
                    if m and m.group(2) in scopes:
                        term.setdefault("syn", []).append(m.group(1).replace('\\"', '"'))
            flush()

    def ground(self, text: str) -> str | None:
        k = norm_key(text)
        if not k:
            return None
        if k in self.by_label:
            return self.by_label[k]
        ids = self.by_syn.get(k, set())
        return next(iter(ids)) if len(ids) == 1 else None


# ---------------------------------------------------------------- model call (as 03j)

class Endpoint:
    """An OpenAI-compatible chat endpoint; remembers a response format the server rejected."""

    def __init__(self, args):
        self.args = args
        self.url = args.base_url.rstrip("/") + "/chat/completions"
        self.key = os.environ.get(args.api_key_env, "") or "EMPTY"
        self.fmt = args.response_format
        self.lock = threading.Lock()

    def _payload(self, system: str, user: str, fmt: str) -> dict:
        a = self.args
        body = {"model": a.model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}]}
        if a.max_tokens:
            body["max_completion_tokens"] = a.max_tokens
        if a.temperature is not None:
            body["temperature"] = a.temperature
        if a.reasoning_effort:
            body["reasoning_effort"] = a.reasoning_effort
        if fmt == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "findings", "strict": True, "schema": SCHEMA}}
        elif fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(self, system: str, user: str) -> tuple[str, dict]:
        last = ""
        for attempt in range(self.args.retries + 1):
            fmt = self.fmt
            req = urllib.request.Request(self.url, data=json.dumps(self._payload(system, user, fmt)).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=self.args.timeout) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:500]
                if e.code == 400 and fmt != "none":
                    with self.lock:                        # the server does not take this format
                        if self.fmt == fmt:
                            self.fmt = FORMATS[FORMATS.index(fmt) + 1]
                            print(f"  note: server rejected response_format {fmt} ({body[:150]!r}); "
                                  f"using {self.fmt} from now on", flush=True)
                    continue
                last = f"HTTP {e.code}: {body[:200]}"
                if e.code not in (408, 409, 429) and e.code < 500:
                    raise RuntimeError(last)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
            else:
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                info = {"finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                        "response_format": fmt, "model": data.get("model")}
                return msg.get("content") or "", info
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(last or "no answer")


def parse_answer(content: str) -> dict:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        return json.loads(s)
    except ValueError:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            return json.loads(s[a:b + 1])
        raise


# ---------------------------------------------------------------- finding text in the section

_SAME = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
         "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
         "“": '"', "”": '"', "„": '"', "″": '"'}


def normalise(text: str) -> tuple[str, list[int]]:
    """Lowercase NFKC text with dashes and quote marks unified and every run of white space as one
    space, and for each of its characters the position in `text` it comes from."""
    out, pos, space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and not space:
                out.append(" ")
                pos.append(i)
            space = True
            continue
        space = False
        for c in unicodedata.normalize("NFKC", ch).lower():
            out.append(_SAME.get(c, c))
            pos.append(i)
    while out and out[-1] == " ":
        out.pop()
        pos.pop()
    return "".join(out), pos


def key_of(s: str) -> str:
    return normalise(s)[0].strip()


def span_of(norm: str, pos: list[int], needle: str, lo: int = 0, hi: int | None = None) -> tuple[int, int] | None:
    """(start, end) in the original text of `needle` (normalised) within norm[lo:hi]."""
    if not needle:
        return None
    k = norm.find(needle, lo, len(norm) if hi is None else hi)
    if k < 0:
        return None
    return pos[k], pos[k + len(needle) - 1] + 1


def flat(s: str) -> str:
    """Line breaks as spaces, one character for one, so offsets stay valid."""
    return s.replace("\r", " ").replace("\n", " ")


STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with", "was", "were", "is", "are", "be",
        "been", "had", "has", "have", "by", "for", "from", "as", "both", "his", "her", "their", "its"}
TOKEN = re.compile(r"[^\W_]+")


def stem(w: str) -> str:
    """A rough stem: common English endings removed (hyperostoses -> hyperost, thickening -> thicken)."""
    for end in ("ations", "ation", "ings", "ing", "oses", "osis", "ies", "es", "ed", "ly", "s"):
        if len(w) > len(end) + 3 and w.endswith(end):
            return w[:-len(end)]
    return w


def same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    sa, sb = stem(a), stem(b)
    return sa == sb or (min(len(sa), len(sb)) >= 4 and (sa.startswith(sb) or sb.startswith(sa)))


def quote_parts(text: str, norm: str, pos: list[int], ev: tuple[int, int], quote: str) -> tuple[int, int] | None:
    """The stretch of the evidence holding a quote that is not in it as one piece:
    1. pieces cut at "..." found in order ("oral ... candidiasis");
    2. otherwise the shortest stretch holding all of the quote's content words, word endings
       aside ("hyperostoses in the maxilla" in "hyperostoses in the occipital calvarium, maxilla")."""
    lo, hi = bisect.bisect_left(pos, ev[0]), bisect.bisect_left(pos, ev[1])
    pieces = [key_of(x) for x in re.split(r"\.\.\.+|\u2026", quote) if key_of(x)]
    if len(pieces) > 1:
        start, end, k = None, None, lo
        for piece in pieces:
            j = norm.find(piece, k, hi)
            if j < 0:
                break
            start = j if start is None else start
            end, k = j + len(piece), j + len(piece)
        else:
            return pos[start], pos[end - 1] + 1
    want = list(dict.fromkeys(w for w in TOKEN.findall(key_of(quote)) if w not in STOP))
    if not want:
        return None
    toks = [(m.start() + ev[0], m.end() + ev[0], m.group().lower()) for m in TOKEN.finditer(text[ev[0]:ev[1]])]
    hits = [[i for i, w in enumerate(want) if same_word(w, t)] for _, _, t in toks]
    best, count, have, left = None, 0, collections.Counter(), 0
    for right, h in enumerate(hits):                  # shortest window of tokens covering every word
        for i in h:
            have[i] += 1
            count += have[i] == 1
        while count == len(want):
            if best is None or toks[right][1] - toks[left][0] < best[1] - best[0]:
                best = (toks[left][0], toks[right][1])
            for i in hits[left]:
                have[i] -= 1
                count -= have[i] == 0
            left += 1
    return best


def spans_all(norm: str, pos: list[int], needle: str) -> list[tuple[int, int]]:
    """Every (start, end) of `needle` (normalised) in the original text."""
    out, k = [], norm.find(needle) if needle else -1
    while k >= 0:
        out.append((pos[k], pos[k + len(needle) - 1] + 1))
        k = norm.find(needle, k + 1)
    return out

def select_parts(sections: list[dict], parts: str) -> list[dict]:
    want = {p.strip() for p in parts.split(",") if p.strip()}
    return [s for s in sections if not s.get("part") or s["part"] in want]




# ---------------------------------------------------------------- tables: joining, splitting, cells

def table_key(s: dict) -> tuple:
    """The chunks of one long table share this key ("Table 2", "Table 2 (part 2)")."""
    return (s["pmid"], s["kind"], re.sub(r"\s*\(part \d+\)\s*$", "", (s.get("title") or "").strip()))


def group_tables(sections: list[dict]) -> list[list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for s in sections:
        groups.setdefault(table_key(s), []).append(s)
    return list(groups.values())


def read_text(args, s: dict) -> str:
    with open(os.path.join(args.sections, s["path"].replace("\\", "/")), encoding="utf-8", newline="") as fh:
        return fh.read()


def join_chunks(texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """The chunks one after the other (a repeated header is kept: offsets stay exact), and where each is."""
    out, spans, at = [], [], 0
    for i, t in enumerate(texts):
        if i:
            out.append("\n")
            at += 1
        spans.append((at, at + len(t)))
        out.append(t)
        at += len(t)
    return "".join(out), spans


def header_and_rows(text: str) -> tuple[str, list[str]]:
    """The table's caption, footnotes and header (up to and including the first tab-separated line and
    the lines continuing it), and the remaining lines."""
    lines = text.split("\n")
    first = next((i for i, ln in enumerate(lines) if "\t" in ln), None)
    if first is None:
        return text, []
    end = first + 1
    while end < len(lines) and "\t" not in lines[end] and lines[end].strip():
        end += 1                                     # a header cell broken over lines (PDF tables)
    return "\n".join(lines[:end]), lines[end:]


def blocks_of(text: str, max_chars: int) -> list[str]:
    """The table as one block, or, when longer than max_chars, in blocks of rows (each starting at a
    tab-separated line), every block preceded by the header."""
    if len(text) <= max_chars:
        return [text]
    head, rows = header_and_rows(text)
    blocks, cur, size = [], [], len(head)
    for ln in rows:
        starts_row = "\t" in ln
        if cur and starts_row and size + len(ln) + 1 > max_chars:
            blocks.append(head + "\n" + "\n".join(cur))
            cur, size = [], len(head)
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        blocks.append(head + "\n" + "\n".join(cur))
    return blocks


def lines_with_pos(text: str) -> list[tuple[int, str]]:
    out, at = [], 0
    for ln in text.split("\n"):
        out.append((at, ln))
        at += len(ln) + 1
    return out


def cells_of(start: int, line: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of each tab-separated cell of a line."""
    out, at = [], start
    for c in line.split("\t"):
        out.append((at, at + len(c), c))
        at += len(c) + 1
    return out


def same_label(a: str, b: str) -> bool:
    ka, kb = key_of(a), key_of(b)
    if not ka or not kb:
        return False
    return ka == kb or (len(kb) >= 3 and (kb in ka or ka in kb))


def extends(a: str, b: str) -> bool:
    """One label is the other continued by more letters or digits ("MS239 IV:2" and "MS239 IV:20",
    "Patient 1" and "Patient 10"): different patients, not the same label written longer."""
    ka, kb = key_of(a), key_of(b)
    short, long_ = sorted((ka, kb), key=len)
    if not short or short == long_ or short not in long_:
        return False
    i = long_.find(short)
    before, after = long_[i - 1:i], long_[i + len(short):i + len(short) + 1]
    return bool(re.match(r"[0-9a-z]", after)) or bool(re.match(r"[0-9a-z]", before))


NUMERIC = re.compile(r"^[<>≤≥~]?\s*\d+([.,]\d+)?(\s*[-–]\s*\d+([.,]\d+)?)?\s*-?%?$")


def squash(s: str) -> str:
    """A label's key without spaces: PDF tables join broken labels with or without one."""
    return key_of(s).replace(" ", "")


def grid_of(text: str) -> list[list[dict]]:
    """The table rebuilt as rows of cells, each cell {"text", "segs": [(start, end), ...]}.

    PDF tables break cells over lines: a header "Patient 2M | Patient 1M | reference range" comes as
    "\tPatient", "2M\tPatient", "1M\t...", "2B\treference", "range", a row label
    "CD3+ T lymphocytes, absolute counts" as "CD3+TLymphocytes," then "absolutecounts\t1781\t...",
    and a description cell as several lines ("OD: cataract", "OS: phthisis", "OU: RD\tOD: HM").
    The number of cells of a row (N) is the one, from the largest number of cells of a tab-separated
    line up to three more, that makes the most rows complete (a row whose last cell is broken over
    lines has fewer tabs on each line than cells).
    A line short of N cells continues the row in progress (its first piece goes on that row's last
    cell). A line without tabs after a complete row goes on that row's last cell (the header's, a
    description's last line), unless the next tab-separated line is a complete row by itself and
    this line ends with "," "-" "/" "(" "+" "&" or the next line starts in lower case: then it starts that
    row's label. A line with an empty first cell after a complete row continues that row's cells
    column by column. Lines between "Rows are tab-separated..." and the first tab-separated line
    start the header's first cell ("Patient" over "number"); the caption before them, and the lines
    after the last tab-separated line (footnotes), are left out. A first cell of one capital letter
    followed by a cell starting in lower case is a word the PDF cut after its first letter
    ("G" | "lycine"): the two are joined in the first cell."""
    lines = []
    for at, ln in lines_with_pos(text):
        ln = ln.rstrip("\r")
        lines.append((at, ln))
    tab_lines = [i for i, (_, ln) in enumerate(lines) if "\t" in ln]
    if not tab_lines:
        return []
    widest = max(lines[i][1].count("\t") + 1 for i in tab_lines)
    best = None
    for n in range(widest, widest + 4):
        rows = _grid(lines, tab_lines, n)
        full = sum(len(r) == n for r in rows)
        score = (full - (len(rows) - full), -n)
        if best is None or score > best[0]:
            best = (score, rows)
    return best[1]


SKIP_LINE = re.compile(r"^\s*(Supplementary file:|Rows are tab-separated)", re.I)


def _grid(lines: list[tuple[int, str]], tab_lines: list[int], n: int) -> list[list[dict]]:
    rows, cur, pending = [], None, []
    marker = max((i for i, (_, ln) in enumerate(lines[:tab_lines[0]])
                  if ln.lower().startswith("rows are tab-separated")), default=None)
    if marker is not None:
        pending = [cells_of(at, ln)[0] for at, ln in lines[marker + 1:tab_lines[0]] if ln.strip()]
    next_tab = {}
    nxt = None
    for i in range(len(lines) - 1, -1, -1):
        next_tab[i] = nxt
        if "\t" in lines[i][1]:
            nxt = i

    def cell(seg):
        return {"text": seg[2].strip(), "segs": [(seg[0], seg[1])] if seg[2].strip() else []}

    def extend(c, seg):
        if seg[2].strip():
            c["text"] = (c["text"] + " " + seg[2].strip()).strip()
            c["segs"].append((seg[0], seg[1]))

    def starts_next_label(i: int, ln: str) -> bool:
        j = next_tab.get(i)
        if j is None:
            return False
        nl = lines[j][1]
        if nl.count("\t") + 1 < n:
            return False                             # the next row is built from broken lines itself
        first = nl.split("\t", 1)[0].strip()
        return ln.rstrip().endswith((",", "-", "/", "(", "+", "&")) or (first[:1].islower())

    for i in range(tab_lines[0], tab_lines[-1] + 1):
        at, ln = lines[i]
        if not ln.strip() or (("\t" not in ln) and SKIP_LINE.match(ln)):
            continue                                 # the caption of the next page of a joined table
        segs = cells_of(at, ln)
        if "\t" not in ln:
            if cur is not None and not pending and (len(cur) < n or not rows or not starts_next_label(i, ln)):
                extend(cur[-1], segs[0])             # a broken cell, the header's or the row's last cell
            else:
                pending.append(segs[0])              # the start of the next row's label
            continue
        k = len(segs)
        if cur is not None and len(cur) < n and k < n and len(cur) + k - 1 <= n and not pending:
            extend(cur[-1], segs[0])                 # the row in progress, continued
            cur += [cell(x) for x in segs[1:]]
            continue
        if cur is not None and len(cur) >= n and not segs[0][2].strip() and k < n and not pending:
            for j, x in enumerate(segs[1:], 1):      # an empty first cell: the row's cells continued
                if j < len(cur):
                    extend(cur[j], x)
            continue
        if cur is not None:
            rows.append(cur)
        cur = [cell(x) for x in segs]
        if pending:
            first = {"text": "", "segs": []}
            for x in pending + [segs[0]]:
                extend(first, x)
            cur[0], pending = first, []
    if cur is not None:
        rows.append(cur)
    for r in rows:                                   # a drop capital cut off: "G" | "lycine" -> "Glycine"
        if len(r) > 1 and re.fullmatch(r"[A-Z]", r[0]["text"]) and r[1]["text"][:1].islower():
            r[0] = {"text": r[0]["text"] + r[1]["text"], "segs": r[0]["segs"] + r[1]["segs"]}
            r[1] = {"text": "", "segs": []}
    return rows


def row_span(row: list[dict]) -> tuple[int, int]:
    segs = [s for c in row for s in c["segs"]]
    return (min(s[0] for s in segs), max(s[1] for s in segs)) if segs else (0, 0)


def label_score(cell_text: str, label: str) -> int:
    """2 the same label, 1 one contains the other (3 characters or more, or whole words: "2M" in
    "Patient 2M"), 0 not the same."""
    a, b = squash(cell_text), squash(label)
    if not a or not b:
        return 0
    if a == b:
        return 2
    if len(min(a, b, key=len)) >= 3 and (a in b or b in a):
        return 1
    ta, tb = TOKEN.findall(key_of(cell_text)), TOKEN.findall(key_of(label))
    if not tb or len(tb) >= len(ta) or NUMERIC.match(cell_text.strip()):
        return 0                                     # the label's words within a longer cell only
    return 1 if any(ta[i:i + len(tb)] == tb for i in range(len(ta))) else 0


def column_candidates(grid: list[list[dict]], label: str, head: int = 3) -> list[int]:
    """The columns a column label names, best first: a header cell, or the header cells of a column
    read top to bottom ("Reference" over "[2]"), the same as the label; then one containing it or
    contained in it. With several partial matches, the closest in length first."""
    top = []
    for r in grid[:head]:                            # header rows: before the first row of values
        cells = [c["text"] for c in r if c["text"]]
        if top and sum(bool(NUMERIC.match(c)) for c in cells) >= 2:
            break
        top.append(r)
    width = max((len(r) for r in top), default=0)
    found = {}
    for j in range(width):
        texts = [r[j]["text"] for r in top if j < len(r) and r[j]["text"]]
        for t in texts + [" ".join(texts)]:
            sc = label_score(t, label)
            if sc:
                gap = abs(len(squash(t)) - len(squash(label)))
                if j not in found or (sc, -gap) > found[j]:
                    found[j] = (sc, -gap)
    best = max((v[0] for v in found.values()), default=0)
    return sorted((j for j, v in found.items() if v[0] == best), key=lambda j: -found[j][1])


def row_candidates(grid: list[list[dict]], label: str, label_cols: int = 3) -> list[int]:
    """The rows a row label names, best first. The label is looked for in the first cell, and also
    in the next two (a table may start with a family column before the patient's: "F | Pat | ..."
    with "1 | 1.1"); the same label first, then one containing the other; at equal match, the label
    further left. A row with an empty first cell takes the first cell of the row above it (a
    patient's or feature's row continued over several rows, or a family's second patient)."""
    found, last = [], ""
    for i, r in enumerate(grid):
        own = r[0]["text"] if r else ""
        last = own or last
        texts = [last] + [c["text"] for c in r[1:label_cols]]
        best = None
        for j, t in enumerate(texts):
            if j and NUMERIC.match(t) and not NUMERIC.match(label.strip()):
                continue                             # a value, not a label
            sc = label_score(t, label)
            if sc and (best is None or (sc, -j) > (best[0], -best[2])):
                best = (sc, -abs(len(squash(t)) - len(squash(label))), j)
        if best:
            found.append((best[0], best[1], -best[2], i))
    if not found:
        return []
    top = max(f[0] for f in found)
    return [i for sc, gap, col, i in sorted(found, key=lambda f: (-f[0], -f[2], -f[1], f[3])) if sc == top]


def search_grid(grid, row, col, vkey, value, norm, pos):
    """(rank, evidence_found, (start, end)) of the cell in one grid; rank 0 when not in its row."""
    rank = {"verified": 3, "shifted": 2, "in_row": 1}
    best = (0, None, None)
    for r_label, c_label in ((row, col), (col, row)):
        rows_i = row_candidates(grid, r_label)
        cols_j = column_candidates(grid, c_label)
        for i in rows_i:
            cells = grid[i]
            for j in cols_j:
                if j < len(cells) and squash(cells[j]["text"]) == vkey and cells[j]["segs"]:
                    segs = cells[j]["segs"]
                    return (rank["verified"], "verified", (segs[0][0], segs[-1][1]))
            for j in cols_j:                         # the value is one part of its cell: "otitis externa" in
                if j < len(cells) and vkey in squash(cells[j]["text"]) and cells[j]["segs"]:   # "S. aureus skin abscesses, otitis externa"
                    segs = cells[j]["segs"]
                    return (rank["verified"], "verified", (segs[0][0], segs[-1][1]))
            for j in cols_j:
                for k in (j - 1, j + 1):
                    if 0 < k < len(cells) and squash(cells[k]["text"]) == vkey and cells[k]["segs"] \
                            and best[0] < rank["shifted"]:
                        segs = cells[k]["segs"]
                        best = (rank["shifted"], "shifted", (segs[0][0], segs[-1][1]))
            if best[0] < rank["in_row"]:
                for c in cells[1:]:
                    if squash(c["text"]) == vkey and c["segs"]:
                        best = (rank["in_row"], "in_row", (c["segs"][0][0], c["segs"][-1][1]))
                        break
                else:
                    lo_hi = row_span(cells)
                    lo, hi = bisect.bisect_left(pos, lo_hi[0]), bisect.bisect_left(pos, lo_hi[1])
                    hit = span_of(norm, pos, key_of(value), lo, hi)
                    if hit:
                        best = (rank["in_row"], "in_row", hit)
        if rows_i:
            break                                    # the labels the right way round matched a row
    return best


_GRIDS: dict[str, list] = {}


def locate_cell(text: str, f: dict, patients_in: str, labels: list[str]) -> tuple[str, tuple[int, int] | None]:
    """Where the evidence cell is: (evidence_found, (start, end) of the value in the text).

    The table is rebuilt as a grid (grid_of); the row is the one whose first cell is the row label,
    the column the one whose header is the column label (the same label first, then one containing
    the other). When the row label is not a row's first cell, the two labels are tried the other way
    round (a model may swap them). The best result over the candidate rows and columns is kept."""
    ev = f["evidence"]
    row, col, value = ev["row_label"], ev["column_label"], ev["cell_value"]
    norm, pos = normalise(text)
    vkey = squash(value)
    if not vkey:
        return "no", None
    grid = _GRIDS.get(text)
    if grid is None:
        grid = _GRIDS[text] = grid_of(text)
    rank = {"verified": 3, "shifted": 2, "in_row": 1}
    best = (0, None, None)
    # the grid as rebuilt, then with each row's empty cells left out: some PDF tables put an empty
    # cell between all cells and start every other row with one ("\tValine\t\t256" after
    # "Leucine\t\t165"), so their labels and columns line up only without the empty cells
    for g in (grid, [[c for c in r if c["text"]] for r in grid]):
        found = search_grid(g, row, col, vkey, value, norm, pos)
        if found[0] > best[0]:
            best = found
        if best[0] == rank["verified"]:
            break
    if best[1]:
        return best[1], best[2]
    hit = span_of(norm, pos, key_of(value))
    return ("anywhere", hit) if hit else ("no", None)


def quote_place(text: str, cell: tuple[int, int] | None, f: dict) -> tuple[str, tuple[int, int] | None]:
    """Is the quote in the row label, column label or cell value (yes / parts / no), and where it is in
    the text when it is in the located cell."""
    ev, q = f["evidence"], f["quote"]
    qk = key_of(q)
    if cell:
        norm, pos = normalise(text)
        lo, hi = bisect.bisect_left(pos, cell[0]), bisect.bisect_left(pos, cell[1])
        at = span_of(norm, pos, qk, lo, hi)
        if at:
            return "yes", at
        at = quote_parts(text, norm, pos, cell, q)
        if at:
            return "parts", at
    for part in (ev["row_label"], ev["column_label"], ev["cell_value"]):
        if qk and qk in key_of(part):
            return "yes", None
    words = [w for w in TOKEN.findall(qk) if w not in STOP]
    pool = TOKEN.findall(key_of(" ".join((ev["row_label"], ev["column_label"], ev["cell_value"]))))
    if words and all(any(same_word(w, t) for t in pool) for w in words):
        return "parts", None
    return "no", None


# ---------------------------------------------------------------- one answer -> rows

def short_label(label: str) -> str:
    """The patient's label as an identifier: dashes unified, and notes in brackets at its end left out
    when what remains holds a digit ("1435-IV-5 (U) (14 y)" -> "1435-IV-5"), spaces a PDF put inside
    the identifier removed ("5 5 8 -III-1" -> "558-III-1"), a sex after a comma or space left out ("1, F", "1 M" -> "1"),
    so it pairs with the
    same patient in the text and in the phenopackets. The full label is kept in other_names."""
    s = " ".join(label.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"})).split())
    s = re.sub(r"(?<=\d) (?=[\d-])|(?<=-) (?=\w)| (?=-)", "", s)   # PDF spacing: "5 5 8 -III-1" -> "558-III-1"
    cut = re.sub(r"(\s*\([^()]*\))+$", "", s).strip()
    cut = re.sub(r"(\s*[,/]\s*|\s+)(F|M|female|male)$", "", cut, flags=re.I).strip()   # "1, F", "1 M" -> "1"
    return cut if cut and re.search(r"\d", cut) else s


def clean(data: dict) -> tuple[dict, list[dict], list[str]]:
    """The table judgement and its findings in a fixed shape; each finding's patient mapped to a listed
    patient label. Problems are returned as notes."""
    notes = []
    labels = [str(x).strip() for x in data.get("patient_labels") or [] if str(x).strip()]
    head = {"reason": " ".join(str(data.get("reason") or "").split()),
            "patients_in": str(data.get("patients_in") or "").strip().lower(), "patient_labels": labels,
            "legend": [{"symbol": str(x.get("symbol", "")), "meaning": str(x.get("meaning", ""))}
                       for x in data.get("legend") or [] if isinstance(x, dict)]}
    findings, seen = [], set()
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        ev = f.get("evidence") if isinstance(f.get("evidence"), dict) else {}
        ev = {k: str(ev.get(k) or "").strip() for k in ("row_label", "column_label", "cell_value")}
        who = str(f.get("patient") or "").strip()
        quote = str(f.get("phenotype_quote") or "").strip()
        term = str(f.get("standard_term") or "").strip() or quote
        if not quote and not term or term.lower().rstrip(".") in PLACEHOLDERS:
            continue
        note = []
        match = (next((l for l in labels if key_of(l) == key_of(who)), None)      # the same label first:
                 or next((l for l in labels if same_label(l, who) and not extends(l, who)), None))   # "IV:2" is not "IV:20"
        if match:
            who = match
        else:
            note.append("patient not listed")
        if not (same_label(ev["row_label"], who) or same_label(ev["column_label"], who)):
            note.append("patient not a label of its cell")
        status = ("excluded" if str(f.get("phenotype_status") or "").strip().lower() in ("normal", "absent")
                  else "observed")                   # "normal": the phenotype is absent or normal
        k = (who.lower(), quote.lower(), term.lower(), status, ev["row_label"].lower(), ev["column_label"].lower())
        if k in seen:
            continue
        seen.add(k)
        findings.append({"patient": short_label(who) or "individual", "full_label": who, "evidence": ev,
                         "quote": quote, "term": term, "status": status, "note": "; ".join(note)})
    if head["patients_in"] == "none" and findings:   # only when the prompt asks for patients_in
        notes.append(f"table judged without individual patients but {len(findings)} findings given")
    return head, findings, notes


def summary_line(findings: list[dict]) -> str:
    """The table's patients in one line, as OntoGPT's first pass writes it, for 03d. Table findings are
    written with their standard term: a table quote ("<1%", "Valine") says nothing to the merge step."""
    per = collections.OrderedDict()
    for f in findings:
        name = f["term"] or f["quote"]
        per.setdefault(f["patient"], []).append(
            (name if f["status"] == "observed" else "no " + name).replace(";", ","))
    return "patients: " + "; ".join(f"{p}: " + ", ".join(dict.fromkeys(q)) for p, q in per.items())


def rows_of(group: list[dict], res: dict, lookup, args, place_of: dict, n_of: dict) -> list[dict]:
    text, chunks = res["input_text"], res["chunks"]
    rows = []
    heads, findings = [], []
    for raw in res["raw_json"]:
        head, fs, _ = clean(parse_answer(raw))
        heads.append(head)
        findings += fs
    seen, uniq = set(), []                      # blocks of one table repeat nothing, but be safe
    for f in findings:
        k = (f["patient"].lower(), f["quote"].lower(), f["term"].lower(), f["status"],
             f["evidence"]["row_label"].lower(), f["evidence"]["column_label"].lower())
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    findings = uniq
    reason = " | ".join(dict.fromkeys(h["reason"] for h in heads if h["reason"]))
    labels = list(dict.fromkeys(l for h in heads for l in h["patient_labels"]))
    pin = orientation(heads, findings)
    first_pass = " ".join(summary_line(findings).split()) if findings else ""
    by_sid = {s["section_id"]: s for s in group}
    for f in findings:
        ev = f["evidence"]
        found, cell = locate_cell(text, f, pin, labels)
        in_ev, qspan = quote_place(text, cell, f)
        where = qspan or cell
        # the chunk (section) holding the cell, and offsets within it
        sid = chunks[0][0]
        if where:
            for c_sid, a, b in chunks:
                if a <= where[0] < b:
                    sid, base = c_sid, a
                    break
            else:
                base = chunks[0][1]
            span = f"{where[0] - base}:{where[1] - base - 1}"
        else:
            span = ""
        feature = ev["column_label"] if same_label(ev["row_label"], f["patient"]) else ev["row_label"]
        ctx = f"{f['patient']} · {feature}: {ev['cell_value']}".replace("\r", " ").replace("\n", " ")
        ctx = " ".join(ctx.split())
        k = key_of(f["quote"])
        cn, cpos = normalise(ctx)
        at = span_of(cn, cpos, k) if k else None
        if not at:
            at = (0, len(ctx))
        problems = [x for x in (f["note"],
                                {"shifted": "value in a neighbouring column", "anywhere": "cell not in its row",
                                 "no": "cell not found"}.get(found, ""),
                                "quote not in evidence" if in_ev == "no" else "") if x]
        bare_term = re.sub(r"\s*\([^()]*\)\s*$", "", f["term"]).strip()   # "Eosinophilia (high eosinophils)"
        hid = lookup.ground(f["term"]) or (bare_term and lookup.ground(bare_term)) or "AUTO:" + urlquote(f["term"])

        s = by_sid.get(sid, group[0])
        rows.append({"pmid": s["pmid"], "section_id": sid, "kind": s["kind"], "patient_id": f["patient"],
                     "hpo_id": hid, "label": f["term"], "status": f["status"],
                     "grounded": "yes" if hid.startswith("HP:") else "no", "span": f["quote"], "ontogpt_id": hid,
                     "chosen_span": span, "located_spans": span, "unlocated": "no" if span else "yes",
                     "context_source": "text" if span else "none", "context": ctx,
                     "context_start": at[0], "context_end": at[1],
                     "other_names": f["full_label"] if f["full_label"] != f["patient"] else "", "sex": "", "vital_status": "", "age_at_last_examination": "",
                     "section_position": place_of.get((s["pmid"], sid), ""), "section_count": n_of.get(s["pmid"], ""),
                     "section_part": s.get("part", ""), "section_title": " ".join((s.get("title") or "").split()),
                     "first_pass": first_pass,
                     "evidence": " | ".join(" ".join(ev[k2].split()) for k2 in ("row_label", "column_label", "cell_value")),
                     "evidence_found": found, "quote_in_evidence": in_ev, "check": "; ".join(problems),
                     "table_reason": reason, "patients_in": pin})
    return rows


def orientation(heads: list[dict], findings: list[dict]) -> str:
    """rows / columns / none: as the model stated it (when the prompt asks for patients_in), else from
    the findings: patients in rows when a finding's patient is its row label."""
    given = next((h["patients_in"] for h in heads if h["patients_in"] in ("rows", "columns")), "")
    if given:
        return given
    rows = sum(same_label(f["evidence"]["row_label"], f["patient"]) for f in findings)
    cols = sum(same_label(f["evidence"]["column_label"], f["patient"]) for f in findings)
    if not findings:
        return "none"
    return "rows" if rows >= cols else "columns"


# ---------------------------------------------------------------- one table

def out_path(args, group: list[dict]) -> str:
    s = group[0]
    return os.path.join(args.out, f"PMID_{s['pmid']}", s["section_id"] + ".json")


def is_done(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return "raw_json" in json.load(fh)
    except (OSError, ValueError):
        return False


def ask(endpoint, system: str, user: str, args) -> tuple[str | None, dict, str]:
    content, info, problem = "", {}, ""
    for _ in range(1 + args.json_retries):
        try:
            content, info = endpoint.chat(system, user)
        except RuntimeError as e:
            return None, {}, str(e)[:300]
        if info["finish_reason"] == "length":
            problem = f"answer cut off at --max-tokens {args.max_tokens}"
            continue
        try:
            data = parse_answer(content)
        except ValueError:
            problem = "invalid JSON in the answer" if content.strip() else "empty answer"
            continue
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            return content, info, ""
        problem = "JSON without a 'findings' list"
    return None, info, f"{problem} ({1 + args.json_retries} attempts)"


def run_one(group: list[dict], args, endpoint, system: str) -> tuple[list[dict], str]:
    dst = out_path(args, group)
    if is_done(dst):
        return group, "skipped"
    texts = [read_text(args, s) for s in group]
    text, spans = join_chunks(texts)
    answers, infos, notes = [], [], []
    for block in blocks_of(text, args.max_chars):
        content, info, problem = ask(endpoint, system, block, args)
        if content is None:
            return group, problem
        answers.append(content)
        infos.append(info)
        head, fs, n = clean(parse_answer(content))
        notes += n
    res = {"input_text": text, "chunks": [[s["section_id"], a, b] for s, (a, b) in zip(group, spans)],
           "raw_json": answers, "notes": notes, "calls": infos, "extraction": "03t_extract_table_findings"}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(dst + ".part", dst)
    parsed = [clean(parse_answer(a)) for a in answers]
    n_find = sum(len(fs) for _, fs, _ in parsed)
    pin = orientation([h for h, _, _ in parsed], [f for _, fs, _ in parsed for f in fs])
    return group, (f"done: {n_find} findings" + (f", patients in {pin}" if pin != "none" else ", no individual patients")
                   + (f" ({len(answers)} blocks)" if len(answers) > 1 else "")
                   + (f" ({'; '.join(notes)})" if notes else ""))


# ---------------------------------------------------------------- collect

def collect(groups: list[list[dict]], everything: list[dict], args, lookup) -> None:
    n_of = collections.Counter(s["pmid"] for s in everything)
    seen_n, place_of = collections.Counter(), {}
    for s in everything:
        seen_n[s["pmid"]] += 1
        place_of[(s["pmid"], s["section_id"])] = seen_n[s["pmid"]]
    rows, n_tables, judged = [], 0, collections.Counter()
    for group in groups:
        path = out_path(args, group)
        if not is_done(path):
            continue
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        n_tables += 1
        r = rows_of(group, res, lookup, args, place_of, n_of)
        parsed = [clean(parse_answer(a)) for a in res["raw_json"]]
        judged["with individual patients" if any(h["patient_labels"] or fs for h, fs, _ in parsed)
               else "without"] += 1
        rows += r
    out = os.path.join(args.out, "predictions.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    ev = collections.Counter(r["evidence_found"] for r in rows)
    st = collections.Counter(r["status"] for r in rows)
    qin = collections.Counter(r["quote_in_evidence"] for r in rows)
    g = sum(r["grounded"] == "yes" for r in rows)
    print(f"{n_tables} tables: {judged['with individual patients']} with individual patients, {judged['without']} without")
    print(f"{n} findings ({st['observed']} present, {st['excluded']} absent) for "
          f"{len({(r['pmid'], r['patient_id']) for r in rows})} patients")
    print(f"Cells: verified {ev['verified']}, neighbouring column {ev['shifted']}, in the row {ev['in_row']}, "
          f"elsewhere {ev['anywhere']}, not found {ev['no']}; quote in its cell or labels: "
          f"yes {qin['yes']}, in pieces {qin['parts']}, no {qin['no']}")
    print(f"Grounded by HPO lookup {g}, not {n - g} (AUTO:, for 03b)")
    problems = collections.Counter(p for r in rows for p in r["check"].split("; ") if p)
    if problems:
        print("Checks: " + ", ".join(f"{k} {v}" for k, v in problems.most_common()))
    print(f"Wrote {out}")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", help="folder written by 02_prepare_sections.py")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--hpo", help="hp.obo, for grounding the terms by lookup")
    ap.add_argument("--synonyms", choices=["exact", "all"], default="exact",
                    help="synonyms used for the lookup: exact (with labels; default) or all")
    ap.add_argument("--model", help="model name as the server knows it (curl <base-url>/models)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible endpoint (default $OPENAI_BASE_URL or http://localhost:8000/v1)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the key")
    ap.add_argument("--response-format", choices=FORMATS, default="json_schema")
    ap.add_argument("--max-tokens", type=int, default=16384, help="cap on the answer, reasoning included")
    ap.add_argument("--temperature", type=float, help="not sent unless given")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="for gpt-oss / gpt-5 models")
    ap.add_argument("--prompt", help="file with instructions replacing the built-in ones")
    ap.add_argument("--print-prompt", action="store_true", help="print the built-in instructions and exit")
    ap.add_argument("--kinds", default=TABLE_KINDS, help=f"section kinds read as tables (default {TABLE_KINDS})")
    ap.add_argument("--parts", default=DEFAULT_PARTS, help=f"article parts (default {DEFAULT_PARTS}; all: {ALL_PARTS})")
    ap.add_argument("--pmids", help="file with one PMID per line: only these papers")
    ap.add_argument("--limit-papers", type=int, help="only the first N papers (trial run)")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="a longer table is sent in blocks of rows, each with its header (default 12000)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per request (default 900)")
    ap.add_argument("--retries", type=int, default=3, help="retries after timeouts and server errors")
    ap.add_argument("--json-retries", type=int, default=2,
                    help="new calls when the answer is cut off, empty, not JSON or has no findings list (default 2)")
    ap.add_argument("--collect-only", action="store_true", help="no model calls: rebuild predictions.tsv")
    args = ap.parse_args()
    if args.print_prompt:
        print(DEFAULT_PROMPT)
        return 0
    missing = [o for o in ("sections", "out", "hpo") if not getattr(args, o)]
    if not args.collect_only and not args.model:
        missing.append("model")
    if missing:
        ap.error("required: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    system = DEFAULT_PROMPT
    if args.prompt:
        with open(args.prompt, encoding="utf-8") as fh:
            system = fh.read()

    with open(os.path.join(args.sections, "index.tsv"), encoding="utf-8") as fh:
        everything = list(csv.DictReader(fh, delimiter="\t"))
    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    sections = select_parts([r for r in everything if r["kind"] in kinds], args.parts)
    if args.pmids:
        with open(args.pmids, encoding="utf-8") as fh:
            wanted = {ln.strip().replace("PMID:", "") for ln in fh if ln.strip()}
        sections = [s for s in sections if s["pmid"] in wanted]
    if args.limit_papers:
        keep = set(sorted({s["pmid"] for s in sections}, key=int)[:args.limit_papers])
        sections = [s for s in sections if s["pmid"] in keep]
    groups = group_tables(sections)
    print(f"{len(groups)} tables ({len(sections)} sections) from {len({s['pmid'] for s in sections})} papers")
    lookup = HPOLookup(args.hpo, args.synonyms)
    os.makedirs(args.out, exist_ok=True)

    if not args.collect_only:
        endpoint = Endpoint(args)
        failures, done, start = [], 0, time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, g, args, endpoint, system) for g in groups]
            for fut in as_completed(futures):
                g, status = fut.result()
                done += 1
                ok = status.startswith("done") or status == "skipped"
                if not ok:
                    failures.append((g, status))
                print(f"  [{done}/{len(futures)}, {time.time() - start:.0f}s] PMID {g[0]['pmid']} {g[0]['section_id']}"
                      f"{f' (+{len(g) - 1} chunks)' if len(g) > 1 else ''}: {status if ok else 'FAILED: ' + status[:150]}",
                      flush=True)
        fail_path = os.path.join(args.out, "failures.tsv")
        if failures:
            with open(fail_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["pmid", "section_id", "error"])
                w.writerows([[g[0]["pmid"], g[0]["section_id"], st] for g, st in failures])
            print(f"{len(failures)} tables failed (see {fail_path}); rerun the same command to retry them")
        elif os.path.exists(fail_path):
            os.remove(fail_path)
    collect(groups, everything, args, lookup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
