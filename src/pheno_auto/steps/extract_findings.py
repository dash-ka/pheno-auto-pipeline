#!/usr/bin/env python3
"""Step 3 (replaces 03_run_ontogpt.py + 03a_locate_context.py). One model call per section; every
finding comes with its patient, its present/absent status and the sentence that states it.

The model answers, for one section:

  {"patients": [{"name": "index case", "other_names": ["proband"], "sex": "male", "vital_status": "alive"}],
   "findings": [{"patient": "index case",
                 "evidence": "Urinary analyses revealed hypercalciuria, 5.67 mmol/24 hours ...",
                 "phenotype_quote": "hypercalciuria",
                 "phenotype_status": "present",
                 "standard_term": "hypercalciuria"}]}

  patients   who the section describes; their other names and sex are what the patient
             merging (03d / 03e) needs; vital_status is written as a column
  findings   one per (patient, abnormality): the verbatim sentence (evidence), the verbatim words
             within it (quote), present or absent, and a short medical term

No rule-based locating: each finding is placed where its evidence sentence occurs in the
section; a sentence that occurs more than once (the same words in two patients' paragraphs) is
taken where that patient's other findings are. The checks, done in code, are written as columns:
  evidence_found     exact   the evidence occurs in the section (case, spacing, line breaks,
                             dashes and quote marks ignored)
                     approx  not word for word: the section sentence sharing most of its words
                             (at least --min-overlap of them) is used instead
                     no      not found: the quotation is searched in the whole section instead
  quote_in_evidence  yes     the quote occurs in the evidence as it is
                     parts   in pieces: cut at "..." and found in order, or all of its content
                             words (word endings aside) within the evidence, as when a list is
                             split ("hyperostoses in the maxilla" from "hyperostoses in the
                             occipital calvarium, maxilla, mandible"); the stretch holding them
                             is the quote's place
                     no      not in the evidence: the finding is placed on the whole evidence
  check              what went wrong, if anything ("patient not listed", "evidence not found",
                     "quote not in evidence", ...)

Input
  --sections   folder written by 02_prepare_sections.py (index.tsv and the section texts)
  --hpo        hp.obo: each term is looked up among HPO labels and exact synonyms (as 03j does);
               terms not found get "AUTO:<term>", for 03b to send to BeLink

Output (in --out)
  PMID_<pmid>/<section_id>.json   the model's answer and the call's details (a finished section
                                  is not asked again: stop and rerun to resume)
  predictions.tsv                 one row per finding, with 03a_locate_context.py's columns, so
                                  relink.py, merge_patients.py, 03e and 04/05 run on
                                  it unchanged; plus vital_status, evidence, evidence_found, quote_in_evidence,
                                  check
  failures.tsv                    sections without a usable answer (rerun to retry them)

  status is "observed" (present) or "excluded" (absent). 04 scores both unless told
  --status observed; use that to compare with runs whose template had present findings only.

Usage (single line)
  python extract_findings.py --sections ../belink-test/phenopacket-auto/sections --hpo hp.obo --out findings_gptoss120b --model gpt-oss-120b --base-url http://localhost:8000/v1 --pmids curation/prose_pmids.txt --workers 4 --temperature 0
  then
  python relink.py --pred findings_gptoss120b/predictions.tsv --linker belink_linker.py --mode all --out findings_gptoss120b/predictions_belink.tsv
  python merge_patients.py --pred findings_gptoss120b/predictions_belink.tsv --out findings_gptoss120b/predictions_merged_llm.tsv --model gpt-oss-120b --base-url http://localhost:8000/v1

Retries: timeouts, connection errors and HTTP 408/409/429/5xx --retries times; an answer cut off,
empty, not JSON or without a "findings" list --json-retries more times. --collect-only rebuilds
predictions.tsv from the saved answers (after changing --hpo, say). --print-prompt shows the
instructions; --prompt FILE replaces them.

Standard library only.
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
You are an expert clinician. Your task is to extract patient-level clinical phenotypes from the provided section of an article.
Study the section text thoroughly; analyze who are the individual patients whose clinical phenotypes are being discussed; extract individual patients and the associated clinical phenotypes the text reports for each patient.
Return your answer as the following JSON object:

{"patients": # a list of the individual patients studied in this section, structured as follows
[{
	"name": <the identifier the text uses for this individual patient, copied exactly, such as "Patient 1", "P7", "II-3", "BAB3022", "proband" or "son". When the text gives both a code or number and a role word for the same patient (for example "the proband (NAPO-6 in Fig. 1)" or "Patient 2, the younger sister"), use the code.>,
	"other_names": [<the other names the text uses for this patient, each copied exactly; [] if none>],
	"sex": <"male" or "female"; "" if not stated>,
	"vital_status": <"alive" or "dead"; "" if not stated>
}],
"findings": # a list of abnormal clinical phenotypes associated with an individual patient from this section (one entry per patient's clinical phenotype), structured as follows
[{
	"patient": <an individual patient (one person) whose own clinical phenotype is reported in the provided text. This must be one of the names in "patients">,
	"evidence": <the sentence of the section that states this abnormal clinical phenotype about this patient, copied verbatim and whole from the text, character by character>,
    "phenotype_quote": <an abnormal human phenotype the section states about this patient, such as a sign, symptom, an abnormal phenotypic finding or another phenotypic abnormality reported on this patient. This is the shortest run of words within the evidence that indicates the abnormal phenotype of this patient, copied verbatim from the text. Do not rephrase, normalise or expand abbreviations. Patient phenotypes should not be inferred from numerical values, that is the text should explicitly indicate the abnormal phenotype (e.g., "low", "high", "elevated", "reduced" or outside the normal reference range).>,
	"phenotype_status": <"present" if the text states this clinical phenotype was observed in this patient and is abnormal, or "absent" if the text states this clinical phenotype was looked for in this patient but was absent or normal.>,
	"standard_term": <the standard medical term for this phenotypic abnormality being quoted>
}]}

N.B. Only the clinical phenotypes associated with individual patients described in this section should be extracted. If the section reports no individual patient's phenotypes answer {"patients": [], "findings": []}.
Do not extract aggregate phenotypes that cannot be associated with individual patients, or phenotypes of patients studied or cited in other publications.
Do not extract genes, variants, treatments, procedures, genomic finidings, or normal phenotypic findings.
Apply the one phenotype_quote per single abnormal phenotype rule: this ensures that each entry in findings quotes exactly one abnormal phenotype of a patient.

Article section:
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["patients", "findings"],
    "properties": {
        "patients": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "other_names", "sex", "vital_status"],
            "properties": {"name": {"type": "string"},
                           "other_names": {"type": "array", "items": {"type": "string"}},
                           "sex": {"type": "string"},
                           "vital_status": {"type": "string"}}}},
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["patient", "evidence", "phenotype_quote", "phenotype_status", "standard_term"],
            "properties": {"patient": {"type": "string"},
                           "evidence": {"type": "string"},
                           "phenotype_quote": {"type": "string"},
                           "phenotype_status": {"type": "string", "enum": ["present", "absent"]},
                           "standard_term": {"type": "string"}}}},
    },
}

COLS = ["pmid", "section_id", "kind", "patient_id", "hpo_id", "label", "status", "grounded", "span",
        "ontogpt_id", "chosen_span", "located_spans", "unlocated", "context_source", "context",
        "context_start", "context_end",
        "other_names", "sex", "vital_status", "age_at_last_examination",
        "section_position", "section_count", "section_part", "section_title", "first_pass",
        "evidence", "evidence_found", "quote_in_evidence", "check"]


# ---------------------------------------------------------------- HPO lookup (as 03j)

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


SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n\s*")
WORD = re.compile(r"[a-z0-9]+")


def sentences(text: str) -> list[tuple[int, int]]:
    """Rough sentence (and line) boundaries in `text`."""
    out, start = [], 0
    for m in SENT_END.finditer(text):
        if m.start() > start:
            out.append((start, m.start()))
        start = m.end()
    if start < len(text):
        out.append((start, len(text)))
    return [(a, b) for a, b in out if text[a:b].strip()]


def best_sentence(text: str, sents: list[tuple[int, int]], evidence: str, min_overlap: float):
    """The sentence sharing most of the evidence's words (at least min_overlap of them)."""
    want = collections.Counter(WORD.findall(key_of(evidence)))
    if not want:
        return None
    best, best_score = None, 0.0
    n = sum(want.values())
    for a, b in sents:
        have = collections.Counter(WORD.findall(key_of(text[a:b])))
        shared = sum((want & have).values())
        score = shared / n - 0.01 * max(0, sum(have.values()) - n) / n
        if score > best_score:
            best, best_score = (a, b), score
    return best if best and best_score >= min_overlap else None


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


def place(text: str, norm: str, pos: list[int], sents, f: dict, args, near: float | None = None) -> dict:
    """Where the finding is in the section: chosen_span, context (the evidence sentence) and the
    quote's place in it, and the checks. When the evidence occurs more than once (the same
    sentence in two patients' paragraphs: "Cerebellar signs and tremor were absent."), the
    occurrence closest to `near` (where this patient's other findings are) is taken."""
    problems = []
    ev_key, q_key = key_of(f["evidence"]), key_of(f["quote"])
    occ = spans_all(norm, pos, ev_key)
    ev = min(occ, key=lambda o: abs(o[0] - near)) if occ and near is not None else (occ[0] if occ else None)
    found = "exact" if ev else ""
    if not ev and f["evidence"].strip():
        ev = best_sentence(text, sents, f["evidence"], args.min_overlap)
        found = "approx" if ev else "no"
        problems.append("evidence not word for word" if ev else "evidence not found")
    elif not f["evidence"].strip():
        found = "no"
        problems.append("no evidence given")
    q = None
    if ev:
        # the quote within the evidence (normalised positions of the evidence's span)
        lo, hi = bisect.bisect_left(pos, ev[0]), bisect.bisect_left(pos, ev[1])
        q = span_of(norm, pos, q_key, lo, hi)
    in_ev = "yes" if q is not None else "no"
    if ev and q is None:
        q = quote_parts(text, norm, pos, ev, f["quote"])
        in_ev = "parts" if q else "no"
        if not q:
            problems.append("quote not in evidence")
    if not ev:                                  # no evidence: the quote anywhere in the section
        q = span_of(norm, pos, q_key)
        if q:
            ev = next(((a, b) for a, b in sents if a <= q[0] < b), (q[0], q[1]))
            ev = (min(ev[0], q[0]), max(ev[1], q[1]))
    if not ev:
        problems.append("quote not found")
        ctx = flat(f["quote"])
        return {"chosen_span": "", "located_spans": "", "unlocated": "yes", "context_source": "none",
                "context": ctx, "context_start": 0, "context_end": len(ctx),
                "evidence_found": found, "quote_in_evidence": "no", "check": "; ".join(problems)}
    # the context: the evidence, at most --context-chars on each side of the quote
    a, b = ev
    qa, qb = q if q else (a, b)
    lo, hi = max(a, qa - args.context_chars), min(b, qb + args.context_chars)
    while lo > a and lo < qa and not text[lo - 1].isspace():
        lo += 1
    while hi < b and hi > qb and not text[hi].isspace():
        hi -= 1
    ctx = flat(text[lo:hi])
    span = f"{qa}:{qb - 1}"
    return {"chosen_span": span, "located_spans": span, "unlocated": "no", "context_source": "text",
            "context": ctx, "context_start": qa - lo, "context_end": qb - lo,
            "evidence_found": found, "quote_in_evidence": in_ev, "check": "; ".join(problems)}


# ---------------------------------------------------------------- one answer -> rows

def clean(data: dict) -> tuple[list[dict], list[dict], list[str]]:
    """Patients (one per name) and findings in a fixed shape; each finding's patient mapped to a
    listed patient's name. Problems are returned as notes."""
    notes, patients, by_name = [], [], {}
    for p in data.get("patients") or []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or p.get("patient_id") or "").strip()
        if not name:
            continue
        q = by_name.get(name.lower())
        if q is None:
            q = {"name": name, "other_names": [], "sex": "", "vital_status": "", "age_at_last_examination": ""}
            patients.append(q)
            by_name[name.lower()] = q
        for n in p.get("other_names") or []:
            n = str(n).strip()
            if n and n.lower() != name.lower() and n not in q["other_names"]:
                q["other_names"].append(n)
        q["sex"] = q["sex"] or str(p.get("sex") or "").strip()
        q["vital_status"] = q["vital_status"] or str(p.get("vital_status") or "").strip()
        q["age_at_last_examination"] = q["age_at_last_examination"] or str(p.get("age_at_last_examination") or "").strip()
    alias = {}
    for q in patients:                              # other names point to the patient, unless shared
        for n in q["other_names"]:
            alias.setdefault(n.lower(), []).append(q["name"])
    findings, seen = [], set()
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        who = str(f.get("patient") or "").strip()
        quote = str(f.get("phenotype_quote") or f.get("quote") or "").strip()
        term = str(f.get("standard_term") or f.get("term") or "").strip() or quote
        if not quote and not term or term.lower().rstrip(".") in PLACEHOLDERS:
            continue
        note = ""
        if who.lower() in by_name:
            who = by_name[who.lower()]["name"]
        elif len(alias.get(who.lower(), [])) == 1:
            who = alias[who.lower()][0]
        else:
            note = "patient not listed"
            if who and who.lower() not in by_name:
                q = {"name": who or "individual", "other_names": [], "sex": "", "vital_status": "", "age_at_last_examination": ""}
                patients.append(q)
                by_name[q["name"].lower()] = q
                notes.append(f"finding for {who!r}, who is not among the patients: added")
            who = who or "individual"
        status = "excluded" if str(f.get("phenotype_status") or f.get("status") or "").strip().lower() == "absent" else "observed"
        evidence = str(f.get("evidence") or "").strip()
        k = (who.lower(), quote.lower(), term.lower(), status, key_of(evidence))
        if k in seen:
            continue
        seen.add(k)
        findings.append({"patient": who, "evidence": evidence, "quote": quote, "term": term,
                         "status": status, "note": note})
    return patients, findings, notes


def summary_line(patients: list[dict], findings: list[dict]) -> str:
    """The section's patients in one line, as OntoGPT's first pass writes it, for 03d."""
    parts = []
    for p in patients:
        facts = [x for x in (p["sex"], p["vital_status"], p["age_at_last_examination"]) if x]
        facts += [(f["quote"] if f["status"] == "observed" else "no " + f["quote"]).replace(";", ",")
                  for f in findings if f["patient"] == p["name"]]
        name = p["name"] + (f" ({', '.join(p['other_names'])})" if p["other_names"] else "")
        parts.append(f"{name}: " + ", ".join(facts))
    return "patients: " + "; ".join(parts)


def rows_of(s: dict, res: dict, lookup: HPOLookup, args, place_info: dict) -> list[dict]:
    text = res["input_text"]
    patients, findings, _ = clean(parse_answer(res["raw_json"]))
    norm, pos = normalise(text)
    sents = sentences(text)
    first_pass = " ".join(summary_line(patients, findings).split())
    by_name = {p["name"]: p for p in patients}
    # where each patient's findings are: the middle of those whose evidence occurs only once
    spots = collections.defaultdict(list)
    for f in findings:
        occ = spans_all(norm, pos, key_of(f["evidence"]))
        if len(occ) == 1:
            spots[f["patient"]].append(occ[0][0])
    near = {who: sorted(v)[len(v) // 2] for who, v in spots.items()}
    rows = []
    for f in findings:
        p = by_name[f["patient"]]
        hid = lookup.ground(f["term"]) or "AUTO:" + urlquote(f["term"])
        where = place(text, norm, pos, sents, f, args, near.get(f["patient"]))
        if f["note"]:
            where["check"] = "; ".join(x for x in (f["note"], where["check"]) if x)
        rows.append({"pmid": s["pmid"], "section_id": s["section_id"], "kind": s["kind"], "patient_id": f["patient"],
                     "hpo_id": hid, "label": f["term"], "status": f["status"],
                     "grounded": "yes" if hid.startswith("HP:") else "no", "span": f["quote"], "ontogpt_id": hid,
                     **where, "evidence": flat(f["evidence"]),
                     "other_names": ";".join(p["other_names"]), "sex": p["sex"],
                     "vital_status": p["vital_status"], "age_at_last_examination": p["age_at_last_examination"],
                     **place_info, "first_pass": first_pass})
    return rows


# ---------------------------------------------------------------- one section

def out_path(args, s: dict) -> str:
    return os.path.join(args.out, f"PMID_{s['pmid']}", s["section_id"] + ".json")


def is_done(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return "raw_json" in json.load(fh)
    except (OSError, ValueError):
        return False


def run_one(s: dict, args, endpoint: Endpoint, system: str) -> tuple[dict, str]:
    dst = out_path(args, s)
    if is_done(dst):
        return s, "skipped"
    with open(os.path.join(args.sections, s["path"].replace("\\", "/")), encoding="utf-8", newline="") as fh:
        text = fh.read()
    user = "Section text:\n<<<\n" + text + "\n>>>"
    content, info, data, problem = "", {}, None, ""
    for _ in range(1 + args.json_retries):
        try:
            content, info = endpoint.chat(system, user)
        except RuntimeError as e:
            return s, str(e)[:300]
        if info["finish_reason"] == "length":
            problem = f"answer cut off at --max-tokens {args.max_tokens}"
            continue
        try:
            data = parse_answer(content)
        except ValueError:
            problem = "invalid JSON in the answer" if content.strip() else "empty answer"
            continue
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            break
        problem, data = "JSON without a 'findings' list", None
    if data is None:
        return s, f"{problem} ({1 + args.json_retries} attempts)"
    _, _, notes = clean(data)
    res = {"input_text": text, "raw_json": content, "notes": notes, "extraction": "03f_extract_findings", **info}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(dst + ".part", dst)
    return s, "done" + (f" ({'; '.join(notes)})" if notes else "")


# ---------------------------------------------------------------- collect

def collect(sections: list[dict], everything: list[dict], args, lookup: HPOLookup) -> None:
    n_of = collections.Counter(s["pmid"] for s in everything)
    seen_n, place_of = collections.Counter(), {}
    for s in everything:                     # place of each section in its paper, over all its sections
        seen_n[s["pmid"]] += 1
        place_of[(s["pmid"], s["section_id"])] = seen_n[s["pmid"]]
    rows, n_files = [], 0
    for s in sections:
        path = out_path(args, s)
        if not is_done(path):
            continue
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        n_files += 1
        info = {"section_position": place_of[(s["pmid"], s["section_id"])], "section_count": n_of[s["pmid"]],
                "section_part": s.get("part", ""), "section_title": " ".join((s.get("title") or "").split())}
        rows += rows_of(s, res, lookup, args, info)
    out = os.path.join(args.out, "predictions.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    ev = collections.Counter(r["evidence_found"] for r in rows)
    st = collections.Counter(r["status"] for r in rows)
    g = sum(r["grounded"] == "yes" for r in rows)
    qin = collections.Counter(r["quote_in_evidence"] for r in rows)
    unl = sum(r["unlocated"] == "yes" for r in rows)
    print(f"{n} findings ({st['observed']} present, {st['excluded']} absent) of "
          f"{len({(r['pmid'], r['section_id'], r['patient_id']) for r in rows})} patient entries in {n_files} sections")
    print(f"Evidence: word for word {ev['exact']}, nearest sentence {ev['approx']}, not found {ev['no']}; "
          f"quote in the evidence: as one piece {qin['yes']}, in pieces {qin['parts']}, not {qin['no']}; "
          f"not placed in the text {unl}")
    print(f"Grounded by HPO lookup {g}, not {n - g} (AUTO:, for 03b)")
    problems = collections.Counter(p for r in rows for p in r["check"].split("; ") if p)
    if problems:
        print("Checks: " + ", ".join(f"{k} {v}" for k, v in problems.most_common()))
    print(f"Wrote {out}")


# ---------------------------------------------------------------- main

def select_parts(sections: list[dict], parts: str) -> list[dict]:
    want = {p.strip() for p in parts.split(",") if p.strip()}
    return [s for s in sections if not s.get("part") or s["part"] in want]


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
    ap.add_argument("--kinds", default=DEFAULT_KINDS, help=f"section kinds (default {DEFAULT_KINDS}; all: {ALL_KINDS})")
    ap.add_argument("--parts", default=DEFAULT_PARTS, help=f"article parts (default {DEFAULT_PARTS}; all: {ALL_PARTS})")
    ap.add_argument("--pmids", help="file with one PMID per line: only these papers")
    ap.add_argument("--limit-papers", type=int, help="only the first N papers (trial run)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per request (default 900)")
    ap.add_argument("--retries", type=int, default=3, help="retries after timeouts and server errors")
    ap.add_argument("--json-retries", type=int, default=2,
                    help="new calls when the answer is cut off, empty, not JSON or has no findings list (default 2)")
    ap.add_argument("--min-overlap", type=float, default=0.8,
                    help="evidence not found word for word: share of its words a section sentence must have (default 0.8)")
    ap.add_argument("--context-chars", type=int, default=200,
                    help="characters of the evidence kept on each side of the quote (default 200)")
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
    print(f"{len(sections)} sections from {len({s['pmid'] for s in sections})} papers")
    lookup = HPOLookup(args.hpo, args.synonyms)
    os.makedirs(args.out, exist_ok=True)

    if not args.collect_only:
        endpoint = Endpoint(args)
        failures, done, start = [], 0, time.time()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(run_one, s, args, endpoint, system) for s in sections]
            for fut in as_completed(futures):
                s, status = fut.result()
                done += 1
                ok = status.startswith("done") or status == "skipped"
                if not ok:
                    failures.append((s, status))
                print(f"  [{done}/{len(futures)}, {time.time() - start:.0f}s] PMID {s['pmid']} {s['section_id']}: "
                      f"{status if ok else 'FAILED: ' + status[:150]}", flush=True)
        fail_path = os.path.join(args.out, "failures.tsv")
        if failures:
            with open(fail_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, delimiter="\t")
                w.writerow(["pmid", "section_id", "error"])
                w.writerows([[s["pmid"], s["section_id"], st] for s, st in failures])
            print(f"{len(failures)} sections failed (see {fail_path}); rerun the same command to retry them")
        elif os.path.exists(fail_path):
            os.remove(fail_path)
    collect(sections, everything, args, lookup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
