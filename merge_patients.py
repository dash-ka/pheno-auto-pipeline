#!/usr/bin/env python3
"""Step 3d. Let the model decide which patient names of a paper are the same individual.

Every section of a paper is extracted on its own, so one patient appears under several names:
"proband" in the abstract, "NAPO-6" in the results, "the boy" in a figure legend. 04/05 match
predicted to gold patients by name, so each patient needs one name per paper.
03e_merge_patients.py does this with rules; this step asks the model instead (an alternative
to 03e: run one or the other on the same input).

Input
  --pred   the predictions of 03a_locate_context.py or relink.py: one row per finding, with
           its section's first-pass line (first_pass), position, part and title, and its
           patient's other names and sex. Nothing else is read.

What the model is asked, one call per paper
  The paper's sections that have patients, in reading order. For each section: its position
  in the paper, kind, part and title; the patients the section describes, with what it says
  about each (the first_pass column: OntoGPT's raw_completion_output for the section, given
  whole, its "patients:" prefix dropped and one patient per line at the semicolons); and the
  names used for patients in that section, one entry per patient_id its findings are recorded
  under, with the other names and sex recorded for it. The model groups the names; how the
  descriptions and the names correspond is left to it, nothing is matched in code. No findings
  and no article sentences are sent, and nothing about how the extraction was done.
  (For outputs of 03j_extract_json.py the descriptions are a summary 03j built from its answer,
  so they repeat the entries' quotations.)
  The model returns groups of entries, one per individual, with the name to keep:
    {"patients": [{"patient_id": "NAPO-6", "entries": ["E1", "E4"], "is_individual": true,
                   "reason": "the abstract's proband is NAPO-6 in the results"}]}
  --print-prompt shows the instructions.

Checks on the answer, before anything is renamed
  * every entry is placed exactly once: an entry the model left out stays a patient of its
    own, an entry placed twice stays in its first group, unknown entry ids are ignored;
  * the kept name must be one the group's entries use (a name or one of its other names), so
    the model cannot invent one; otherwise the group's most frequent coded name is kept (as in
    03e), or its most frequent name;
  * two groups never share a name (as 04 compares names, "Patient 1" = "P1"): the second keeps
    its own most frequent name instead;
  * a group with is_individual false (an entry about several people, "both sibs") is not renamed.
  A paper whose names are all one already (by 04's comparison) is not sent. A paper whose call
  fails keeps its names; rerunning the command retries it.

Output
  --out    the same predictions, every row and column kept, with patient_id renamed and three
           columns added:
             original_patient_id   the name before this step
             merged_names          all names of the patient in the paper, "; "-joined
             merge_reason          llm: <the model's reason> | llm: several people: <reason> |
                                   not sent: one name | llm failed: <error> | "" (kept as it was)
  <out>.llm.json   for each paper sent: the request, the model's answer, the corrections made
           to it, the instructions, and details of the call. A rerun reuses a paper's answer
           while its request and the instructions are unchanged (--redo asks again).

The model is reached through any OpenAI-compatible endpoint, by default vLLM on localhost:8000.
The answer format is enforced by the server (--response-format json_schema, vLLM's guided
decoding); a server that rejects it (HTTP 400) is asked again with json_object, then with none
(JSON taken from the text). Timeouts and server errors are retried --retries times; an answer
that is cut off, empty or not usable JSON is asked for again --json-retries times.

Usage
  python merge_patients.py --pred work/ontogpt/predictions_belink.tsv \\
      --model gpt-oss-120b --out work/ontogpt/predictions_merged_llm.tsv
Then score the merged file with 04/05 as before.

Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

FORMATS = ["json_schema", "json_object", "none"]

# Names are compared as 04_evaluate.py compares them (the same code as in 03e_merge_patients.py).
ID_STOPWORDS = {"patient", "patients", "individual", "case", "subject", "proband", "pt", "p", "no", "number", "id",
                "the", "family", "fam", "f", "only", "affected", "index", "her", "his", "s"}   # as in 04_evaluate.py
AGE = r"\b\d+(?:\.\d+)?\s*-?\s*(?:years?|yrs?|y|months?|mos?|weeks?|wks?|days?)\b"   # "4-year-old" is not a code
def id_key(patient_id: str) -> str:
    """As 04_evaluate.py: 'Patient 1' = 'P1' = '1'; role words keep their words ('brother')."""
    tokens = re.findall(r"[a-z]+|\d+", str(patient_id).lower())
    kept = [t.lstrip("0") or "0" if t.isdigit() else t for t in tokens if t not in ID_STOPWORDS]
    return "".join(kept) or "".join(tokens)              # "proband" alone keeps its word


def is_coded(name: str) -> bool:
    name = re.sub(AGE, " ", name, flags=re.I)             # "4-year-old boy" is a description, not a code
    return bool(re.search(r"\d", name)) or bool(re.fullmatch(r"\s*[IVX]+[-:.]\s*\d+\s*", name))



PROMPT = """\
You resolve patient names across the sections of one biomedical article.

The sections of the article were read one at a time, and each section refers to its patients in \
its own way: "the proband" in the abstract, "NAPO-6" in the results, "the boy" in a figure \
legend. You get the sections that describe patients, in reading order. For each section: where \
it is in the article (position, kind, part, title); the patients it describes, with what it says \
about each of them; and the names used for patients in that section, each with an id (E1, E2, \
...), the other names it has in that section, and sex when known.

Your task: decide which of these names refer to the same individual across the article, and \
group the name ids by individual.
- Every name id goes into exactly one group. A name that is nobody else stays alone in its group.
- Names with the same code (in the same family) are the same person.
- Put names together only when the evidence shows they are the same person.
- When in doubt, keep names apart: joining two people is worse than leaving one person under two names.
- A name that stands for several people rather than one ("the patients", "both sibs") goes alone in a group with is_individual false.

For each group give
- patient_id: the name the article uses for this individual, copied exactly from the group's \
names (a name or one of its other names), with nothing added: no sex, age or description in \
brackets. Prefer its code ("NAPO-6", "Patient 2", "II-3"; with its family, such as \
"Family 2 III-4", when the same code occurs in several families and that form is among the \
names); without a code, its most specific role word. Two groups never get the same patient_id.
- entries: the name ids, such as "E1".
- is_individual: true for one person, false for a name that stands for several people.
- reason: one short sentence on the evidence.

Answer with one JSON object and nothing else:
{"patients": [{"patient_id": "...", "entries": ["E1", "..."], "is_individual": true, "reason": "..."}]}
"""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["patients"],
    "properties": {"patients": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["patient_id", "entries", "is_individual", "reason"],
        "properties": {"patient_id": {"type": "string"},
                       "entries": {"type": "array", "items": {"type": "string"}},
                       "is_individual": {"type": "boolean"},
                       "reason": {"type": "string"}}}}},
}


# ---------------------------------------------------------------- the model

class Endpoint:
    """An OpenAI-compatible chat endpoint; remembers a response format the server rejected."""

    def __init__(self, args, schema: dict, schema_name: str):
        self.args = args
        self.schema, self.schema_name = schema, schema_name
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
                                       "json_schema": {"name": self.schema_name, "strict": True,
                                                       "schema": self.schema}}
        elif fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def chat(self, system: str, user: str) -> tuple[str, dict]:
        """The answer's text and some details of the call. Raises RuntimeError on failure."""
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
    """The JSON object in the answer (code fences and text around it are tolerated)."""
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except ValueError:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            return json.loads(s[a:b + 1])
        raise



# ---------------------------------------------------------------- the entries of a paper

def build_entries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """One entry per (section, name) in the order of the rows (the extraction's order), and the
    paper's sections with entries, in reading order."""
    entries: dict[tuple, dict] = {}
    secs: dict[str, dict] = {}
    for r in rows:
        sid = r["section_id"]
        secs.setdefault(sid, {"section_id": sid, "kind": r.get("kind", ""), "part": r.get("section_part", ""),
                              "title": r.get("section_title", ""), "position": r.get("section_position", ""),
                              "count": r.get("section_count", ""), "first_pass": r.get("first_pass", "")})
        e = entries.setdefault((sid, r["patient_id"]), {"section_id": sid, "name": r["patient_id"],
                                                         "other_names": [], "sex": ""})
        for o in (r.get("other_names") or "").split(";"):
            o = o.strip()
            if o and o != e["name"] and o not in e["other_names"]:
                e["other_names"].append(o)
        e["sex"] = e["sex"] or (r.get("sex") or "").strip()
    first = {sid: i for i, sid in enumerate(secs)}
    order = sorted(secs, key=lambda sid: (int(secs[sid]["position"] or 10 ** 6), first[sid]))
    out = [e for sid in order for e in entries.values() if e["section_id"] == sid]
    for i, e in enumerate(out, 1):
        e["id"] = f"E{i}"
    return out, [secs[sid] for sid in order]


def describe(pmid: str, entries: list[dict], secs: list[dict]) -> str:
    """The request for one paper: its sections in reading order, each with the patients it describes
    (the first-pass line, whole, one patient per line) and the names used in it."""
    n_all = next((s["count"] for s in secs if s["count"]), "")
    lines = [f"Article PMID {pmid}: {len(secs)} sections with patients"
             + (f" (of {n_all} sections)" if n_all else "") + f", {len(entries)} entries.", ""]
    for s in secs:
        where = ", ".join(x for x in (s["kind"], f"part {s['part']}" if s["part"] else "",
                                      f'title "{s["title"]}"' if s["title"] else "") if x)
        at = f"Section {s['position']} of {s['count']}" if s["position"] else "Section"
        lines.append(f"{at}: {s['section_id']} ({where})")
        described = re.sub(r"^\s*patients\s*:\s*", "", s["first_pass"] or "", flags=re.I)
        items = [x.strip() for x in described.split(";") if x.strip()]
        lines.append("  Patients described in this section:")
        lines += [f"    - {x}" for x in items] or ["    (none given)"]
        lines.append("  Names used for patients in this section:")
        for e in entries:
            if e["section_id"] == s["section_id"]:
                lines.append(f"    {e['id']} | name: {e['name']} | other names: {'; '.join(e['other_names']) or '-'}"
                             f" | sex: {e['sex'] or '-'}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- the model's answer

def ask(endpoint, user: str, cached: dict | None, args) -> tuple[dict | None, str, str, dict]:
    """(answer or None, "asked" | "cached" | the problem, the raw answer, call details)."""
    if (cached and not args.redo and cached.get("request") == user and cached.get("prompt") == PROMPT
            and isinstance(cached.get("answer"), dict)):
        return cached["answer"], "cached", cached.get("raw", ""), cached.get("info", {})
    problem, content, info = "", "", {}
    for _ in range(1 + args.json_retries):
        try:
            content, info = endpoint.chat(PROMPT, user)
        except RuntimeError as e:
            return None, str(e)[:300], "", {}
        if info.get("finish_reason") == "length":
            problem = f"answer cut off at --max-tokens {args.max_tokens}"
            continue
        try:
            data = parse_answer(content)
        except ValueError:
            problem = "invalid JSON in the answer" if content.strip() else "empty answer"
            continue
        if isinstance(data, dict) and isinstance(data.get("patients"), list):
            return data, "asked", content, info
        problem = "JSON without a 'patients' list"
    return None, f"{problem} ({1 + args.json_retries} attempts)", content, info


def fallback_name(names: Counter) -> str:
    """As 03e: the most frequent coded name, else the most frequent name (ties: the longer)."""
    coded = [n for n in names if is_coded(n)]
    return max(coded or list(names), key=lambda n: (names[n], len(n)))


def apply_answer(entries: list[dict], answer: dict, row_count: Counter) -> tuple[dict, list]:
    """(section, name) -> (kept name, merged names, reason), and notes on what was corrected."""
    by_id = {e["id"]: e for e in entries}
    placed: dict[str, int] = {}
    groups, notes = [], []
    for g in answer.get("patients") or []:
        if not isinstance(g, dict):
            continue
        ids = []
        for x in g.get("entries") or []:
            x = str(x).strip()
            if x not in by_id:
                notes.append(f"unknown entry {x!r} ignored")
            elif x in placed:
                notes.append(f"{x} placed twice: kept in its first group")
            else:
                placed[x] = len(groups)
                ids.append(x)
        if ids:
            groups.append({"ids": ids, "want": str(g.get("patient_id") or "").strip(),
                           "individual": g.get("is_individual") is not False, "reason": str(g.get("reason") or "")})
    for e in entries:
        if e["id"] not in placed:
            notes.append(f"{e['id']} ({e['name']}) left out by the model: kept as its own patient")
            groups.append({"ids": [e["id"]], "want": e["name"], "individual": True, "reason": "not placed by the model"})

    result, used_keys, kept_as = {}, set(), {}
    for g in groups:
        members = [by_id[x] for x in g["ids"]]
        names = Counter()
        for e in members:
            names[e["name"]] += row_count[(e["section_id"], e["name"])] or 1
            for o in e["other_names"]:
                names[o] += 0
        merged = "; ".join(sorted(names))
        if not g["individual"]:
            for e in members:
                used_keys.add(id_key(e["name"]))
                result[(e["section_id"], e["name"])] = (e["name"], merged, "llm: several people: " + g["reason"])
            continue
        allowed = {}                                      # name key -> the group's commonest spelling of it
        for n in sorted(names, key=lambda n: (-names[n], -len(n))):
            allowed.setdefault(id_key(n), n)
        want = g["want"]
        if want and want not in names and re.sub(r"\s*\([^()]*\)\s*$", "", want) in names:
            notes.append(f"name {want!r} for {', '.join(g['ids'])}: the added bracket was dropped")
            want = re.sub(r"\s*\([^()]*\)\s*$", "", want)
        keep = want if want in names else (allowed.get(id_key(want)) if want else None)
        if keep is None:
            if want:
                notes.append(f"name {want!r} for {', '.join(g['ids'])} is not one of their names")
            keep = fallback_name(names)
        if id_key(keep) in used_keys:                     # two groups must not end up as one in 04
            own = Counter({e["name"]: names[e["name"]] for e in members})
            free = [n for n in sorted(own, key=lambda n: (-own[n], -len(n))) if id_key(n) not in used_keys]
            if not free and id_key(keep) in kept_as:
                # every name of this group is the same identifier as the other group's ("2.1" and
                # "Patient 2.1"): 04 would count them as one patient anyway, so they are joined
                keep = kept_as[id_key(keep)]
                notes.append(f"name {keep!r} already taken by another patient with the same identifier: "
                             f"{', '.join(g['ids'])} joined with it")
                for e in members:
                    result[(e["section_id"], e["name"])] = (keep, merged, "llm: " + g["reason"] + " (joined: same identifier)")
                continue
            notes.append(f"name {keep!r} already taken by another patient: "
                         + (f"{', '.join(g['ids'])} keep {free[0]!r}" if free else f"{', '.join(g['ids'])} not renamed"))
            if not free:
                for e in members:
                    result[(e["section_id"], e["name"])] = (e["name"], merged, "llm: " + g["reason"] + " (name taken)")
                continue
            keep = free[0]
        used_keys.add(id_key(keep))
        kept_as.setdefault(id_key(keep), keep)
        for e in members:
            result[(e["section_id"], e["name"])] = (keep, merged, "llm: " + g["reason"])
    return result, notes


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predictions TSV from 03a_locate_context.py or relink.py")
    ap.add_argument("--out", help="rewritten predictions TSV")
    ap.add_argument("--model", help="model name as the server knows it (curl <base-url>/models)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible endpoint (default $OPENAI_BASE_URL or http://localhost:8000/v1)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the key")
    ap.add_argument("--response-format", choices=FORMATS, default="json_schema")
    ap.add_argument("--max-tokens", type=int, default=16384, help="cap on the answer, reasoning included")
    ap.add_argument("--temperature", type=float, help="not sent unless given")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="for gpt-oss / gpt-5 models")
    ap.add_argument("--workers", type=int, default=4, help="papers sent in parallel (default 4)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per request (default 900)")
    ap.add_argument("--retries", type=int, default=3, help="retries after timeouts and server errors")
    ap.add_argument("--json-retries", type=int, default=2, help="new calls for an unusable answer (default 2)")
    ap.add_argument("--pmid", action="append", help="only this paper (repeatable); the others keep their names")
    ap.add_argument("--redo", action="store_true", help="ask again even when an answer is saved")
    ap.add_argument("--print-prompt", action="store_true", help="print the instructions and exit")
    args = ap.parse_args()
    if args.print_prompt:
        print(PROMPT)
        return 0
    missing = [o for o in ("pred", "out", "model") if not getattr(args, o)]
    if missing:
        ap.error("required: " + ", ".join("--" + o for o in missing))

    with open(args.pred, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if "first_pass" not in cols:
        raise SystemExit(f"{args.pred} has no first_pass column: pass the output of 03a_locate_context.py or "
                         "relink.py (rerun 03a if the file was made by an older version)")
    for r in rows:                                         # a rerun on merged output starts from the originals
        if r.get("original_patient_id"):
            r["patient_id"] = r["original_patient_id"]
        r["patient_id"] = (r.get("patient_id") or "").strip() or "individual"
    rows_by = defaultdict(list)
    for r in rows:
        rows_by[r["pmid"]].append(r)

    log_path = os.path.splitext(args.out)[0] + ".llm.json"
    log = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as fh:
                log = json.load(fh)
        except (OSError, ValueError):
            log = {}

    todo, result = {}, {}                                 # result: (pmid, section, name) -> (kept, merged, reason)
    for pmid, prow in sorted(rows_by.items(), key=lambda x: (len(x[0]), x[0])):
        if args.pmid and pmid not in args.pmid:
            continue
        entries, secs = build_entries(prow)
        if len({id_key(n) for e in entries for n in [e["name"], *e["other_names"]]}) < 2:
            for e in entries:                              # one name only: nothing to decide
                result[(pmid, e["section_id"], e["name"])] = (e["name"], e["name"], "not sent: one name")
            continue
        todo[pmid] = (entries, describe(pmid, entries, secs))
    print(f"{len(todo)} papers to send, {sum(len(v[0]) for v in todo.values())} entries "
          f"({len(rows_by) - len(todo)} papers have a single name or were not selected)")

    def save_log():
        with open(log_path + ".part", "w", encoding="utf-8") as fh:
            json.dump(log, fh, ensure_ascii=False, indent=1)
        os.replace(log_path + ".part", log_path)

    endpoint = Endpoint(args, schema=SCHEMA, schema_name="patient_groups")
    n_failed, n_notes, start = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(ask, endpoint, user, log.get(pmid), args): pmid for pmid, (_, user) in todo.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            pmid = futures[fut]
            entries, user = todo[pmid]
            answer, status, raw, info = fut.result()
            if answer is None:
                n_failed += 1
                for e in entries:
                    result[(pmid, e["section_id"], e["name"])] = (e["name"], "", f"llm failed: {status}")
                log[pmid] = {"request": user, "prompt": PROMPT, "error": status, "raw": raw, "info": info}
                save_log()
                print(f"  [{i}/{len(futures)}, {time.time() - start:.0f}s] PMID {pmid}: FAILED: {status}", flush=True)
                continue
            row_count = Counter((r["section_id"], r["patient_id"]) for r in rows_by[pmid])
            res, notes = apply_answer(entries, answer, row_count)
            result.update({(pmid,) + k: v for k, v in res.items()})
            n_notes += len(notes)
            log[pmid] = {"request": user, "prompt": PROMPT, "answer": answer, "corrections": notes, "raw": raw,
                         "info": info}
            save_log()
            n_kept = len({v[0] for v in res.values()})
            print(f"  [{i}/{len(futures)}, {time.time() - start:.0f}s] PMID {pmid}: {len(entries)} entries -> "
                  f"{n_kept} names ({status})" + (f"; corrections: {' | '.join(notes)}" if notes else ""), flush=True)

    out_cols = [c for c in cols if c not in ("original_patient_id", "merged_names", "merge_reason")]
    out_cols += ["original_patient_id", "merged_names", "merge_reason"]
    changed = 0
    for r in rows:
        old = r["patient_id"]
        kept, merged, reason = result.get((r["pmid"], r["section_id"], old), (old, "", ""))
        r.update(original_patient_id=old, patient_id=kept, merged_names=merged, merge_reason=reason)
        changed += kept != old
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    before = sum(len({r["original_patient_id"] for r in v}) for v in rows_by.values())
    after = sum(len({r["patient_id"] for r in v}) for v in rows_by.values())
    print(f"{len(rows_by)} papers: {before} patient names -> {after} after merging; {changed} of {len(rows)} rows renamed"
          + (f"; {n_notes} corrections to the model's answers" if n_notes else ""))
    if n_failed:
        print(f"{n_failed} papers failed and keep their names; rerun the same command to retry them")
    print(f"Wrote {args.out} and {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
