#!/usr/bin/env python3
"""Step 3b (optional). Ground the extracted findings with your own entity linker.

Reads the predictions table written by extract_findings.py / extract_table_findings.py (or their
concatenation, concat_predictions.py): one row per finding, with the model's quotation and the
sentence or cell it was found in, sends the findings to the `link()`
function of your linker module, and writes the same table with the linker's grounding added.
04_evaluate.py and 05_compare_linkers.py accept the output unchanged. The extraction JSON is not
read again: finding the quotation and its context is done once, in 03a.

Modes
  --mode ungrounded   (default) keep OntoGPT's HP ids; ask your model only for the
                      findings OntoGPT could not ground (the AUTO: ones)
  --mode all          ask your model for every finding; OntoGPT's ids are kept only
                      in the `ontogpt_id` column for comparison (05 needs this mode)

--term-mention-kinds table,suppl_table links the findings of those section kinds from their standard
term (the `label` column) alone: the mention is sent with "term_only": true, and BeLink searches its
index with the term's vector (no mixing with the quotation) and gives its re-ranker the term as both
the mention and the context. The `span` and `context` columns are left as they are, for the app.

Columns added or changed (the others are 03a's, unchanged)
  hpo_id        the id used for evaluation: with --mode all the linker's id (AUTO: when it
                declined); with --mode ungrounded OntoGPT's id, or the linker's where OntoGPT had none
  grounded      yes when hpo_id is an HP id
  linker_id     the linker's id ("" when it declined or was not asked)
  linker_label  the label the linker returned for it (BeLink: the HPO preferred name)
  linker_score  the linker's score (BeLink: re-ranker probability of the chosen option)
  source        linker = hpo_id came from the linker; ontogpt = from OntoGPT

Usage
  BELINK_INDEX=hpo/index pheno-auto relink --pred run/both/predictions.tsv --mode all \\
      --term-mention-kinds table,suppl_table --out run/both/predictions_belink.tsv
  (pheno-auto passes --linker belink_linker.py, the BeLink adapter, unless another is given)

The linker module must define

    def link(mentions: list[dict]) -> list[dict | None]

Each mention is
    {"text": the model's quotation, "llm_term": str, "context": the sentence it was found in,
     "start": int, "end": int, "unlocated": bool, "context_source": "text" | "first_pass" | "none",
     "located_spans": ["start:end", ...],
     "pmid": str, "section_id": str, "patient_id": str, "status": "observed" | "excluded",
     "ontogpt_id": OntoGPT's HP id for it, or "" if OntoGPT did not ground it,
     "term_only": true for findings of --term-mention-kinds (link from llm_term alone)}
`text` is the model's quotation when the template provides one, otherwise OntoGPT's phrase;
`llm_term` is the model's interpreted term for it; context[start:end] is the matched wording
(see 03a_locate_context.py for how the quotation and its sentence are found). The same
quotation with the same term and status, for the same patient in the same section, is sent
once.

Return, in the same order, either None (no link) or
{"id": "HP:0001250", "score": 0.93, "label": "Seizure"}; `score` and `label` are optional.

Standard library only (your linker brings its own dependencies).
"""
from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import os
import sys

NEEDED = ["pmid", "section_id", "patient_id", "label", "status", "span", "ontogpt_id", "context",
          "context_start", "context_end", "unlocated", "context_source", "located_spans"]
LINKER_COLS = ["linker_id", "linker_label", "linker_score", "source"]


def load_linker(path: str):
    spec = importlib.util.spec_from_file_location("user_linker", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "link", None)):
        raise SystemExit(f"{path} must define a function link(mentions) -> list")
    return mod.link


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions TSV from extract_findings / extract_table_findings / concat_predictions")
    ap.add_argument("--linker", required=True, help="Python file defining link(mentions)")
    ap.add_argument("--out", required=True, help="output predictions TSV")
    ap.add_argument("--mode", choices=["ungrounded", "all"], default="ungrounded")
    ap.add_argument("--min-score", type=float, default=None, help="discard links scored below this")
    ap.add_argument("--term-mention-kinds", default="",
                    help="section kinds whose findings are linked from their standard term instead of "
                         "their quotation, e.g. table,suppl_table (a table quote such as \"Valine\" or "
                         "\"44\" does not say what is abnormal; the term \"hypovalinemia\" does)")
    args = ap.parse_args()
    term_kinds = {k.strip() for k in args.term_mention_kinds.split(",") if k.strip()}

    with open(args.pred, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        in_cols = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in NEEDED if c not in in_cols]
    if missing:
        raise SystemExit(f"{args.pred} lacks the columns {', '.join(missing)}: pass the predictions of "
                         "extract_findings, extract_table_findings or concat_predictions")

    mentions, todo, seen = [], [], {}                     # todo: (row index, mention index)
    for ri, row in enumerate(rows):
        tid = row["ontogpt_id"]
        is_hp = tid.startswith("HP:")
        row.update(hpo_id=tid, grounded="yes" if is_hp else "no", linker_id="", linker_label="", linker_score="",
                   source="ontogpt")                      # start from OntoGPT's grounding (also on a rerun)
        if not (args.mode == "all" or not is_hp):
            continue
        term_only = row.get("kind") in term_kinds and bool(row["label"])
        quote = row["label"] if term_only else (row["span"] or row["label"])
        key = (row["pmid"], row["section_id"], row["patient_id"], quote.lower(), row["label"].lower(), row["status"])
        if key not in seen:
            seen[key] = len(mentions)
            mentions.append({"text": quote.replace("\r", " ").replace("\n", " "), "llm_term": row["label"],
                             "context": row["context"], "start": int(row["context_start"] or 0),
                             "end": int(row["context_end"] or 0), "unlocated": row["unlocated"] == "yes",
                             "context_source": row["context_source"],
                             "located_spans": [x for x in row["located_spans"].split(";") if x],
                             "pmid": row["pmid"], "section_id": row["section_id"], "status": row["status"],
                             "patient_id": row["patient_id"], "ontogpt_id": tid if is_hp else "",
                             "term_only": term_only})
        todo.append((ri, seen[key]))

    print(f"{len(rows)} extracted phrases; sending {len(mentions)} distinct mentions to the linker ({args.mode})")
    if mentions:
        src_n = collections.Counter(m["context_source"] for m in mentions)
        print(f"Context: sentence in the text {src_n['text']}, sentence in the first-pass summary"
              f" {src_n['first_pass']}, none (bare quotation) {src_n['none']}")
    results = load_linker(args.linker)(mentions) if mentions else []
    if len(results) != len(mentions):
        raise SystemExit(f"link() returned {len(results)} results for {len(mentions)} mentions")

    linked = changed = 0
    for ri, mi in todo:
        r, row = results[mi], rows[ri]
        ok = isinstance(r, dict) and str(r.get("id", "")).startswith("HP:")
        if ok and args.min_score is not None and r.get("score") is not None and float(r["score"]) < args.min_score:
            ok = False
        if ok:
            linked += 1
            changed += row["hpo_id"] != r["id"]
            row.update(hpo_id=r["id"], grounded="yes", source="linker", linker_id=r["id"],
                       linker_label=str(r.get("label") or ""),
                       linker_score="" if r.get("score") is None else f"{float(r['score']):.4f}")
        elif args.mode == "all":
            # your model declined: do not fall back to OntoGPT's id, so the comparison stays clean
            row.update(hpo_id="AUTO:" + row["label"].replace(" ", "%20"), grounded="no", source="linker")

    # 03a's columns, with the linker's after ontogpt_id (the order 03b has always written)
    base = [c for c in in_cols if c not in LINKER_COLS]
    at = base.index("ontogpt_id") + 1 if "ontogpt_id" in base else len(base)
    cols = base[:at] + LINKER_COLS + base[at:]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    grounded = sum(1 for r in rows if r["grounded"] == "yes")
    print(f"Linker grounded {linked} of {len(todo)} phrases it was asked about"
          + (f" ({changed} differ from OntoGPT's id)" if args.mode == "all" else ""))
    print(f"Overall {grounded} of {len(rows)} phrases now have an HPO id")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
