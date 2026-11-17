#!/usr/bin/env python3
"""Pack one run for the review app (viewer/patient_phenotype_review.html): the predictions,
BeLink's candidates, the gold and the section texts of the papers in the run, in one .zip to
drop on the page.

  pheno-auto review-zip --pred run/predictions_merged_llm.tsv --candidates run/predictions_belink.candidates.jsonl --gold gold.tsv --sections sections --out run/review.zip

Papers are those of the predictions, plus those of the gold that have sections (a paper where
nothing was extracted still shows, with its text and all its gold missed). Only one predictions table goes in: the app reads every TSV with patient_id and hpo_id columns
as predictions. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predictions TSV (merged, from 03d or 03e)")
    ap.add_argument("--candidates", help="BeLink's candidates JSONL from 03b (optional)")
    ap.add_argument("--gold", help="gold TSV (phenopacket phenotypes per patient); optional")
    ap.add_argument("--sections", required=True, help="sections folder with index.tsv")
    ap.add_argument("--out", required=True, help="the .zip to write")
    args = ap.parse_args()
    with open(args.pred, encoding="utf-8", newline="") as fh:
        pmids = {r["pmid"] for r in csv.DictReader(fh, delimiter="\t")}
    gold_pmids = set()
    if args.gold:                                               # papers with gold but no findings too
        with open(args.gold, encoding="utf-8", newline="") as fh:
            gold_pmids = {r["pmid"] for r in csv.DictReader(fh, delimiter="\t")}
    with open(os.path.join(args.sections, "index.tsv"), encoding="utf-8", newline="") as fh:
        have = {r["pmid"] for r in csv.DictReader(fh, delimiter="\t")}
    empty = sorted((gold_pmids & have) - pmids, key=int)
    if empty:
        print(f"Papers with gold and sections but no predicted findings (included, all gold missed): {', '.join(empty)}")
    pmids |= gold_pmids & have
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.pred, os.path.basename(args.pred))
        if args.candidates:
            z.write(args.candidates, "candidates.jsonl")
        if args.gold:
            z.write(args.gold, "gold/" + os.path.basename(args.gold))
        with open(os.path.join(args.sections, "index.tsv"), encoding="utf-8", newline="") as fh:
            for s in csv.DictReader(fh, delimiter="\t"):
                p = os.path.join(args.sections, s["path"].replace("\\", "/"))
                if s["pmid"] in pmids and os.path.exists(p):
                    z.write(p, f"PMID_{s['pmid']}/{s['section_id']}.txt")
                    n += 1
    print(f"Wrote {args.out}: {len(pmids)} papers, {n} section texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
