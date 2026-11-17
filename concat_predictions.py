#!/usr/bin/env python3
"""Put several predictions tables together (a prose run of 03f and a table run of 03t, say) before
relinking and merging, so each patient's findings from the text and from the tables end up under one
name.

  python concat_predictions.py runs/prose/predictions.tsv runs/tables/predictions.tsv runs/both/predictions.tsv

The last argument is the output. Columns are the union of the inputs' columns, in the order they
first appear; a column a table lacks is left empty for its rows. Rows are kept as they are, in input
order. Standard library only.
"""
from __future__ import annotations

import csv
import os
import sys


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    *inputs, out = sys.argv[1:]
    cols, rows = [], []
    for path in inputs:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for c in reader.fieldnames or []:
                if c not in cols:
                    cols.append(c)
            part = list(reader)
        rows += part
        print(f"{path}: {len(part)} rows")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}: {len(rows)} rows, {len(cols)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
