"""End-to-end smoke test: `pheno-auto run` on a two-patient example.

A stub OpenAI-compatible server stands in for vLLM (it answers the text, table and merging calls
with fixed JSON), and a stub linker stands in for BeLink. The example text is synthetic.
"""
from __future__ import annotations

import csv
import http.server
import json
import os
import re
import subprocess
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
SRC = os.path.join(HERE, "..", "src")
PMID = "99999999"

CASE_TEXT = ("Case report\n"
             "A 7 year old boy was investigated after his 18-month old sister was found to have marked "
             "renal insufficiency and nephrocalcinosis. He too had extensive nephrocalcinosis and partial "
             "distal renal tubular acidosis.\n")
TABLE_TEXT = ("Supplementary file: s1.pdf / page 3 table 1: Table S1. Clinical features\n"
              "Footnotes: +, present; -, absent.\n"
              "Rows are tab-separated; the first row is the column header.\n"
              "Clinical feature\tSister\tBoy\n"
              "Nephrocalcinosis\t+\t+\n"
              "Polyuria\t+\t-\n")

TEXT_ANSWER = {
    "patients": [{"name": "boy", "other_names": ["7 year old boy"], "sex": "male", "vital_status": ""},
                 {"name": "sister", "other_names": ["18-month old sister"], "sex": "female", "vital_status": ""}],
    "findings": [
        {"patient": "sister", "evidence": "A 7 year old boy was investigated after his 18-month old sister was found "
         "to have marked renal insufficiency and nephrocalcinosis.", "phenotype_quote": "marked renal insufficiency",
         "phenotype_status": "present", "standard_term": "renal insufficiency"},
        {"patient": "sister", "evidence": "A 7 year old boy was investigated after his 18-month old sister was found "
         "to have marked renal insufficiency and nephrocalcinosis.", "phenotype_quote": "nephrocalcinosis",
         "phenotype_status": "present", "standard_term": "nephrocalcinosis"},
        {"patient": "boy", "evidence": "He too had extensive nephrocalcinosis and partial distal renal tubular acidosis.",
         "phenotype_quote": "extensive nephrocalcinosis", "phenotype_status": "present",
         "standard_term": "nephrocalcinosis"},
        {"patient": "boy", "evidence": "He too had extensive nephrocalcinosis and partial distal renal tubular acidosis.",
         "phenotype_quote": "partial distal renal tubular acidosis", "phenotype_status": "present",
         "standard_term": "partial distal renal tubular acidosis"}]}

TABLE_ANSWER = {
    "reason": "Clinical features of the two siblings, one column each.",
    "patient_labels": ["Sister", "Boy"],
    "legend": [{"symbol": "+", "meaning": "present"}, {"symbol": "-", "meaning": "absent"}],
    "findings": [
        {"patient": "Sister", "evidence": {"row_label": "Nephrocalcinosis", "column_label": "Sister", "cell_value": "+"},
         "phenotype_quote": "Nephrocalcinosis", "phenotype_status": "present", "standard_term": "Nephrocalcinosis"},
        {"patient": "Boy", "evidence": {"row_label": "Nephrocalcinosis", "column_label": "Boy", "cell_value": "+"},
         "phenotype_quote": "Nephrocalcinosis", "phenotype_status": "present", "standard_term": "Nephrocalcinosis"},
        {"patient": "Sister", "evidence": {"row_label": "Polyuria", "column_label": "Sister", "cell_value": "+"},
         "phenotype_quote": "Polyuria", "phenotype_status": "present", "standard_term": "Polyuria"},
        {"patient": "Boy", "evidence": {"row_label": "Polyuria", "column_label": "Boy", "cell_value": "-"},
         "phenotype_quote": "Polyuria", "phenotype_status": "normal", "standard_term": "Polyuria"}]}


def merge_answer(request: str) -> dict:
    """One group per patient: entries whose name (or other names) say boy / sister."""
    groups: dict[str, list[str]] = {}
    for eid, rest in re.findall(r"(E\d+) \| name: (.*)", request):
        who = "Boy" if "boy" in rest.lower() else "Sister"
        groups.setdefault(who, []).append(eid)
    return {"patients": [{"patient_id": who, "entries": ids, "is_individual": True, "reason": "same sibling"}
                         for who, ids in groups.items()]}


class Stub(http.server.BaseHTTPRequestHandler):
    calls: list[str] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if "resolve patient names" in system:
            kind, answer = "merge", merge_answer(user)
        elif "article table" in system:
            kind, answer = "table", TABLE_ANSWER
        else:
            kind, answer = "text", TEXT_ANSWER
        Stub.calls.append(kind)
        data = {"model": "stub", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}]}
        out = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def pheno(*args: str) -> str:
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run([sys.executable, "-m", "pheno_auto", *args], capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"pheno-auto {args[0]} failed:\n{res.stdout}\n{res.stderr}"
    return res.stdout


def rows(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def make_sections(sec) -> None:
    (sec / f"PMID_{PMID}").mkdir(parents=True)
    (sec / f"PMID_{PMID}" / "01_case_case-report.txt").write_text(CASE_TEXT, encoding="utf-8")
    (sec / f"PMID_{PMID}" / "02_suppl_table_s1-pdf-page-3-table-1.txt").write_text(TABLE_TEXT, encoding="utf-8")
    with open(sec / "index.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pmid", "section_id", "kind", "part", "title", "n_chars", "path"])
        w.writerow([PMID, "01_case_case-report", "case", "body", "Case report", len(CASE_TEXT),
                    f"PMID_{PMID}/01_case_case-report.txt"])
        w.writerow([PMID, "02_suppl_table_s1-pdf-page-3-table-1", "suppl_table", "suppl_file",
                    "s1.pdf / page 3 table 1", len(TABLE_TEXT), f"PMID_{PMID}/02_suppl_table_s1-pdf-page-3-table-1.txt"])


def test_run(tmp_path, server):
    sec, out = tmp_path / "sections", tmp_path / "run"
    make_sections(sec)
    hpo, gold = os.path.join(FIX, "hp_mini.obo"), os.path.join(FIX, "gold.tsv")
    assert "extract-text" in pheno("help")
    log = pheno("run", "--sections", str(sec), "--out", str(out), "--hpo", hpo, "--gold", gold,
                "--model", "stub", "--base-url", server, "--workers", "1",
                "--linker", os.path.join(FIX, "stub_linker.py"))
    assert "Done:" in log

    text, tables = rows(out / "text" / "predictions.tsv"), rows(out / "tables" / "predictions.tsv")
    assert len(text) == 4 and all(r["evidence_found"] == "exact" for r in text)
    assert len(tables) == 4 and {r["evidence_found"] for r in tables} == {"verified"}
    assert [r["status"] for r in tables if r["patient_id"] == "Boy" and r["label"] == "Polyuria"] == ["excluded"]
    linked = rows(out / "both" / "predictions_belink.tsv")
    assert len(linked) == 8 and all(r["linker_id"].startswith("HP:") for r in linked)
    merged = rows(out / "both" / "predictions_merged_llm.tsv")
    assert {r["patient_id"] for r in merged} == {"Boy", "Sister"}
    assert (out / "review.zip").stat().st_size > 0
    assert Stub.calls.count("text") == 1 and Stub.calls.count("table") == 1 and Stub.calls.count("merge") == 1

    # a rerun reuses every saved answer: no new LLM calls
    pheno("run", "--sections", str(sec), "--out", str(out), "--hpo", hpo, "--model", "stub",
          "--base-url", server, "--linker", os.path.join(FIX, "stub_linker.py"))
    assert len(Stub.calls) == 3


def test_steps_alone(tmp_path):
    """Each step also runs alone, with its own options (here: the prompts and a review zip without gold)."""
    assert "expert clinician" in pheno("extract-text", "--print-prompt")
    assert "article table" in pheno("extract-tables", "--print-prompt")
    assert "resolve patient names" in pheno("merge", "--print-prompt")
