"""BeLink-HPO as the linker of relink.py.

BeLink-HPO (https://github.com/dash-ka/BeLink-HPO) links a mention in two steps:
  1. retrieval  SapBERT embeddings of every HPO name and synonym; the query is the
                mention vector mixed with the vector of a "standard name" for it
                (alpha * mention + (1 - alpha) * standard name)
  2. re-ranking Qwen3-4B with a LoRA adapter sees the sentence, the mention and the
                first 10 distinct retrieved concepts as a multiple-choice question, and
                picks one of them or "None of the above".

What this adapter gives it, per mention from relink.py:
  mention        observed findings: the exact quotation (observed_spans); excluded findings:
                 OntoGPT's term without the negation (excluded_phenotypes), because the
                 re-ranker was trained on positive mentions only (negated ones were skipped)
  standard name  OntoGPT's medical term for the quotation (observed_phenotypes). This is
                 the role the LLM-generated standard names play in BeLink's own evaluation.
  context        the sentence that contains the quotation (BeLink was trained on sentences);
                 the bare mention when the quotation was not found in the section

Use
  pip install git+https://github.com/dash-ka/BeLink-HPO.git      (torch, transformers, peft)
  # once: the retrieval index (needs hp.obo; prepare-obo also writes preferred_names.json)
  belink-hpo prepare-obo --ontology_path hp.obo --output_dir belink_index
  belink-hpo embed-terms --ontology belink_index/terminology.json.gz --out_vectors belink_index/hpo_vec.npy

  BELINK_INDEX=belink_index python relink.py --mode all --sections work/sections \
      --ontogpt work/ontogpt --linker belink_linker.py --out work/ontogpt/predictions_belink.tsv
  python 05_compare_linkers.py --gold work/gold.tsv --hpo hp.json --index work/sections/index.tsv \
      --pred work/ontogpt/predictions_belink.tsv --log work/ontogpt/predictions_belink.candidates.jsonl \
      --out work/compare_belink

Term-only mentions (relink.py --term-mention-kinds, for table findings): a mention with
"term_only": true is linked from its standard term alone. Retrieval searches the index with the
term's own vector (no mixing with a mention vector), and the re-ranker is given the term as both
the mention and the context. A table's quote ("Valine", "44") and its cell context do not say what
is abnormal; the term ("hypovalinemia") does.

Settings (environment variables, since relink.py passes only the mentions)
  BELINK_INDEX            folder with hpo_vec.npy, ontology_mapping.txt, preferred_names.json
                          (default: belink_index)
  BELINK_ADAPTER          re-ranker adapter, Hub id or local folder
                          (default: Dash00/qwen3-4b-lora-hpo-reranker)
  BELINK_TOP_K            names retrieved per mention (default 30; the re-ranker is shown the
                          first 10 distinct concepts among them, as in BeLink's evaluation)
  BELINK_ALPHA            weight of the mention vector in the query (default 0.6)
  BELINK_STANDARD_NAME    1 = use OntoGPT's term as the standard name (default), 0 = mention only
  BELINK_CONTEXT          sentence (default) | window (03b's +-200 characters) | none (mention only)
  BELINK_EXCLUDED_MENTION term (default: OntoGPT's term) | span (the quotation, with its negation)
  BELINK_NUMERIC_TERM     1 = when the mention contains a number (a quoted lab value), the re-ranker
                          is shown "mention (OntoGPT's term)", e.g. "creatinine: 454 mcmol/L
                          (elevated creatinine)", since the value alone does not say which way it is
                          abnormal; retrieval is unchanged (default 1; 0 = the mention alone)
  BELINK_ADD_ONTOGPT      1 = if OntoGPT's own HP id is not among the 10 options, it replaces the
                          10th, so the re-ranker can still choose it (default 0)
  BELINK_4BIT             1 = load the re-ranker in 4 bit (needs bitsandbytes; default 0)
  BELINK_BATCH            re-ranker batch size (default 16)
  BELINK_LOG              JSONL with every mention, its candidates, the options shown and the
                          choice (default: <03b --out without .tsv>.candidates.jsonl, or
                          belink_candidates.jsonl when that is not known)

Returned to relink.py per mention: {"id", "score" (re-ranker probability of the chosen
option), "label"} or None when the re-ranker answers "None of the above".
"""
from __future__ import annotations

import json
import os
import re
import sys

MAX_OPTIONS = 10
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9(\[])|\n+")
_ABBREV = re.compile(r"\b(?:Fig|Figs|Dr|Mr|Mrs|Ms|Prof|vs|al|i\.e|e\.g|approx|No|Ref|Refs|Tab|ca)\.$", re.I)
_state: dict = {}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _log_path() -> str:
    if os.environ.get("BELINK_LOG"):
        return os.environ["BELINK_LOG"]
    if "--out" in sys.argv:                                 # same folder and name as 03b's output
        out = sys.argv[sys.argv.index("--out") + 1]
        return re.sub(r"\.tsv$", "", out) + ".candidates.jsonl"
    return "belink_candidates.jsonl"


# --------------------------------------------------------------------------- #
# Mention, standard name and context
# --------------------------------------------------------------------------- #
def sentence_around(ctx: str, start: int, end: int) -> str:
    """The sentence of ctx that contains ctx[start:end] (several if the quotation spans them)."""
    bounds = [0]
    for m in _SENT_END.finditer(ctx):
        if "\n" not in m.group() and _ABBREV.search(ctx[:m.start()]):
            continue                                        # "Fig. 2", "et al. 2010"
        bounds.append(m.end())
    bounds.append(len(ctx))
    lo = max(b for b in bounds if b <= start)
    hi = min(b for b in bounds if b >= end)
    sent = ctx[lo:hi].strip()
    # the window of 03b may cut a sentence: drop a partial first or last word
    if lo == 0 and ctx[:1].islower() and " " in sent and sent.index(" ") < start:
        sent = sent.split(" ", 1)[1]
    return " ".join(sent.split())


def prepare(m: dict, use_standard: bool, context_mode: str, excluded_mention: str,
            numeric_term: bool = True) -> dict:
    span, term = (m.get("text") or "").strip(), (m.get("llm_term") or "").strip()
    if m.get("term_only"):                     # a table finding: the standard term is all BeLink sees
        t = term or span
        return {"mention": t, "rerank_mention": t, "standard_name": None, "context": t, "term_only": True}
    found = m.get("context", "") != (m.get("text") or "")   # 03b gives the bare phrase when not found
    if m.get("status") == "excluded" and excluded_mention == "term" and term:
        mention = term
    else:
        mention = span or term
    if context_mode == "none" or not found:
        context = mention
    elif context_mode == "window":
        context = " ".join(m["context"].split())
    else:
        context = sentence_around(m["context"], int(m.get("start", 0)), int(m.get("end", 0))) or mention
    standard = term if use_standard and term else None          # kept even when equal to the mention
    # A quoted value ("creatinine: 454 mcmol/L") does not say which way it is abnormal; the
    # re-ranker sees only the mention, so it is shown with OntoGPT's term: "quote (term)".
    shown = mention
    if numeric_term and term and any(ch.isdigit() for ch in mention) and term.lower() != mention.lower():
        shown = f"{mention} ({term})"
    return {"mention": mention, "rerank_mention": shown, "standard_name": standard, "context": context,
            "term_only": False}


# --------------------------------------------------------------------------- #
# Models (loaded once)
# --------------------------------------------------------------------------- #
def _load():
    if _state:
        return _state
    from belink_hpo import CandidateRetriever, HPOReranker
    from belink_hpo.ontology import load_preferred_names

    index = _env("BELINK_INDEX", "belink_index")
    need = ["hpo_vec.npy", "ontology_mapping.txt", "preferred_names.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(index, f))]
    if missing:
        raise SystemExit(f"BELINK_INDEX={index}: missing {', '.join(missing)}. Build it with\n"
                         f"  belink-hpo prepare-obo --ontology_path hp.obo --output_dir {index}\n"
                         f"  belink-hpo embed-terms --ontology {index}/terminology.json.gz "
                         f"--out_vectors {index}/hpo_vec.npy")
    preferred = load_preferred_names(os.path.join(index, "preferred_names.json"))
    print(f"[belink] loading retriever from {index}", file=sys.stderr)
    _state["retriever"] = CandidateRetriever.from_files(os.path.join(index, "hpo_vec.npy"),
                                                        os.path.join(index, "ontology_mapping.txt"))
    adapter = _env("BELINK_ADAPTER", "Dash00/qwen3-4b-lora-hpo-reranker")
    print(f"[belink] loading re-ranker {adapter}", file=sys.stderr)
    _state["reranker"] = HPOReranker.from_pretrained(adapter, load_in_4bit=_env("BELINK_4BIT", "0") == "1",
                                                     preferred_names=preferred)
    _state["preferred"] = preferred
    return _state


def first_concepts(cands: list[dict], n: int = MAX_OPTIONS) -> list[dict]:
    out, seen = [], set()
    for c in cands:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out[:n]


# --------------------------------------------------------------------------- #
# relink.py entry point
# --------------------------------------------------------------------------- #
def link(mentions: list[dict]) -> list[dict | None]:
    if not mentions:
        return []
    st = _load()
    retriever, reranker, preferred = st["retriever"], st["reranker"], st["preferred"]
    top_k, alpha = int(_env("BELINK_TOP_K", "30")), float(_env("BELINK_ALPHA", "0.6"))
    use_standard = _env("BELINK_STANDARD_NAME", "1") == "1"
    context_mode = _env("BELINK_CONTEXT", "sentence")
    excluded_mention = _env("BELINK_EXCLUDED_MENTION", "term")
    add_ontogpt = _env("BELINK_ADD_ONTOGPT", "0") == "1"

    numeric_term = _env("BELINK_NUMERIC_TERM", "1") == "1"
    items = [prepare(m, use_standard, context_mode, excluded_mention, numeric_term) for m in mentions]

    # retrieval: one query per distinct (mention, standard name); term-only findings (tables) are
    # searched with the term's own vector, without mixing in a standard name
    queries = sorted({(it["mention"], it["standard_name"] or "") for it in items if not it["term_only"]})
    terms = sorted({it["mention"] for it in items if it["term_only"]})
    print(f"[belink] retrieving {top_k} names for {len(queries)} distinct queries"
          + (f" and {len(terms)} standard terms alone (tables)" if terms else ""), file=sys.stderr)
    by_query = {}
    if queries:
        ranked = retriever.retrieve([q[0] for q in queries], [q[1] or None for q in queries] if use_standard else None,
                                    top_k=top_k, rocchio=True, alpha=alpha)
        by_query.update({("q",) + q: r for q, r in zip(queries, ranked)})
    if terms:
        ranked = retriever.retrieve(terms, None, top_k=top_k, rocchio=False, alpha=1.0)
        by_query.update({("t", t): r for t, r in zip(terms, ranked)})

    batch, injected = [], []
    for m, it in zip(mentions, items):
        cands = by_query[("t", it["mention"])] if it["term_only"] else \
            by_query[("q", it["mention"], it["standard_name"] or "")]
        options = first_concepts(cands)
        added = False
        oid = (m.get("ontogpt_id") or "").upper()
        if add_ontogpt and oid.startswith("HP:") and oid not in {c["id"] for c in options}:
            name = (preferred.get(oid) or oid).lower()
            options = options[: MAX_OPTIONS - 1] + [{"id": oid, "name": name}]
            added = True
        injected.append(added)
        batch.append({"context": it["context"], "mention": it["rerank_mention"], "candidates": options,
                      "_retrieved": cands})

    print(f"[belink] re-ranking {len(batch)} mentions", file=sys.stderr)
    results = reranker.rerank_batch([{k: b[k] for k in ("context", "mention", "candidates")} for b in batch],
                                    batch_size=int(_env("BELINK_BATCH", "16")), progress=True)

    out, log = [], _log_path()
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        for m, it, b, r, added in zip(mentions, items, batch, results, injected):
            fh.write(json.dumps({
                "pmid": m.get("pmid"), "section_id": m.get("section_id"), "status": m.get("status"),
                "patient_id": m.get("patient_id"),
                "text": m.get("text"), "llm_term": m.get("llm_term"), "ontogpt_id": m.get("ontogpt_id", ""),
                "mention": it["mention"], "rerank_mention": it["rerank_mention"],
                "standard_name": it["standard_name"], "context": it["context"], "term_only": it["term_only"],
                "retrieved": [c["id"] for c in b["_retrieved"]],
                "options": [[letter, oid, name] for letter, oid, name in r.options]
                           + [[chr(ord("A") + len(r.options)), None, "None of the above."]],   # as in the prompt
                "ontogpt_added": added,
                "chosen": r.id, "chosen_name": r.name, "letter": r.letter, "confidence": round(r.confidence, 4),
            }, ensure_ascii=False) + "\n")
            out.append(None if r.id is None else
                       {"id": r.id, "score": r.confidence, "label": preferred.get(r.id, r.name)})
    print(f"[belink] {sum(o is not None for o in out)} of {len(out)} linked; "
          f"{sum(o is None for o in out)} answered 'None of the above'; candidates in {log}", file=sys.stderr)
    return out
