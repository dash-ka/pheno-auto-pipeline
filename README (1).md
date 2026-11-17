# pheno-auto-pipeline

Patient-level phenotype extraction from case reports. Starting from a paper split into sections
and tables, it produces one list of HPO terms per patient, each with the sentence or table cell
it came from, and a review app to inspect the result.

```
sections folder (from 02_prepare_sections.py)
  -> extract-text     LLM, one call per text section: patients, findings, evidence sentence, present/absent
  -> extract-tables   LLM, one call per table: patient labels, legend, findings with their cell
  -> concat           text + table findings in one table
  -> relink           BeLink grounding: SapBERT retrieval + Qwen3-4B LoRA re-ranker
  -> merge            LLM, one call per paper: which patient names are the same person
  -> review-zip       a .zip for the review app (pheno-auto view)
```

The LLM steps call an OpenAI-compatible server: vLLM serving `gpt-oss-120b`, in its own environment.

## 1. Install

Two environments: the pipeline (with BeLink) and vLLM (which pins its own torch and CUDA).

```bash
# the pipeline
python3 -m venv ~/envs/pheno-auto
source ~/envs/pheno-auto/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/Dash00/pheno-auto-pipeline.git"
pheno-auto help

# the vLLM environment (from the pipeline environment; creates ~/envs/vllm)
pheno-auto vllm-install --env ~/envs/vllm
```

If the server's NVIDIA driver is older than the CUDA of the default torch wheel, install a
matching torch first, then the pipeline, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`
(`nvidia-smi` shows the highest CUDA version the driver supports). For vLLM, pin a version with
`--vllm "vllm==0.11.0"`.

## 2. HPO and the BeLink index (once)

```bash
pheno-auto prepare-hpo --out ~/hpo                        # latest HPO release
pheno-auto prepare-hpo --out ~/hpo --obo /path/to/hp.obo  # or a given hp.obo (e.g. the one of earlier runs)
```

This writes `~/hpo/hp.obo` and the BeLink retrieval index `~/hpo/index/`. The extraction's exact
lookup and BeLink both use this hp.obo. The SapBERT encoder and the re-ranker
(`Dash00/qwen3-4b-lora-hpo-reranker`, on Qwen3-4B) are downloaded from Hugging Face on first use.

## 3. Run

Input: a folder written by `02_prepare_sections.py`, with `index.tsv` and the section texts
(`PMID_<pmid>/<section>.txt`). Upload it to the server, e.g. to `~/data/sections50`.

```bash
# terminal 1 (tmux): the LLM server
pheno-auto vllm-serve --env ~/envs/vllm --gpu 0

# terminal 2: the pipeline
source ~/envs/pheno-auto/bin/activate
pheno-auto run --sections ~/data/sections50 --hpo-dir ~/hpo --out ~/runs/run1 \
    --pmids ~/data/sections50/pmids50.txt --gold ~/data/sections50/gold50.tsv --belink-gpu 1
```

- `--pmids` (optional) limits the run to these papers; `--gold` (optional) adds the gold standard
  to the review zip.
- `--belink-gpu 1` puts BeLink on a second GPU while vLLM holds GPU 0. **With a single GPU**, run
  the steps in two parts and stop vLLM in between:
  ```bash
  pheno-auto run ... --steps text,tables,concat
  # stop vLLM (Ctrl-C in terminal 1)
  pheno-auto run ... --steps relink
  # start vLLM again
  pheno-auto run ... --steps merge,review
  ```
- **Resuming.** Every model answer is saved (one JSON per section or table, `.llm.json` for
  merging), and a rerun asks only for what is missing. After an interruption, rerun the same command.
  To ask again, delete the JSON files concerned or use a new `--out`.
- `pheno-auto run -h` lists all options (`--model`, `--base-url`, `--workers`, `--temperature`,
  `--max-tokens`, `--table-max-tokens`, `--reasoning-effort`, ...).

### Output

```
<out>/text/PMID_<pmid>/<section>.json   the model's answer per text section
<out>/text/predictions.tsv              one row per finding, with evidence and checks
<out>/tables/...                        the same for tables
<out>/both/predictions.tsv              text + tables
<out>/both/predictions_belink.tsv       + BeLink: linker_id, linker_label, linker_score
<out>/both/predictions_belink.candidates.jsonl   BeLink's options and choice per mention
<out>/both/predictions_merged_llm.tsv   final: one name per patient (original_patient_id kept)
<out>/both/predictions_merged_llm.llm.json       the merging answers
<out>/review.zip                        for the review app
```

In `predictions_merged_llm.tsv`, `ontogpt_id` is the exact HPO lookup's id and `linker_id` with
`linker_score` is BeLink's; `hpo_id` holds BeLink's id for every finding.

### Review

```bash
pheno-auto view          # opens the app (one HTML file); drop review.zip on the page
```
On a remote server, copy `review.zip` and the HTML (its path is printed) to your computer.

## The steps one by one

Each step is a script in `src/pheno_auto/steps/` and keeps its own options
(`pheno-auto <step> -h`):

| Command | Script | Reads | Writes |
|---|---|---|---|
| `extract-text` | extract_findings.py | sections, hp.obo, LLM | `PMID_*/<section>.json`, `predictions.tsv` |
| `extract-tables` | extract_table_findings.py | table sections, hp.obo, LLM | `PMID_*/<table>.json`, `predictions.tsv` |
| `concat` | concat_predictions.py | prediction tables | one predictions table |
| `relink` | relink.py + belink_linker.py | predictions, `BELINK_INDEX` | `+ linker_id, linker_score`, candidates (`BELINK_LOG`) |
| `merge` | merge_patients.py | predictions, LLM | renamed patients, `.llm.json` |
| `review-zip` | make_review_zip.py | predictions, sections, gold (optional) | `review.zip` |

`relink` uses BeLink unless `--linker` names another module defining `link(mentions)`. BeLink's
settings are environment variables (`BELINK_INDEX`, `BELINK_ADAPTER`, `BELINK_TOP_K`,
`BELINK_ALPHA`, ...; see `belink_linker.py`).

## vLLM settings

`pheno-auto vllm-serve` defaults to a 64k context with no cap on the answer. The earlier run50
used a 32k context with every answer capped at 16,384 tokens; to reproduce it:

```bash
pheno-auto vllm-serve --env ~/envs/vllm --max-model-len 32768 --answer-cap 16384
```

Other vLLM options go after `--`, e.g. `pheno-auto vllm-serve --gpu 0,1 -- --tensor-parallel-size 2`.

## Tests

```bash
pip install --no-deps -e ".[test]" && pip install pytest && pytest -q
```

The smoke test runs `pheno-auto run` on a synthetic two-patient paper with a stub LLM server and a
stub linker (no GPU, no BeLink download).

## Citation

BeLink: Shlyk, D., Montanelli, S., & Hunter, L. (2026). BeLink: Biomedical Entity Linking Meets
Generative Re-Ranking. SIGIR '26.
