# pheno-auto-pipeline

Patient-level phenotype extraction from case reports. Starting from a paper split into sections
and tables, it produces one list of HPO terms per patient, each with the sentence or table cell
it came from, and a review app to inspect the result.

```
articles (JATS XML + supplementary files)
  -> prepare-sections one text file per section and table, with index.tsv
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

**Check the GPU driver first.** `nvidia-smi` shows, top right, the highest CUDA version the driver
supports. PyTorch must be built for that version or lower; a plain `pip install torch` takes the
newest build (CUDA 13), and on an older driver BeLink then silently runs on the CPU.

```bash
nvidia-smi                                  # e.g. "CUDA Version: 12.4"

python3 -m venv ~/envs/pheno-auto
source ~/envs/pheno-auto/bin/activate
pip install --upgrade pip

# 1. torch 2.8.0 (as in run50), built for the driver: cu128 if it shows 12.8 or newer, else cu126
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, 'GPU usable:', torch.cuda.is_available())"
#    GPU usable: False -> pip install --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 2. the versions BeLink used in run50, then the pipeline
pip install -r https://raw.githubusercontent.com/Dash00/pheno-auto-pipeline/main/requirements/belink-run50.txt
pip install --no-deps "git+https://github.com/Dash00/pheno-auto-pipeline.git"
pheno-auto help

# the vLLM environment (skip if you have one; creates ~/envs/vllm)
pheno-auto vllm-install --env ~/envs/vllm
```

`requirements/belink-run50.txt` pins what the BeLink step loads, as in the environment of run50:
belink-hpo at commit `af084c7`, torch 2.8.0, transformers 5.17.0, peft 0.21.2, accelerate 1.15.0
and their companions. For a private repository, put the file on the server and use `-r <file>`,
and install the package with your token or SSH key. For vLLM, pin a version with `--vllm "vllm==..."`.

## 2. HPO and the BeLink index (once)

```bash
pheno-auto prepare-hpo --out ~/hpo                        # latest HPO release
pheno-auto prepare-hpo --out ~/hpo --obo /path/to/hp.obo  # or a given hp.obo (e.g. the one of earlier runs)
```

This writes `~/hpo/hp.obo` and the BeLink retrieval index `~/hpo/index/`. The extraction's exact
lookup and BeLink both use this hp.obo. The SapBERT encoder and the re-ranker
(`Dash00/qwen3-4b-lora-hpo-reranker`, on Qwen3-4B) are downloaded from Hugging Face on first use.

## 3. The sections folder

The pipeline reads a sections folder: `index.tsv` (pmid, section_id, kind, part, title, n_chars,
path) and one text file per section, table or figure legend. If you already have one (e.g.
`sections50`), upload it and skip this step. To make one from downloaded articles:

```bash
pip install openpyxl xlrd pdfplumber          # supplementary .xlsx/.xls/.pdf (or: pip install "...[sections]")
pheno-auto prepare-sections --articles ~/literature/articles --out ~/data/sections_new \
    --supplementary --max-supp-chunks 350 --max-chars 20000
cp ~/data/pmids.txt ~/data/sections_new/      # optional: the papers to run
```

**Settings of `sections50`** (the input of run50): it was made with

```bash
python 02_prepare_sections.py --articles phenopacket-auto/literature/articles/ \
    --out phenopacket-auto/work/sections_v2 --supplementary --max-supp-chunks 350 --max-chars 20000
```

and the 50 papers of run50 were then taken from `sections_v2`. Two options differ from the
defaults: `--max-chars 20000` (default 12000; sections and tables cut into pieces of at most 20,000
characters) and
`--max-supp-chunks 350` (default 200). Use the same options to get the same section files, and so
the same text per model call. The cap is confirmed by `sections50` itself: 13 units are over
12,000 characters (the longest 19,885), and the supplementary texts of PMID 19776401 and 22772371
were cut into parts although they total 22,420 and 29,182 characters, which only a 20,000 cap does. The script is the same as `prepare_sections.py` here (only its usage
lines differ). `sections50` was made on Windows, with pdfplumber 0.11.10, openpyxl 3.1.5 and xlrd
2.0.2; to read supplementary files the same way, install those versions:

```bash
pip install pdfplumber==0.11.10 openpyxl==3.1.5 xlrd==2.0.2
```

Made on Windows, its files end their lines with CRLF (`\r\n`); the same command on Linux writes
LF (`\n`). The text is otherwise the same, but character offsets differ (a line break counts as 2
characters), and the model sees `\r\n` where it would see `\n`. To compare a rebuilt folder with
`sections50`, ignore that difference: `diff -rq --strip-trailing-cr`. To rerun run50 exactly, use
the `sections50` files themselves.

`--articles` is a folder of `PMID_<pmid>/` folders, each with `fulltext/*.xml` (the JATS XML from
PMC) and, for `--supplementary`, `supplementary/` with the supplementary files (.xlsx .xls .csv
.tsv .txt .docx .pdf .html, also inside .zip; .doc/.rtf need LibreOffice's `soffice` on PATH).
This is the layout `fetch_phenopacket_literature.py` writes. The run ends with counts per section
kind and the supplementary files read and not read; `skipped.tsv` lists the sections left out
(acknowledgements, funding, ...). `pheno-auto prepare-sections -h` shows the options
(`--max-chars`, `--max-table-chars`, `--per-paragraph`, ...).

## 4. Run

Input: a sections folder (step 3), e.g. `~/data/sections50`.

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
| `prepare-sections` | prepare_sections.py | articles (XML + supplementary files) | `index.tsv`, one text per section/table |
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
