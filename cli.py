"""pheno-auto: patient-level phenotype extraction from sectioned case reports.

Pipeline (input: a folder written by 02_prepare_sections.py, with index.tsv and the section texts)

  pheno-auto run --sections DIR --out DIR      every step below, in order
  pheno-auto extract-text   ...                LLM: findings of the running text   (extract_findings.py)
  pheno-auto extract-tables ...                LLM: findings of the tables         (extract_table_findings.py)
  pheno-auto concat ...                        text + table predictions            (concat_predictions.py)
  pheno-auto relink ...                        BeLink grounding                    (relink.py + belink_linker.py)
  pheno-auto merge ...                         LLM: one name per patient           (merge_patients.py)
  pheno-auto review-zip ...                    a .zip for the review app           (make_review_zip.py)
  pheno-auto view                              open the review app

Setup
  pheno-auto prepare-hpo --out DIR             download hp.obo and build the BeLink index
  pheno-auto vllm-install --env DIR            create the vLLM environment
  pheno-auto vllm-serve --env DIR              start the LLM server (gpt-oss-120b)

`pheno-auto <command> -h` shows a command's options.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(HERE, "steps")
VIEWER = os.path.join(HERE, "viewer", "patient_phenotype_review.html")
LINKER = os.path.join(STEPS_DIR, "belink_linker.py")

STEPS = {                       # command -> script in steps/
    "extract-text": "extract_findings.py",
    "extract-tables": "extract_table_findings.py",
    "concat": "concat_predictions.py",
    "relink": "relink.py",
    "merge": "merge_patients.py",
    "review-zip": "make_review_zip.py",
}
RUN_STEPS = ["text", "tables", "concat", "relink", "merge", "review"]
HPO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/{}/hp.obo"
GPT_OSS = "openai/gpt-oss-120b"


# ---------------------------------------------------------------- one step

def run_step(command: str, args: list[str]) -> int:
    """Run one step script as `python <script> args` would, in this process."""
    import runpy

    script = os.path.join(STEPS_DIR, STEPS[command])
    if command == "relink":
        if "--linker" not in args:
            args = args + ["--linker", LINKER]
        else:
            i = args.index("--linker") + 1
            if i < len(args) and args[i] == "belink":
                args[i] = LINKER
    sys.argv = [f"pheno-auto {command}"] + list(args)
    sys.path.insert(0, STEPS_DIR)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            return e.code or 0
        print(e.code, file=sys.stderr)
        return 1
    return 0


def sub(command: str, args: list[str], env: dict | None = None) -> None:
    """Run a step in its own process (GPU memory is freed when BeLink is done), printing the command."""
    cmd = [sys.executable, "-m", "pheno_auto", command, *args]
    print("\n+ pheno-auto " + " ".join([command, *args]), flush=True)
    res = subprocess.run(cmd, env=env)
    if res.returncode:
        raise SystemExit(f"pheno-auto {command} failed (exit {res.returncode})")


# ---------------------------------------------------------------- the whole pipeline

def cmd_run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="pheno-auto run",
        description="Run the pipeline on a sections folder: extract-text, extract-tables, concat, relink, "
                    "merge, review-zip. A step whose output exists is resumed (sections already answered "
                    "are not asked again).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sections", required=True, help="folder from 02_prepare_sections.py (index.tsv + texts)")
    ap.add_argument("--out", required=True, help="run folder: text/, tables/, both/, review.zip")
    ap.add_argument("--hpo-dir", help="folder from `pheno-auto prepare-hpo` (hp.obo and index/)")
    ap.add_argument("--hpo", help="hp.obo (default: <hpo-dir>/hp.obo)")
    ap.add_argument("--belink-index", help="BeLink index (default: <hpo-dir>/index)")
    ap.add_argument("--pmids", help="file with one PMID per line: only these papers")
    ap.add_argument("--gold", help="gold TSV, added to the review zip (optional)")
    ap.add_argument("--model", default="gpt-oss-120b", help="model name as the server knows it")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--max-tokens", type=int, default=16384, help="answer cap for text sections and merging")
    ap.add_argument("--table-max-tokens", type=int, default=32768, help="answer cap for tables")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    ap.add_argument("--belink-gpu", help="CUDA_VISIBLE_DEVICES for BeLink, e.g. 1 when vLLM holds GPU 0")
    ap.add_argument("--belink-adapter", default="Dash00/qwen3-4b-lora-hpo-reranker")
    ap.add_argument("--linker", help="another linker module for relink (default: BeLink)")
    ap.add_argument("--steps", default=",".join(RUN_STEPS),
                    help="the steps to run, comma-separated, from: " + ",".join(RUN_STEPS))
    a = ap.parse_args(argv)

    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in RUN_STEPS]
    if bad:
        ap.error(f"unknown steps {bad}; choose from {RUN_STEPS}")
    if not os.path.exists(os.path.join(a.sections, "index.tsv")):
        ap.error(f"{a.sections} has no index.tsv: pass a folder written by 02_prepare_sections.py")
    hpo = a.hpo or (a.hpo_dir and os.path.join(a.hpo_dir, "hp.obo"))
    index = a.belink_index or (a.hpo_dir and os.path.join(a.hpo_dir, "index"))
    if {"text", "tables"} & set(steps) and not (hpo and os.path.exists(hpo)):
        ap.error("give --hpo-dir (from `pheno-auto prepare-hpo`) or --hpo hp.obo")
    if "relink" in steps and not a.linker and not (index and os.path.isdir(index)):
        ap.error("give --hpo-dir or --belink-index (the BeLink index, from `pheno-auto prepare-hpo`)")

    out = a.out
    both = os.path.join(out, "both")
    llm = ["--model", a.model, "--base-url", a.base_url, "--temperature", str(a.temperature)]
    if a.reasoning_effort:
        llm += ["--reasoning-effort", a.reasoning_effort]
    sel = ["--pmids", a.pmids] if a.pmids else []
    cands = os.path.join(both, "predictions_belink.candidates.jsonl")
    for step in steps:
        if step == "text":
            sub("extract-text", ["--sections", a.sections, "--hpo", hpo, "--out", os.path.join(out, "text"),
                                 *llm, "--workers", str(a.workers), "--max-tokens", str(a.max_tokens), *sel])
        elif step == "tables":
            sub("extract-tables", ["--sections", a.sections, "--hpo", hpo, "--out", os.path.join(out, "tables"),
                                   *llm, "--workers", str(a.workers), "--max-tokens", str(a.table_max_tokens), *sel])
        elif step == "concat":
            parts = [os.path.join(out, d, "predictions.tsv") for d in ("text", "tables")]
            parts = [p for p in parts if os.path.exists(p)]
            if not parts:
                raise SystemExit("concat: no text/ or tables/ predictions in " + out)
            sub("concat", [*parts, os.path.join(both, "predictions.tsv")])
        elif step == "relink":
            env = dict(os.environ, BELINK_LOG=cands, BELINK_ADAPTER=a.belink_adapter)
            if index:
                env["BELINK_INDEX"] = index
            if a.belink_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(a.belink_gpu)
            sub("relink", ["--pred", os.path.join(both, "predictions.tsv"), "--mode", "all",
                           "--term-mention-kinds", "table,suppl_table",
                           "--out", os.path.join(both, "predictions_belink.tsv"),
                           *(["--linker", a.linker] if a.linker else [])], env=env)
        elif step == "merge":
            sub("merge", ["--pred", os.path.join(both, "predictions_belink.tsv"),
                          "--out", os.path.join(both, "predictions_merged_llm.tsv"),
                          "--model", a.model, "--base-url", a.base_url, "--max-tokens", str(a.max_tokens),
                          "--temperature", str(a.temperature), "--workers", str(a.workers)])
        elif step == "review":
            sub("review-zip", ["--pred", os.path.join(both, "predictions_merged_llm.tsv"),
                               *(["--candidates", cands] if os.path.exists(cands) else []),
                               *(["--gold", a.gold] if a.gold else []),
                               "--sections", a.sections, "--out", os.path.join(out, "review.zip")])
    print(f"\nDone: {', '.join(steps)} -> {out}")
    return 0


# ---------------------------------------------------------------- HPO and the BeLink index

def cmd_prepare_hpo(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="pheno-auto prepare-hpo",
        description="Put hp.obo in --out and build BeLink's retrieval index from it in <out>/index. "
                    "The extraction's lookup and BeLink must use the same hp.obo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, help="folder for hp.obo and index/")
    ap.add_argument("--release", default="latest",
                    help="HPO release tag, e.g. v2025-10-22 (see github.com/obophenotype/human-phenotype-ontology/releases)")
    ap.add_argument("--obo", help="use this hp.obo instead of downloading one")
    ap.add_argument("--no-index", action="store_true", help="only hp.obo, no BeLink index")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    obo = os.path.join(a.out, "hp.obo")
    if a.obo:
        if os.path.abspath(a.obo) != os.path.abspath(obo):
            shutil.copyfile(a.obo, obo)
    elif not os.path.exists(obo):
        url = HPO_URL.format("latest/download" if a.release == "latest" else f"download/{a.release}")
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, obo)
    with open(obo, encoding="utf-8") as fh:
        version = next((ln.strip() for ln in fh if ln.startswith("data-version")), "data-version: ?")
    print(f"{obo}: {version}")
    if a.no_index:
        return 0
    belink = shutil.which("belink-hpo") or os.path.join(os.path.dirname(sys.executable), "belink-hpo")
    index = os.path.join(a.out, "index")
    for cmd in ([belink, "prepare-obo", "--ontology_path", obo, "--output_dir", index],
                [belink, "embed-terms", "--ontology", os.path.join(index, "terminology.json.gz"),
                 "--out_vectors", os.path.join(index, "hpo_vec.npy")]):
        print("+ " + " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            raise SystemExit("building the BeLink index failed")
    need = ["hpo_vec.npy", "ontology_mapping.txt", "preferred_names.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(index, f))]
    if missing:
        raise SystemExit(f"{index} lacks {', '.join(missing)}")
    with open(os.path.join(a.out, "HPO_VERSION"), "w", encoding="utf-8") as fh:
        fh.write(version + "\n")
    print(f"Ready: {obo} and {index}  (pass --hpo-dir {a.out} to `pheno-auto run`)")
    return 0


# ---------------------------------------------------------------- vLLM

def cmd_vllm_install(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="pheno-auto vllm-install",
        description="Create a separate virtual environment with vLLM (vLLM pins its own torch/CUDA, so it is "
                    "kept apart from this package's environment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--env", default=os.path.expanduser("~/envs/vllm"), help="folder of the new environment")
    ap.add_argument("--python", default="python3", help="Python to create it with (vLLM supports 3.10-3.12)")
    ap.add_argument("--vllm", default="vllm>=0.10.1", help="pip requirement for vLLM (gpt-oss needs >= 0.10.1)")
    a = ap.parse_args(argv)
    py = os.path.join(a.env, "bin", "python")
    steps = []
    if not os.path.exists(py):
        steps.append([a.python, "-m", "venv", a.env])
    steps += [[py, "-m", "pip", "install", "--upgrade", "pip"], [py, "-m", "pip", "install", a.vllm]]
    for cmd in steps:
        print("+ " + " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            raise SystemExit("vLLM installation failed")
    subprocess.run([py, "-c", "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, "
                              "'| CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count())"])
    print(f"Ready: start the server with  pheno-auto vllm-serve --env {a.env}")
    return 0


def cmd_vllm_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="pheno-auto vllm-serve",
        description="Start vLLM with gpt-oss-120b on an OpenAI-compatible endpoint (runs in the foreground: "
                    "use tmux or a second terminal). Extra vLLM options go after --.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--env", default=os.path.expanduser("~/envs/vllm"), help="the vLLM environment")
    ap.add_argument("--model", default=GPT_OSS, help="model on the Hugging Face Hub, or a local folder")
    ap.add_argument("--served-name", default="gpt-oss-120b", help="name the pipeline asks for (--model of run)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-model-len", type=int, default=65536, help="context: prompt + answer")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--answer-cap", type=int,
                    help="cap every answer at this many tokens, whatever the client asks (run50 used 16384 "
                         "with --max-model-len 32768)")
    a, extra = ap.parse_known_args(argv)
    extra = [x for x in extra if x != "--"]
    vllm = os.path.join(a.env, "bin", "vllm")
    if not os.path.exists(vllm):
        raise SystemExit(f"no vLLM in {a.env}: run  pheno-auto vllm-install --env {a.env}")
    cmd = [vllm, "serve", a.model, "--served-model-name", a.served_name, "--max-model-len", str(a.max_model_len),
           "--gpu-memory-utilization", str(a.gpu_memory_utilization), "--port", str(a.port)]
    if a.answer_cap:
        cmd += ["--override-generation-config", '{"max_new_tokens": %d}' % a.answer_cap]
    cmd += extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    print(f"+ CUDA_VISIBLE_DEVICES={a.gpu} " + " ".join(cmd), flush=True)
    os.execve(vllm, cmd, env)
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help", "list"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("--version", "version"):
        from . import __version__
        print(__version__)
        return 0
    if cmd in STEPS:
        return run_step(cmd, rest)
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "prepare-hpo":
        return cmd_prepare_hpo(rest)
    if cmd == "vllm-install":
        return cmd_vllm_install(rest)
    if cmd == "vllm-serve":
        return cmd_vllm_serve(rest)
    if cmd == "view":
        print(VIEWER)
        print("Open it in a browser and drop the run's review.zip on the page.")
        if "--no-browser" not in rest:
            webbrowser.open("file://" + VIEWER)
        return 0
    print(f"unknown command {cmd!r}; `pheno-auto help` lists them", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
