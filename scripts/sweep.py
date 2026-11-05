"""Run a list of Ollama models over dev and collect one summary table (E2 zero-shot).

    python scripts/sweep.py                      # default model list: 50-case smoke, then 200 dev cases
    python scripts/sweep.py --limit 0 --models qwen3:4b   # full dev for a finalist
    python scripts/sweep.py --models qwen3:1.7b llama3.2:3b --limit 100
    python scripts/sweep.py --fewshot 2          # same sweep with 2 train examples (E3)

For each model: `ollama pull`, run extract/run.py (resumable), score with
extract/score.py --subset, append a row to runs/sweep_<split>_k<k>.csv.
A model is skipped after the smoke run if fewer than 80% of outputs are usable
(valid, or valid after extract/postprocess.repair) or the mean time per case is
over 120 s; the reason goes in the table. Output is capped at 1,500 tokens
(extract/run.py --max-tokens), so a runaway generation ends quickly and counts as
one invalid output instead of hitting the 600 s request timeout.

Screening uses the first 200 dev cases for every model (dev order is already
shuffled by the generator, so this is a random sample); finalists then get the
full dev split. Results for one model and split are resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.score import score  # noqa: E402

OLLAMA = shutil.which("ollama") or str(pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")

# Small open-weight, non-Claude models that fit a 4 GB GPU at Ollama's default 4-bit.
DEFAULT = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen2.5:1.5b", "qwen2.5:3b",
           "llama3.2:1b", "llama3.2:3b", "gemma3:1b", "gemma3:4b", "phi4-mini",
           "smollm2:1.7b", "granite3.3:2b"]
COLS = ["model", "think", "n", "valid_rate", "usable_rate", "field_f1", "hallucination_rate", "exact_match",
        "correction_accuracy", "triage_accuracy", "under_triage_rate", "silent_under_triage_rate",
        "over_triage_rate", "latency_mean_s", "note"]


def tag(model, split, k):
    return f"{model.replace('/', '_').replace(':', '-')}_{split}_k{k}"


def run(model, split, k, limit, think):
    cmd = [sys.executable, "-m", "extract.run", "--backend", "ollama", "--model", model,
           "--split", split, "--fewshot", str(k)]
    if limit:
        cmd += ["--limit", str(limit)]
    if think:
        cmd += ["--think", think]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ROOT / "runs" / f"{tag(model, split, k)}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=50)
    ap.add_argument("--limit", type=int, default=200, help="cases per model after the smoke run (0 = all dev)")
    a = ap.parse_args()
    if a.split == "test":
        raise SystemExit("sweeps are for dev; the test split is run once, by hand, in E6")

    out = ROOT / "runs" / f"sweep_{a.split}_k{a.fewshot}{'' if a.limit == 200 else f'_n{a.limit or "all"}'}.csv"
    done = {r["model"] for r in csv.DictReader(out.open(encoding="utf-8"))} if out.exists() else set()
    for model in a.models:
        if model in done:
            print(f"== {model}: already in {out.name}, skipping")
            continue
        print(f"== {model}")
        think = "off" if model.startswith("qwen3") else None
        row = {"model": model, "think": think or "", "note": ""}
        try:
            subprocess.run([OLLAMA, "pull", model], check=True)
            pred = run(model, a.split, a.fewshot, a.smoke, think)
            rep, _ = score(pred, ROOT / "examples" / "synthetic_handovers_v2.csv", a.split, subset=True)
            lat = (rep["latency_s"] or {}).get("mean", 0)
            if rep["overall"]["usable_rate"] < 0.8 or lat > 120:
                row["note"] = f"stopped after smoke: usable {rep['overall']['usable_rate']:.2f}, mean latency {lat:.0f}s"
            else:
                pred = run(model, a.split, a.fewshot, a.limit or None, think)
                rep, _ = score(pred, ROOT / "examples" / "synthetic_handovers_v2.csv", a.split, subset=True)
            (pred.with_suffix(".report.json")).write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
            o = rep["overall"]
            row.update({k: (round(o[k], 4) if isinstance(o.get(k), float) else o.get(k)) for k in COLS if k in o})
            row["latency_mean_s"] = round((rep["latency_s"] or {}).get("mean", 0), 2)
        except Exception as e:                           # keep the sweep going
            row["note"] = f"failed: {type(e).__name__}: {e}"[:200]
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"   -> {row}")
    print(f"summary: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
