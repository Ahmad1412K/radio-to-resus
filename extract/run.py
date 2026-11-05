"""Inference harness for the SLM experiments (plan item P6).

Runs a model over a split and writes predictions in the format extract/score.py
reads. Resumable: cases already in the output file are skipped.

Backends
  ollama   local Ollama server (default http://localhost:11434), JSON-schema
           constrained via the "format" field
  openai   any OpenAI-compatible server (vLLM, llama.cpp server, LM Studio),
           constrained via response_format json_schema
  gold     no model; echoes the gold record (tests the plumbing)

Examples
  python -m extract.run --backend ollama --model qwen3:4b --split dev --limit 50
  python -m extract.run --backend openai --base-url http://gpu-box:8000/v1 \
        --model Qwen/Qwen3-4B --split dev --fewshot 2
  python -m extract.run --backend gold --split dev --limit 20

Few-shot examples come from the template TRAIN set by default, so no dev/test
case is ever shown to the model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import random
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical, decoding_schema, messages, system_prompt  # noqa: E402

csv.field_size_limit(10**8)


# ------------------------------------------------------------------ backends
class Ollama:
    def __init__(self, model, base_url, constrained, num_ctx, timeout, think=None, max_tokens=1500):
        self.model, self.url = model, base_url.rstrip("/") + "/api/chat"
        self.constrained, self.num_ctx, self.think, self.max_tokens = constrained, num_ctx, think, max_tokens
        self.client = httpx.Client(timeout=timeout)

    def __call__(self, msgs):
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx,
                            "num_predict": self.max_tokens}}   # a runaway stops here, scored invalid
        if self.constrained:
            body["format"] = decoding_schema()
        if self.think is not None:                       # reasoning models (Qwen3 etc.)
            body["think"] = self.think
        r = self.client.post(self.url, json=body)
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"], {"prompt_tokens": j.get("prompt_eval_count"),
                                         "output_tokens": j.get("eval_count")}


class OpenAICompat:
    def __init__(self, model, base_url, constrained, api_key, timeout):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self.constrained = constrained
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key or 'none'}"})

    def __call__(self, msgs):
        body = {"model": self.model, "messages": msgs, "temperature": 0, "seed": 0, "max_tokens": 1500}
        if self.constrained:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "handover", "schema": decoding_schema(), "strict": True}}
        r = self.client.post(self.url, json=body)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        return j["choices"][0]["message"]["content"], {"prompt_tokens": u.get("prompt_tokens"),
                                                       "output_tokens": u.get("completion_tokens")}


class GoldEcho:
    def __init__(self, gold):
        self.gold = gold

    def __call__(self, msgs):
        return self.gold[msgs[-1]["content"]], {}


# ------------------------------------------------------------------ few-shot
def pick_examples(k, source, seed=0):
    if k <= 0:
        return []
    path = ROOT / "examples" / ("synthetic_handovers_train_v0.csv" if source == "train" else source)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rng = random.Random(seed)
    # prefer varied examples: one with a correction, one with injuries, then random
    picks = []
    pools = [[r for r in rows if r["has_retraction"] == "True"],
             [r for r in rows if json.loads(r["said_record_json"]).get("injuries")],
             rows]
    for pool in pools:
        if len(picks) >= k:
            break
        picks.append(rng.choice(pool))
    while len(picks) < k:
        picks.append(rng.choice(rows))
    return [(r["text"], canonical(json.loads(r["said_record_json"]))) for r in picks]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai", "gold"], required=True)
    ap.add_argument("--model", default="gold")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--gold", default=str(ROOT / "examples" / "synthetic_handovers_v2.csv"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--fewshot-source", default="train")
    ap.add_argument("--unconstrained", action="store_true", help="no JSON-schema constraint")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--think", choices=["on", "off"], default=None,
                    help="Ollama only: toggle reasoning for thinking models (default: model's own default)")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="Ollama only: output-token cap (a full record is ~550-900 tokens)")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.split == "test":
        print("NOTE: running on the frozen test split. Only do this for the final run (E6).")
    rows = [r for r in csv.DictReader(open(a.gold, encoding="utf-8")) if r["split"] == a.split]
    if a.limit:
        rows = rows[:a.limit]
    examples = pick_examples(a.fewshot, a.fewshot_source)
    constrained = not a.unconstrained

    if a.backend == "ollama":
        model = Ollama(a.model, a.base_url or "http://localhost:11434", constrained, a.num_ctx, a.timeout,
                       think=None if a.think is None else a.think == "on", max_tokens=a.max_tokens)
    elif a.backend == "openai":
        model = OpenAICompat(a.model, a.base_url or "http://localhost:8000/v1", constrained, a.api_key, a.timeout)
    else:
        model = GoldEcho({r["text"]: r["said_record_json"] for r in rows})

    tag = f"{a.model.replace('/', '_').replace(':', '-')}_{a.split}_k{a.fewshot}{'' if constrained else '_free'}"
    out = pathlib.Path(a.out or ROOT / "runs" / f"{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["case_id"] for l in out.open(encoding="utf-8") if l.strip()}
    meta = {"model": a.model, "backend": a.backend, "split": a.split, "fewshot": a.fewshot,
            "constrained": constrained, "num_ctx": a.num_ctx, "think": a.think, "max_tokens": a.max_tokens, "gold": a.gold,
            "prompt_sha256": hashlib.sha256(system_prompt(examples).encode()).hexdigest(),
            "started": dt.datetime.now().isoformat(timespec="seconds")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    todo = [r for r in rows if r["case_id"] not in done]
    print(f"{len(rows)} cases, {len(done)} already done, {len(todo)} to run -> {out}")
    errors = 0
    with out.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                text, usage = model(messages(r["text"], examples))
                err = None
            except Exception as e:                       # keep going; scored as invalid
                text, usage, err = None, {}, f"{type(e).__name__}: {e}"[:300]
                errors += 1
            rec = {"case_id": r["case_id"], "output": text, "latency_s": round(time.perf_counter() - t0, 3), **usage}
            if err:
                rec["error"] = err
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  last {rec['latency_s']:.1f}s  errors {errors}")
    print(f"done. score with: python -m extract.score {out} --gold {a.gold} --split {a.split}")


if __name__ == "__main__":
    main()
