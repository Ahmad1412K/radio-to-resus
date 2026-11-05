"""Build the template training split for the SLM (plan item P1).

    python scripts/make_train_split.py            # 10,000 cases, seed 20261027
    python scripts/make_train_split.py --n 5000

Uses the same template generator as v0 (no LLM), with a different seed and
case IDs T00001... so they never clash with the H-numbered dev/test cases.

Leakage guard: any training row whose text, said record or true record is
identical to a test case (any version: v0, v1, v2) is dropped and replaced,
so the final file has exactly --n rows. Overlap with dev is only reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.generate import main as generate  # noqa: E402

EX = ROOT / "examples"
csv.field_size_limit(10**8)


def load_eval_keys():
    keys = {"test": {"text": set(), "said": set(), "true": set()},
            "dev": {"text": set(), "said": set(), "true": set()}}
    for name in ("v0", "v1", "v2"):
        for r in csv.DictReader((EX / f"synthetic_handovers_{name}.csv").open(encoding="utf-8")):
            k = keys[r["split"]]
            k["text"].add(r["text"].strip().lower())
            k["said"].add(r["said_record_json"])
            k["true"].add(r["true_record_json"])
    return keys


def clashes(r, k):
    return [f for f, col in (("text", "text"), ("said", "said_record_json"), ("true", "true_record_json"))
            if (r[col].strip().lower() if f == "text" else r[col]) in k[f]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=str(EX / "synthetic_handovers_train_v0.csv"))
    a = ap.parse_args()

    keys = load_eval_keys()
    tmp = pathlib.Path(a.out).with_suffix(".tmp.csv")
    kept, dropped, dev_overlap = [], Counter(), Counter()
    seed, extra = a.seed, 0
    while len(kept) < a.n:
        need = a.n - len(kept) + 50
        _, rows, _ = generate(n=need, seed=seed, out=tmp, split_mode="train", id_prefix="T")
        for r in rows:
            c = clashes(r, keys["test"])
            if c:
                dropped.update(c)
                continue
            dev_overlap.update(clashes(r, keys["dev"]))
            kept.append(r)
            if len(kept) == a.n:
                break
        seed += 1
        extra += 1
    tmp.unlink(missing_ok=True)

    seen = Counter(r["said_record_json"] for r in kept)
    for i, r in enumerate(kept, 1):
        r["case_id"] = f"T{i:05d}"
    out = pathlib.Path(a.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)

    lv = Counter(str(r["true_level"]) for r in kept)
    print(f"wrote {out.relative_to(ROOT)}: {len(kept)} rows (generator passes: {extra})")
    print("dropped for matching a test case:", dict(dropped) or "none")
    print("identical to a dev case (kept, reported only):", dict(dev_overlap) or "none")
    print("duplicate said records within train:", sum(v - 1 for v in seen.values() if v > 1))
    print("levels:", dict(sorted(lv.items())))
    print("retractions: %.1f%%  borderline: %.1f%%  out of scope: %d" % (
        100 * sum(r["has_retraction"] for r in kept) / len(kept),
        100 * sum(r["borderline"] for r in kept) / len(kept),
        sum(not r["in_scope"] for r in kept)))
    print("criteria targeted:", len({r["target_criterion"] for r in kept}))


if __name__ == "__main__":
    main()
