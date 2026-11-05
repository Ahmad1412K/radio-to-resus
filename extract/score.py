"""Score extractor output against the gold "said" records (plan item P4).

    python -m extract.score PRED.jsonl [--gold examples/synthetic_handovers_v2.csv]
                            [--split dev] [--out report.json] [--cases cases.csv]

Predictions: one JSON object per line,
    {"case_id": "H00001", "output": "<raw model text>"}   (+ optional "latency_s")

The raw text is parsed here, so formatting failures count against the model.

Metrics
-------
validity      parse rate; schema-valid rate (Pydantic, including cross-field checks)
fields        every leaf of the record, per case:
                correct      gold said, prediction equal
                wrong        gold said, prediction a different value
                missed       gold said, prediction null
                hallucinated gold null ("not said"), prediction filled in
                ok_null      both null
              precision = correct / (correct + wrong + hallucinated)
              recall    = correct / (correct + wrong + missed)
              hallucination rate = hallucinated / (hallucinated + ok_null)
              List fields (injuries, allergies, medications, background,
              shock_signs, interventions, retractions) are compared as
              multisets of items; null vs [] is checked first. For the three
              lists that cannot be null (shock_signs, interventions,
              retractions), [] is treated as "not said".
corrections   cases with a planted correction: corrected value right AND the
              retraction logged
exact match   whole record identical
triage        three-valued engine (triage/engine.py) run on the predicted record
              and on the gold said record, so only extraction errors are measured.
              Under-triage = predicted level less urgent than gold.
              Silent under-triage = under-triage the engine did NOT flag as
              provisional (no open question that could raise the level). This
              is the headline safety metric.
breakdowns    style (v2 only), length band, borderline, correction, scope
CIs           bootstrap 95% intervals over cases (1,000 resamples)

Invalid outputs are scored as an all-null record (they miss everything but
hallucinate nothing); metrics on valid outputs only are reported as well.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.postprocess import repair  # noqa: E402
from extract.prompt_schema import canonical, decoding_schema  # noqa: E402
from triage.engine import URGENCY, triage as engine_triage  # noqa: E402

csv.field_size_limit(10**8)

LIST_FIELDS = {"injuries", "allergies", "medications", "background",
               "signs.shock_signs", "treatment.interventions", "retractions"}
# These lists cannot be null in the schema, so [] just means "nothing said":
# an empty prediction must not get credit for matching an empty gold list.
NON_NULLABLE_LISTS = {"signs.shock_signs", "treatment.interventions", "retractions"}
EMPTY = canonical({})


# ------------------------------------------------------------------ parsing
def parse_output(raw: str, fixes: list | None = None):
    """Return (record or None, status) with status in valid/repaired/invalid_json/invalid_schema.

    "repaired": the JSON broke a cross-field rule of the schema and was made valid by
    extract/postprocess.repair (which only nulls or drops contradicting details).
    The names of the fixes are appended to `fixes` if a list is given."""
    if raw is None:
        return None, "invalid_json"
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        return None, "invalid_json"
    try:
        obj = json.loads(s[a:b + 1])
    except json.JSONDecodeError:
        return None, "invalid_json"
    obj.pop("schema_version", None)
    try:
        return canonical(obj), "valid"
    except Exception:
        pass
    if not isinstance(obj, dict):
        return None, "invalid_schema"
    rec, applied = repair(obj)
    if fixes is not None:
        fixes.extend(applied)
    return (canonical(rec), "repaired") if rec is not None else (None, "invalid_schema")


# ------------------------------------------------------------------ flattening
def _leaf_paths(node, prefix=""):
    for name, sub in node.get("properties", {}).items():
        p = f"{prefix}.{name}" if prefix else name
        if p in LIST_FIELDS:
            yield p
            continue
        obj = next((alt for alt in [sub] + sub.get("anyOf", []) if alt.get("type") == "object"), None)
        if obj is not None:
            yield from _leaf_paths(obj, p)
        else:
            yield p


LEAVES = list(_leaf_paths(decoding_schema()))


def get(rec, path):
    cur = rec
    for k in path.split("."):
        if cur is None:
            return None
        cur = cur.get(k)
    return cur


def norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().lower()
        try:
            return float(t)
        except ValueError:
            return re.sub(r"\s+", " ", t)
    if isinstance(v, dict):
        return tuple(sorted((k, norm(x)) for k, x in v.items()))
    return v


def compare_case(gold, pred):
    """Per-field outcomes for one case: {path: outcome}, plus list item counts."""
    out, items = {}, defaultdict(Counter)
    for p in LEAVES:
        g, q = get(gold, p), get(pred, p)
        if p in NON_NULLABLE_LISTS:
            g, q = g or None, q or None
        if p in LIST_FIELDS:
            if g is None and q is None:
                out[p] = "ok_null"
                continue
            if g is None:
                out[p] = "hallucinated"
                items[p]["fp"] += len(q)
                continue
            if q is None:
                out[p] = "missed"
                items[p]["fn"] += len(g)
                continue
            gc, qc = Counter(map(norm, g)), Counter(map(norm, q))
            tp = sum((gc & qc).values())
            items[p]["tp"] += tp
            items[p]["fp"] += sum(qc.values()) - tp
            items[p]["fn"] += sum(gc.values()) - tp
            out[p] = "correct" if gc == qc else "wrong"
        else:
            if g is None and q is None:
                out[p] = "ok_null"
            elif g is None:
                out[p] = "hallucinated"
            elif q is None:
                out[p] = "missed"
            else:
                out[p] = "correct" if norm(g) == norm(q) else "wrong"
    return out, items


# ------------------------------------------------------------------ triage
def level_of(rec):
    """Three-valued engine decision (level, possible_level, questions)."""
    return engine_triage(rec)


def triage_outcome(g, p):
    if g == p:
        return "correct"
    if "out_of_scope" in (g, p):
        return "scope_error"
    return "under" if URGENCY[p] < URGENCY[g] else "over"


# ------------------------------------------------------------------ corrections
def correction_ok(gold, pred):
    if not gold["retractions"]:
        return None
    pr = Counter(norm(r) for r in pred["retractions"])
    for r in gold["retractions"]:
        if pr[norm(r)] < 1 or norm(get(pred, r["field"])) != norm(get(gold, r["field"])):
            return False
    return True


# ------------------------------------------------------------------ aggregation
def summarise(cases):
    c = Counter()
    for k in cases:
        c.update(k["fields"].values())
    prec_den = c["correct"] + c["wrong"] + c["hallucinated"]
    rec_den = c["correct"] + c["wrong"] + c["missed"]
    P = c["correct"] / prec_den if prec_den else math.nan
    R = c["correct"] / rec_den if rec_den else math.nan
    tri = Counter(k["triage"] for k in cases)
    corr = [k["correction_ok"] for k in cases if k["correction_ok"] is not None]
    n = len(cases)
    return {
        "n": n,
        "valid_rate": sum(k["status"] == "valid" for k in cases) / n if n else math.nan,
        "usable_rate": sum(k["status"] in ("valid", "repaired") for k in cases) / n if n else math.nan,
        "field_precision": P,
        "field_recall": R,
        "field_f1": 0.0 if c["correct"] == 0 and rec_den else (2 * P * R / (P + R) if P + R else math.nan),
        "hallucination_rate": c["hallucinated"] / (c["hallucinated"] + c["ok_null"]) if c["hallucinated"] + c["ok_null"] else math.nan,
        "exact_match": sum(k["exact"] for k in cases) / n if n else math.nan,
        "correction_accuracy": sum(corr) / len(corr) if corr else math.nan,
        "n_corrections": len(corr),
        "triage_accuracy": tri["correct"] / n if n else math.nan,
        "under_triage_rate": tri["under"] / n if n else math.nan,
        "over_triage_rate": tri["over"] / n if n else math.nan,
        "scope_error_rate": tri["scope_error"] / n if n else math.nan,
        "silent_under_triage_rate": sum(k["silent_under"] for k in cases) / n if n else math.nan,
        "provisional_rate": sum(k["provisional"] for k in cases) / n if n else math.nan,
        "outcome_counts": dict(c),
    }


HEADLINE = ["valid_rate", "usable_rate", "field_f1", "hallucination_rate", "exact_match",
            "correction_accuracy", "triage_accuracy", "under_triage_rate", "silent_under_triage_rate"]


def bootstrap(cases, reps=1000, seed=0):
    rng = random.Random(seed)
    vals = defaultdict(list)
    for _ in range(reps):
        s = summarise([cases[rng.randrange(len(cases))] for _ in cases])
        for k in HEADLINE:
            if not math.isnan(s[k]):
                vals[k].append(s[k])
    ci = {}
    for k, v in vals.items():
        v.sort()
        ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    return ci


def per_field(cases):
    out = {}
    for p in LEAVES:
        c = Counter(k["fields"][p] for k in cases)
        pd_ = c["correct"] + c["wrong"] + c["hallucinated"]
        rd = c["correct"] + c["wrong"] + c["missed"]
        out[p] = {**c,
                  "precision": c["correct"] / pd_ if pd_ else None,
                  "recall": c["correct"] / rd if rd else None,
                  "hallucination_rate": c["hallucinated"] / (c["hallucinated"] + c["ok_null"])
                  if c["hallucinated"] + c["ok_null"] else None}
    return out


def breakdowns(cases):
    groups = defaultdict(list)
    for k in cases:
        for g in k["groups"]:
            groups[g].append(k)
    return {g: {m: summarise(v)[m] for m in ("n",) + tuple(HEADLINE)} for g, v in sorted(groups.items())}


def groups_of(row):
    g = [f"length:{row.get('length_band') or 'n/a'}",
         f"borderline:{row['borderline']}",
         f"correction:{row['has_retraction']}",
         f"in_scope:{row['in_scope']}"]
    if row.get("llm_style"):
        try:
            st = json.loads(row["llm_style"])
            g += [f"pace:{st.get('pace')}", f"fillers:{st.get('fillers')}",
                  f"structure:{st.get('structure')}", f"exchange:{st.get('exchange')}"]
        except json.JSONDecodeError:
            pass
    return g


# ------------------------------------------------------------------ main
def score(pred_path, gold_path, split, subset=False):
    """subset=True scores only the cases present in the predictions (quick runs)."""
    gold_rows = {r["case_id"]: r for r in csv.DictReader(open(gold_path, encoding="utf-8"))
                 if split in (None, "all") or r["split"] == split}
    preds = {}
    for line in open(pred_path, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            preds[o["case_id"]] = o
    if subset:
        gold_rows = {k: v for k, v in gold_rows.items() if k in preds}
    missing = [c for c in gold_rows if c not in preds]
    cases = []
    for cid, row in gold_rows.items():
        gold = canonical(json.loads(row["said_record_json"]))
        o = preds.get(cid, {})
        fixes = []
        pred, status = parse_output(o.get("output"), fixes)
        if cid not in preds:
            status = "missing"
        scored = pred if pred is not None else EMPTY
        fields, items = compare_case(gold, scored)
        g_dec, p_dec = level_of(gold), level_of(scored)
        g_lvl, p_lvl = g_dec.level, p_dec.level
        tri = triage_outcome(g_lvl, p_lvl)
        cases.append({
            "case_id": cid, "status": status, "fixes": fixes, "fields": fields, "items": items,
            "exact": scored == gold, "correction_ok": correction_ok(gold, scored),
            "gold_level": g_lvl, "pred_level": p_lvl, "triage": tri,
            "silent_under": tri == "under" and not p_dec.provisional,
            "provisional": p_dec.provisional, "n_questions": len(p_dec.ask),
            "true_level": row["true_level"], "groups": groups_of(row),
            "latency_s": o.get("latency_s"),
        })
    item_tot = defaultdict(Counter)
    for k in cases:
        for p, c in k["items"].items():
            item_tot[p].update(c)
    items = {p: {**c, "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None,
                 "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None}
             for p, c in item_tot.items()}
    lat = [k["latency_s"] for k in cases if k["latency_s"] is not None]
    report = {
        "predictions": str(pred_path), "gold": str(gold_path), "split": split,
        "missing_predictions": len(missing),
        "status_counts": dict(Counter(k["status"] for k in cases)),
        "repairs": dict(Counter(f for k in cases for f in k["fixes"]).most_common()),
        "overall": summarise(cases),
        "ci95": bootstrap(cases),
        "valid_only": summarise([k for k in cases if k["status"] == "valid"]) if any(k["status"] == "valid" for k in cases) else None,
        "per_field": per_field(cases),
        "list_items": items,
        "breakdowns": breakdowns(cases),
        "triage_confusion": dict(Counter(f"{k['gold_level']}->{k['pred_level']}" for k in cases)),
        "latency_s": {"mean": sum(lat) / len(lat), "max": max(lat)} if lat else None,
    }
    return report, cases


def print_summary(r):
    o, ci = r["overall"], r["ci95"]
    print(f"cases {o['n']}  status {r['status_counts']}  missing {r['missing_predictions']}")
    if r.get("repairs"):
        print("  repairs:", ", ".join(f"{k} {v}" for k, v in list(r["repairs"].items())[:6]))
    for k in HEADLINE:
        lo_hi = ci.get(k)
        extra = f"  [{lo_hi[0]:.3f}-{lo_hi[1]:.3f}]" if lo_hi else ""
        print(f"  {k:<26}{o[k]:.3f}{extra}")
    print(f"  over_triage_rate      {o['over_triage_rate']:.3f}   scope_error_rate {o['scope_error_rate']:.3f}"
          f"   provisional_rate {o['provisional_rate']:.3f}")
    worst = sorted((v["recall"], p) for p, v in r["per_field"].items() if v.get("recall") is not None)[:5]
    print("  lowest-recall fields:", ", ".join(f"{p} {x:.2f}" for x, p in worst))
    hall = sorted(((v["hallucination_rate"], p) for p, v in r["per_field"].items()
                   if v.get("hallucination_rate") is not None), reverse=True)[:5]
    print("  most-hallucinated fields:", ", ".join(f"{p} {x:.3f}" for x, p in hall if x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--gold", default=str(ROOT / "examples" / "synthetic_handovers_v2.csv"))
    ap.add_argument("--split", default="dev", help="dev | test | train | all")
    ap.add_argument("--out", help="write the full JSON report here")
    ap.add_argument("--cases", help="write a per-case CSV here")
    ap.add_argument("--subset", action="store_true", help="score only cases present in PRED (quick runs)")
    a = ap.parse_args()
    if a.split == "test":
        print("NOTE: scoring the frozen test split. Only do this once, for the final run (E6).")
    report, cases = score(a.pred, a.gold, a.split, a.subset)
    print_summary(report)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if a.cases:
        with open(a.cases, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "status", "exact", "gold_level", "pred_level", "triage", "silent_under", "n_questions", "true_level",
                        "correction_ok", "n_wrong", "n_missed", "n_hallucinated", "error_fields"])
            for k in cases:
                bad = [p for p, x in k["fields"].items() if x in ("wrong", "missed", "hallucinated")]
                c = Counter(k["fields"].values())
                w.writerow([k["case_id"], k["status"], k["exact"], k["gold_level"], k["pred_level"], k["triage"], k["silent_under"], k["n_questions"],
                            k["true_level"], k["correction_ok"], c["wrong"], c["missed"], c["hallucinated"],
                            ";".join(bad)])


if __name__ == "__main__":
    main()
