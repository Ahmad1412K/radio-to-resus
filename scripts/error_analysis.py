"""Error analysis of extractor runs: where the record goes wrong and what it does to triage.

    python scripts/error_analysis.py runs/qwen3-4b_dev_k0.jsonl
    python scripts/error_analysis.py runs/qwen3-4b_dev_k0.jsonl runs/llama3.2-3b_dev_k0.jsonl ...

For each run it writes runs/error_analysis/<run>/:
  per_field.csv     correct / wrong / missed / hallucinated per field, sorted by errors
  under_triage.csv  every under-triaged case: levels, criteria lost, empty-record flag
  corrections.csv   every spoken correction and how the model handled it
  summary.json      the error patterns below, as counts and rates
and one table over all runs given: runs/error_analysis/patterns_<split>.csv.

The patterns were found by reading Qwen3 4B's full-dev errors (E2); each one is a
rule that can be checked on any run, so the same table compares models and, later,
prompt changes (E3) and fine-tuning (E4). Dev only: the test split is not analysed.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical  # noqa: E402
from extract.score import EMPTY, get, norm, parse_output, score  # noqa: E402
from triage.engine import triage  # noqa: E402

GOLD = ROOT / "examples" / "synthetic_handovers_v2.csv"
OUT = ROOT / "runs" / "error_analysis"


def filled(node) -> int:
    """Number of leaves that are not null (a non-empty list counts as one)."""
    if isinstance(node, dict):
        return sum(filled(v) for v in node.values())
    if isinstance(node, list):
        return 1 if node else 0
    return node is not None


def rate(hits, total):
    return {"n": total, "hits": hits, "rate": round(hits / total, 4) if total else None}


def load(pred_path, split):
    rep, cases = score(pred_path, GOLD, split)
    rows = {r["case_id"]: r for r in csv.DictReader(open(GOLD, encoding="utf-8")) if r["split"] == split}
    preds = {}
    for line in open(pred_path, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            preds[o["case_id"]] = parse_output(o["output"])[0]
    gold = {cid: canonical(json.loads(rows[cid]["said_record_json"])) for cid in rows}
    return rep, cases, rows, gold, preds


def correction_kind(g, p, r):
    """How one spoken correction was handled (r is a gold retraction entry)."""
    if p is None:
        return "no usable output"
    value_ok = norm(get(p, r["field"])) == norm(get(g, r["field"]))
    logs = p["retractions"]
    if value_ok and any(norm(x) == norm(r) for x in logs):
        return "fully right"
    if value_ok and any(re.search("[a-z]", f"{x['retracted_value']}{x['corrected_value']}") for x in logs):
        return "value right, log in words"
    if value_ok and logs:
        return "value right, log entry wrong"
    if value_ok:
        return "value right, not logged"
    if str(get(p, r["field"])) == r["retracted_value"]:
        return "kept the retracted value"
    if get(p, r["field"]) is None:
        return "value left null"
    return "other wrong value"


def analyse(pred_path, split):
    rep, cases, rows, gold, preds = load(pred_path, split)
    P = {k["case_id"]: preds.get(k["case_id"]) or EMPTY for k in cases}
    G = {k["case_id"]: gold[k["case_id"]] for k in cases}
    style = {cid: json.loads(rows[cid]["llm_style"] or "{}") for cid in G}
    pat = {}

    # 1. empty record: (almost) nothing extracted from a full handover
    empty = {cid for cid, p in P.items() if filled(p) <= 3 and filled(G[cid]) > 3}
    pat["empty_record"] = rate(len(empty), len(cases))
    pat["empty_record_vitals_first"] = rate(
        sum(style[c].get("structure", "").startswith("leads with vital") for c in empty), len(empty))

    # 2. a nested object left null although part of it was said
    def miss(path, cond):
        ids = [c for c in G if cond(G[c]) and get(G[c], path) is not None]
        return rate(sum(get(P[c], path) is None for c in ids), len(ids))
    pat["gcs_total_only_missed"] = miss("signs.gcs.total", lambda g: g["signs"]["gcs"] and g["signs"]["gcs"]["eye"] is None)
    pat["gcs_with_components_missed"] = miss("signs.gcs.total", lambda g: g["signs"]["gcs"] and g["signs"]["gcs"]["eye"] is not None)
    pat["no_loc_missed"] = miss("mechanism.loss_of_consciousness.occurred",
                                lambda g: (g["mechanism"]["loss_of_consciousness"] or {}).get("occurred") is False)
    pat["impact_speed_missed"] = miss("mechanism.vehicle.impact_speed_mph", lambda g: True)
    pat["burn_mechanism_missed"] = miss("mechanism.type", lambda g: g["mechanism"]["type"] == "burn")

    # 3. digit amputations recorded on the hand/foot instead of finger/toe (L3-05 needs finger/toe)
    amp = [(c, i) for c in G for i in G[c]["injuries"] or []
           if i["type"] == "amputation" and i["region"] in ("finger", "toe")]
    pat["amputation_region_hand_or_foot"] = rate(
        sum(any(j["type"] == "amputation" and j["region"] in ("hand", "foot") for j in P[c]["injuries"] or [])
            for c, _ in amp), len(amp))

    # 4. "one-fourteen" read as 140: systolic 110-119 written as (x - 100) * 10
    sbp = [c for c in G if 110 <= (get(G[c], "signs.vitals.systolic_bp_mmhg") or 0) <= 119]
    pat["sbp_110_119_read_as_1x0"] = rate(
        sum(get(P[c], "signs.vitals.systolic_bp_mmhg") == (get(G[c], "signs.vitals.systolic_bp_mmhg") - 100) * 10
            for c in sbp), len(sbp))

    # 5. a "normal" value written although nothing was said (rule 1 of the prompt)
    for name, path, val in [("hemorrhage_none_unsaid", "signs.hemorrhage", "none"),
                            ("airway_patent_unsaid", "signs.airway", "patent"),
                            ("abdominal_pain_false_unsaid", "signs.abdominal_pain", False),
                            ("trend_stable_unsaid", "signs.trend", "stable")]:
        ids = [c for c in G if get(G[c], path) is None]
        pat[name] = rate(sum(get(P[c], path) == val for c in ids), len(ids))

    # 6. spoken corrections
    corr = []
    for c in G:
        for r in G[c]["retractions"]:
            corr.append({"case_id": c, "field": r["field"], "retracted": r["retracted_value"],
                         "corrected": r["corrected_value"], "model_value": get(P[c], r["field"]),
                         "crosses_threshold": rows[c]["retraction_crosses_threshold"],
                         "kind": correction_kind(G[c], preds.get(c), r)})
    kinds = Counter(x["kind"] for x in corr)
    pat["corrections"] = {"n": len(corr), **dict(kinds.most_common())}
    pat["correction_value_right"] = rate(sum(x["kind"].startswith("value right") or x["kind"] == "fully right"
                                             for x in corr), len(corr))
    cross = [x for x in corr if x["crosses_threshold"] == "True"]
    pat["kept_retracted_when_crosses_threshold"] = rate(
        sum(x["kind"] == "kept the retracted value" for x in cross), len(cross))

    # 7. under-triage: which criteria were lost, and how many came from empty records
    under = []
    for k in cases:
        if k["triage"] != "under":
            continue
        c = k["case_id"]
        lost = [x for x in triage(G[c]).fired if x not in triage(P[c]).fired]
        under.append({"case_id": c, "gold_level": k["gold_level"], "pred_level": k["pred_level"],
                      "provisional": k["provisional"], "empty_record": c in empty, "criteria_lost": " ".join(lost)})
    pat["under_triage"] = {"n": len(under), "from_empty_record": sum(u["empty_record"] for u in under),
                           "criteria_lost": dict(Counter(x for u in under for x in u["criteria_lost"].split()).most_common())}

    o = rep["overall"]
    summary = {"run": pathlib.Path(pred_path).stem, "split": split,
               "headline": {m: o[m] for m in ("field_f1", "hallucination_rate", "correction_accuracy",
                                              "triage_accuracy", "under_triage_rate", "silent_under_triage_rate")},
               "outcome_counts": o["outcome_counts"], "repairs": rep["repairs"], "patterns": pat}

    d = OUT / summary["run"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    pf = sorted(rep["per_field"].items(), key=lambda kv: -sum(kv[1].get(x, 0) for x in ("wrong", "missed", "hallucinated")))
    with open(d / "per_field.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "correct", "wrong", "missed", "hallucinated", "ok_null", "precision", "recall"])
        for f, v in pf:
            w.writerow([f] + [v.get(x, 0) for x in ("correct", "wrong", "missed", "hallucinated", "ok_null")]
                       + [None if v[x] is None else round(v[x], 4) for x in ("precision", "recall")])
    for name, data in (("under_triage", under), ("corrections", corr)):
        with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            if data:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    if a.split == "test":
        raise SystemExit("error analysis is for dev; the test split is run once, in E6")
    table = []
    for p in a.runs:
        s = analyse(pathlib.Path(p), a.split)
        row = {"run": s["run"]}
        for k, v in s["patterns"].items():
            if isinstance(v, dict) and "rate" in v:
                row[k] = v["rate"]
        row["under_triage_n"] = s["patterns"]["under_triage"]["n"]
        row["under_from_empty_record"] = s["patterns"]["under_triage"]["from_empty_record"]
        table.append(row)
        print(f"{s['run']}: {OUT / s['run']}")
    out = OUT / f"patterns_{a.split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"table: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
