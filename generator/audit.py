"""Audit the generated CSV: distribution, coverage, and text-record consistency."""
import csv
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from generator.generate import para, words  # noqa: E402

rows = list(csv.DictReader((ROOT / "examples" / "synthetic_handovers_v0.csv").open(encoding="utf-8")))
print(f"rows: {len(rows)}   columns: {len(rows[0])}")

print("\ntrue level:", dict(sorted(Counter(r["true_level"] for r in rows).items())))
print("split     :", dict(Counter(r["split"] for r in rows)))

test = [r for r in rows if r["split"] == "test" and r["in_scope"] == "True"]
lv1 = sum(1 for r in test if r["true_level"] == "1")
print(f"test set in scope: {len(test)}   true Level I: {lv1}")
fires = Counter(c for r in test for c in r["true_criteria"].split(";") if c)
low = {k: n for k, n in fires.items() if n < 20}
print(f"criteria firing in test set: {len(fires)}/37   fewest: {min(fires.values())}   under 20: {low or 'none'}")

pct = lambda k: 100 * sum(1 for r in rows if r[k] == "True") / len(rows)  # noqa: E731
print(f"borderline {pct('borderline'):.1f}%   with retraction {pct('has_retraction'):.1f}%   "
      f"retraction crossing a threshold {pct('retraction_crosses_threshold'):.1f}%")
print("omissions:", dict(Counter(o for r in rows for o in r["omitted"].split(";") if o).most_common()))
mism = sum(1 for r in rows if r["in_scope"] == "True" and str(r["said_level_naive"]) != r["true_level"])
print(f"said-only naive level differs from true level: {mism} cases ({100 * mism / len(rows):.1f}%)")


# ---- every value in the as-said record must be audible in the text
def forms(n):
    n = int(round(float(n)))
    return {str(n), para(n), words(n)}


bad, invalid = [], 0
for r in rows:
    said = json.loads(r["said_record_json"])
    try:
        HandoverRecord.model_validate(said)
    except Exception:
        invalid += 1
    t = r["text"].lower()
    v = said["signs"]["vitals"]
    checks = {"age": said["identification"]["age_years"], "sbp": v["systolic_bp_mmhg"], "dbp": v["diastolic_bp_mmhg"],
              "hr": v["heart_rate_bpm"], "rr": v["resp_rate_per_min"], "spo2": v["spo2_percent"],
              "gcs": (said["signs"]["gcs"] or {}).get("total"), "eta": said["identification"]["eta_minutes"]}
    for k, val in checks.items():
        if val is not None and not any(re.search(r"(?<![\w-])" + re.escape(f) + r"(?!\w)(?!-(?!year))", t) for f in forms(val)):
            bad.append((r["case_id"], k, val))
print(f"\nas-said records failing schema validation: {invalid}")
print(f"stated values missing from their text: {len(bad)}", bad[:5])
