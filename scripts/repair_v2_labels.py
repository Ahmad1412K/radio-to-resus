"""Repair v2 "said" labels so they match what the v2 text actually says.

The v2 texts are Claude rewrites of the template texts; the rewrite sometimes
dropped a statement (e.g. "no obvious injuries") or added one ("signs of
shock", "stable"). scripts/audit_v2_labels.py proposes corrections; every
proposal was reviewed by hand (2026-10-27). Accepted proposals are applied
here; rejected ones are listed in REJECT with the reason.

Only said_record_json (and said_level_naive, derived from it) changes. The
true record and true level describe the patient, not the text, and stay as
they are.

    python scripts/audit_v2_labels.py      # candidates
    python scripts/repair_v2_labels.py     # apply + log
    python scripts/freeze_test.py          # re-freeze (test labels changed)

Writes:
    examples/synthetic_handovers_v2.csv          (in place; original kept as
                                                  examples/archive/synthetic_handovers_v2_prerepair.csv)
    examples/splits/v2_label_repairs.csv         one row per change
"""
from __future__ import annotations

import csv
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generator"))
from extract.prompt_schema import canonical  # noqa: E402
from generate import _fill, label, level  # noqa: E402

csv.field_size_limit(10**8)
V2 = ROOT / "examples" / "synthetic_handovers_v2.csv"
BACKUP = ROOT / "examples" / "archive" / "synthetic_handovers_v2_prerepair.csv"
CANDS = ROOT / "runs" / "v2_label_candidates.csv"
LOG = ROOT / "examples" / "splits" / "v2_label_repairs.csv"

# (case_id, field) -> why the proposal was rejected after reading the text
REJECT = {
    **{(c, "allergies"): "text says 'allergies? none known' (detector missed the phrasing)"
       for c in ("H01259", "H01693", "H01827", "H02559")},
    ("H01960", "medications"): "text says 'medications? none'",
    **{(c, "background"): "text says 'diabetic' (detector looked for 'diabetes')"
       for c in ("H00203", "H00339", "H00374", "H00398", "H00549")},
    **{(c, "signs.vitals.spo2"): "SpO2 given as 'N percent' in the vitals list"
       for c in ("H00022", "H00203", "H00236")},
    **{(c, "identification.eta_minutes"): "text says 'eta? <n>' without 'minutes'"
       for c in ("H01649", "H01755")},
    ("H00029", "signs.neuro"): "'moving all four?' is the hospital's question; crew does not answer",
}


def apply(said, field, old, new):
    S, I, M = said["signs"], said["identification"], said["mechanism"]
    val = None if new == "null" else json.loads(new) if new not in ("remove", "false/false") else new
    if field == "injuries":
        said["injuries"] = val
    elif field == "signs.gcs.components":
        S["gcs"].update(eye=None, verbal=None, motor=None)
    elif field == "signs.abdomen":
        S["abdominal_pain"] = S["abdomen_firm_or_distended"] = None
    elif field == "signs.neuro":
        S["paralysis"] = S["bilateral_sensory_loss"] = None
    elif field == "mechanism.loss_of_consciousness":
        M["loss_of_consciousness"] = None
    elif field == "identification.eta_minutes":
        I["eta_minutes"] = None
    elif field == "signs.vitals.spo2":
        S["vitals"]["spo2_percent"] = S["vitals"]["spo2_on_room_air"] = None
    elif field.startswith("signs."):
        S[field.split(".", 1)[1]] = val
    elif field in ("allergies", "medications", "background"):
        if new == "remove":
            said[field] = [x for x in said[field] if x != old]
        else:
            said[field] = val
    else:
        raise ValueError(f"no rule for {field} -> {new}")


def main():
    if LOG.exists():
        raise SystemExit(f"{LOG.relative_to(ROOT)} exists: repair already applied. "
                         f"To redo it, restore {BACKUP.relative_to(ROOT)} and delete the log.")
    rows = list(csv.DictReader(open(V2, encoding="utf-8")))
    cands = list(csv.DictReader(open(CANDS, encoding="utf-8")))
    by_id = {r["case_id"]: r for r in rows}
    if not BACKUP.exists():
        BACKUP.parent.mkdir(exist_ok=True)
        shutil.copy2(V2, BACKUP)

    log, rejected = [], 0
    for c in cands:
        key = (c["case_id"], c["field"])
        if key in REJECT:
            rejected += 1
            log.append({**{k: c[k] for k in ("case_id", "split", "field", "old", "new")},
                        "decision": "rejected", "reason": REJECT[key]})
            continue
        if c["new"] == "?":
            raise SystemExit(f"unreviewed open proposal: {key}")
        r = by_id[c["case_id"]]
        said = json.loads(r["said_record_json"])
        apply(said, c["field"], c["old"], c["new"])
        canonical(said)                                   # still a valid record
        r["said_record_json"] = json.dumps(said)
        log.append({**{k: c[k] for k in ("case_id", "split", "field", "old", "new")},
                    "decision": "applied", "reason": c["reason"]})

    changed = {l["case_id"] for l in log if l["decision"] == "applied"}
    lvl_moves = 0
    for cid in changed:
        r = by_id[cid]
        new_lvl = str(level(label(_fill(json.loads(r["said_record_json"])))))
        lvl_moves += new_lvl != r["said_level_naive"]
        r["said_level_naive"] = new_lvl

    with V2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with LOG.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)

    split = {s: len({l["case_id"] for l in log if l["decision"] == "applied" and l["split"] == s}) for s in ("dev", "test")}
    print(f"applied {len(log) - rejected} changes in {len(changed)} cases "
          f"(dev {split['dev']}, test {split['test']}); rejected {rejected}")
    print(f"said_level_naive changed in {lvl_moves} cases")
    print(f"log -> {LOG.relative_to(ROOT)}; backup -> {BACKUP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
