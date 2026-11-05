"""Find v2 cases where the text and the gold "said" record disagree.

The v2 texts are Claude rewrites of the template texts. The rewrite sometimes
dropped a statement the label still records, or added one the label lacks.
This script proposes label corrections; each proposal is reviewed by hand
before scripts/repair_v2_labels.py applies it.

    python scripts/audit_v2_labels.py            # writes runs/v2_label_candidates.csv

Detectors are deliberately liberal about wording (they only need to know
whether a topic is mentioned at all), and every proposal carries the text
snippet that justifies it.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical  # noqa: E402
from extract.regex_baseline import words_to_digits  # noqa: E402

csv.field_size_limit(10**8)
OUT = ROOT / "runs" / "v2_label_candidates.csv"


def snip(text, pat, width=70):
    m = re.search(pat, text)
    if not m:
        return ""
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return text[a:b].replace("\n", " ")


def sentences(text, pat):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if re.search(pat, s)]


def crew_says(text, pat):
    """True if pat appears in a sentence that is not a hospital read-back ('copy, ...')."""
    return any(not re.match(r"^(?:uh,? |um,? )*copy\b", s) for s in sentences(text, pat))


def propose(cid, t, g):
    """Yield (field, old, new, reason, snippet)."""
    S, I, M = g["signs"], g["identification"], g["mechanism"]

    # ---------------- injuries
    inj_pat = (r"no (?:other |obvious |visible )?injur|no injuries|injur(?:y|ies)? (?:noted|found)"
               r"|none obvious|none found|none other|nothing we've found|nothing obvious"
               r"|injuries\?\s*(?:(?:uh|um),?\s*)*(?:none|no|nothing)")
    if g["injuries"] == [] and not re.search(inj_pat, t):
        yield "injuries", "[]", "null", "'no injuries' not said in v2", snip(t, r"injur")
    if g["injuries"] is None and re.search(r"no (?:other |obvious |visible )?injuries", t):
        yield "injuries", "null", "[]", "v2 says no injuries", snip(t, r"no (?:other |obvious |visible )?injuries")

    # ---------------- GCS
    gcs = S["gcs"]
    if gcs and gcs["total"] is not None and not re.search(r"\bgcs\b|glasgow", t):
        yield "signs.gcs", json.dumps(gcs), "null", "GCS not said in v2", ""
    elif gcs and gcs["eye"] is not None:
        e, v, m = gcs["eye"], gcs["verbal"], gcs["motor"]
        if not re.search(rf"\b{e}\b\W+(?:\w+\W+){{0,3}}{v}\b\W+(?:\w+\W+){{0,3}}{m}\b", t):
            yield "signs.gcs.components", f"{e}/{v}/{m}", "null", "GCS components not said in v2", snip(t, r"\bgcs\b")
    if gcs is None and re.search(r"\bgcs\b", t):
        yield "signs.gcs", "null", "?", "v2 states a GCS the label lacks", snip(t, r"\bgcs\b")

    # ---------------- simple sign statements: (field, label value, mention pattern, add-pattern -> value)
    checks = [
        ("signs.airway", S["airway"], r"airway|intubat|et tube|supraglottic|i-gel",
         [(r"airway(?:'s| is)? (?:is )?patent|patent airway", "patent")]),
        ("signs.pneumothorax_or_hemothorax", S["pneumothorax_or_hemothorax"], r"breath sounds|pneumo|hemothorax",
         [(r"breath sounds (?:are )?equal|no pneumothorax", False)]),
        ("signs.hemorrhage", S["hemorrhage"], r"bleed|hemorrhag",
         [(r"no (?:active |obvious |external )?bleeding|not bleeding", "none"), (r"bleeding (?:is )?controlled|bleeding's controlled", "controlled")]),
        ("signs.trend", S["trend"], r"\bstable\b|improving|deteriorat|getting worse",
         [(r"(?<!un)\bstable\b", "stable"), (r"improving", "improving"), (r"deteriorat|getting worse", "deteriorating")]),
    ]
    for f, val, mention, adds in checks:
        if val is not None and val != "advanced_airway" and not re.search(mention, t):
            yield f, json.dumps(val), "null", "not said in v2", ""
        if val is None:
            for pat, new in adds:
                if crew_says(t, pat):
                    yield f, "null", json.dumps(new), "v2 crew states it", snip(t, pat)
                    break

    # abdomen
    abd_mention = r"abdom|belly|tummy"
    if (S["abdominal_pain"] is not None or S["abdomen_firm_or_distended"] is not None) and not re.search(abd_mention, t):
        yield "signs.abdomen", f"{S['abdominal_pain']}/{S['abdomen_firm_or_distended']}", "null", "abdomen not said in v2", ""
    if S["abdominal_pain"] is None and S["abdomen_firm_or_distended"] is None and crew_says(t, r"abdomen(?:'s| is)? soft|no abdominal pain"):
        yield "signs.abdomen", "None/None", "false/false", "v2 crew says abdomen soft / no pain", snip(t, r"abdomen(?:'s| is)? soft|no abdominal pain")

    # neuro
    if (S["paralysis"] is False or S["bilateral_sensory_loss"] is False) and not re.search(r"moving all|sensation|move", t):
        yield "signs.neuro", f"{S['paralysis']}/{S['bilateral_sensory_loss']}", "null", "neuro exam not said in v2", ""
    if S["paralysis"] is None and S["bilateral_sensory_loss"] is None and crew_says(t, r"moving all (?:4|four)"):
        yield "signs.neuro", "None/None", "false/false", "v2 crew says moving all four", snip(t, r"moving all (?:4|four)")

    # LOC
    loc = M["loss_of_consciousness"]
    if loc and loc["occurred"] is not None and not re.search(r"conscious|\bloc\b|passed out|knocked out", t):
        yield "mechanism.loss_of_consciousness", json.dumps(loc), "null", "LOC not said in v2", ""
    if (loc is None or loc["occurred"] is None) and crew_says(t, r"no (?:loss of consciousness|loc)\b"):
        yield "mechanism.loss_of_consciousness", "null", '{"occurred": false}', "v2 crew says no LOC", snip(t, r"no (?:loss of consciousness|loc)\b")

    # shock
    if S["clinical_signs_of_shock"] is None and crew_says(t, r"(?<!no )signs of shock"):
        yield "signs.clinical_signs_of_shock", "null", "true", "v2 crew says signs of shock", snip(t, r"signs of shock")
    if S["clinical_signs_of_shock"] is True and not re.search(r"shock", t):
        yield "signs.clinical_signs_of_shock", "true", "null", "shock not said in v2", ""
    words = {"pale": r"\bpale\b", "cool_clammy": r"clammy|cool", "diaphoretic": r"diaphoretic|sweaty",
             "delayed_cap_refill": r"cap(?:illary)? refill", "weak_thready_pulse": r"thready|weak pulse",
             "altered_mentation": r"altered"}
    for sgn in S["shock_signs"]:
        if not re.search(words[sgn], t):
            yield "signs.shock_signs", sgn, "remove", "shock sign not said in v2", ""

    # ETA
    eta = I["eta_minutes"]
    if eta is not None and not re.search(rf"\b{eta}\b[^.?!]{{0,15}}(?:minute|min\b|out|away)|eta[^.?!]{{0,25}}\b{eta}\b|see you in {eta}\b", t):
        yield "identification.eta_minutes", str(eta), "null", "ETA not said in v2", snip(t, rf"\b{eta}\b")

    # AMBO lists
    for f, empty_pat, mention in (("allergies", r"no (?:known )?allergies|\bnka\b|no allergies", r"allerg|\bnka\b"),
                                  ("medications", r"no (?:regular )?medications|no meds|not on any", r"medicat|\bmeds\b|takes|\bon\b"),
                                  ("background", r"no (?:significant |past )?(?:medical )?history|no history|no medical history", r"history")):
        val = g[f]
        if val == [] and not re.search(empty_pat, t):
            yield f, "[]", "null", "'none' not said in v2", snip(t, mention)
        for item in val or []:
            key = {"atrial fibrillation": r"a-fib|afib|atrial fib"}.get(item, re.escape(item))
            if not re.search(key, t):
                yield f, item, "remove", "item not said in v2", ""

    # interventions
    tx_words = {"splint": r"splint", "analgesia": r"analges|fentanyl|pain med|morphine given",
                "iv_fluids": r"fluids|\biv\b", "tourniquet": r"tourniquet", "pelvic_binder": r"binder",
                "spinal_motion_restriction": r"spinal|collar|c-spine", "endotracheal_intubation": r"intubat|et tube",
                "blood_transfusion": r"blood|transfus", "needle_decompression": r"needle|decompress",
                "supraglottic_airway": r"supraglottic|i-gel|king"}
    for tx in g["treatment"]["interventions"]:
        if tx in tx_words and not re.search(tx_words[tx], t):
            yield "treatment.interventions", tx, "remove", "intervention not said in v2", ""

    # SpO2
    v = S["vitals"]
    if v["spo2_percent"] is not None and not re.search(r"\bsats?\b|spo2|saturat|oxygen|\bo2\b|pulse ox", t):
        yield "signs.vitals.spo2", str(v["spo2_percent"]), "null", "SpO2 not said in v2", ""

    # retractions
    if g["retractions"] and not re.search(r"correction|sorry|i mean|scratch that|wait", t):
        yield "retractions", json.dumps(g["retractions"]), "?", "no correction phrase in v2", ""


def main():
    rows = list(csv.DictReader(open(ROOT / "examples" / "synthetic_handovers_v2.csv", encoding="utf-8")))
    out = []
    for r in rows:
        t = words_to_digits(r["text"])
        g = canonical(json.loads(r["said_record_json"]))
        for f, old, new, why, sn in propose(r["case_id"], t, g):
            out.append({"case_id": r["case_id"], "split": r["split"], "field": f, "old": old, "new": new,
                        "reason": why, "snippet": sn, "decision": ""})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    c = Counter((o["field"], o["new"] if o["new"] in ("null", "remove") else "add/change") for o in out)
    print(f"{len(out)} candidates in {len({o['case_id'] for o in out})} cases -> {OUT.relative_to(ROOT)}")
    for k, n in c.most_common():
        print(f"  {n:4d}  {k[0]:40s} {k[1]}")


if __name__ == "__main__":
    main()
