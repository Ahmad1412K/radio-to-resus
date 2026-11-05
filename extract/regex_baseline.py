"""E0: rule/regex baseline extractor.

    python -m extract.regex_baseline --split dev --out runs/e0_regex_v2_dev.jsonl

No model. Converts spoken numbers to digits, collapses self-corrections, then
fills fields with hand-written patterns. Its vocabulary (drug names, injury
words, region words) comes from the generator, so it is an optimistic floor:
it knows the closed vocabulary that a real system would not.

Output format is the same as the model runs (one JSON line per case with the
raw output text), so extract/score.py scores it the same way.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical  # noqa: E402

csv.field_size_limit(10**8)

# ------------------------------------------------------------------ spoken numbers
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
TEENS = {w: 10 + i for i, w in enumerate("ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
NUMWORDS = set(UNITS) | set(TEENS) | set(TENS) | {"hundred", "oh"}


def _two_digit(tok, i):
    """Parse a 1-2 word number < 100 at tok[i]; return (value, next index) or None."""
    if i >= len(tok):
        return None
    w = tok[i]
    if w in TENS:
        if i + 1 < len(tok) and tok[i + 1] in UNITS and tok[i + 1] != "zero":
            return TENS[w] + UNITS[tok[i + 1]], i + 2
        return TENS[w], i + 1
    if w in TEENS:
        return TEENS[w], i + 1
    if w in UNITS:
        return UNITS[w], i + 1
    return None


def _number(tok, i):
    """Parse one spoken number at tok[i]; return (value, next index) or None."""
    w = tok[i]
    if w in UNITS and i + 1 < len(tok) and tok[i + 1] == "hundred":
        base, j = UNITS[w] * 100, i + 2
        if j < len(tok) and tok[j] == "and":
            j += 1
        rest = _two_digit(tok, j)
        return (base + rest[0], rest[1]) if rest else (base, j)
    if w == "one" and i + 1 < len(tok):            # medic style: "one twenty-four", "one-oh-five"
        nxt = tok[i + 1]
        if nxt == "oh" and i + 2 < len(tok) and tok[i + 2] in UNITS:
            return 100 + UNITS[tok[i + 2]], i + 3
        if nxt in TENS or nxt in TEENS:
            rest = _two_digit(tok, i + 1)
            return 100 + rest[0], rest[1]
    return _two_digit(tok, i)


def words_to_digits(text: str) -> str:
    """'BP one twenty-four over eighty, GCS fifteen' -> 'bp 124 over 80, gcs 15'."""
    t = text.lower().replace("’", "'")
    parts = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^a-z\d\s]|\s+", t.replace("-", " - "))
    out, i = [], 0
    words_only = [p for p in parts]
    while i < len(words_only):
        p = words_only[i]
        if p in NUMWORDS and p not in ("hundred", "oh"):
            # gather the word run, skipping single spaces and hyphens
            run, idx = [], i
            while idx < len(words_only) and (words_only[idx] in NUMWORDS or words_only[idx] in (" ", " - ", "-")
                                             or words_only[idx] == "and" and run and run[-1] == "hundred"):
                if words_only[idx].strip(" -"):
                    run.append(words_only[idx])
                idx += 1
            k, nums = 0, []
            while k < len(run):
                r = _number(run, k)
                if r is None:
                    nums.append(run[k])
                    k += 1
                else:
                    nums.append(str(r[0]))
                    k = r[1]
            out.append(" ".join(nums))
            # keep trailing whitespace that the run swallowed
            if idx > i and words_only[idx - 1].isspace():
                out.append(" ")
            i = idx
        else:
            out.append(p)
            i += 1
    s = "".join(out)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\b(uh|um)\b,?", " ", s)
    s = s.replace("...", ", ")
    return re.sub(r"[ \t]+", " ", s)


# ------------------------------------------------------------------ helpers
SIDE = r"(?:(left|right)\s+(?:side of the\s+)?)?"
REGIONS = {
    "head": "head", "scalp": "head", "face": "face", "neck": "neck", "shoulder": "shoulder_axilla",
    "axilla": "shoulder_axilla", "chest": "chest", "abdomen": "abdomen", "flank": "flank", "back": "back",
    "pelvis": "pelvis", "groin": "groin_perineum", "buttock": "buttock", "upper arm": "upper_arm",
    "elbow": "elbow", "forearm": "forearm", "wrist": "wrist", "hand": "hand", "fingers": "finger",
    "finger": "finger", "thigh": "thigh", "knee": "knee", "lower leg": "lower_leg", "ankle": "ankle",
    "foot": "foot", "toes": "toe", "toe": "toe",
}
REGION_RE = "|".join(sorted(map(re.escape, REGIONS), key=len, reverse=True))
BONES = {"femur": ("femur", "thigh"), "humerus": ("humerus", "upper_arm"), "tib": ("tibia", "lower_leg"),
         "tibia": ("tibia", "lower_leg"), "radius": ("radius", "forearm"), "rib": ("rib", "chest"),
         "skull": ("skull", "head")}
ALLERGIES = ["penicillin", "sulfa", "morphine", "codeine", "latex"]
MEDS = ["apixaban", "metformin", "lisinopril", "albuterol", "sertraline", "warfarin"]
HISTORY = {"depression": "depression", "a-fib": "atrial fibrillation", "afib": "atrial fibrillation",
           "atrial fibrillation": "atrial fibrillation", "asthma": "asthma", "diabetes": "diabetes",
           "diabetic": "diabetes", "hypertension": "hypertension"}
SHOCK = {"pale": "pale", "cool and clammy": "cool_clammy", "clammy": "cool_clammy", "diaphoretic": "diaphoretic",
         "cap refill": "delayed_cap_refill", "weak thready pulse": "weak_thready_pulse",
         "altered": "altered_mentation"}
TX = {"tourniquet": "tourniquet", "splint": "splint", "analgesia": "analgesia", "fentanyl": "analgesia",
      "binder": "pelvic_binder", "iv fluids": "iv_fluids", "fluids running": "iv_fluids",
      "spinal motion restriction": "spinal_motion_restriction", "spinal precautions": "spinal_motion_restriction",
      "collar": "spinal_motion_restriction", "needle decompression": "needle_decompression",
      "et tube": "endotracheal_intubation", "intubated": "endotracheal_intubation",
      "supraglottic": "supraglottic_airway", "i-gel": "supraglottic_airway",
      "blood transfusion": "blood_transfusion", "running blood": "blood_transfusion"}
MECH = [  # order matters: first match wins
    (r"near drowning", "near_drowning"), (r"near hanging", "near_hanging"),
    (r"electrical", "electrical"), (r"pedestrian", "pedestrian_struck"), (r"bicycl", "bicyclist_struck"),
    (r"motorcycle", "motorcycle_crash"), (r"\batv\b", "atv_crash"),
    (r"\bmvc\b|motor vehicle|car crash|rollover", "motor_vehicle_crash"),
    (r"gunshot|\bgsw\b", "gunshot"), (r"\bstab(?:bed| wound)?\b", "stab"), (r"penetrating", "other_penetrating"),
    (r"\bburns?\b", "burn"), (r"crush", "crush"), (r"assault", "assault_blunt"),
    (r"fell|fall\b|falls\b", "fall"), (r"blunt|struck by", "other_blunt"),
]


def first(pattern, s, group=1, cast=int):
    m = re.search(pattern, s)
    if not m:
        return None
    try:
        return cast(m.group(group))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ extraction
def extract(text: str) -> dict:
    s = words_to_digits(text)
    rec = canonical({})
    I, M, S, V = rec["identification"], rec["mechanism"], rec["signs"], rec["signs"]["vitals"]

    # self-corrections: "120, correction, 90" -> keep 90, remember the pair
    corrections = []

    def _fix(m):
        corrections.append((m.group(1), m.group(2), m.start()))
        return m.group(2)
    s = re.sub(r"(\d+)(?:[ ,.]+(?:uh|um|sorry|wait))*[ ,.]+(?:correction|sorry|i mean|scratch that)[ ,.]+(?:(?:uh|um)[ ,.]+)*(\d+)",
               _fix, s)

    # I
    I["age_years"] = first(r"(\d+)[- ]year[- ]old", s)
    sx = first(r"year[- ]old (male|female)", s, cast=str) or first(r"\b(male|female)\b", s, cast=str)
    I["sex"] = sx
    I["arrival_mode"] = "ems_interfacility" if "interfacility" in s or "transfer from" in s else "ems_scene"
    if re.search(r"\bpregnant\b", s):
        I["pregnant"] = True
        I["gestation_weeks"] = first(r"(\d+) weeks", s)
    eta = first(r"eta[^\d]{0,12}(\d+)", s) or first(r"(\d+) minutes? (?:out|away)", s)
    if eta is None:
        for m in re.finditer(r"(\d+) minutes?\b", s):
            before = s[max(0, m.start() - 40):m.start()]
            if not re.search(r"loc|conscious|extrication|for about|about", before):
                eta = int(m.group(1))
                break
    I["eta_minutes"] = eta

    # M
    for pat, mt in MECH:
        if re.search(pat, s):
            M["type"] = mt
            break
    if M["type"] == "fall":
        if re.search(r"same[- ]level|ground[- ]level|from standing", s):
            M["fall"] = {"fall_type": "same_level", "height_ft": None}
        else:
            h = first(r"(?:fell|fall from|fall of)[^\d]{0,15}(\d+)(?: |-)f(?:ee|oo)t", s) or first(r"(\d+)[- ]f(?:oo|ee)t fall", s) \
                or first(r"(\d+) feet", s)
            M["fall"] = {"fall_type": "from_height" if h is not None else None, "height_ft": h}
    if M["type"] in ("motor_vehicle_crash", "pedestrian_struck", "bicyclist_struck", "motorcycle_crash", "atv_crash"):
        veh = {}
        veh["impact_speed_mph"] = first(r"(?:mvc|crash|struck|collision)[^.\d]{0,12}\bat (?:about )?(\d+)", s) \
            or first(r"(\d+) (?:mph|miles per hour)", s)
        veh["extrication_minutes"] = first(r"extrication[^\d]{0,20}(\d+)", s)
        occ = first(r"occupant intrusion (\d+)|intrusion[^\d]{0,15}(\d+)[^.]{0,20}occupant", s)
        if occ is not None:
            veh["intrusion_occupant_site_in"] = occ
        else:
            veh["intrusion_any_site_in"] = first(r"intrusion[^\d]{0,15}(\d+)", s)
        if "complete ejection" in s or "fully ejected" in s or "full ejection" in s:
            veh["ejection"] = "complete"
        elif "partial ejection" in s:
            veh["ejection"] = "partial"
        if "same compartment" in s:
            veh["death_same_compartment"] = True
        if re.search(r"\bthrown\b", s) and M["type"] in ("pedestrian_struck", "bicyclist_struck"):
            veh["patient_thrown"] = True
        if "run over" in s or "ran over" in s:
            veh["run_over"] = True
        if "separated" in s:
            veh["rider_separated"] = True
        if "rollover" in s:
            veh["rollover"] = True
        M["vehicle"] = veh if any(v is not None for v in veh.values()) else None
    if re.search(r"no (?:loss of consciousness|loc)\b|(?:loss of consciousness|loc)\?\s*(?:uh,? )?no\b", s):
        M["loss_of_consciousness"] = {"occurred": False, "duration_minutes": None}
    elif re.search(r"positive loc|loss of consciousness|\bloc\b", s):
        d = first(r"(?:loc|consciousness)[^\d.]{0,25}(\d+) minutes?", s, cast=float)
        M["loss_of_consciousness"] = {"occurred": True, "duration_minutes": d}

    # injuries
    inj = []
    if "no obvious injuries" in s or "no injuries" in s:
        inj = []
    for m in re.finditer(rf"(possible |suspected )?(open |closed )?{SIDE}(femur|humerus|tib|tibia|radius|rib|skull) fractures?", s):
        bone, region = BONES[m.group(4)]
        inj.append({"type": "fracture", "region": region, "side": m.group(3), "bone": bone,
                    "suspected": True if m.group(1) else None,
                    "open_fracture": {"open ": True, "closed ": False}.get(m.group(2) or "")})
    for m in re.finditer(rf"(possible |suspected )?(open |closed )?(?:fracture|deformity) (?:to|of) (?:the |a )?{SIDE}({REGION_RE})", s):
        region = REGIONS[m.group(4)]
        bone = {"thigh": "femur", "upper_arm": "humerus"}.get(region)
        inj.append({"type": "fracture", "region": region, "side": m.group(3), "bone": bone,
                    "suspected": True if (m.group(1) or "deformity" in m.group(0)) else None,
                    "open_fracture": {"open ": True, "closed ": False}.get(m.group(2) or "")})
    for m in re.finditer(rf"(possible |suspected )?(open |closed )?{SIDE}({REGION_RE}) fracture", s):
        region = REGIONS[m.group(4)]
        if region in ("pelvis", "head"):
            continue
        bone = {"thigh": "femur", "upper_arm": "humerus"}.get(region)
        inj.append({"type": "fracture", "region": region, "side": m.group(3), "bone": bone,
                    "suspected": True if m.group(1) else None,
                    "open_fracture": {"open ": True, "closed ": False}.get(m.group(2) or "")})
    if re.search(r"unstable pelvis", s):
        inj.append({"type": "fracture", "region": "pelvis", "bone": "pelvis", "pelvis_unstable": True,
                    "suspected": True if re.search(r"(possible|suspected) unstable pelvis", s) else None})
    elif "pelvic fracture" in s:
        inj.append({"type": "fracture", "region": "pelvis", "bone": "pelvis"})
    if re.search(r"(open|depressed)[a-z ]*skull fracture", s) and not any(i.get("bone") == "skull" for i in inj):
        inj.append({"type": "fracture", "region": "head", "bone": "skull"})
    for i in inj:
        if i.get("bone") == "skull":
            pre = s[max(0, s.find("skull") - 40):s.find("skull")]
            i["open_fracture"] = True if "open" in pre else None
            i["depressed"] = True if "depressed" in pre else None
    simple = [("laceration", "laceration"), ("contusion", "contusion"), ("abrasion", "abrasion"),
              ("stab wound", "penetrating_wound"), ("gunshot wound", "penetrating_wound"),
              ("penetrating wound", "penetrating_wound"), ("degloving", "degloving"),
              ("crush injury", "crush"), ("mangled", "mangled_extremity"), ("seatbelt sign", "seatbelt_sign")]
    for word, typ in simple:
        for m in re.finditer(rf"{word}(?: injury)?,? (?:to|of|across) (?:the |a )?{SIDE}({REGION_RE})", s):
            inj.append({"type": typ, "region": REGIONS[m.group(2)], "side": m.group(1)})
        for m in re.finditer(rf"{SIDE}({REGION_RE}) {word}", s):
            inj.append({"type": typ, "region": REGIONS[m.group(2)], "side": m.group(1)})
        if typ == "mangled_extremity":
            for m in re.finditer(rf"mangled {SIDE}({REGION_RE})", s):
                inj.append({"type": typ, "region": REGIONS[m.group(2)], "side": m.group(1)})
    for m in re.finditer(rf"(?:no distal pulses in the|pulseless)\s+{SIDE}({REGION_RE})", s):
        inj.append({"type": "pulseless_extremity", "region": REGIONS[m.group(2)], "side": m.group(1)})
    for m in re.finditer(rf"{SIDE}({REGION_RE}) is pulseless", s):
        inj.append({"type": "pulseless_extremity", "region": REGIONS[m.group(2)], "side": m.group(1)})
    for m in re.finditer(rf"amputation of (\d+) (fingers|toes)(?: on the| of the|,)? {SIDE}", s):
        inj.append({"type": "amputation", "region": REGIONS[m.group(2)], "side": m.group(3),
                    "digits_involved": int(m.group(1))})
    for m in re.finditer(rf"amputation,? (?:of )?(?:the )?{SIDE}({REGION_RE})", s):
        if m.group(2) not in ("fingers", "toes", "finger", "toe"):
            inj.append({"type": "amputation", "region": REGIONS[m.group(2)], "side": m.group(1)})
    # de-duplicate (type, region, side)
    seen, uniq = set(), []
    for i in inj:
        k = (i["type"], i["region"], i.get("side"))
        if k not in seen:
            seen.add(k)
            uniq.append({k2: v for k2, v in i.items() if v is not None})
    if uniq:
        rec["injuries"] = uniq
    elif "no obvious injuries" in s or "no injuries" in s:
        rec["injuries"] = []

    # burns
    pct = first(r"burns?,?[^\d.]{0,25}(\d+)(?: |-)?(?:percent|%)", s, cast=float) or \
        first(r"(\d+)(?: |-)?(?:percent|%) burns", s, cast=float)
    if pct is not None or "inhalation injury" in s:
        rec["burns"] = {"tbsa_percent": pct,
                        "inhalation_injury": False if "no inhalation injury" in s else (True if "inhalation injury" in s else None),
                        "electrical": True if M["type"] == "electrical" else None}

    # S: vitals
    m = re.search(r"(?:bp|blood pressure)(?: is| of)?:?\s*(\d{2,3})(?:\s*(?:over|/)\s*(\d{2,3}))?", s)
    if m:
        V["systolic_bp_mmhg"] = int(m.group(1))
        V["diastolic_bp_mmhg"] = int(m.group(2)) if m.group(2) else None
    V["heart_rate_bpm"] = first(r"(?:heart rate|\bhr\b|pulse)(?: is| of)?:?\s*(\d{2,3})", s)
    V["resp_rate_per_min"] = first(r"(?:resp(?:iratory)? rate|resps|respirations|\brr\b)(?: is| of)?:?\s*(\d{1,2})", s)
    sp = first(r"(?:sats|spo2|saturation)(?: is| of)?:?\s*(\d{2,3})", s)
    if sp is not None and sp <= 100:
        V["spo2_percent"] = sp
        V["spo2_on_room_air"] = False if re.search(r"on oxygen|on o2|non-rebreather", s) else True
    if V["diastolic_bp_mmhg"] and V["systolic_bp_mmhg"] and V["diastolic_bp_mmhg"] > V["systolic_bp_mmhg"]:
        V["diastolic_bp_mmhg"] = None

    # GCS
    m = re.search(r"gcs(?: is| of)?:?\s*(\d{1,2})(?:[,. ]+(?:that's )?(?:eyes? )?(\d)[,. ]+(?:verbal )?(\d)[,. ]+(?:motor )?(\d))?", s)
    if m and 3 <= int(m.group(1)) <= 15:
        g = {"total": int(m.group(1))}
        if m.group(2):
            e, v_, mo = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if e + v_ + mo == g["total"] and e <= 4 and v_ <= 5 and mo <= 6:
                g.update(eye=e, verbal=v_, motor=mo)
        if "prior to intubation" in s or "before intubation" in s:
            g["before_intubation"] = True
        S["gcs"] = g

    # other signs
    if re.search(r"airway (?:is )?compromised|compromised airway", s):
        S["airway"] = "compromised"
    elif re.search(r"et tube|intubated|supraglottic|i-gel|advanced airway", s):
        S["airway"] = "advanced_airway"
    elif re.search(r"airway (?:is )?patent", s):
        S["airway"] = "patent"
    if "flail" in s:
        S["flail_chest"] = True
    if re.search(r"no pneumothorax|breath sounds equal", s):
        S["pneumothorax_or_hemothorax"] = False
    elif "pneumothorax" in s:
        S["pneumothorax_or_hemothorax"] = True
    S["shock_signs"] = sorted({v for k, v in SHOCK.items() if k in s and not (k == "clammy" and "cool and clammy" in s)})
    if "signs of shock" in s:
        S["clinical_signs_of_shock"] = True
    if re.search(r"(?:not controlled|uncontrolled|can't control|active bleeding)", s) and "no active bleeding" not in s:
        S["hemorrhage"] = "active_uncontrolled"
    elif re.search(r"bleeding (?:is )?controlled", s):
        S["hemorrhage"] = "controlled"
    elif re.search(r"no (?:active )?bleeding", s):
        S["hemorrhage"] = "none"
    if re.search(r"(?:can't|unable to|cannot) move (?:his |her |the )?legs", s):
        S["paralysis"] = True
    if re.search(r"no sensation bilaterally|bilateral sensory loss", s):
        S["bilateral_sensory_loss"] = True
    if re.search(r"spinal cord injury", s):
        S["suspected_spinal_cord_injury"] = True
    if re.search(r"moving all (?:4|four)", s):
        S["paralysis"] = S["paralysis"] or False
        S["bilateral_sensory_loss"] = S["bilateral_sensory_loss"] or False
    if re.search(r"abdomen (?:is )?(?:firm|distended)|firm abdomen", s):
        S["abdomen_firm_or_distended"] = True
    if re.search(r"(?<!no )abdominal pain", s):
        S["abdominal_pain"] = True
    if re.search(r"abdomen soft|soft abdomen|no abdominal pain", s):
        S["abdominal_pain"] = S["abdominal_pain"] or False
        S["abdomen_firm_or_distended"] = S["abdomen_firm_or_distended"] or False
    for w in ("deteriorating", "improving", "stable"):
        if re.search(rf"(?<!un)\b{w}\b", s):
            S["trend"] = w
            break

    # T
    rec["treatment"]["interventions"] = sorted({v for k, v in TX.items() if k in s})
    if re.search(r"(?:required|requires|needing|needs) (?:a )?blood", s):
        rec["treatment"]["blood_transfusion_required_to_maintain_vitals"] = True

    # AMBO
    if re.search(r"no known allergies|\bnka\b", s):
        rec["allergies"] = []
    found = [a for a in ALLERGIES if re.search(rf"allerg\w* to {a}|{a} allergy|allergic,? {a}|\b{a}\b", s)]
    if found:
        rec["allergies"] = found
    if re.search(r"no (?:regular )?medications|no meds", s):
        rec["medications"] = []
    found = [d for d in MEDS if d in s]
    if found:
        rec["medications"] = found
    if re.search(r"no (?:significant )?(?:medical )?history", s):
        rec["background"] = []
    found = sorted({v for k, v in HISTORY.items() if k in s})
    if found:
        rec["background"] = found

    # map corrections to the field whose value is the corrected number
    paths = {"identification.age_years": I["age_years"], "signs.vitals.systolic_bp_mmhg": V["systolic_bp_mmhg"],
             "signs.vitals.heart_rate_bpm": V["heart_rate_bpm"], "signs.vitals.resp_rate_per_min": V["resp_rate_per_min"],
             "signs.vitals.spo2_percent": V["spo2_percent"]}
    for wrong, right, _ in corrections:
        for p, val in paths.items():
            if val is not None and str(val) == right:
                rec["retractions"].append({"field": p, "retracted_value": wrong, "corrected_value": right})
                break

    # keep the record valid; drop whatever breaks the schema
    try:
        return canonical(rec)
    except Exception:
        rec["injuries"] = None if not rec["injuries"] else rec["injuries"]
        for key in ("injuries", "burns"):
            try:
                return canonical(rec)
            except Exception:
                rec[key] = None
        return canonical({"identification": {"age_years": I["age_years"]}, "signs": {"vitals": V}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "examples" / "synthetic_handovers_v2.csv"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.gold, encoding="utf-8")) if r["split"] == a.split]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            t0 = time.perf_counter()
            out = json.dumps(extract(r["text"]))
            fh.write(json.dumps({"case_id": r["case_id"], "output": out,
                                 "latency_s": round(time.perf_counter() - t0, 5)}) + "\n")
    print(f"wrote {a.out}: {len(rows)} cases")


if __name__ == "__main__":
    main()
