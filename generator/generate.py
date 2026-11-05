"""Template-based generator of synthetic pre-hospital radio handovers (v0).

For every case it produces
  * the TRUE patient record (complete; ground truth for triage),
  * the AS-SAID record (what the crew actually said; ground truth for extraction),
  * the radio TEXT rendered from the as-said record,
  * the activation label computed from the true record with all 37 criteria.

v0 limits (see the README in the output folder): template phrasing rather than an
LLM, and hand-set distributions rather than MIMIC-IV-ED / NEMSIS samples.
"""
from __future__ import annotations

import copy
import csv
import json
import pathlib
import random
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from generator.rules_full import label, level  # noqa: E402

VERSION = "template-v0"

# =============================================================== spoken numbers
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + ("" if o == 0 else "-" + ONES[o])
    h, r = divmod(n, 100)
    return ONES[h] + " hundred" + ("" if r == 0 else " " + words(r))


def para(n: int) -> str:
    """How paramedics say numbers: 108 -> one-oh-eight, 142 -> one-forty-two."""
    n = int(round(n))
    if n < 100:
        return words(n)
    h, r = divmod(n, 100)
    if r == 0:
        return ONES[h] + " hundred"
    if r < 10:
        return f"{ONES[h]}-oh-{ONES[r]}"
    return f"{ONES[h]}-{words(r)}"


class Speaker:
    def __init__(self, rng):
        self.rng = rng
        self.style = "spoken" if rng.random() < 0.7 else "digits"
        self.filler_rate = rng.choice([0.0, 0.1, 0.25])

    def num(self, n, kind="vital"):
        n = int(round(n))
        if self.style == "digits":
            return str(n)
        return para(n) if kind == "vital" else words(n)

    def filler(self):
        return self.rng.choice(["uh, ", "um, ", ""]) if self.rng.random() < self.filler_rate else ""


# =============================================================== true case
def base(rng, age=None):
    sex = rng.choice(["male", "male", "female"])
    sbp = rng.randint(112, 148)
    return {
        "identification": {"age_years": age if age is not None else rng.randint(15, 64), "sex": sex,
                           "pregnant": False if sex == "female" else None, "gestation_weeks": None,
                           "arrival_mode": "ems_scene", "eta_minutes": rng.choice([3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25])},
        "mechanism": {"type": None, "description": None, "fall": None, "vehicle": None,
                      "loss_of_consciousness": {"occurred": False, "duration_minutes": None}},
        "injuries": [], "burns": None,
        "signs": {"vitals": {"systolic_bp_mmhg": sbp, "diastolic_bp_mmhg": rng.randint(62, min(94, sbp - 25)),
                             "heart_rate_bpm": rng.randint(62, 98), "resp_rate_per_min": rng.randint(12, 20),
                             "spo2_percent": rng.randint(95, 100), "spo2_on_room_air": True},
                  "gcs": {"eye": 4, "verbal": 5, "motor": 6, "total": 15, "before_intubation": None},
                  "airway": "patent", "flail_chest": False, "pneumothorax_or_hemothorax": False,
                  "clinical_signs_of_shock": False, "shock_signs": [], "hemorrhage": "none",
                  "paralysis": False, "bilateral_sensory_loss": False, "suspected_spinal_cord_injury": False,
                  "abdominal_pain": False, "abdomen_firm_or_distended": False, "trend": rng.choice(["stable", "stable", "improving"])},
        "treatment": {"interventions": [], "blood_transfusion_required_to_maintain_vitals": False},
        "allergies": rng.choice([[], [], [], ["penicillin"], ["sulfa"], ["morphine"], ["latex"], ["codeine"]]),
        "medications": rng.choice([[], [], ["lisinopril"], ["metformin"], ["albuterol"], ["sertraline"], ["warfarin"], ["apixaban"]]),
        "background": rng.choice([[], [], ["hypertension"], ["diabetes"], ["asthma"], ["depression"], ["atrial fibrillation"]]),
        "other": None, "retractions": [],
    }


def set_gcs(c, rng, total):
    best = None
    for e in range(1, 5):
        for vb in range(1, 6):
            mo = total - e - vb
            if 1 <= mo <= 6:
                best = (e, vb, mo) if best is None or rng.random() < 0.4 else best
    c["signs"]["gcs"].update({"eye": best[0], "verbal": best[1], "motor": best[2], "total": total})


def add(c, typ, region, side=None, **kw):
    inj = {"type": typ, "region": region, "side": side, "suspected": None, "bone": None,
           "open_fracture": None, "depressed": None, "pelvis_unstable": None, "digits_involved": None}
    inj.update(kw)
    c["injuries"].append(inj)


def tx(c, *items):
    for i in items:
        if i not in c["treatment"]["interventions"]:
            c["treatment"]["interventions"].append(i)


def side(rng):
    return rng.choice(["left", "right"])


def shock(c, rng, n=None):
    c["signs"]["clinical_signs_of_shock"] = True
    c["signs"]["shock_signs"] = rng.sample(["pale", "cool_clammy", "diaphoretic", "delayed_cap_refill",
                                             "weak_thready_pulse", "altered_mentation"], n or rng.randint(2, 3))


def near(rng, lo, hi, border_lo, border_hi, p=0.45):
    """Sample a value, near the threshold border p of the time (borderline cases)."""
    return (rng.randint(border_lo, border_hi), True) if rng.random() < p else (rng.randint(lo, hi), False)


def mech_blunt(c, rng):
    """A background blunt mechanism that by itself sits at Level III or below."""
    k = rng.choice(["mvc", "fall", "assault"])
    if k == "mvc":
        c["mechanism"]["type"] = "motor_vehicle_crash"
        c["mechanism"]["vehicle"] = {"impact_speed_mph": rng.choice([None, 25, 30, 35, 40]), "extrication_minutes": None,
                                     "intrusion_occupant_site_in": None, "intrusion_any_site_in": None,
                                     "ejection": "none", "death_same_compartment": False, "patient_thrown": None,
                                     "run_over": None, "rider_separated": None, "rollover": rng.random() < 0.2}
    elif k == "fall":
        c["mechanism"]["type"] = "fall"
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": rng.choice([6, 8, 10, 12, 15])}
    else:
        c["mechanism"]["type"] = "assault_blunt"


VEH_NONE = {"impact_speed_mph": None, "extrication_minutes": None, "intrusion_occupant_site_in": None,
            "intrusion_any_site_in": None, "ejection": None, "death_same_compartment": None, "patient_thrown": None,
            "run_over": None, "rider_separated": None, "rollover": None}


def veh(**kw):
    d = dict(VEH_NONE)
    d.update(kw)
    return d


# =============================================================== scenarios (target criterion -> builder)
def s_L1_01(c, r):
    mech_blunt(c, r)
    g, b = near(r, 3, 8, 7, 8)
    set_gcs(c, r, g)
    if r.random() < 0.5:
        c["signs"]["gcs"]["before_intubation"] = True
        c["signs"]["airway"] = "advanced_airway"
        tx(c, "endotracheal_intubation")
    return b


def s_L1_02(c, r):
    mech_blunt(c, r)
    v = c["signs"]["vitals"]
    v["systolic_bp_mmhg"], b = near(r, 66, 89, 85, 89)
    v["diastolic_bp_mmhg"] = r.randint(38, v["systolic_bp_mmhg"] - 22)
    v["heart_rate_bpm"] = r.randint(100, 118)
    shock(c, r)
    c["signs"]["trend"] = r.choice(["deteriorating", "stable"])
    tx(c, "iv_fluids")
    return b


def s_L1_03(c, r):
    mech_blunt(c, r)
    v = c["signs"]["vitals"]
    v["heart_rate_bpm"], b = near(r, 121, 150, 121, 124)
    v["systolic_bp_mmhg"] = r.randint(92, 112)
    v["diastolic_bp_mmhg"] = r.randint(50, v["systolic_bp_mmhg"] - 25)
    shock(c, r)
    tx(c, "iv_fluids")
    return b


def s_L1_04(c, r):
    mech_blunt(c, r)
    low = r.random() < 0.3
    c["signs"]["vitals"]["resp_rate_per_min"], b = near(r, 5, 9, 8, 9) if low else near(r, 30, 40, 30, 32)
    return b


def s_L1_05(c, r):
    mech_blunt(c, r)
    c["signs"]["vitals"]["spo2_percent"], b = near(r, 78, 89, 87, 89)
    c["signs"]["vitals"]["spo2_on_room_air"] = True
    return b


def s_L1_06(c, r):
    mech_blunt(c, r)
    k = r.choice(["compromised", "ett", "sga", "flail", "ptx"])
    s = c["signs"]
    if k == "compromised":
        s["airway"] = "compromised"
    elif k in ("ett", "sga"):
        s["airway"] = "advanced_airway"
        tx(c, "endotracheal_intubation" if k == "ett" else "supraglottic_airway")
        set_gcs(c, r, r.randint(9, 12))
    elif k == "flail":
        s["flail_chest"] = True
        add(c, "fracture", "chest", side(r), bone="rib")
    else:
        s["pneumothorax_or_hemothorax"] = True
        if r.random() < 0.5:
            tx(c, "needle_decompression")
    return False


def s_L1_07(c, r):
    c["mechanism"]["type"] = r.choice(["motor_vehicle_crash", "crush", "other_blunt"])
    if c["mechanism"]["type"] == "motor_vehicle_crash":
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([30, 40]))
    region = r.choice(["thigh", "lower_leg", "upper_arm", "forearm"])
    add(c, "laceration", region, side(r))
    if r.random() < 0.5:
        c["signs"]["hemorrhage"] = "active_uncontrolled"
    else:
        c["signs"]["hemorrhage"] = "controlled"
        tx(c, "tourniquet")
    return False


def s_L1_08(c, r):
    c["mechanism"]["type"] = r.choice(["stab", "gunshot"])
    add(c, "penetrating_wound", r.choice(["head", "neck", "chest", "abdomen", "back", "flank", "groin_perineum",
                                          "buttock", "shoulder_axilla"]), side(r))
    return False


def s_L1_09(c, r):
    c["mechanism"]["type"] = r.choice(["motorcycle_crash", "crush", "motor_vehicle_crash"])
    c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([25, 35, 45])) if "crash" in c["mechanism"]["type"] else None
    reg, sd = r.choice(["lower_leg", "forearm"]), side(r)
    add(c, "pulseless_extremity", reg, sd)
    if r.random() < 0.6:
        add(c, "fracture", reg, sd, bone={"lower_leg": "tibia", "forearm": "radius"}[reg])
    return False


def s_L1_10(c, r):
    c["mechanism"]["type"] = r.choice(["crush", "other_blunt"])
    add(c, "amputation", r.choice(["forearm", "lower_leg", "upper_arm", "thigh"]), side(r))
    c["signs"]["hemorrhage"] = "controlled"
    tx(c, "tourniquet")
    return False


def s_L1_11(c, r):
    c["mechanism"]["type"] = "fall"
    c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": r.choice([6, 8, 10, 12, 15])}
    s = c["signs"]
    k = r.choice(["paralysis", "sensory", "sci"])
    s["paralysis"] = k == "paralysis"
    s["bilateral_sensory_loss"] = k == "sensory"
    s["suspected_spinal_cord_injury"] = True
    tx(c, "spinal_motion_restriction")
    return False


def s_L1_12(c, r):
    c["mechanism"]["type"] = "burn" if r.random() < 0.85 else "electrical"
    if r.random() < 0.5:
        t, b = near(r, 21, 60, 21, 24)
        c["burns"] = {"tbsa_percent": t, "inhalation_injury": False, "electrical": c["mechanism"]["type"] == "electrical"}
    else:
        b = False
        c["burns"] = {"tbsa_percent": r.randint(3, 18), "inhalation_injury": True,
                      "electrical": c["mechanism"]["type"] == "electrical"}
    return b


def s_L1_13(c, r):
    c["mechanism"]["type"] = r.choice(["motorcycle_crash", "pedestrian_struck", "fall"])
    if c["mechanism"]["type"] == "fall":
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": r.choice([10, 15, 18])}
    else:
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([25, 35, 45]))
    add(c, "fracture", "pelvis", None, bone="pelvis", pelvis_unstable=True)
    tx(c, "pelvic_binder")
    return False


def s_L1_14(c, r):
    c["mechanism"]["type"] = r.choice(["assault_blunt", "fall", "other_blunt"])
    if c["mechanism"]["type"] == "fall":
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": r.choice([6, 10, 12])}
    opened = r.random() < 0.5
    add(c, "fracture", "head", None, bone="skull", open_fracture=opened, depressed=not opened or r.random() < 0.3)
    set_gcs(c, r, r.randint(13, 15))
    return False


def s_L1_15(c, r):
    mech_blunt(c, r)
    a, b = r.choice([("femur", "femur"), ("femur", "humerus"), ("humerus", "humerus")])
    reg = {"femur": "thigh", "humerus": "upper_arm"}
    sides = ["left", "right"] if a == b else [side(r), side(r)]
    add(c, "fracture", reg[a], sides[0], bone=a, open_fracture=False)
    add(c, "fracture", reg[b], sides[1], bone=b, open_fracture=False)
    tx(c, "splint", "analgesia")
    return False


def s_L1_16(c, r):
    mech_blunt(c, r)
    c["identification"]["arrival_mode"] = "ems_interfacility"
    c["treatment"]["blood_transfusion_required_to_maintain_vitals"] = True
    tx(c, "blood_transfusion")
    v = c["signs"]["vitals"]
    v["systolic_bp_mmhg"] = r.randint(92, 105)
    v["diastolic_bp_mmhg"] = r.randint(50, 65)
    return False


def s_L2_01(c, r):
    mech_blunt(c, r)
    g, b = near(r, 9, 13, 13, 13)
    set_gcs(c, r, g)
    return b


def s_L2_02(c, r):
    c["mechanism"]["type"] = "fall"
    h, b = near(r, 21, 45, 21, 23)
    c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": h}
    if r.random() < 0.6:
        add(c, "fracture", r.choice(["ankle", "wrist", "foot"]), side(r), open_fracture=False)
    return b


def s_L2_03(c, r):
    c["mechanism"]["type"] = "motor_vehicle_crash"
    k = r.choice(["extrication", "intrusion_occ", "intrusion_any", "ejection", "death"])
    vv, b = veh(impact_speed_mph=r.choice([None, 40, 50, 60])), False
    if k == "extrication":
        vv["extrication_minutes"], b = near(r, 21, 45, 21, 23)
    elif k == "intrusion_occ":
        vv["intrusion_occupant_site_in"], b = near(r, 13, 24, 13, 14)
    elif k == "intrusion_any":
        vv["intrusion_any_site_in"], b = near(r, 19, 30, 19, 20)
    elif k == "ejection":
        vv["ejection"] = r.choice(["partial", "complete"])
    else:
        vv["death_same_compartment"] = True
    c["mechanism"]["vehicle"] = vv
    if r.random() < 0.5:
        add(c, "contusion", r.choice(["chest", "face", "knee"]), side(r))
    return b


def s_L2_04(c, r):
    k = r.choice(["ped", "bike", "moto", "atv"])
    b = False
    if k in ("ped", "bike"):
        c["mechanism"]["type"] = "pedestrian_struck" if k == "ped" else "bicyclist_struck"
        sub = r.choice(["thrown", "run_over", "speed"])
        if sub == "speed":
            sp, b = near(r, 21, 45, 21, 22)
            c["mechanism"]["vehicle"] = veh(impact_speed_mph=sp, patient_thrown=False, run_over=False)
        else:
            c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([None, 15, 25]), patient_thrown=sub == "thrown",
                                            run_over=sub == "run_over")
    else:
        c["mechanism"]["type"] = "motorcycle_crash" if k == "moto" else "atv_crash"
        if r.random() < 0.5:
            sp, b = near(r, 20, 60, 20, 21)
            c["mechanism"]["vehicle"] = veh(impact_speed_mph=sp, rider_separated=False)
        else:
            c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([10, 15]), rider_separated=True,
                                            rollover=(k == "atv" and r.random() < 0.5))
    add(c, "abrasion", r.choice(["forearm", "knee", "hand", "flank"]), side(r))
    return b


def s_L2_05(c, r):
    c["mechanism"]["type"] = r.choice(["assault_blunt", "other_blunt"])
    d, b = near(r, 6, 20, 6, 6)
    c["mechanism"]["loss_of_consciousness"] = {"occurred": True, "duration_minutes": d}
    set_gcs(c, r, r.randint(9, 12))
    add(c, "laceration", r.choice(["head", "face"]), None)
    return b


def s_L2_06(c, r):
    mech_blunt(c, r)
    bone = r.choice(["femur", "humerus"])
    add(c, "fracture", {"femur": "thigh", "humerus": "upper_arm"}[bone], side(r), bone=bone, open_fracture=False)
    tx(c, "splint", "analgesia")
    return False


def s_L2_07(c, r):
    mech_blunt(c, r)
    reg = r.choice(["lower_leg", "forearm"])
    add(c, "fracture", reg, side(r), bone={"lower_leg": "tibia", "forearm": "radius"}[reg], open_fracture=True)
    c["signs"]["hemorrhage"] = "controlled"
    tx(c, "splint", "analgesia")
    return False


def s_L2_08(c, r):
    c["mechanism"]["type"] = r.choice(["stab", "gunshot"])
    add(c, "penetrating_wound", r.choice(["upper_arm", "forearm", "thigh", "lower_leg", "elbow", "knee"]), side(r))
    c["signs"]["hemorrhage"] = "controlled"
    return False


def s_L2_09(c, r):
    c["identification"]["sex"] = "female"
    c["identification"]["age_years"] = r.randint(18, 42)
    c["identification"]["pregnant"] = True
    c["identification"]["gestation_weeks"] = r.randint(12, 38)
    c["mechanism"]["type"] = r.choice(["motor_vehicle_crash", "fall"])
    if c["mechanism"]["type"] == "fall":
        c["mechanism"]["fall"] = {"fall_type": "same_level", "height_ft": None}
    else:
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([15, 25, 30]))
    c["signs"]["abdominal_pain"] = True
    return False


def s_L2_10(c, r):
    c["mechanism"]["type"] = r.choice(["crush", "other_blunt", "motorcycle_crash"])
    if c["mechanism"]["type"] == "motorcycle_crash":
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([10, 15]), rider_separated=False)
    add(c, r.choice(["degloving", "crush", "mangled_extremity"]), r.choice(["hand", "forearm", "lower_leg", "foot"]), side(r))
    c["signs"]["hemorrhage"] = "controlled"
    return False


def s_L2_11(c, r):
    c["mechanism"]["type"] = "burn"
    t, b = near(r, 10, 20, 19, 20)
    c["burns"] = {"tbsa_percent": t, "inhalation_injury": False, "electrical": False}
    add(c, r.choice(["laceration", "contusion"]), r.choice(["forearm", "lower_leg", "face"]), side(r))
    return b


def s_L2_12(c, r):
    c["mechanism"]["type"] = "motor_vehicle_crash"
    c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([30, 40, 50]))
    add(c, "seatbelt_sign", r.choice(["chest", "abdomen"]), None)
    return False


def s_L2_13(c, r):
    mech_blunt(c, r)
    c["signs"]["abdominal_pain"] = True
    c["signs"]["abdomen_firm_or_distended"] = r.random() < 0.4
    return False


def s_L3_01(c, r):
    c["mechanism"]["type"] = "fall"
    if r.random() < 0.5:
        c["identification"]["age_years"] = r.randint(55, 64)
        c["mechanism"]["fall"] = {"fall_type": "same_level", "height_ft": None}
        b = False
    else:
        h, b = near(r, 5, 20, 19, 20)
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": h}
    add(c, r.choice(["contusion", "abrasion", "laceration"]), r.choice(["hand", "knee", "face", "forearm"]), side(r))
    return b


def s_L3_02(c, r):
    c["mechanism"]["type"] = r.choice(["assault_blunt", "other_blunt"])
    if r.random() < 0.5:
        d, b = near(r, 1, 4, 4, 4)
        c["mechanism"]["loss_of_consciousness"] = {"occurred": True, "duration_minutes": d}
    else:
        b = False
    add(c, r.choice(["laceration", "contusion"]), r.choice(["face", "head", "forearm"]), None)
    return b


def s_L3_03(c, r):
    c["mechanism"]["type"] = "burn"
    t, b = near(r, 1, 9, 8, 9)
    c["burns"] = {"tbsa_percent": t, "inhalation_injury": False, "electrical": False}
    return b


def s_L3_04(c, r):
    k = r.choice(["mvc", "ped", "moto"])
    b = False
    if k == "mvc":
        c["mechanism"]["type"] = "motor_vehicle_crash"
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=r.choice([None, 15, 20, 25]), ejection="none")
    elif k == "ped":
        c["mechanism"]["type"] = "pedestrian_struck"
        sp, b = near(r, 5, 19, 18, 19)
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=sp, patient_thrown=False, run_over=False)
        add(c, "contusion", r.choice(["thigh", "knee"]), side(r))
    else:
        c["mechanism"]["type"] = r.choice(["motorcycle_crash", "atv_crash"])
        sp, b = near(r, 5, 19, 18, 19)
        c["mechanism"]["vehicle"] = veh(impact_speed_mph=sp, rider_separated=False, rollover=False)
    return b


def s_L3_05(c, r):
    c["mechanism"]["type"] = r.choice(["crush", "other_blunt"])
    reg = r.choice(["finger", "finger", "toe"])
    add(c, "amputation", reg, side(r), digits_involved=r.randint(2, 4))
    c["signs"]["hemorrhage"] = "controlled"
    return False


def s_L3_06(c, r):
    c["mechanism"]["type"] = r.choice(["stab", "other_penetrating"])
    add(c, "penetrating_wound", r.choice(["hand", "foot", "wrist", "ankle"]), side(r))
    return False


def s_L3_07(c, r):
    c["mechanism"]["type"] = r.choice(["crush", "fall", "other_blunt"])
    if c["mechanism"]["type"] == "fall":
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": r.choice([2, 3, 4])}
    reg = r.choice(["finger", "hand", "foot", "toe", "wrist", "ankle"])
    add(c, "fracture", reg, side(r), open_fracture=True,
        bone="hand_bones" if reg in ("finger", "hand", "wrist") else "foot_bones")
    return False


def s_L3_08(c, r):
    c["mechanism"]["type"] = r.choice(["near_drowning", "near_hanging"])
    return False


def s_none(c, r):
    c["mechanism"]["type"] = "fall"
    if r.random() < 0.6:
        c["identification"]["age_years"] = r.randint(15, 54)
        c["mechanism"]["fall"] = {"fall_type": "same_level", "height_ft": None}
    else:
        c["mechanism"]["fall"] = {"fall_type": "from_height", "height_ft": r.choice([2, 3, 4])}
    k = r.choice(["contusion", "abrasion", "wrist", "ankle"])
    if k in ("wrist", "ankle"):
        add(c, "fracture", k, side(r), open_fracture=False)
        tx(c, "splint")
    else:
        add(c, k, r.choice(["knee", "hand", "forearm", "face"]), side(r))
    return False


SCENARIOS = {k.replace("s_", "").replace("_", "-"): v for k, v in globals().items()
             if k.startswith("s_L") and callable(v)}
SCENARIOS["NONE"] = s_none


def near_miss(c, r):
    """Push one vital close to (but not over) a Level I threshold - realistic hard negatives."""
    v, k = c["signs"]["vitals"], r.choice(["sbp", "hr", "rr", "spo2", "gcs"])
    if k == "sbp":
        v["systolic_bp_mmhg"] = r.randint(90, 95)
        v["diastolic_bp_mmhg"] = r.randint(50, 64)
    elif k == "hr":
        v["heart_rate_bpm"] = r.randint(110, 120)
    elif k == "rr":
        v["resp_rate_per_min"] = r.randint(25, 29)
    elif k == "spo2":
        v["spo2_percent"] = r.randint(90, 92)
    elif c["signs"]["gcs"]["total"] == 15:
        set_gcs(c, r, 14)


# =============================================================== the as-said record + radio text
HOSP = ["Memorial", "County General", "St. Mary's", "Regional", "University Hospital", "Mercy"]
UNITS = ["Medic", "Rescue", "Ambulance", "Unit", "Medic"]

REGION_SAY = {"head": "head", "face": "face", "neck": "neck", "shoulder_axilla": "shoulder", "chest": "chest",
              "abdomen": "abdomen", "flank": "flank", "back": "back", "pelvis": "pelvis", "groin_perineum": "groin",
              "buttock": "buttock", "upper_arm": "upper arm", "elbow": "elbow", "forearm": "forearm", "wrist": "wrist",
              "hand": "hand", "finger": "fingers", "thigh": "thigh", "knee": "knee", "lower_leg": "lower leg",
              "ankle": "ankle", "foot": "foot", "toe": "toes"}
SHOCK_SAY = {"pale": "pale", "cool_clammy": "cool and clammy", "diaphoretic": "diaphoretic",
             "delayed_cap_refill": "cap refill delayed", "weak_thready_pulse": "weak thready pulse",
             "altered_mentation": "altered"}
TX_SAY = {"endotracheal_intubation": "intubated", "supraglottic_airway": "i-gel in place", "surgical_airway": "surgical airway",
          "bag_valve_mask": "bagging", "supplemental_oxygen": "on oxygen", "needle_decompression": "needle decompression done",
          "tourniquet": "tourniquet applied", "wound_packing": "wound packed", "pelvic_binder": "pelvic binder on",
          "blood_transfusion": "running blood", "iv_fluids": "IV fluids going", "spinal_motion_restriction": "in a collar, spinal precautions",
          "splint": "splinted", "analgesia": "given fentanyl for pain", "other": "other care"}


def where(i):
    reg = REGION_SAY[i["region"]]
    return f"{i['side']} {reg}" if i.get("side") in ("left", "right") else reg


def injury_phrase(i, sp, rng, said):
    t = i["type"]
    w = where(i)
    if t == "fracture":
        suspected = rng.random() < 0.5
        said["suspected"] = True if suspected else None
        adj = "open " if i.get("open_fracture") else ("closed " if i.get("open_fracture") is False and rng.random() < 0.3 else "")
        if i.get("open_fracture") is False and adj == "":
            said["open_fracture"] = None
        if i.get("bone") == "pelvis":
            if not i.get("pelvis_unstable"):
                said["pelvis_unstable"] = None
            return ("possible " if suspected else "") + "unstable pelvis" if i.get("pelvis_unstable") else "pelvic fracture"
        if i.get("bone") == "skull":
            kinds = [k for k in ("open", "depressed") if i.get(k if k == "depressed" else "open_fracture")]
            if not i.get("depressed"):
                said["depressed"] = None
            if not i.get("open_fracture"):
                said["open_fracture"] = None
            return " and ".join(kinds) + " skull fracture"
        bone = i.get("bone")
        if bone in ("femur", "humerus", "tibia", "radius", "rib") and rng.random() < 0.7:
            return (("possible " if suspected else "") + f"{adj}{i['side'] + ' ' if i.get('side') else ''}{bone} fracture").strip()
        said["bone"] = None if bone not in ("femur", "humerus") else bone
        return ("deformity to the " if suspected else f"{adj}fracture to the ") + w
    if t == "penetrating_wound":
        return rng.choice([f"penetrating wound to the {w}", f"stab wound to the {w}", f"wound to the {w}"])
    if t == "amputation":
        if i.get("digits_involved"):
            return f"amputation of {words(i['digits_involved'])} {REGION_SAY[i['region']]}, {i['side']} side"
        return f"traumatic amputation, {w}"
    if t == "pulseless_extremity":
        return f"no distal pulses in the {w}"
    if t in ("degloving", "crush", "mangled_extremity"):
        return {"degloving": "degloving injury", "crush": "crush injury", "mangled_extremity": "mangled"}[t] + f" to the {w}"
    if t == "seatbelt_sign":
        return f"seatbelt sign across the {w}"
    return f"{t} to the {w}"


def render(true: dict, rng: random.Random, repair: bool):
    """Return (text, said_record, meta)."""
    sp = Speaker(rng)
    said = copy.deepcopy(true)
    meta = {"omitted": [], "retraction": None}
    S, f = [], sp.filler
    ident, m, s, v = said["identification"], said["mechanism"], said["signs"], said["signs"]["vitals"]

    # repair target chosen up front so the sentence that mentions it can stutter
    cand = [("signs.vitals.systolic_bp_mmhg", v["systolic_bp_mmhg"], 90), ("signs.vitals.heart_rate_bpm", v["heart_rate_bpm"], 120),
            ("signs.vitals.resp_rate_per_min", v["resp_rate_per_min"], 29), ("signs.vitals.spo2_percent", v["spo2_percent"], 90),
            ("identification.age_years", ident["age_years"], None)]
    rep = rng.choice(cand) if repair else None

    def maybe_repair(path, value, render_fn, threshold=None):
        if not rep or rep[0] != path:
            return render_fn(value)
        # wrong value: half the time on the other side of the decision threshold
        if threshold and rng.random() < 0.5:
            wrong = threshold - rng.randint(2, 8) if value >= threshold else threshold + rng.randint(2, 8)
        else:
            wrong = value + rng.choice([-20, -10, 10, 20]) if path != "identification.age_years" else value + rng.choice([-10, 10])
        wrong = max(1, wrong)
        crosses = threshold is not None and ((wrong < threshold) != (value < threshold))
        said["retractions"].append({"field": path, "retracted_value": str(wrong), "corrected_value": str(value)})
        meta["retraction"] = {"field": path, "crosses_threshold": crosses}
        return f"{render_fn(wrong)}, {rng.choice(['uh, correction', 'sorry', 'I mean', 'scratch that'])}, {render_fn(value)}"

    unit = f"{rng.choice(UNITS)} {rng.randint(2, 60)}"
    S.append(f"{rng.choice(HOSP)}, {unit}, {rng.choice(['how copy?', 'come in.', 'do you copy?'])}")
    eta = ident["eta_minutes"]
    if rng.random() < 0.85:
        S.append(f"We're inbound to you, ETA {sp.num(eta, 'count')} minutes.")
    else:
        ident["eta_minutes"] = None
        meta["omitted"].append("eta")
    if ident["arrival_mode"] == "ems_interfacility":
        S.append("This is an interfacility transfer from an outlying hospital.")

    # I - identification
    age = maybe_repair("identification.age_years", ident["age_years"], lambda a: sp.num(a, "count"))
    S.append(f"{f()}We have a {age}-year-old {ident['sex']}.")
    if ident["pregnant"]:
        if ident["gestation_weeks"] and rng.random() < 0.7:
            S.append(f"She's about {sp.num(ident['gestation_weeks'], 'count')} weeks pregnant.")
        else:
            ident["gestation_weeks"] = None
            S.append("She is pregnant.")
    elif ident["sex"] == "female":
        ident["pregnant"] = None

    # M - mechanism
    mt = m["type"]
    fall, vh, loc = m["fall"], m["vehicle"], m["loss_of_consciousness"]
    if mt == "fall":
        if fall["fall_type"] == "same_level":
            S.append(rng.choice(["Ground-level fall.", "Tripped and fell from standing.", "Same-level fall."]))
        else:
            S.append(f"Fell approximately {sp.num(fall['height_ft'], 'count')} feet.")
    elif vh is not None:
        bits = []
        lead = {"motor_vehicle_crash": "Driver in a motor vehicle crash", "pedestrian_struck": "Pedestrian struck by a car",
                "bicyclist_struck": "Cyclist struck by a vehicle", "motorcycle_crash": "Motorcycle crash",
                "atv_crash": "ATV crash"}[mt]
        if vh["impact_speed_mph"] is not None:
            bits.append(f"estimated {sp.num(vh['impact_speed_mph'], 'count')} miles an hour")
        if vh["patient_thrown"]:
            bits.append("patient was thrown")
        if vh["run_over"]:
            bits.append("patient was run over")
        if vh["rider_separated"]:
            bits.append("rider separated from the bike")
        if vh["rollover"]:
            bits.append("rollover")
        if vh["ejection"] in ("partial", "complete"):
            bits.append(f"{vh['ejection']} ejection")
        if vh["extrication_minutes"]:
            bits.append(f"extrication took about {sp.num(vh['extrication_minutes'], 'count')} minutes")
        if vh["intrusion_occupant_site_in"]:
            bits.append(f"about {sp.num(vh['intrusion_occupant_site_in'], 'count')} inches of intrusion into the passenger compartment")
        if vh["intrusion_any_site_in"]:
            bits.append(f"{sp.num(vh['intrusion_any_site_in'], 'count')} inches of intrusion to the front end")
        if vh["death_same_compartment"]:
            bits.append("there was a fatality in the same vehicle")
        S.append(lead + (", " + ", ".join(bits) if bits else "") + ".")
        # unsaid false/none values become None
        for k in ("patient_thrown", "run_over", "rider_separated", "rollover", "death_same_compartment"):
            if not vh[k]:
                vh[k] = None
        if vh["ejection"] == "none":
            vh["ejection"] = None
    else:
        S.append({"assault_blunt": "Assaulted, struck multiple times.", "other_blunt": "Struck by a falling object.",
                  "stab": "Stabbing.", "gunshot": "Gunshot wound.", "other_penetrating": "Impaled on a sharp object.",
                  "burn": "House fire, burn injuries.", "electrical": "Electrical injury, contact with a live line.",
                  "crush": "Crush injury, caught in machinery.", "near_drowning": "Near drowning, pulled from a pool.",
                  "near_hanging": "Near hanging, found by family."}[mt])
    if loc["occurred"]:
        if loc["duration_minutes"] is not None and rng.random() < 0.9:
            S.append(f"{f()}Loss of consciousness for about {sp.num(loc['duration_minutes'], 'count')} minutes.")
        else:
            loc["duration_minutes"] = None
            S.append("Positive loss of consciousness.")
    elif rng.random() < 0.4:
        S.append("No loss of consciousness.")
    else:
        loc["occurred"] = None
        meta["omitted"].append("loss_of_consciousness")

    # I - injuries
    inj_text = []
    for i in said["injuries"]:
        inj_text.append(injury_phrase(i, sp, rng, i))
        if i["type"] != "fracture" or i.get("suspected") is not True:
            i["suspected"] = None
    burns = said["burns"]
    if burns:
        inj_text.append(f"burns to roughly {sp.num(burns['tbsa_percent'], 'count')} percent body surface area")
        if burns["inhalation_injury"]:
            inj_text.append("singed nasal hair, suspect inhalation injury")
        elif rng.random() < 0.3:
            inj_text.append("no signs of inhalation injury")
        else:
            burns["inhalation_injury"] = None
        if not burns["electrical"]:
            burns["electrical"] = None
    vitals_first = rng.random() < 0.15

    def injuries_block():
        if inj_text:
            S.append(f"{f()}Injuries: " + "; ".join(inj_text) + ".")
        elif rng.random() < 0.5:
            S.append("No obvious injuries.")
            said["injuries"] = []
        else:
            said["injuries"] = None
            meta["omitted"].append("injuries")

    # S - signs
    def signs_block():
        parts = []
        target = rep[0] if rep else ""
        sbp_said = True
        if rng.random() < 0.95 or "systolic" in target:
            bp = maybe_repair("signs.vitals.systolic_bp_mmhg", v["systolic_bp_mmhg"], lambda x: sp.num(x), 90)
            if rng.random() < 0.85:
                parts.append(f"BP {bp} over {sp.num(v['diastolic_bp_mmhg'])}")
            else:
                parts.append(f"BP {bp}")
                v["diastolic_bp_mmhg"] = None
        else:
            sbp_said = False
            v["systolic_bp_mmhg"] = v["diastolic_bp_mmhg"] = None
            meta["omitted"].append("bp")
        for key, label_, thr, word in (("heart_rate_bpm", "heart rate", 120, "signs.vitals.heart_rate_bpm"),
                                       ("resp_rate_per_min", "resps", 29, "signs.vitals.resp_rate_per_min")):
            if rng.random() < 0.92 or key in target:
                parts.append(f"{label_} {maybe_repair(word, v[key], lambda x: sp.num(x), thr)}")
            else:
                v[key] = None
                meta["omitted"].append(key)
        if rng.random() < 0.9 or "spo2" in target:
            sat = maybe_repair("signs.vitals.spo2_percent", v["spo2_percent"], lambda x: sp.num(x), 90)
            ra = "on room air" if v["spo2_on_room_air"] else "on oxygen"
            parts.append(f"sats {sat} percent {ra}")
        else:
            v["spo2_percent"] = v["spo2_on_room_air"] = None
            meta["omitted"].append("spo2")
        S.append(f"{f()}Vitals: " + ", ".join(parts) + ".")

        g = s["gcs"]
        if rng.random() < 0.95:
            if rng.random() < 0.3:
                S.append(f"GCS {sp.num(g['total'], 'count')}, that's eyes {sp.num(g['eye'], 'count')}, "
                         f"verbal {sp.num(g['verbal'], 'count')}, motor {sp.num(g['motor'], 'count')}.")
            else:
                g["eye"] = g["verbal"] = g["motor"] = None
                S.append(f"GCS {sp.num(g['total'], 'count')}" + (" prior to intubation." if g["before_intubation"] else "."))
        else:
            s["gcs"] = None
            meta["omitted"].append("gcs")

        if s["airway"] == "compromised":
            S.append("Airway is compromised, gurgling, we're struggling to maintain it.")
        elif s["airway"] == "patent" and rng.random() < 0.3:
            S.append("Airway is patent.")
        elif s["airway"] == "patent":
            s["airway"] = None
        if s["flail_chest"]:
            S.append("Paradoxical chest wall movement, suspect a flail segment.")
        else:
            s["flail_chest"] = None
        if s["pneumothorax_or_hemothorax"]:
            S.append("Decreased breath sounds on one side, suspect pneumothorax.")
        elif rng.random() < 0.15:
            S.append("Breath sounds equal bilaterally.")
        else:
            s["pneumothorax_or_hemothorax"] = None
        if s["shock_signs"]:
            S.append("Skin is " + " and ".join(SHOCK_SAY[x] for x in s["shock_signs"]) + ".")
            if rng.random() < 0.5:
                S.append("He's showing signs of shock." if ident["sex"] == "male" else "She's showing signs of shock.")
            else:
                s["clinical_signs_of_shock"] = None
        else:
            s["clinical_signs_of_shock"] = None
        if s["hemorrhage"] == "active_uncontrolled":
            S.append("Active bleeding we can't control.")
        elif s["hemorrhage"] == "controlled":
            S.append("Bleeding is controlled.")
        elif rng.random() < 0.3:
            S.append("No active bleeding.")
        else:
            s["hemorrhage"] = None
        neuro_pos = s["paralysis"] or s["bilateral_sensory_loss"] or s["suspected_spinal_cord_injury"]
        if neuro_pos:
            if s["paralysis"]:
                S.append("Unable to move the legs.")
            if s["bilateral_sensory_loss"]:
                S.append("No sensation bilaterally below the chest.")
            if s["suspected_spinal_cord_injury"]:
                S.append("Suspect spinal cord injury.")
            for k in ("paralysis", "bilateral_sensory_loss"):
                if not s[k]:
                    s[k] = None
        elif rng.random() < 0.25:
            S.append("Moving all four extremities, sensation intact.")
            s["suspected_spinal_cord_injury"] = None
        else:
            s["paralysis"] = s["bilateral_sensory_loss"] = s["suspected_spinal_cord_injury"] = None
        if s["abdominal_pain"] or s["abdomen_firm_or_distended"]:
            if s["abdominal_pain"]:
                S.append("Complaining of abdominal pain.")
            else:
                s["abdominal_pain"] = None
            if s["abdomen_firm_or_distended"]:
                S.append("Abdomen is firm and distended.")
            else:
                s["abdomen_firm_or_distended"] = None
        elif rng.random() < 0.2:
            S.append("Abdomen soft, non-tender.")
        else:
            s["abdominal_pain"] = s["abdomen_firm_or_distended"] = None

    if vitals_first:
        signs_block()
        injuries_block()
    else:
        injuries_block()
        signs_block()

    # T - treatment
    said_tx = [t for t in said["treatment"]["interventions"] if rng.random() < 0.9]
    said["treatment"]["interventions"] = said_tx
    if said_tx:
        S.append(f"{f()}Treatment: " + ", ".join(TX_SAY[t] for t in said_tx) + ".")
    if said["treatment"]["blood_transfusion_required_to_maintain_vitals"]:
        S.append("Still needing blood to maintain a pressure.")
    else:
        said["treatment"]["blood_transfusion_required_to_maintain_vitals"] = None
    if rng.random() < 0.5:
        S.append({"stable": "Stable en route.", "improving": "Improving en route.",
                  "deteriorating": "Deteriorating en route."}[s["trend"]])
    else:
        s["trend"] = None

    # AMBO - omitted together about half the time (AMBO adherence 47.1%, Elliott et al. 2023)
    if rng.random() < 0.47:
        if said["allergies"]:
            S.append("Allergic to " + " and ".join(said["allergies"]) + ".")
        else:
            S.append("No known allergies.")
        if said["medications"]:
            S.append("Takes " + ", ".join(said["medications"]) + ".")
        elif rng.random() < 0.5:
            S.append("No regular medications.")
        else:
            said["medications"] = None
        if said["background"]:
            S.append("History of " + ", ".join(said["background"]) + ".")
        elif rng.random() < 0.5:
            S.append("No significant medical history.")
        else:
            said["background"] = None
    else:
        said["allergies"] = said["medications"] = said["background"] = None
        meta["omitted"].append("AMBO")
    S.append(rng.choice(["Any questions?", "How copy?", "Over.", "Any orders?"]))

    # never describe the mechanism in free text the generator did not say
    m["description"] = None
    return " ".join(x for x in S if x), said, meta


# =============================================================== main
def build_case(rng, target, out_of_scope):
    for _ in range(80):
        c = base(rng)
        borderline = SCENARIOS[target](c, rng)
        if target[1] in "23" or target == "NONE":
            if rng.random() < 0.2:
                near_miss(c, rng)
        fired = label(c)
        want = 0 if target == "NONE" else int(target[1])
        if level(fired) == want and (target == "NONE" or target in fired):
            if out_of_scope:
                c["identification"]["age_years"] = rng.choice([rng.randint(66, 92), rng.randint(12, 14)])
                if c["identification"]["age_years"] < 15:
                    c["identification"]["pregnant"] = False if c["identification"]["sex"] == "female" else None
                    c["identification"]["gestation_weeks"] = None
                fired = label(c)
            HandoverRecord.model_validate(c)
            return c, fired, borderline
    raise RuntimeError(f"could not build a clean case for {target}")


def main(n=3000, seed=20261010, out=None, split_mode="devtest", id_prefix="H"):
    """split_mode="devtest" gives the 50/50 dev/test split; "train" labels every row train."""
    rng = random.Random(seed)
    # level shares from the data-size plan: I 35%, II 30%, III 20%, none 10%, out of scope 5%
    targets = []
    for lvl, share in (("L1", 0.35), ("L2", 0.30), ("L3", 0.20)):
        crits = sorted(k for k in SCENARIOS if k.startswith(lvl))
        k = round(n * share)
        targets += [crits[i % len(crits)] for i in range(k)]
    targets += ["NONE"] * round(n * 0.10)
    oos = n - len(targets)
    targets += [rng.choice(sorted(SCENARIOS)) for _ in range(oos)]
    flags = [False] * (n - oos) + [True] * oos
    order = list(range(n))
    rng.shuffle(order)

    rows, stats = [], Counter()
    for idx, j in enumerate(order):
        target, oos_flag = targets[j], flags[j]
        true, fired, borderline = build_case(rng, target, oos_flag)
        text, said, meta = render(true, rng, repair=rng.random() < 0.2)
        HandoverRecord.model_validate(said)
        lvl = level(fired)
        said_fired = label(_fill(said))
        rows.append({
            "case_id": f"{id_prefix}{idx + 1:05d}",
            "split": "",
            "target_criterion": target,
            "in_scope": not oos_flag,
            "true_level": "out_of_scope" if oos_flag else lvl,
            "true_criteria": ";".join(fired),
            "said_level_naive": level(said_fired),
            "borderline": borderline,
            "has_retraction": meta["retraction"] is not None,
            "retraction_field": meta["retraction"]["field"] if meta["retraction"] else "",
            "retraction_crosses_threshold": bool(meta["retraction"] and meta["retraction"]["crosses_threshold"]),
            "omitted": ";".join(meta["omitted"]),
            "text": text,
            "said_record_json": json.dumps(said, separators=(",", ":")),
            "true_record_json": json.dumps(true, separators=(",", ":")),
            "generator": VERSION,
            "seed": seed,
        })
        stats[str(rows[-1]["true_level"])] += 1

    # split 50/50 stratified by target criterion
    by_target = {}
    for r_ in rows:
        by_target.setdefault(r_["target_criterion"] + str(r_["in_scope"]), []).append(r_)
    for group in by_target.values():
        rng.shuffle(group)
        for i, r_ in enumerate(group):
            r_["split"] = ("test" if i % 2 == 0 else "dev") if split_mode == "devtest" else "train"

    out = out or ROOT / "examples" / "synthetic_handovers_v0.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out, rows, stats


def _fill(said):
    """For the naive said-level: treat 'not mentioned' as absent/normal (what a careless system would do)."""
    c = copy.deepcopy(said)
    s = c["signs"]
    for k in ("flail_chest", "pneumothorax_or_hemothorax", "clinical_signs_of_shock", "paralysis",
              "bilateral_sensory_loss", "suspected_spinal_cord_injury", "abdominal_pain", "abdomen_firm_or_distended"):
        s[k] = bool(s[k])
    s["gcs"] = s["gcs"] or {"total": None}
    c["injuries"] = c["injuries"] or []
    c["mechanism"]["fall"] = c["mechanism"]["fall"] or {}
    c["mechanism"]["vehicle"] = c["mechanism"]["vehicle"] or {}
    c["mechanism"]["loss_of_consciousness"] = c["mechanism"]["loss_of_consciousness"] or {}
    return c


if __name__ == "__main__":
    out, rows, stats = main()
    print("wrote", out, len(rows), "rows")
