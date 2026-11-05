"""Three-valued triage engine: all 37 criteria (plan item P5).

Replaces the prototype in triage/sketch.py. Works on the records the extractor
produces, where anything the crew did not say is null.

How it decides
--------------
1. Scope: the criteria cover adults 15-64. A stated age outside that range
   gives "out_of_scope". An unstated age is evaluated as adult, with a question.
2. Every criterion returns True, False or None ("can't tell from what was
   said"), using Kleene logic: AND is False if any part is False, OR is True
   if any part is True, otherwise unknown parts make the result None.
   Unsaid findings are never treated as normal.
3. level          = most urgent level with a True criterion (0 = no activation).
   possible_level = most urgent level with a True OR None criterion: the level
                    if every open question came back positive.
   The true level always lies between the two (tested on all generated cases).
4. Questions are generated only for None criteria more urgent than the
   current level, because only those could change the decision.

On complete records (no nulls) the result is identical to the two-valued
labeller in generator/rules_full.py; tests check this on every generated case.

Assumptions (also listed in the schema report; agree them with a clinician)
---------------------------------------------------------------------------
A1 Shock = stated clinical signs of shock, or 2+ listed shock signs.
A2 A stated injury list is taken as complete: injuries not listed are absent.
   (If the crew never mentioned injuries at all, injury criteria are unknown.)
A3 Interventions: a listed intervention was done; an unlisted one is unknown.
A4 Burn criteria are False when the mechanism is known and is not a burn or
   electrical injury and no burns were mentioned.
A5 The handover comes from an ambulance, so an unstated arrival mode counts as
   EMS for the high-risk auto crash rule (L2-03). Interfacility transfer
   (L1-16) still has to be stated.
A6 Wrist/ankle count as distal; upper arm and thigh as proximal limb (zones in
   schema/vocab.py).
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from schema.vocab import ZONE_OF_REGION, AnatomicZone as Z, BodyRegion  # noqa: E402

Tri = Optional[bool]

BLUNT = {"fall", "motor_vehicle_crash", "pedestrian_struck", "bicyclist_struck", "motorcycle_crash",
         "atv_crash", "assault_blunt", "other_blunt", "crush"}
BLUNT_ASSAULT = {"assault_blunt", "other_blunt"}
LONG = {"humerus", "femur"}
DIGITS = {"finger", "toe"}
ADV_AIRWAY = {"endotracheal_intubation", "supraglottic_airway", "surgical_airway"}
URGENCY = {1: 3, 2: 2, 3: 1, 0: 0}


# ------------------------------------------------------------------ Kleene logic
def AND(*xs: Tri) -> Tri:
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def OR(*xs: Tri) -> Tri:
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def NOT(x: Tri) -> Tri:
    return None if x is None else not x


def ANY(xs) -> Tri:
    return OR(*xs) if xs else False


def cmp(x, f) -> Tri:
    return None if x is None else bool(f(x))


def is_(x, *values) -> Tri:
    """x is one of values; None if x is unknown."""
    return None if x is None else x in values


def zone(region) -> Z:
    return ZONE_OF_REGION[BodyRegion(region)]


# ------------------------------------------------------------------ the record view
class View:
    """Read-only helpers over a canonical record dict."""

    def __init__(self, r: dict):
        self.r = r
        self.id = r["identification"]
        self.m = r["mechanism"]
        self.s = r["signs"]
        self.v = r["signs"]["vitals"]
        self.mt = self.m["type"]
        self.fall = self.m["fall"] or {}
        self.veh = self.m["vehicle"] or {}
        self.loc = self.m["loss_of_consciousness"] or {}
        self.inj = r["injuries"]                      # None = never mentioned
        self.tx = set(r["treatment"]["interventions"])
        self.gcs = (self.s["gcs"] or {}).get("total")

    # A3: listed = done, unlisted = unknown
    def done(self, *names) -> Tri:
        return True if self.tx & set(names) else None

    # A2: closed world once a list was given
    def has(self, pred: Callable[[dict], Tri]) -> Tri:
        if self.inj is None:
            return None
        return ANY([pred(i) for i in self.inj])

    def none_except(self, allowed: set) -> Tri:
        """No injuries other than the allowed types."""
        if self.inj is None:
            return None
        return all(i["type"] in allowed for i in self.inj)

    def shock(self) -> Tri:                             # A1
        if self.s["clinical_signs_of_shock"] is True or len(self.s["shock_signs"]) >= 2:
            return True
        if self.s["clinical_signs_of_shock"] is False:
            return False
        return None

    def long_bones(self):
        """(certain count, possible count) of humerus/femur fractures."""
        if self.inj is None:
            return None
        fx = [i for i in self.inj if i["type"] == "fracture"]
        certain = sum(1 for i in fx if i["bone"] in LONG)
        maybe = sum(1 for i in fx if i["bone"] is None and i["region"] in ("upper_arm", "thigh"))
        return certain, certain + maybe

    def loc_minutes(self):
        """'none' if no LOC, a number if LOC with duration, None if unknown."""
        occ = self.loc.get("occurred")
        if occ is False:
            return "none"
        if occ is True:
            return self.loc.get("duration_minutes")
        return None

    def burns(self):
        """Burns dict, {} = known no burns (A4), None = unknown."""
        b = self.r["burns"]
        if b is not None:
            return b
        if self.mt is not None and self.mt not in ("burn", "electrical"):
            return {}
        return None

    def non_burn_injuries(self) -> Tri:
        if self.inj is None:
            return None
        return any(i["type"] != "abrasion" for i in self.inj)


def count_is(rng, f) -> Tri:
    if rng is None:
        return None
    lo, hi = rng
    vals = {f(n) for n in range(lo, hi + 1)}
    return vals.pop() if len(vals) == 1 else None


# ------------------------------------------------------------------ criteria
def _l1_12(x: View) -> Tri:
    b = x.burns()
    if b is None:
        return None
    if not b:
        return False
    tbsa = b.get("tbsa_percent")
    return OR(b.get("inhalation_injury"), AND(cmp(tbsa, lambda t: t > 20), NOT(x.non_burn_injuries())))


def _burn_pct(x: View, f, associated: bool) -> Tri:
    b = x.burns()
    if b is None:
        return None
    if not b:
        return False
    nb = x.non_burn_injuries()
    return AND(cmp(b.get("tbsa_percent"), f), nb if associated else NOT(nb))


def _l2_03(x: View) -> Tri:
    arrival = True if x.id["arrival_mode"] is None else x.id["arrival_mode"] in ("ems_scene", "ems_interfacility")  # A5
    v = x.veh
    return AND(is_(x.mt, "motor_vehicle_crash"), arrival, OR(
        cmp(v.get("extrication_minutes"), lambda t: t > 20),
        cmp(v.get("intrusion_occupant_site_in"), lambda t: t > 12),
        cmp(v.get("intrusion_any_site_in"), lambda t: t > 18),
        is_(v.get("ejection"), "partial", "complete"),
        v.get("death_same_compartment")))


def _l2_04(x: View) -> Tri:
    v, speed = x.veh, x.veh.get("impact_speed_mph")
    return OR(
        AND(is_(x.mt, "pedestrian_struck", "bicyclist_struck"),
            OR(v.get("patient_thrown"), v.get("run_over"), cmp(speed, lambda s: s > 20))),
        AND(is_(x.mt, "motorcycle_crash", "atv_crash"),
            OR(cmp(speed, lambda s: s >= 20), v.get("rider_separated"))),
        AND(is_(x.mt, "atv_crash"), v.get("rollover")))


def _l2_05(x: View) -> Tri:
    lm = x.loc_minutes()
    prolonged = False if lm == "none" else cmp(lm, lambda d: d > 5)
    return AND(is_(x.mt, *BLUNT_ASSAULT), prolonged, cmp(x.gcs, lambda g: g < 13))


def _l3_01(x: View) -> Tri:
    ft, h = x.fall.get("fall_type"), x.fall.get("height_ft")
    return AND(is_(x.mt, "fall"), OR(
        AND(is_(ft, "same_level"), cmp(x.id["age_years"], lambda a: a >= 55)),
        AND(is_(ft, "from_height"), cmp(h, lambda t: 5 <= t <= 20))))


def _l3_02(x: View) -> Tri:
    lm = x.loc_minutes()
    short = True if lm == "none" else cmp(lm, lambda d: d < 5)
    return AND(is_(x.mt, *BLUNT_ASSAULT), short)


def _l3_04(x: View) -> Tri:
    v, speed = x.veh, x.veh.get("impact_speed_mph")
    return OR(
        AND(is_(x.mt, "motor_vehicle_crash"), x.none_except({"abrasion", "contusion"})),
        AND(is_(x.mt, "pedestrian_struck"), cmp(speed, lambda s: s < 20),
            NOT(v.get("patient_thrown")), NOT(v.get("run_over"))),
        AND(is_(x.mt, "motorcycle_crash", "atv_crash"), cmp(speed, lambda s: s < 20), NOT(v.get("rider_separated"))))


@dataclass(frozen=True)
class Rule:
    id: str
    level: int
    text: str
    check: Callable[[View], Tri]
    ask: str


RULES = [
    # --------------------------------------------------------------- Level I
    Rule("L1-01", 1, "GCS < 9 prior to intubation", lambda x: cmp(x.gcs, lambda g: g < 9), "GCS?"),
    Rule("L1-02", 1, "SBP < 90 and signs of shock",
         lambda x: AND(cmp(x.v["systolic_bp_mmhg"], lambda b: b < 90), x.shock()),
         "Blood pressure? Any signs of shock?"),
    Rule("L1-03", 1, "HR > 120 and signs of shock",
         lambda x: AND(cmp(x.v["heart_rate_bpm"], lambda h: h > 120), x.shock()),
         "Heart rate? Any signs of shock?"),
    Rule("L1-04", 1, "RR < 10 or > 29",
         lambda x: cmp(x.v["resp_rate_per_min"], lambda r: r < 10 or r > 29), "Respiratory rate?"),
    Rule("L1-05", 1, "SpO2 on room air < 90%",
         lambda x: AND(x.v["spo2_on_room_air"], cmp(x.v["spo2_percent"], lambda p: p < 90)),
         "SpO2, and is that on room air?"),
    Rule("L1-06", 1, "Airway compromise / advanced airway / flail chest / hemo-pneumothorax",
         lambda x: OR(is_(x.s["airway"], "compromised", "advanced_airway"), x.s["flail_chest"],
                      x.s["pneumothorax_or_hemothorax"], x.done(*ADV_AIRWAY)),
         "Is the airway patent? Any chest injury (flail chest, pneumothorax)?"),
    Rule("L1-07", 1, "Active or uncontrolled hemorrhage / tourniquet",
         lambda x: OR(is_(x.s["hemorrhage"], "active_uncontrolled"), x.done("tourniquet")),
         "Any bleeding, and is it controlled? Tourniquet?"),
    Rule("L1-08", 1, "Penetrating injury to head, neck or t-shirt/boxer area",
         lambda x: x.has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) in (Z.HEAD, Z.NECK, Z.TSHIRT_BOXER)),
         "Any penetrating injuries, and where?"),
    Rule("L1-09", 1, "Pulseless extremity", lambda x: x.has(lambda i: i["type"] == "pulseless_extremity"),
         "Pulses present in all limbs?"),
    Rule("L1-10", 1, "Amputation proximal to wrist or ankle",
         lambda x: x.has(lambda i: i["type"] == "amputation" and zone(i["region"]) == Z.PROXIMAL_LIMB),
         "Any amputation, and where?"),
    Rule("L1-11", 1, "Paralysis / bilateral sensory loss / suspected spinal cord injury",
         lambda x: OR(x.s["paralysis"], x.s["bilateral_sensory_loss"], x.s["suspected_spinal_cord_injury"]),
         "Moving all four limbs, sensation intact?"),
    Rule("L1-12", 1, "Isolated burn > 20% BSA or inhalation injury", _l1_12,
         "Burn size (% BSA)? Any inhalation injury? Other injuries?"),
    Rule("L1-13", 1, "Unstable pelvic fracture",
         lambda x: x.has(lambda i: AND(i["bone"] == "pelvis" if i["bone"] else (None if i["type"] == "fracture" and i["region"] == "pelvis" else False),
                                       i["pelvis_unstable"])),
         "Is the pelvis stable?"),
    Rule("L1-14", 1, "Open or depressed skull fracture",
         lambda x: x.has(lambda i: AND(i["bone"] == "skull" if i["bone"] else (None if i["type"] == "fracture" and i["region"] == "head" else False),
                                       OR(i["open_fracture"], i["depressed"]))),
         "Is the skull fracture open or depressed?"),
    Rule("L1-15", 1, "2 or more long bone fractures (humerus/femur)",
         lambda x: count_is(x.long_bones(), lambda n: n >= 2), "Any long bone fractures?"),
    Rule("L1-16", 1, "Transfer requiring blood to maintain vital signs",
         lambda x: AND(is_(x.id["arrival_mode"], "ems_interfacility"),
                       x.r["treatment"]["blood_transfusion_required_to_maintain_vitals"]),
         "Is this an interfacility transfer, and does the patient need blood to maintain vitals?"),
    # --------------------------------------------------------------- Level II
    Rule("L2-01", 2, "GCS 9-13", lambda x: cmp(x.gcs, lambda g: 9 <= g <= 13), "GCS?"),
    Rule("L2-02", 2, "Fall > 20 ft",
         lambda x: AND(is_(x.mt, "fall"), is_(x.fall.get("fall_type"), "from_height"),
                       cmp(x.fall.get("height_ft"), lambda h: h > 20)),
         "Height of the fall?"),
    Rule("L2-03", 2, "High-risk auto crash", _l2_03,
         "Extrication time? Intrusion? Ejection? Death in the same compartment?"),
    Rule("L2-04", 2, "High-risk pedestrian / cycle / motorcycle / ATV crash", _l2_04,
         "Speed? Was the patient thrown, run over or separated from the vehicle? Rollover?"),
    Rule("L2-05", 2, "Assault/blunt with LOC > 5 min and GCS < 13", _l2_05,
         "Any loss of consciousness, and for how long? GCS?"),
    Rule("L2-06", 2, "One proximal long bone fracture",
         lambda x: count_is(x.long_bones(), lambda n: n == 1), "Any long bone fractures?"),
    Rule("L2-07", 2, "Open fracture proximal to wrist or ankle",
         lambda x: x.has(lambda i: AND(i["type"] == "fracture", i["open_fracture"], i["bone"] != "skull",
                                       zone(i["region"]) in (Z.PROXIMAL_LIMB, Z.TSHIRT_BOXER))),
         "Is the fracture open?"),
    Rule("L2-08", 2, "Penetrating injury between t-shirt/boxer area and wrist/ankle",
         lambda x: x.has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) == Z.PROXIMAL_LIMB),
         "Any penetrating injuries, and where?"),
    Rule("L2-09", 2, "Pregnant with acute abdominal pain",
         lambda x: AND(False if x.id["sex"] == "male" else x.id["pregnant"], x.s["abdominal_pain"]),
         "Is the patient pregnant? Any abdominal pain?"),
    Rule("L2-10", 2, "Degloving / crush / mangled extremity (not digits)",
         lambda x: x.has(lambda i: i["type"] in ("degloving", "crush", "mangled_extremity") and i["region"] not in DIGITS),
         "Any degloving, crush or mangled limb?"),
    Rule("L2-11", 2, "Burns 10-20% BSA with associated injuries",
         lambda x: _burn_pct(x, lambda t: 10 <= t <= 20, associated=True),
         "Burn size (% BSA)? Other injuries?"),
    Rule("L2-12", 2, "Seatbelt sign on chest or abdomen",
         lambda x: x.has(lambda i: i["type"] == "seatbelt_sign" and i["region"] in ("chest", "abdomen")),
         "Any seatbelt sign?"),
    Rule("L2-13", 2, "Blunt abdominal injury with firm/distended abdomen or pain",
         lambda x: AND(is_(x.mt, *BLUNT), OR(x.s["abdomen_firm_or_distended"], x.s["abdominal_pain"])),
         "Abdomen soft? Any abdominal pain?"),
    # --------------------------------------------------------------- Level III
    Rule("L3-01", 3, "Same-level fall at >= 55, or fall 5-20 ft", _l3_01, "Height of the fall? Patient's age?"),
    Rule("L3-02", 3, "Assault/blunt without prolonged LOC", _l3_02, "Any loss of consciousness, and for how long?"),
    Rule("L3-03", 3, "Burns < 10% BSA without other trauma",
         lambda x: _burn_pct(x, lambda t: t < 10, associated=False), "Burn size (% BSA)? Other injuries?"),
    Rule("L3-04", 3, "Low-risk crash", _l3_04, "Speed? Any injuries?"),
    Rule("L3-05", 3, "Amputation of 2+ digits",
         lambda x: x.has(lambda i: AND(i["type"] == "amputation" and i["region"] in DIGITS,
                                       cmp(i["digits_involved"], lambda d: d >= 2))),
         "How many digits are involved?"),
    Rule("L3-06", 3, "Penetrating injury distal to wrist or ankle",
         lambda x: x.has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) == Z.DISTAL_LIMB),
         "Any penetrating injuries, and where?"),
    Rule("L3-07", 3, "Open fracture distal to wrist or ankle",
         lambda x: x.has(lambda i: AND(i["type"] == "fracture", i["open_fracture"], zone(i["region"]) == Z.DISTAL_LIMB)),
         "Is the fracture open?"),
    Rule("L3-08", 3, "Near drowning / near hanging",
         lambda x: is_(x.mt, "near_drowning", "near_hanging"), "Mechanism?"),
]
assert len(RULES) == 37 and len({r.id for r in RULES}) == 37


# ------------------------------------------------------------------ decision
@dataclass
class Decision:
    level: object                          # 1, 2, 3, 0 = no activation, or "out_of_scope"
    possible_level: object                 # level if every unknown came back positive
    fired: list = field(default_factory=list)      # criteria that are True (audit trail)
    unknown: list = field(default_factory=list)    # criteria that are None
    ask: list = field(default_factory=list)        # questions that could raise the level
    notes: list = field(default_factory=list)

    @property
    def provisional(self) -> bool:
        return self.level != self.possible_level


def _canon(record) -> dict:
    if isinstance(record, HandoverRecord):
        return record.model_dump(mode="json")
    return HandoverRecord.model_validate(record).model_dump(mode="json")


def triage(record) -> Decision:
    r = _canon(record)
    age = r["identification"]["age_years"]
    if age is not None and not 15 <= age <= 64:
        return Decision("out_of_scope", "out_of_scope",
                        notes=["outside 15-64: use the paediatric or geriatric criteria"])
    x = View(r)
    res = {rule.id: (rule, rule.check(x)) for rule in RULES}
    fired = [rid for rid, (_, v) in res.items() if v is True]
    unknown = [rid for rid, (_, v) in res.items() if v is None]
    lvl = min((res[i][0].level for i in fired), default=0)
    possible = min((res[i][0].level for i in fired + unknown), default=0)
    ceiling = URGENCY[lvl]
    ask = []
    for rid in unknown:
        rule = res[rid][0]
        if URGENCY[rule.level] > ceiling and rule.ask not in ask:
            ask.append(rule.ask)
    notes = []
    if age is None:
        notes.append("age not stated: adult criteria applied")
        ask.append("Patient's age?")
    return Decision(lvl, possible, fired, unknown, ask, notes)


def level_name(lvl) -> str:
    return {1: "Level I", 2: "Level II", 3: "Level III", 0: "No activation"}.get(lvl, str(lvl))


if __name__ == "__main__":
    for path in sorted((ROOT / "examples").glob("*.json")):
        try:
            d = triage(HandoverRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        extra = f" (could be {level_name(d.possible_level)})" if d.provisional else ""
        print(f"\n{path.stem}\n  -> {level_name(d.level)}{extra}   fired: {', '.join(d.fired) or '-'}")
        for q in d.ask:
            print(f"     ask crew: {q}")
