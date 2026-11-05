"""SUPERSEDED by triage/engine.py (all 37 criteria). Kept as the original prototype.

Workflow
  1. Check scope: the criteria cover adults aged 15-64.
  2. Evaluate every rule. Each rule answers True, False, or None ("can't tell,
     the handover didn't say"). Missing data is never treated as normal.
  3. The activation level is the highest level with at least one True rule.
  4. Any higher-level rule that came back None becomes a question for the crew,
     because the answer could still raise the level. Lower-level gaps are ignored.
  5. Return the level, the rules that fired (the audit trail), and the questions.

Only a representative subset of the 37 criteria is written here, one of each
kind: a vital-sign threshold, a combined condition, a body-zone rule, a count,
a mechanism rule. The real engine would load all of them from criteria.yaml.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from schema.vocab import (  # noqa: E402
    ZONE_OF_REGION, CRITERIA_LONG_BONES, AnatomicZone, Hemorrhage, InjuryType,
    Intervention, MechanismType, VEHICLE_MECHANISMS,
)

Tri = Optional[bool]   # True / False / None = unknown


# ------------------------------------------------------------ three-valued helpers
def lt(x, t) -> Tri:
    return None if x is None else x < t


def gt(x, t) -> Tri:
    return None if x is None else x > t


def AND(*xs: Tri) -> Tri:
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def OR(*xs: Tri) -> Tri:
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def injuries(r: HandoverRecord, pred) -> Tri:
    """True if any injury matches; None if injuries were never mentioned."""
    if r.injuries is None:
        return None
    return any(pred(i) for i in r.injuries)


def shock(r: HandoverRecord) -> Tri:
    # ASSUMPTION to agree with a clinician: stated shock, or 2+ listed shock signs.
    if r.signs.clinical_signs_of_shock is not None:
        return r.signs.clinical_signs_of_shock
    return True if len(r.signs.shock_signs) >= 2 else None


def zone(i) -> AnatomicZone:
    return ZONE_OF_REGION[i.region]


def long_bone_fractures(r: HandoverRecord) -> Optional[int]:
    if r.injuries is None:
        return None
    return sum(1 for i in r.injuries if i.type == InjuryType.FRACTURE and i.bone in CRITERIA_LONG_BONES)


# ------------------------------------------------------------ rules
@dataclass
class Rule:
    id: str
    level: int
    text: str
    check: Callable[[HandoverRecord], Tri]
    ask: str                       # the question to radio back if the answer is unknown


V = lambda r: r.signs.vitals  # noqa: E731

RULES = [
    Rule("L1-02", 1, "SBP < 90 and signs of shock",
         lambda r: AND(lt(V(r).systolic_bp_mmhg, 90), shock(r)), "Blood pressure? Any signs of shock?"),
    Rule("L1-03", 1, "HR > 120 and signs of shock",
         lambda r: AND(gt(V(r).heart_rate_bpm, 120), shock(r)), "Heart rate? Any signs of shock?"),
    Rule("L1-04", 1, "RR < 10 or > 29",
         lambda r: OR(lt(V(r).resp_rate_per_min, 10), gt(V(r).resp_rate_per_min, 29)), "Respiratory rate?"),
    Rule("L1-05", 1, "SpO2 on room air < 90%",
         lambda r: None if V(r).spo2_on_room_air is None else (V(r).spo2_on_room_air and lt(V(r).spo2_percent, 90)),
         "SpO2, and is that on room air?"),
    Rule("L1-01", 1, "GCS < 9 before intubation",
         lambda r: lt(r.signs.gcs.total if r.signs.gcs else None, 9), "GCS?"),
    Rule("L1-07", 1, "Active or uncontrolled hemorrhage / tourniquet",
         lambda r: OR(None if r.signs.hemorrhage is None else r.signs.hemorrhage == Hemorrhage.ACTIVE_UNCONTROLLED,
                      Intervention.TOURNIQUET in r.treatment.interventions),
         "Is the bleeding controlled?"),
    Rule("L1-08", 1, "Penetrating injury to head, neck or t-shirt/boxer area",
         lambda r: injuries(r, lambda i: i.type == InjuryType.PENETRATING_WOUND
                            and zone(i) in (AnatomicZone.HEAD, AnatomicZone.NECK, AnatomicZone.TSHIRT_BOXER)),
         "Any penetrating injuries, and where?"),
    Rule("L1-15", 1, "2 or more long bone fractures (humerus/femur)",
         lambda r: None if long_bone_fractures(r) is None else long_bone_fractures(r) >= 2, "Any long bone fractures?"),

    Rule("L2-01", 2, "GCS 9-13",
         lambda r: None if not r.signs.gcs or r.signs.gcs.total is None else 9 <= r.signs.gcs.total <= 13, "GCS?"),
    Rule("L2-02", 2, "Fall > 20 ft",
         lambda r: False if r.mechanism.type not in (None, MechanismType.FALL)
         else False if r.mechanism.fall and r.mechanism.fall.fall_type and r.mechanism.fall.fall_type.value == "same_level"
         else gt(r.mechanism.fall.height_ft if r.mechanism.fall else None, 20), "Height of the fall?"),
    Rule("L2-04", 2, "Motorcycle/ATV >= 20 mph or rider separation",
         lambda r: False if r.mechanism.type not in (MechanismType.MOTORCYCLE_CRASH, MechanismType.ATV_CRASH)
         else OR(None if not r.mechanism.vehicle or r.mechanism.vehicle.impact_speed_mph is None
                 else r.mechanism.vehicle.impact_speed_mph >= 20,
                 r.mechanism.vehicle.rider_separated if r.mechanism.vehicle else None),
         "Speed, and was the rider separated?"),
    Rule("L2-06", 2, "One proximal long bone fracture",
         lambda r: None if long_bone_fractures(r) is None else long_bone_fractures(r) == 1, "Any long bone fractures?"),
    Rule("L2-07", 2, "Open fracture proximal to wrist or ankle",
         lambda r: injuries(r, lambda i: i.type == InjuryType.FRACTURE and i.open_fracture
                            and zone(i) != AnatomicZone.DISTAL_LIMB), "Is the fracture open?"),

    Rule("L3-01", 3, "Same-level fall at age >= 55, or fall 5-20 ft",
         lambda r: False if r.mechanism.type != MechanismType.FALL or r.mechanism.fall is None
         else OR(AND(r.mechanism.fall.fall_type is not None and r.mechanism.fall.fall_type.value == "same_level",
                     None if r.identification.age_years is None else r.identification.age_years >= 55),
                 AND(None if r.mechanism.fall.height_ft is None else 5 <= r.mechanism.fall.height_ft <= 20)),
         "Height of the fall? Patient's age?"),
    Rule("L3-08", 3, "Near drowning / near hanging",
         lambda r: None if r.mechanism.type is None
         else r.mechanism.type in (MechanismType.NEAR_DROWNING, MechanismType.NEAR_HANGING), "Mechanism?"),
]


# ------------------------------------------------------------ triage
@dataclass
class Decision:
    level: Optional[int]                       # 1, 2, 3, or None = no activation
    fired: list = field(default_factory=list)  # rule ids that came back True (audit trail)
    ask: list = field(default_factory=list)    # questions that could still raise the level
    note: str = ""


def triage(r: HandoverRecord) -> Decision:
    if r.in_adult_scope() is False:
        return Decision(None, note="outside 15-64: use the paediatric or geriatric criteria")

    results = {rule.id: (rule, rule.check(r)) for rule in RULES}
    fired = [rid for rid, (_, res) in results.items() if res is True]
    level = min((results[rid][0].level for rid in fired), default=None)

    # Only unknowns at a HIGHER level than the current one can change the decision.
    ceiling = level if level is not None else 4
    ask = sorted({rule.ask for rule, res in results.values() if res is None and rule.level < ceiling})
    note = "provisional: unknowns could raise the level" if ask else ""
    return Decision(level, fired, ask, note)


if __name__ == "__main__":
    for path in sorted((ROOT / "examples").glob("*.json")):
        rec = HandoverRecord.model_validate_json(path.read_text(encoding="utf-8"))
        d = triage(rec)
        lvl = f"Level {'I' * d.level if d.level != 3 else 'III'}" if d.level else "No activation"
        print(f"\n{path.stem}\n  -> {lvl}   fired: {', '.join(d.fired) or '-'}   {d.note}")
        for q in d.ask:
            print(f"     ask crew: {q}")
