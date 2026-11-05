"""Deterministic repair of extractor output that breaks the schema's cross-field rules.

Constrained decoding guarantees the JSON *shape* (every key, allowed enum values),
but a JSON grammar cannot express the schema's cross-field validators
(schema/handover.py), e.g. "bone is only valid on a fracture" or "GCS components
must sum to the total". A small model that writes `"bone": "femur"` on a
pulseless-limb injury would otherwise lose the whole record.

repair(obj) mirrors each validator and removes the contradicting detail. It
never invents a value: every fix sets something to null (unknown) or drops it,
so the rule engine treats it as "not said" and asks about it. Anything still
invalid after the targeted fixes has the offending leaf nulled (e.g. a number
outside its allowed range); if even that fails the output stays invalid.

The scorer reports both rates: strictly valid (no repair needed) and usable
after repair, and lists which repairs were applied.
"""
from __future__ import annotations

import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pydantic import ValidationError  # noqa: E402
from schema import HandoverRecord  # noqa: E402
from schema.vocab import (  # noqa: E402
    ADVANCED_AIRWAY_INTERVENTIONS, LIMB_REGIONS, REGION_OF_BONE, VEHICLE_MECHANISMS,
)

_V = lambda xs: {getattr(x, "value", x) for x in xs}  # noqa: E731  enum members -> plain strings
VEHICLE = _V(VEHICLE_MECHANISMS)
ADV_AIRWAY = _V(ADVANCED_AIRWAY_INTERVENTIONS)
LIMBS = _V(LIMB_REGIONS)
BONE_REGIONS = {getattr(b, "value", b): _V(rs) for b, rs in REGION_OF_BONE.items()}
LIMB_ONLY = {"pulseless_extremity", "mangled_extremity", "degloving"}


def _d(x):
    return x if isinstance(x, dict) else {}


def targeted(obj: dict) -> list[str]:
    """Apply the validator-mirroring fixes in place; return the names of the fixes made."""
    fixes = []
    ident, mech, signs = _d(obj.get("identification")), _d(obj.get("mechanism")), _d(obj.get("signs"))
    vit, tx = _d(signs.get("vitals")), _d(obj.get("treatment"))

    def fix(name, cond, action):
        if cond:
            action()
            fixes.append(name)

    # identification
    fix("gestation_without_pregnancy", ident.get("gestation_weeks") is not None and ident.get("pregnant") is not True,
        lambda: ident.update(gestation_weeks=None))
    fix("pregnant_male", ident.get("pregnant") is True and ident.get("sex") == "male",
        lambda: ident.update(pregnant=None, gestation_weeks=None))

    # mechanism
    fall, loc = _d(mech.get("fall")), _d(mech.get("loss_of_consciousness"))
    fix("same_level_fall_height", fall.get("fall_type") == "same_level" and fall.get("height_ft") is not None,
        lambda: fall.update(height_ft=None))
    fix("loc_duration_without_loc", loc.get("duration_minutes") is not None and loc.get("occurred") is not True,
        lambda: loc.update(duration_minutes=None))
    fix("fall_details_wrong_mechanism", mech.get("fall") is not None and mech.get("type") not in (None, "fall"),
        lambda: mech.update(fall=None))
    fix("vehicle_details_wrong_mechanism",
        mech.get("vehicle") is not None and mech.get("type") is not None and mech.get("type") not in VEHICLE,
        lambda: mech.update(vehicle=None))

    # injuries
    keep = []
    for inj in obj.get("injuries") or []:
        if not isinstance(inj, dict):
            continue
        t, bone, region = inj.get("type"), inj.get("bone"), inj.get("region")
        if t != "fracture":
            fix("fracture_fields_on_non_fracture",
                any(inj.get(k) is not None for k in ("bone", "open_fracture", "depressed", "pelvis_unstable")),
                lambda: inj.update(bone=None, open_fracture=None, depressed=None, pelvis_unstable=None))
        fix("depressed_not_skull", inj.get("depressed") is not None and inj.get("bone") != "skull",
            lambda: inj.update(depressed=None))
        fix("unstable_not_pelvis", inj.get("pelvis_unstable") is not None and inj.get("bone") != "pelvis",
            lambda: inj.update(pelvis_unstable=None))
        fix("digits_not_amputation", inj.get("digits_involved") is not None and t != "amputation",
            lambda: inj.update(digits_involved=None))
        bone = inj.get("bone")
        fix("bone_not_in_region", bone in BONE_REGIONS and region not in BONE_REGIONS[bone],
            lambda: inj.update(bone=None))
        if t in LIMB_ONLY and region not in LIMBS:
            fixes.append("limb_injury_off_limb_dropped")
            continue
        keep.append(inj)
    if obj.get("injuries") is not None:
        obj["injuries"] = keep

    # signs
    gcs = _d(signs.get("gcs"))
    parts = [gcs.get(k) for k in ("eye", "verbal", "motor")]
    fix("gcs_components_mismatch",
        all(isinstance(p, int) for p in parts) and isinstance(gcs.get("total"), int) and sum(parts) != gcs["total"],
        lambda: gcs.update(eye=None, verbal=None, motor=None))
    fix("room_air_without_spo2", vit.get("spo2_on_room_air") is not None and vit.get("spo2_percent") is None,
        lambda: vit.update(spo2_on_room_air=None))
    sbp, dbp = vit.get("systolic_bp_mmhg"), vit.get("diastolic_bp_mmhg")
    fix("diastolic_above_systolic", isinstance(sbp, int) and isinstance(dbp, int) and dbp > sbp,
        lambda: vit.update(diastolic_bp_mmhg=None))
    fix("shock_signs_but_shock_false", bool(signs.get("shock_signs")) and signs.get("clinical_signs_of_shock") is False,
        lambda: signs.update(clinical_signs_of_shock=None))

    # cross-section
    given = set(tx.get("interventions") or [])
    fix("advanced_airway_status_conflict",
        bool(given & ADV_AIRWAY) and signs.get("airway") not in (None, "advanced_airway"),
        lambda: signs.update(airway=None))
    fix("tourniquet_with_no_hemorrhage", "tourniquet" in given and signs.get("hemorrhage") == "none",
        lambda: signs.update(hemorrhage=None))
    burns = _d(obj.get("burns"))
    fix("electrical_burn_wrong_mechanism",
        burns.get("electrical") and mech.get("type") not in (None, "electrical", "burn"),
        lambda: burns.update(electrical=None))
    return fixes


def _null_at(obj, loc) -> bool:
    """Set the value at a pydantic error location to None. Only leaves (not whole sections)."""
    cur = obj
    for k in loc[:-1]:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return False
    last = loc[-1]
    try:
        if isinstance(cur[last], (dict, list)):
            return False
    except (KeyError, IndexError, TypeError):
        return False
    cur[last] = None
    return True


def repair(obj: dict):
    """Return (validated record dict or None, list of fixes applied)."""
    obj = copy.deepcopy(obj)
    obj.pop("schema_version", None)
    try:
        return HandoverRecord.model_validate(obj).model_dump(mode="json"), []
    except ValidationError:
        pass
    fixes = targeted(obj)
    for _ in range(10):                                  # leftovers: out-of-range numbers, bad strings
        try:
            return HandoverRecord.model_validate(obj).model_dump(mode="json"), fixes
        except ValidationError as e:
            changed = False
            for err in e.errors():
                if err["type"] != "value_error" and _null_at(obj, err["loc"]):
                    fixes.append(f"nulled:{'.'.join(map(str, err['loc']))}")
                    changed = True
            if not changed:
                return None, fixes
    return None, fixes
