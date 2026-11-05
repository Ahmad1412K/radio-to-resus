"""All 37 criteria, two-valued, for labelling TRUE patient records.

True records are complete (every finding is known), so no three-valued logic is
needed here. This is the ground-truth labeller for the generator; the
three-valued engine for extracted records is triage/engine.py.

Assumptions match the schema report (t-shirt/boxer boundary, wrist/ankle = distal,
shock = stated or 2+ signs, boundary values at exactly 20 mph / 5 min not covered).
"""
from schema.vocab import ZONE_OF_REGION, BodyRegion, AnatomicZone as Z

BLUNT = {"fall", "motor_vehicle_crash", "pedestrian_struck", "bicyclist_struck", "motorcycle_crash",
         "atv_crash", "assault_blunt", "other_blunt", "crush"}
LONG = {"humerus", "femur"}
DIGITS = {"finger", "toe"}


def zone(region):
    return ZONE_OF_REGION[BodyRegion(region)]


def label(c: dict) -> list[str]:
    s, v, ident, m = c["signs"], c["signs"]["vitals"], c["identification"], c["mechanism"]
    inj = c["injuries"] or []
    gcs = (s["gcs"] or {}).get("total")
    shock = bool(s["clinical_signs_of_shock"]) or len(s["shock_signs"]) >= 2
    tx = set(c["treatment"]["interventions"])
    mt = m["type"]
    fall, veh, loc = m["fall"] or {}, m["vehicle"] or {}, m["loss_of_consciousness"] or {}
    burns = c["burns"] or {}
    tbsa = burns.get("tbsa_percent")
    non_burn_injuries = [i for i in inj if i["type"] not in ("abrasion",)]

    def has(pred):
        return any(pred(i) for i in inj)

    fractures = [i for i in inj if i["type"] == "fracture"]
    n_long = sum(1 for i in fractures if i.get("bone") in LONG)
    speed = veh.get("impact_speed_mph")
    loc_min = loc.get("duration_minutes") if loc.get("occurred") else None
    fired = []

    def rule(rid, cond):
        if cond:
            fired.append(rid)

    # ------------------------------------------------------------- Level I
    rule("L1-01", gcs is not None and gcs < 9)
    rule("L1-02", v["systolic_bp_mmhg"] is not None and v["systolic_bp_mmhg"] < 90 and shock)
    rule("L1-03", v["heart_rate_bpm"] is not None and v["heart_rate_bpm"] > 120 and shock)
    rule("L1-04", v["resp_rate_per_min"] is not None and (v["resp_rate_per_min"] < 10 or v["resp_rate_per_min"] > 29))
    rule("L1-05", v["spo2_on_room_air"] is True and v["spo2_percent"] is not None and v["spo2_percent"] < 90)
    rule("L1-06", s["airway"] in ("compromised", "advanced_airway") or s["flail_chest"] or s["pneumothorax_or_hemothorax"]
         or bool(tx & {"endotracheal_intubation", "supraglottic_airway", "surgical_airway"}))
    rule("L1-07", s["hemorrhage"] == "active_uncontrolled" or "tourniquet" in tx)
    rule("L1-08", has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) in (Z.HEAD, Z.NECK, Z.TSHIRT_BOXER)))
    rule("L1-09", has(lambda i: i["type"] == "pulseless_extremity"))
    rule("L1-10", has(lambda i: i["type"] == "amputation" and zone(i["region"]) == Z.PROXIMAL_LIMB))
    rule("L1-11", s["paralysis"] or s["bilateral_sensory_loss"] or s["suspected_spinal_cord_injury"])
    rule("L1-12", bool(burns) and (burns.get("inhalation_injury") or
                                   (tbsa is not None and tbsa > 20 and not non_burn_injuries)))
    rule("L1-13", has(lambda i: i.get("bone") == "pelvis" and i.get("pelvis_unstable")))
    rule("L1-14", has(lambda i: i.get("bone") == "skull" and (i.get("open_fracture") or i.get("depressed"))))
    rule("L1-15", n_long >= 2)
    rule("L1-16", ident["arrival_mode"] == "ems_interfacility" and c["treatment"]["blood_transfusion_required_to_maintain_vitals"])

    # ------------------------------------------------------------- Level II
    rule("L2-01", gcs is not None and 9 <= gcs <= 13)
    rule("L2-02", mt == "fall" and fall.get("fall_type") == "from_height" and (fall.get("height_ft") or 0) > 20)
    rule("L2-03", mt == "motor_vehicle_crash" and ident["arrival_mode"] in ("ems_scene", "ems_interfacility") and (
        (veh.get("extrication_minutes") or 0) > 20 or (veh.get("intrusion_occupant_site_in") or 0) > 12
        or (veh.get("intrusion_any_site_in") or 0) > 18 or veh.get("ejection") in ("partial", "complete")
        or veh.get("death_same_compartment")))
    rule("L2-04", (mt in ("pedestrian_struck", "bicyclist_struck") and
                   (veh.get("patient_thrown") or veh.get("run_over") or (speed is not None and speed > 20)))
         or (mt in ("motorcycle_crash", "atv_crash") and ((speed is not None and speed >= 20) or veh.get("rider_separated")))
         or (mt == "atv_crash" and veh.get("rollover")))
    rule("L2-05", mt in ("assault_blunt", "other_blunt") and loc_min is not None and loc_min > 5 and gcs is not None and gcs < 13)
    rule("L2-06", n_long == 1)
    rule("L2-07", has(lambda i: i["type"] == "fracture" and i.get("open_fracture") and i.get("bone") != "skull"
                      and zone(i["region"]) in (Z.PROXIMAL_LIMB, Z.TSHIRT_BOXER)))
    rule("L2-08", has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) == Z.PROXIMAL_LIMB))
    rule("L2-09", ident["pregnant"] is True and s["abdominal_pain"])
    rule("L2-10", has(lambda i: i["type"] in ("degloving", "crush", "mangled_extremity") and i["region"] not in DIGITS))
    rule("L2-11", tbsa is not None and 10 <= tbsa <= 20 and bool(non_burn_injuries))
    rule("L2-12", has(lambda i: i["type"] == "seatbelt_sign" and i["region"] in ("chest", "abdomen")))
    rule("L2-13", mt in BLUNT and (s["abdomen_firm_or_distended"] or s["abdominal_pain"]))

    # ------------------------------------------------------------- Level III
    rule("L3-01", mt == "fall" and ((fall.get("fall_type") == "same_level" and (ident["age_years"] or 0) >= 55)
                                    or (fall.get("fall_type") == "from_height" and 5 <= (fall.get("height_ft") or 0) <= 20)))
    rule("L3-02", mt in ("assault_blunt", "other_blunt") and (not loc.get("occurred") or (loc_min is not None and loc_min < 5)))
    rule("L3-03", tbsa is not None and tbsa < 10 and not non_burn_injuries)
    rule("L3-04", (mt == "motor_vehicle_crash" and not [i for i in inj if i["type"] not in ("abrasion", "contusion")])
         or (mt == "pedestrian_struck" and speed is not None and speed < 20 and not veh.get("patient_thrown") and not veh.get("run_over"))
         or (mt in ("motorcycle_crash", "atv_crash") and speed is not None and speed < 20 and not veh.get("rider_separated")))
    rule("L3-05", has(lambda i: i["type"] == "amputation" and i["region"] in DIGITS and (i.get("digits_involved") or 0) >= 2))
    rule("L3-06", has(lambda i: i["type"] == "penetrating_wound" and zone(i["region"]) == Z.DISTAL_LIMB))
    rule("L3-07", has(lambda i: i["type"] == "fracture" and i.get("open_fracture") and zone(i["region"]) == Z.DISTAL_LIMB))
    rule("L3-08", mt in ("near_drowning", "near_hanging"))
    return fired


def level(fired: list[str]) -> int:
    """1, 2 or 3; 0 = no activation."""
    return min((int(f[1]) for f in fired), default=0)
