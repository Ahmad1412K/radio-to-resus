"""Write example handover records (one per activation level, plus a speaker correction)
and export the JSON Schema used for constrained decoding."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402

EXAMPLES = {
    # Level I: gunshot to the chest with hypotension and signs of shock.
    "level1_gunshot_chest": {
        "identification": {"age_years": 27, "sex": "male", "arrival_mode": "ems_scene", "eta_minutes": 8},
        "mechanism": {"type": "gunshot", "description": "single gunshot wound, left chest"},
        "injuries": [{"type": "penetrating_wound", "region": "chest", "side": "left"}],
        "signs": {
            "vitals": {"systolic_bp_mmhg": 82, "heart_rate_bpm": 134, "resp_rate_per_min": 28,
                       "spo2_percent": 91, "spo2_on_room_air": False},
            "gcs": {"eye": 4, "verbal": 5, "motor": 6, "total": 15},
            "airway": "patent",
            "clinical_signs_of_shock": True,
            "shock_signs": ["pale", "cool_clammy"],
            "pneumothorax_or_hemothorax": True,
            "trend": "deteriorating",
        },
        "treatment": {"interventions": ["supplemental_oxygen", "iv_fluids", "wound_packing"]},
        "allergies": None, "medications": None, "background": None,
    },
    # Level II: car occupant, prolonged extrication, one femur fracture.
    "level2_mvc_extrication": {
        "identification": {"age_years": 46, "sex": "female", "arrival_mode": "ems_scene", "eta_minutes": 15},
        "mechanism": {"type": "motor_vehicle_crash",
                      "vehicle": {"extrication_minutes": 25, "intrusion_occupant_site_in": 10,
                                  "ejection": "none"},
                      "loss_of_consciousness": {"occurred": False}},
        "injuries": [{"type": "fracture", "region": "thigh", "side": "right", "bone": "femur",
                      "suspected": True, "open_fracture": False}],
        "signs": {
            "vitals": {"systolic_bp_mmhg": 118, "diastolic_bp_mmhg": 76, "heart_rate_bpm": 104,
                       "resp_rate_per_min": 20, "spo2_percent": 97, "spo2_on_room_air": True},
            "gcs": {"total": 15},
            "airway": "patent",
            "clinical_signs_of_shock": False,
            "trend": "stable",
        },
        "treatment": {"interventions": ["splint", "analgesia", "spinal_motion_restriction"]},
        "allergies": [], "medications": ["lisinopril"], "background": ["hypertension"],
    },
    # Level III: 58-year-old same-level fall; the AMBO half was never mentioned.
    "level3_same_level_fall": {
        "identification": {"age_years": 58, "sex": "female", "arrival_mode": "ems_scene"},
        "mechanism": {"type": "fall", "fall": {"fall_type": "same_level"},
                      "loss_of_consciousness": {"occurred": False}},
        "injuries": [{"type": "fracture", "region": "wrist", "side": "left", "bone": "radius",
                      "suspected": True, "open_fracture": False}],
        "signs": {"vitals": {"systolic_bp_mmhg": 142, "heart_rate_bpm": 88, "resp_rate_per_min": 16},
                  "gcs": {"total": 15}},
        "treatment": {"interventions": ["splint"]},
    },
    # Speaker correction: "BP is ninety - sorry, one-fifty over ninety".
    "retraction_blood_pressure": {
        "identification": {"age_years": 35, "sex": "male", "arrival_mode": "ems_scene"},
        "mechanism": {"type": "assault_blunt",
                      "loss_of_consciousness": {"occurred": True, "duration_minutes": 2}},
        "injuries": [{"type": "laceration", "region": "head"}],
        "signs": {"vitals": {"systolic_bp_mmhg": 150, "diastolic_bp_mmhg": 90, "heart_rate_bpm": 96},
                  "gcs": {"total": 14}},
        "retractions": [{"field": "signs.vitals.systolic_bp_mmhg",
                         "retracted_value": "90", "corrected_value": "150"}],
    },
}

out = ROOT / "examples"
out.mkdir(exist_ok=True)
for name, data in EXAMPLES.items():
    rec = HandoverRecord.model_validate(data)
    (out / f"{name}.json").write_text(rec.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
    print("wrote", name)

schema_path = ROOT / "schema" / "handover.schema.json"
schema_path.write_text(json.dumps(HandoverRecord.model_json_schema(), indent=2), encoding="utf-8")
print("wrote", schema_path.name)
