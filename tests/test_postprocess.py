"""extract/postprocess.repair: fixes only null or drop contradicting details."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from extract.postprocess import repair  # noqa: E402


def test_valid_record_untouched():
    rec, fixes = repair({"identification": {"age_years": 40}})
    assert fixes == [] and rec["identification"]["age_years"] == 40


def test_fracture_fields_on_other_injury():
    rec, fixes = repair({"injuries": [{"type": "pulseless_extremity", "region": "thigh", "bone": "femur",
                                       "digits_involved": 1}]})
    i = rec["injuries"][0]
    assert i["type"] == "pulseless_extremity" and i["bone"] is None and i["digits_involved"] is None
    assert "fracture_fields_on_non_fracture" in fixes


def test_cross_section_conflicts():
    rec, fixes = repair({"signs": {"airway": "patent", "hemorrhage": "none"},
                         "treatment": {"interventions": ["endotracheal_intubation", "tourniquet"]}})
    assert rec["signs"]["airway"] is None and rec["signs"]["hemorrhage"] is None
    assert rec["treatment"]["interventions"] == ["endotracheal_intubation", "tourniquet"]


def test_gcs_and_vitals():
    rec, _ = repair({"signs": {"gcs": {"eye": 4, "verbal": 5, "motor": 5, "total": 15},
                               "vitals": {"systolic_bp_mmhg": 80, "diastolic_bp_mmhg": 95, "spo2_on_room_air": True}}})
    assert rec["signs"]["gcs"]["total"] == 15 and rec["signs"]["gcs"]["eye"] is None
    assert rec["signs"]["vitals"]["diastolic_bp_mmhg"] is None and rec["signs"]["vitals"]["spo2_on_room_air"] is None


def test_never_invents_values():
    raw = {"mechanism": {"type": "assault_blunt", "vehicle": {"impact_speed_mph": 30},
                         "loss_of_consciousness": {"occurred": None, "duration_minutes": 5}}}
    rec, _ = repair(raw)
    assert rec["mechanism"]["vehicle"] is None and rec["mechanism"]["loss_of_consciousness"]["occurred"] is None
    assert rec["mechanism"]["loss_of_consciousness"]["duration_minutes"] is None
