import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical  # noqa: E402
from extract.score import compare_case, correction_ok, parse_output, summarise, triage_outcome  # noqa: E402

GOLD = canonical({
    "identification": {"age_years": 40, "sex": "male"},
    "signs": {"vitals": {"systolic_bp_mmhg": 85, "heart_rate_bpm": 130}, "clinical_signs_of_shock": True},
    "allergies": [],
    "retractions": [{"field": "signs.vitals.systolic_bp_mmhg", "retracted_value": "95", "corrected_value": "85"}],
})


def _case(pred):
    fields, items = compare_case(GOLD, pred)
    return {"fields": fields, "items": items, "status": "valid", "exact": pred == GOLD,
            "correction_ok": correction_ok(GOLD, pred), "triage": "correct",
            "silent_under": False, "provisional": False}


def test_parse_handles_fences_and_garbage():
    assert parse_output('```json\n{"identification": {"age_years": 40}}\n```')[1] == "valid"
    assert parse_output("no json here")[1] == "invalid_json"
    rec, status = parse_output('{"identification": {"age_years": 400}}')
    assert status == "repaired" and rec["identification"]["age_years"] is None   # out-of-range leaf nulled
    assert parse_output('{"identification": "x"}')[1] == "invalid_schema"   # a section cannot be repaired


def test_perfect_prediction():
    s = summarise([_case(copy.deepcopy(GOLD))])
    assert s["field_f1"] == 1 and s["hallucination_rate"] == 0 and s["correction_accuracy"] == 1


def test_outcome_types():
    p = copy.deepcopy(GOLD)
    p["signs"]["vitals"]["heart_rate_bpm"] = 100          # wrong
    p["identification"]["sex"] = None                      # missed
    p["identification"]["eta_minutes"] = 10                # hallucinated
    p["allergies"] = None                                  # missed ([] was said)
    f, _ = compare_case(GOLD, p)
    assert f["signs.vitals.heart_rate_bpm"] == "wrong"
    assert f["identification.sex"] == "missed"
    assert f["identification.eta_minutes"] == "hallucinated"
    assert f["allergies"] == "missed"


def test_empty_prediction_gets_no_credit():
    s = summarise([_case(canonical({}))])
    assert s["field_f1"] == 0 and s["hallucination_rate"] == 0


def test_correction_requires_value_and_log():
    p = copy.deepcopy(GOLD)
    p["retractions"] = []
    assert correction_ok(GOLD, p) is False
    p = copy.deepcopy(GOLD)
    p["signs"]["vitals"]["systolic_bp_mmhg"] = 95
    assert correction_ok(GOLD, p) is False


def test_triage_direction():
    assert triage_outcome(1, 3) == "under"
    assert triage_outcome(3, 1) == "over"
    assert triage_outcome(2, 0) == "under"
    assert triage_outcome("out_of_scope", 1) == "scope_error"
