import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical  # noqa: E402
from scripts.error_analysis import correction_kind, filled  # noqa: E402

GOLD = canonical({
    "signs": {"vitals": {"systolic_bp_mmhg": 85}},
    "retractions": [{"field": "signs.vitals.systolic_bp_mmhg", "retracted_value": "95", "corrected_value": "85"}],
})
R = GOLD["retractions"][0]


def test_filled_counts_leaves_and_lists():
    assert filled(canonical({})) == 0
    assert filled(GOLD) == 2                      # one vital + one non-empty list


def test_correction_kinds():
    assert correction_kind(GOLD, copy.deepcopy(GOLD), R) == "fully right"
    p = copy.deepcopy(GOLD)
    p["retractions"] = [{**R, "retracted_value": "ninety-five", "corrected_value": "eighty-five"}]
    assert correction_kind(GOLD, p, R) == "value right, log in words"
    p["retractions"] = []
    assert correction_kind(GOLD, p, R) == "value right, not logged"
    p["signs"]["vitals"]["systolic_bp_mmhg"] = 95
    assert correction_kind(GOLD, p, R) == "kept the retracted value"
    p["signs"]["vitals"]["systolic_bp_mmhg"] = None
    assert correction_kind(GOLD, p, R) == "value left null"
    assert correction_kind(GOLD, None, R) == "no usable output"
