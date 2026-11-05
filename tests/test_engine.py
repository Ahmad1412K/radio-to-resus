import csv
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.rules_full import label  # noqa: E402
from triage.engine import AND, NOT, OR, RULES, URGENCY, triage  # noqa: E402

csv.field_size_limit(10**8)
ROWS = list(csv.DictReader((ROOT / "examples" / "synthetic_handovers_v0.csv").open(encoding="utf-8")))


def test_kleene():
    assert AND(True, None) is None and AND(False, None) is False
    assert OR(True, None) is True and OR(False, None) is None
    assert NOT(None) is None


def test_all_37_criteria_match_yaml():
    import yaml
    ids = [c["id"] for c in yaml.safe_load((ROOT / "schema" / "criteria.yaml").read_text())["criteria"]]
    assert sorted(ids) == sorted(r.id for r in RULES)


@pytest.mark.parametrize("r", ROWS, ids=lambda r: r["case_id"])
def test_complete_records_match_reference_labeller(r):
    true = json.loads(r["true_record_json"])
    d = triage(true)
    if r["in_scope"] == "True":
        assert sorted(d.fired) == sorted(label(true))
        assert str(d.level) == r["true_level"]
    else:
        assert d.level == "out_of_scope"


@pytest.mark.parametrize("r", [r for r in ROWS if r["in_scope"] == "True"], ids=lambda r: r["case_id"])
def test_true_level_lies_between_certain_and_possible(r):
    d = triage(json.loads(r["said_record_json"]))
    t = URGENCY[int(r["true_level"])]
    assert URGENCY[d.level] <= t <= URGENCY[d.possible_level]
    if URGENCY[d.level] < t:
        assert d.provisional and d.ask


def test_unsaid_is_not_normal():
    d = triage({"identification": {"age_years": 30}})
    assert d.level == 0 and d.possible_level == 1 and d.provisional
    assert "GCS?" in d.ask and "Respiratory rate?" in d.ask


def test_questions_only_above_current_level():
    d = triage({"identification": {"age_years": 30}, "signs": {"vitals": {"resp_rate_per_min": 35}}})
    assert d.level == 1 and d.fired == ["L1-04"] and d.ask == [] and not d.provisional


def test_out_of_scope_and_unknown_age():
    assert triage({"identification": {"age_years": 80}}).level == "out_of_scope"
    assert "Patient's age?" in triage({}).ask
