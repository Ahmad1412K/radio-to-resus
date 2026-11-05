import pathlib
import sys
import typing

import pytest
import yaml
from pydantic import BaseModel, ValidationError

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord, Injury, Vitals, GCS  # noqa: E402


# ----------------------------------------------------------- coverage of the criteria
def _resolve(model: type, path: str):
    """Follow a dot path such as 'injuries[].bone' through the Pydantic models."""
    current = model
    for part in path.split("."):
        name = part.removesuffix("[]")
        assert isinstance(current, type) and issubclass(current, BaseModel), f"{path}: '{name}' is not inside a model"
        assert name in current.model_fields, f"{path}: no field '{name}' on {current.__name__}"
        ann = current.model_fields[name].annotation
        while True:  # unwrap Optional[...] / List[...] / Annotated[...]
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            origin = typing.get_origin(ann)
            if origin in (typing.Union, list, typing.Annotated) or str(origin) == "types.UnionType":
                ann = args[0]
                continue
            break
        current = ann
    return current


CRITERIA = yaml.safe_load((ROOT / "schema" / "criteria.yaml").read_text(encoding="utf-8"))


def test_every_criterion_is_mapped():
    levels = [c["level"] for c in CRITERIA["criteria"]]
    assert levels.count(1) == 16 and levels.count(2) == 13 and levels.count(3) == 8


@pytest.mark.parametrize("crit", CRITERIA["criteria"], ids=lambda c: c["id"])
def test_criterion_fields_exist(crit):
    for path in crit["fields"]:
        _resolve(HandoverRecord, path)


# ----------------------------------------------------------- examples
@pytest.mark.parametrize("path", sorted((ROOT / "examples").glob("*.json")), ids=lambda p: p.stem)
def test_examples_validate(path):
    HandoverRecord.model_validate_json(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------- null is not negative
def test_not_mentioned_differs_from_none_known():
    a = HandoverRecord.model_validate({"allergies": None})
    b = HandoverRecord.model_validate({"allergies": []})
    assert a.allergies is None and b.allergies == []


def test_empty_record_is_valid():
    HandoverRecord()  # nothing mentioned is a legal (if useless) handover


# ----------------------------------------------------------- validators reject contradictions
@pytest.mark.parametrize("bad, cls", [
    ({"eye": 4, "verbal": 5, "motor": 6, "total": 12}, GCS),
    ({"spo2_on_room_air": True}, Vitals),
    ({"systolic_bp_mmhg": 80, "diastolic_bp_mmhg": 95}, Vitals),
    ({"type": "laceration", "region": "head", "bone": "skull"}, Injury),
    ({"type": "fracture", "region": "thigh", "bone": "humerus"}, Injury),
    ({"type": "fracture", "region": "thigh", "bone": "femur", "depressed": True}, Injury),
    ({"type": "fracture", "region": "head", "bone": "skull", "pelvis_unstable": True}, Injury),
    ({"type": "penetrating_wound", "region": "chest", "digits_involved": 2}, Injury),
    ({"type": "pulseless_extremity", "region": "chest"}, Injury),
])
def test_contradictions_rejected(bad, cls):
    with pytest.raises(ValidationError):
        cls.model_validate(bad)


@pytest.mark.parametrize("bad", [
    {"unknown_field": 1},
    {"identification": {"sex": "male", "pregnant": True}},
    {"identification": {"gestation_weeks": 20}},
    {"mechanism": {"type": "fall", "fall": {"fall_type": "same_level", "height_ft": 12}}},
    {"mechanism": {"type": "stab", "fall": {"height_ft": 10}}},
    {"mechanism": {"type": "gunshot", "vehicle": {"impact_speed_mph": 30}}},
    {"mechanism": {"loss_of_consciousness": {"occurred": False, "duration_minutes": 3}}},
    {"signs": {"clinical_signs_of_shock": False, "shock_signs": ["pale"]}},
    {"signs": {"airway": "patent"}, "treatment": {"interventions": ["endotracheal_intubation"]}},
    {"signs": {"hemorrhage": "none"}, "treatment": {"interventions": ["tourniquet"]}},
    {"signs": {"vitals": {"heart_rate_bpm": 400}}},
])
def test_record_level_contradictions_rejected(bad):
    with pytest.raises(ValidationError):
        HandoverRecord.model_validate(bad)


def test_adult_scope():
    assert HandoverRecord.model_validate({"identification": {"age_years": 30}}).in_adult_scope() is True
    assert HandoverRecord.model_validate({"identification": {"age_years": 70}}).in_adult_scope() is False
    assert HandoverRecord().in_adult_scope() is None
