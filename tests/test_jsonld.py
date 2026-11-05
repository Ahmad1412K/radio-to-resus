import json
import pathlib
import sys

import pytest
from pyld import jsonld

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from schema.jsonld.build import build_context, build_vocab  # noqa: E402
from schema.jsonld.convert import to_jsonld  # noqa: E402
from schema.jsonld.bindings import STATED_LIST_FIELDS  # noqa: E402

EXAMPLES = sorted((ROOT / "examples").glob("*.json"))
NS = "https://w3id.org/radio-to-resus/vocab#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def load(p):
    return HandoverRecord.model_validate_json(p.read_text(encoding="utf-8"))


def leaves(obj):
    """Count the values a record actually states (None and empty lists state nothing)."""
    if isinstance(obj, dict):
        return sum(leaves(v) for k, v in obj.items() if k != "schema_version")
    if isinstance(obj, list):
        return sum(leaves(v) for v in obj)
    return 0 if obj is None else 1


def triples(doc):
    nq = jsonld.to_rdf(doc, {"format": "application/n-quads"})
    return [l for l in nq.strip().splitlines() if l]


def test_generated_files_are_current():
    ctx = json.loads((ROOT / "schema/jsonld/context.jsonld").read_text(encoding="utf-8"))
    voc = json.loads((ROOT / "schema/jsonld/vocab.jsonld").read_text(encoding="utf-8"))
    assert ctx == build_context(), "run: python -m schema.jsonld.build"
    assert voc == build_vocab(), "run: python -m schema.jsonld.build"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_no_value_lost_in_rdf(path):
    rec = load(path)
    doc = to_jsonld(rec, record_id="urn:test:1")
    data = [t for t in triples(doc)
            if RDF_TYPE not in t and " _:" not in t.split(" ", 2)[2][:3] and "recordedAt" not in t
            and "Stated>" not in t]
    # each nested section is linked by a triple whose object is a blank node; drop those
    data = [t for t in data if not t.split(" ", 2)[2].startswith("_:")]
    assert len(data) == leaves(rec.model_dump()), "a field was dropped or duplicated in JSON-LD"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_stated_flags_keep_empty_lists(path):
    rec = load(path)
    nq = "\n".join(triples(to_jsonld(rec, record_id="urn:test:1")))
    for f in STATED_LIST_FIELDS:
        expected = "true" if getattr(rec, f) is not None else "false"
        assert f'<{NS}HandoverRecord.{f}Stated> "{expected}"' in nq


def test_enum_values_become_distinct_iris():
    rec = HandoverRecord.model_validate({
        "mechanism": {"type": "motor_vehicle_crash", "vehicle": {"ejection": "none"}},
        "signs": {"hemorrhage": "none"},
    })
    nq = "\n".join(triples(to_jsonld(rec, record_id="urn:test:1")))
    assert f"<{NS}Ejection.none>" in nq
    assert f"<{NS}Hemorrhage.none>" in nq
    assert f"<{NS}MechanismType.motor_vehicle_crash>" in nq


def test_vitals_carry_loinc_and_units_in_vocab():
    voc = {n["@id"]: n for n in build_vocab()["@graph"]}
    sbp = voc["r2r:Vitals.systolic_bp_mmhg"]
    assert sbp["skos:exactMatch"]["@id"] == "loinc:8480-6/"
    assert sbp["r2r:ucumUnit"] == "mm[Hg]"
    assert voc["r2r:BodyRegion.chest"]["r2r:anatomicZone"]["@id"] == "r2r:AnatomicZone.tshirt_boxer"
    assert voc["r2r:Bone.femur"]["r2r:countsAsLongBone"] is True
