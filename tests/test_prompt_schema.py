import csv
import json
import pathlib
import sys

import jsonschema
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract.prompt_schema import canonical, compact_schema, decoding_schema, messages  # noqa: E402

csv.field_size_limit(10**8)


def _paths(node, prefix=""):
    for name, sub in node.get("properties", {}).items():
        p = f"{prefix}.{name}" if prefix else name
        yield p
        for alt in [sub] + sub.get("anyOf", []):
            target = alt.get("items", alt)
            if target.get("type") == "object":
                yield from _paths(target, p)


def test_compact_schema_lists_every_field():
    text = compact_schema()
    for p in _paths(decoding_schema()):
        assert f"{p.split('.')[-1]}:" in text, p


def test_decoding_schema_requires_every_key():
    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                assert set(n["required"]) == set(n["properties"])
                assert n["additionalProperties"] is False
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for x in n:
                walk(x)
    walk(decoding_schema())


@pytest.mark.parametrize("name", ["synthetic_handovers_v2.csv", "synthetic_handovers_train_v0.csv"])
def test_gold_records_fit_decoding_schema(name):
    v = jsonschema.Draft202012Validator(decoding_schema())
    rows = list(csv.DictReader((ROOT / "examples" / name).open(encoding="utf-8")))
    for r in rows[:500]:
        assert not list(v.iter_errors(canonical(json.loads(r["said_record_json"])))), r["case_id"]


def test_messages_shape():
    m = messages("County General, Medic 12...", examples=[("text", {"identification": {"age_years": 40}})])
    assert [x["role"] for x in m] == ["system", "user"]
    assert "Example 1 JSON:" in m[0]["content"]
