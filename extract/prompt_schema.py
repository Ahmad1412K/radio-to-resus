"""Compact schema and prompt for the SLM extractor (plan item P3).

Everything is derived from ``schema.HandoverRecord`` so the prompt can never
drift from the schema the rule engine reads.

Three outputs:
  compact_schema()      short, human-style field list for the prompt
  decoding_schema()     JSON Schema for constrained decoding (refs inlined,
                        every key required, so the model always emits the
                        full skeleton and uses null for "not said")
  system_prompt()       rules + compact schema (+ optional few-shot examples)

    python -m extract.prompt_schema            # print sizes and the compact schema
    python -m extract.prompt_schema --write    # also write extract/decoding_schema.json
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402

DROP = {"schema_version"}          # added after extraction, not generated

# Short hints for fields whose meaning is not obvious from the name.
NOTES = {
    "identification.arrival_mode": "interfacility = transfer from another hospital",
    "identification.eta_minutes": "minutes until arrival",
    "mechanism.description": "free text only if the type list does not fit",
    "mechanism.vehicle.intrusion_occupant_site_in": "inches, at the patient's seat",
    "mechanism.vehicle.intrusion_any_site_in": "inches, anywhere in the vehicle",
    "mechanism.vehicle.rider_separated": "motorcycle/ATV rider thrown from the vehicle",
    "mechanism.loss_of_consciousness.occurred": "true / false only if LOC was mentioned",
    "injuries": "null = injuries not mentioned; [] = crew said no injuries",
    "injuries[].suspected": "true when the crew said 'suspected' / 'possible'",
    "injuries[].bone": "fractures only",
    "injuries[].open_fracture": "fractures only",
    "injuries[].depressed": "skull fractures only",
    "injuries[].pelvis_unstable": "pelvic fractures only",
    "injuries[].digits_involved": "amputations of fingers/toes: how many",
    "burns": "null unless burns were mentioned",
    "burns.tbsa_percent": "% total body surface area",
    "signs.vitals.spo2_on_room_air": "true if the crew said 'on room air'",
    "signs.gcs": "null if GCS not mentioned",
    "signs.gcs.before_intubation": "true when the GCS was taken before intubation",
    "signs.shock_signs": "[] if none mentioned",
    "signs.trend": "how the patient is doing en route",
    "treatment.interventions": "[] if none mentioned",
    "treatment.blood_transfusion_required_to_maintain_vitals": "interfacility transfers",
    "allergies": "null = not mentioned; [] = 'no known allergies'",
    "medications": "null = not mentioned; [] = 'no medications'",
    "background": "past medical history; null = not mentioned; [] = 'no history'",
    "retractions": "values the speaker corrected; [] if none",
    "retractions[].field": "dot path, e.g. signs.vitals.systolic_bp_mmhg",
    "retractions[].retracted_value": "the wrong value, digits as a string",
    "retractions[].corrected_value": "the corrected value, digits as a string",
}

RULES = """\
You convert an ambulance radio handover into one JSON object.

Rules:
1. Record only what the speaker said. If something was not mentioned, use null.
   Never fill in a normal or likely value. null means "not said", which is different
   from false or [] ("said to be absent").
2. Numbers are spoken as words or medic style ("one-oh-eight", "one twenty over
   eighty"). Write them as digits: 108, 120 and 80.
3. If the speaker corrects a value ("ninety, correction, one fifty"), put the
   corrected value in the record and add an entry to "retractions" with the field
   path, the wrong value and the corrected value, both as strings.
4. Use only the allowed values listed in the schema. Use lower_snake_case.
5. Ignore radio procedure ("how do you copy", read-backs, sign-offs) and fillers
   ("uh", "um"). A read-back by the hospital is not new information.
6. Output the JSON object only, with every key present, no comments and no extra text.

Conventions (how common phrases are recorded):
- arrival_mode is "ems_interfacility" if the call says it is a transfer from another
  hospital, otherwise "ems_scene" (every call comes from an ambulance).
- spo2_on_room_air is true when an SpO2 is given, unless the speaker says the patient
  is on oxygen (then false).
- "Moving all four, sensation intact": paralysis false and bilateral_sensory_loss false.
- "Abdomen soft" / "no abdominal pain": abdominal_pain false and abdomen_firm_or_distended false.
- "Breath sounds equal" / "no pneumothorax": pneumothorax_or_hemothorax false.
- "No bleeding" / "no active bleeding": hemorrhage "none".
- "Signs of shock": clinical_signs_of_shock true; list the named signs in shock_signs.
- "Positive LOC" without a duration: occurred true, duration_minutes null.
- "No obvious injuries": injuries [].
- Fractures: a thigh fracture is bone "femur", an upper-arm fracture "humerus";
  "unstable pelvis" is a fracture, region "pelvis", bone "pelvis", pelvis_unstable true;
  "possible"/"suspected" sets suspected true; "closed" sets open_fracture false.
- "Stab wound", "gunshot wound" and "penetrating wound" are injury type "penetrating_wound".
- "No pulse" / "pulseless" in a limb is injury type "pulseless_extremity".
"""


# --------------------------------------------------------------- JSON Schema helpers
def _raw_schema() -> dict:
    s = HandoverRecord.model_json_schema()
    for k in DROP:
        s["properties"].pop(k, None)
        if k in s.get("required", []):
            s["required"].remove(k)
    return s


META = ("$defs", "title", "default", "description")


def _inline(node, defs, is_props=False):
    """Inline $refs and drop metadata keys. Inside a "properties" map the keys are
    field names (one of them is literally "description"), so nothing is dropped there."""
    if isinstance(node, dict):
        if not is_props and "$ref" in node:
            target = defs[node["$ref"].split("/")[-1]]
            return _inline(copy.deepcopy(target), defs)
        if is_props:
            return {k: _inline(v, defs) for k, v in node.items()}
        return {k: _inline(v, defs, is_props=(k == "properties"))
                for k, v in node.items() if k not in META}
    if isinstance(node, list):
        return [_inline(x, defs) for x in node]
    return node


def _require_all(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        for v in node.values():
            _require_all(v)
    elif isinstance(node, list):
        for x in node:
            _require_all(x)
    return node


def decoding_schema() -> dict:
    s = _raw_schema()
    out = _inline(s, s.get("$defs", {}))
    return _require_all(out)


# --------------------------------------------------------------- compact rendering
def _type_str(node) -> tuple[str, dict | None]:
    """Return (short type text, nested object node or None)."""
    if "anyOf" in node:
        parts = [p for p in node["anyOf"] if p.get("type") != "null"]
        nullable = len(parts) < len(node["anyOf"])
        if len(parts) == 1:
            t, nested = _type_str(parts[0])
            return (t + " | null" if nullable else t), nested
        return " | ".join(_type_str(p)[0] for p in parts) + (" | null" if nullable else ""), None
    if "enum" in node:
        return " | ".join(json.dumps(v) for v in node["enum"]), None
    if "const" in node:
        return json.dumps(node["const"]), None
    t = node.get("type")
    if t == "object":
        return "object", node
    if t == "array":
        inner, nested = _type_str(node.get("items", {}))
        if nested is not None:
            return "list of objects", nested
        return f"list of ({inner})" if "|" in inner else f"list of {inner}", None
    rng = ""
    if "minimum" in node or "maximum" in node:
        rng = f" {node.get('minimum', '')}-{node.get('maximum', '')}"
    return {"integer": "int", "number": "number", "boolean": "true | false",
            "string": "text"}.get(t, str(t)) + rng, None


def _render(node, path, lines, indent):
    for name, sub in node.get("properties", {}).items():
        p = f"{path}.{name}" if path else name
        t, nested = _type_str(sub)
        note = NOTES.get(p)
        lines.append(f"{'  ' * indent}{name}: {t}" + (f"   # {note}" if note else ""))
        if nested is not None:
            child = p + "[]" if t.startswith("list") else p
            _render(nested, child, lines, indent + 1)


def compact_schema() -> str:
    lines: list[str] = []
    _render(decoding_schema(), "", lines, 0)
    return "\n".join(lines)


def canonical(record: dict) -> dict:
    """Validate a record and return it with every key present (null where unsaid).
    Used for gold labels, few-shot examples and model outputs alike."""
    m = HandoverRecord.model_validate(record).model_dump(mode="json")
    for k in DROP:
        m.pop(k, None)
    return m


def system_prompt(examples: list[tuple[str, dict]] | None = None) -> str:
    parts = [RULES, "Schema (every key must appear; null where not said):", compact_schema()]
    for i, (text, record) in enumerate(examples or [], 1):
        parts += [f"\nExample {i} handover:", text, f"Example {i} JSON:",
                  json.dumps(record, separators=(",", ":"))]
    return "\n".join(parts)


def messages(text: str, examples=None) -> list[dict]:
    return [{"role": "system", "content": system_prompt(examples)},
            {"role": "user", "content": text}]


# --------------------------------------------------------------- CLI
if __name__ == "__main__":
    full = json.dumps(HandoverRecord.model_json_schema())
    dec = decoding_schema()
    comp = compact_schema()
    sp = system_prompt()
    print(compact_schema())
    print("\n--- sizes (characters; ~4 characters per token) ---")
    print(f"full JSON Schema      {len(full):>7,}")
    print(f"compact schema        {len(comp):>7,}")
    print(f"system prompt total   {len(sp):>7,}  (~{len(sp)//4:,} tokens)")
    if "--write" in sys.argv:
        p = ROOT / "extract" / "decoding_schema.json"
        p.write_text(json.dumps(dec, indent=1), encoding="utf-8")
        print("wrote", p.relative_to(ROOT))
