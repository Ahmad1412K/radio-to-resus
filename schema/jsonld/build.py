"""Generate the JSON-LD context and vocabulary from the Pydantic models.

Both files are derived, never hand-edited, so they cannot drift from the schema:
    python -m schema.jsonld.build
"""
from __future__ import annotations

import enum
import json
import pathlib
import typing

from pydantic import BaseModel

from ..handover import HandoverRecord
from .. import vocab as V
from .bindings import FIELDS_AWAITING_SNOMED, LOINC, PREFIXES, STATED_LIST_FIELDS, UCUM

HERE = pathlib.Path(__file__).resolve().parent
XSD = {bool: "xsd:boolean", int: "xsd:integer", float: "xsd:decimal", str: "xsd:string"}


def inner_type(annotation):
    """Strip Optional[...], List[...] and Annotated[...] to the element type."""
    ann = annotation
    while True:
        origin = typing.get_origin(ann)
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if origin in (typing.Union, list, typing.Annotated) or str(origin) == "types.UnionType":
            ann = args[0]
            continue
        if origin is typing.Literal:
            return str
        return ann


def term(model: type, field: str) -> str:
    return f"r2r:{model.__name__}.{field}"


def enum_term(e: type, member) -> str:
    return f"r2r:{e.__name__}.{member.value}"


def models_and_enums(root=HandoverRecord):
    models, enums, stack = [], [], [root]
    while stack:
        m = stack.pop()
        if m in models:
            continue
        models.append(m)
        for f in m.model_fields.values():
            t = inner_type(f.annotation)
            if isinstance(t, type) and issubclass(t, BaseModel):
                stack.append(t)
            elif isinstance(t, type) and issubclass(t, enum.Enum) and t not in enums:
                enums.append(t)
    return models, enums


# --------------------------------------------------------------------------- context
def scoped_context(model: type) -> dict:
    ctx = {}
    for name, f in model.model_fields.items():
        t = inner_type(f.annotation)
        entry = {"@id": term(model, name)}
        if isinstance(t, type) and issubclass(t, BaseModel):
            entry["@context"] = scoped_context(t)
        elif isinstance(t, type) and issubclass(t, enum.Enum):
            entry["@type"] = "@vocab"
            entry["@context"] = {m.value: enum_term(t, m) for m in t}
        ctx[name] = entry
    return ctx


def build_context() -> dict:
    ctx = {"@version": 1.1, **PREFIXES}
    ctx.update(scoped_context(HandoverRecord))
    for f in STATED_LIST_FIELDS:
        ctx[f"{f}Stated"] = {"@id": f"r2r:HandoverRecord.{f}Stated", "@type": "xsd:boolean"}
    ctx["recordedAt"] = {"@id": "r2r:HandoverRecord.recordedAt", "@type": "xsd:dateTime"}
    return {"@context": ctx}


# --------------------------------------------------------------------------- vocabulary
def label(name: str) -> str:
    return name.replace("_", " ")


def build_vocab() -> dict:
    models, enums = models_and_enums()
    graph = []

    for m in models:
        graph.append({"@id": f"r2r:{m.__name__}", "@type": "rdfs:Class", "rdfs:label": m.__name__,
                      "rdfs:comment": (m.__doc__ or "").strip().split("\n")[0] or m.__name__})
        for name, f in m.model_fields.items():
            key = f"{m.__name__}.{name}"
            t = inner_type(f.annotation)
            node = {"@id": term(m, name), "@type": "rdf:Property", "rdfs:label": label(name),
                    "rdfs:domain": {"@id": f"r2r:{m.__name__}"}}
            if isinstance(t, type) and issubclass(t, BaseModel):
                node["rdfs:range"] = {"@id": f"r2r:{t.__name__}"}
            elif isinstance(t, type) and issubclass(t, enum.Enum):
                node["rdfs:range"] = {"@id": f"r2r:{t.__name__}"}
            elif t in XSD:
                node["rdfs:range"] = {"@id": XSD[t]}
            if f.description:
                node["rdfs:comment"] = f.description
            if key in LOINC:
                code, name_ = LOINC[key]
                node["skos:exactMatch"] = {"@id": f"loinc:{code}/"}
                node["r2r:loincName"] = name_
            if key in UCUM:
                node["r2r:ucumUnit"] = UCUM[key]
            graph.append(node)

    for f in STATED_LIST_FIELDS:
        graph.append({"@id": f"r2r:HandoverRecord.{f}Stated", "@type": "rdf:Property",
                      "rdfs:label": f"{label(f)} stated",
                      "rdfs:comment": f"True when the handover addressed {label(f)}, even if the list is empty.",
                      "rdfs:domain": {"@id": "r2r:HandoverRecord"}, "rdfs:range": {"@id": "xsd:boolean"}})

    for e in enums:
        graph.append({"@id": f"r2r:{e.__name__}", "@type": ["rdfs:Class", "skos:ConceptScheme"],
                      "rdfs:label": e.__name__,
                      **({"r2r:snomedBinding": "pending UMLS licence"} if e.__name__ in FIELDS_AWAITING_SNOMED else {})})
        for m in e:
            node = {"@id": enum_term(e, m), "@type": ["skos:Concept", f"r2r:{e.__name__}"],
                    "skos:prefLabel": label(m.value), "skos:inScheme": {"@id": f"r2r:{e.__name__}"}}
            # Rule-relevant knowledge lives in the vocabulary as data.
            if e is V.BodyRegion:
                node["r2r:anatomicZone"] = {"@id": enum_term(V.AnatomicZone, V.ZONE_OF_REGION[m])}
            if e is V.Bone:
                node["r2r:countsAsLongBone"] = m in V.CRITERIA_LONG_BONES
            graph.append(node)

    zone = V.AnatomicZone
    graph.append({"@id": f"r2r:{zone.__name__}", "@type": ["rdfs:Class", "skos:ConceptScheme"],
                  "rdfs:label": zone.__name__,
                  "rdfs:comment": "Landmarks used by the activation criteria (t-shirt/boxer area, wrist/ankle)."})
    for m in zone:
        graph.append({"@id": enum_term(zone, m), "@type": ["skos:Concept", f"r2r:{zone.__name__}"],
                      "skos:prefLabel": label(m.value), "skos:inScheme": {"@id": f"r2r:{zone.__name__}"}})
    graph.append({"@id": "r2r:anatomicZone", "@type": "rdf:Property", "rdfs:label": "anatomic zone",
                  "rdfs:domain": {"@id": "r2r:BodyRegion"}, "rdfs:range": {"@id": "r2r:AnatomicZone"}})
    graph.append({"@id": "r2r:countsAsLongBone", "@type": "rdf:Property",
                  "rdfs:label": "counts as long bone",
                  "rdfs:comment": "The criteria count only the humerus and femur as long bones.",
                  "rdfs:domain": {"@id": "r2r:Bone"}, "rdfs:range": {"@id": "xsd:boolean"}})
    for p in ("ucumUnit", "loincName"):
        graph.append({"@id": f"r2r:{p}", "@type": "rdf:Property", "rdfs:range": {"@id": "xsd:string"}})

    return {"@context": {**PREFIXES}, "@graph": graph}


def main():
    (HERE / "context.jsonld").write_text(json.dumps(build_context(), indent=2), encoding="utf-8")
    (HERE / "vocab.jsonld").write_text(json.dumps(build_vocab(), indent=2), encoding="utf-8")
    models, enums = models_and_enums()
    print(f"context.jsonld and vocab.jsonld written: {len(models)} classes, {len(enums)} enumerations")


if __name__ == "__main__":
    main()
