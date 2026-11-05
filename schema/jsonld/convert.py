"""Turn a validated HandoverRecord into its JSON-LD view.

The extraction model never sees JSON-LD: it decodes the plain schema, which keeps
the grammar small. This function adds the semantics afterwards.
"""
from __future__ import annotations

import enum
import json
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from ..handover import HandoverRecord
from .bindings import STATED_LIST_FIELDS

CONTEXT_FILE = pathlib.Path(__file__).resolve().parent / "context.jsonld"


def _node(obj):
    if isinstance(obj, BaseModel):
        out = {"@type": f"r2r:{type(obj).__name__}"}
        for name in type(obj).model_fields:
            value = getattr(obj, name)
            if value is None:
                continue                      # not mentioned: no triple
            if isinstance(value, list) and not value:
                continue                      # empty lists are carried by <field>Stated
            out[name] = _node(value)
        return out
    if isinstance(obj, list):
        return [_node(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def to_jsonld(record: HandoverRecord, record_id: Optional[str] = None,
              context: str | dict = "inline", recorded_at: Optional[datetime] = None) -> dict:
    """context='inline' embeds the context; pass a URL string to reference a published one."""
    doc = _node(record)
    doc.pop("schema_version", None)
    for f in STATED_LIST_FIELDS:
        doc[f"{f}Stated"] = getattr(record, f) is not None
    if context == "inline":
        context = json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))["@context"]
    stamp = (recorded_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    return {"@context": context, "@id": record_id or f"urn:uuid:{uuid.uuid4()}",
            **doc, "recordedAt": stamp}
