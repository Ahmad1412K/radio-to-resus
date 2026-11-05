"""Write the JSON-LD view of every example record to examples/jsonld/."""
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import HandoverRecord  # noqa: E402
from schema.jsonld.convert import to_jsonld  # noqa: E402

out = ROOT / "examples" / "jsonld"
out.mkdir(exist_ok=True)
stamp = datetime(2026, 10, 8, tzinfo=timezone.utc)
for p in sorted((ROOT / "examples").glob("*.json")):
    rec = HandoverRecord.model_validate_json(p.read_text(encoding="utf-8"))
    doc = to_jsonld(rec, record_id=f"urn:radio-to-resus:example:{p.stem}", recorded_at=stamp)
    (out / f"{p.stem}.jsonld").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", p.stem)
