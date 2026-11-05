"""Freeze the held-out test set (plan item P2).

    python scripts/freeze_test.py            # write examples/splits/test_freeze.json
    python scripts/freeze_test.py --verify   # check nothing in the test set has changed
    python scripts/freeze_test.py --reason "why"   # re-freeze after a deliberate change

Re-freezing over an existing manifest that differs requires --reason; every
re-freeze is appended to the manifest's "history" (date, reason, old and new
all-rows hashes), so changes to the test set stay documented.

The manifest records:
  - the test case IDs (shared by v0, v1 and v2, which are the same cases),
  - a SHA-256 per test row and per version (case_id, text, said and true records),
  - a SHA-256 over all test rows per version,
  - SHA-256 hashes of the real recordings and their transcripts.

Dev rows are not frozen: they are for model and prompt selection.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EX = ROOT / "examples"
OUT = EX / "splits" / "test_freeze.json"
VERSIONS = {
    "v0": EX / "synthetic_handovers_v0.csv",
    "v1": EX / "synthetic_handovers_v1.csv",
    "v2": EX / "synthetic_handovers_v2.csv",
}
REAL = ["sample_audio.flac", "sample_audio.m4a", "Test_sample.flac", "Test_sample.m4a", "test.mp4"]
ROW_FIELDS = ("case_id", "text", "said_record_json", "true_record_json")

csv.field_size_limit(10**8)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def row_hash(r: dict) -> str:
    return sha("\x1f".join(r[k] for k in ROW_FIELDS).encode("utf-8"))


def build() -> dict:
    man = {"note": "Held-out test set. Do not tune on these cases. Verify with --verify.",
           "versions": {}, "real_recordings": {}}
    ids_by_version = {}
    for v, path in VERSIONS.items():
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["split"] == "test"]
        rows.sort(key=lambda r: r["case_id"])
        hashes = {r["case_id"]: row_hash(r) for r in rows}
        ids_by_version[v] = list(hashes)
        man["versions"][v] = {
            "file": path.name,
            "n_test": len(rows),
            "all_rows_sha256": sha("".join(hashes.values()).encode()),
            "row_sha256": hashes,
        }
    ids = ids_by_version["v0"]
    assert all(x == ids for x in ids_by_version.values()), "test IDs differ between versions"
    man["test_case_ids"] = ids
    for name in REAL:
        p = EX / name
        if p.exists():
            man["real_recordings"][name] = sha(p.read_bytes())
    for p in sorted((EX / "transcripts").glob("*")):
        man["real_recordings"]["transcripts/" + p.name] = sha(p.read_bytes())
    return man


def main() -> int:
    current = build()
    if "--verify" not in sys.argv:
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
        current["history"] = (old or {}).get("history", [])
        if old and any(old["versions"][v]["all_rows_sha256"] != current["versions"][v]["all_rows_sha256"]
                       for v in current["versions"]) or (old and old["test_case_ids"] != current["test_case_ids"]):
            if "--reason" not in sys.argv:
                print("test set differs from the frozen manifest; re-freeze needs --reason \"...\"")
                return 1
            current["history"].append({
                "date": dt.date.today().isoformat(),
                "reason": sys.argv[sys.argv.index("--reason") + 1],
                "all_rows_sha256": {v: {"old": old["versions"][v]["all_rows_sha256"],
                                        "new": current["versions"][v]["all_rows_sha256"]}
                                    for v in current["versions"]},
            })
        OUT.write_text(json.dumps(current, indent=1), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}: {len(current['test_case_ids'])} test cases, "
              f"{len(current['real_recordings'])} real recording files")
        return 0
    frozen = json.loads(OUT.read_text(encoding="utf-8"))
    ok = frozen["test_case_ids"] == current["test_case_ids"]
    for v, f in frozen["versions"].items():
        same = f["all_rows_sha256"] == current["versions"][v]["all_rows_sha256"]
        print(f"{v}: {'OK' if same else 'CHANGED'}")
        ok &= same
    for k, h in frozen["real_recordings"].items():
        if k not in current["real_recordings"]:          # recordings are kept out of git
            print(f"real recording not present, not checked: {k}")
        elif current["real_recordings"][k] != h:
            print(f"real recording changed: {k}")
            ok = False
    print("test set intact" if ok else "TEST SET CHANGED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
