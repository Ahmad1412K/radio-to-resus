"""Rewrite the template handovers (v0) as realistic radio speech with Claude Sonnet 5.5.

The facts are fixed: the model receives the AS-SAID record and must say all of it,
and nothing else. Every output is validated against that record before it is
accepted, so the ground truth in v0 stays valid for v1.

Usage (needs ANTHROPIC_API_KEY, or an `ant auth login` profile):
    python generator/llm_paraphrase.py dryrun            # build prompts, estimate cost, no API calls
    python generator/llm_paraphrase.py pilot 10          # 10 synchronous calls, validated, printed
    python generator/llm_paraphrase.py submit --yes      # submit all cases as one Message Batch (50% price)
    python generator/llm_paraphrase.py collect           # wait for the batch, validate, retry failures, write v1 CSV
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import random
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.generate import para, words  # noqa: E402

MODEL = "claude-sonnet-5-5"
EFFORT = "low"                 # a rewriting task; raise to "medium" if the pilot shows quality problems
MAX_TOKENS = 4000
SRC = ROOT / "examples" / "synthetic_handovers_v0.csv"
OUT = ROOT / "examples" / "synthetic_handovers_v1.csv"
STATE = ROOT / "generator" / "llm_state"
STATE.mkdir(exist_ok=True)
csv.field_size_limit(10**8)

SYSTEM = """You write realistic transcripts of paramedic radio pre-notification calls to a hospital emergency department, for a research corpus used to test speech recognition and information extraction.

You are given FACTS: a JSON record of exactly what the paramedic says in this call. Write the call as it would actually sound over the radio.

Hard rules - the transcript is checked automatically against FACTS and rejected if any rule is broken:
1. Say every fact in FACTS. Keep every number exactly as given.
2. Say nothing clinical that is not in FACTS: no extra vital signs, findings, injuries, treatments, medications, allergies, history, or patient names. A field that is absent from FACTS was not mentioned on the radio - do not mention that topic at all, not even to say it is unknown or normal.
3. If FACTS contains "retractions", the paramedic first says the retracted value, then corrects it to the corrected value in the same breath (for example "BP eighty-eight, uh, correction, one-oh-eight over sixty-two"). Say the wrong value first, then the right one.
4. Follow the STYLE notes for wording, pace and number style.
5. Write plain spoken text only: no headings, no markdown, no speaker labels, no stage directions. Short hospital replies ("Go ahead", "Copy") may appear inline as their own sentences, but the hospital never adds clinical facts.
6. A unit call sign and hospital name may be invented; they are the only numbers or names not in FACTS."""

STYLE_POOL = {
    "pace": ["rushed and clipped, under pressure", "calm and methodical", "slightly breathless, working on the patient",
             "experienced and terse, heavy use of radio shorthand", "a little disorganised, adds details as remembered"],
    "numbers": ["paramedic style (one-oh-eight, ninety-five)", "mostly digits as spoken words (one hundred eight)",
                "mix of styles"],
    "structure": ["follows IMIST-AMBO order", "leads with vital signs, then the story", "starts with mechanism, wanders a little"],
    "fillers": ["no fillers", "occasional 'uh' and 'um'", "frequent fillers and self-interruptions"],
    "exchange": ["one continuous report", "hospital says 'go ahead' after the call sign, otherwise one report",
                 "short back-and-forth, hospital asks for the next item"],
}


# ------------------------------------------------------------------ facts and prompts
def strip_nulls(x):
    if isinstance(x, dict):
        out = {k: strip_nulls(v) for k, v in x.items() if v is not None and k != "schema_version"}
        return {k: v for k, v in out.items() if v not in ({},)}
    if isinstance(x, list):
        return [strip_nulls(v) for v in x]
    return x


def facts_for(row) -> dict:
    said = json.loads(row["said_record_json"])
    f = strip_nulls(said)
    if not f.get("retractions"):
        f.pop("retractions", None)
    tr = f.get("treatment", {})
    if not tr.get("interventions"):
        tr.pop("interventions", None)
    if not tr:
        f.pop("treatment", None)
    sg = f.get("signs", {})
    if not sg.get("shock_signs"):
        sg.pop("shock_signs", None)
    return f


def style_for(case_id) -> dict:
    rng = random.Random(int(hashlib.sha256(case_id.encode()).hexdigest(), 16))
    return {k: rng.choice(v) for k, v in STYLE_POOL.items()}


def user_prompt(row) -> str:
    return ("FACTS:\n" + json.dumps(facts_for(row), indent=1) +
            "\n\nSTYLE:\n" + "\n".join(f"- {k}: {v}" for k, v in style_for(row["case_id"]).items()) +
            "\n\nWrite the radio transcript.")


def params(row, feedback: str | None = None) -> dict:
    messages = [{"role": "user", "content": user_prompt(row)}]
    if feedback:
        messages[0]["content"] += ("\n\nA previous attempt was rejected for these reasons; avoid them:\n" + feedback)
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": EFFORT},
        "messages": messages,
    }


# ------------------------------------------------------------------ validation
def forms(n):
    n = int(round(float(n)))
    out = {str(n), para(n), words(n)}
    if 100 <= n < 1000:
        h, r = divmod(n, 100)
        out.add(f"{words(h)} {words(r)}" if r >= 10 else f"{words(h)} oh {words(r)}" if r else out.pop())
        out.add(f"{words(h)} hundred and {words(r)}" if r else f"{words(h)} hundred")
    return {f.lower() for f in out}


def heard(text, value) -> bool:
    t = text.lower().replace("‑", "-")
    for f in forms(value):
        f2 = re.escape(f).replace(r"\-", r"[-\s]")
        if re.search(r"(?<![\w-])" + f2 + r"(?!\w)(?!-(?!year|old|minute|week|foot|feet|inch|percent))", t):
            return True
    return False


# topic -> words that would reveal the topic was mentioned
TOPIC_WORDS = {
    "allergies": r"allerg",
    "medications": r"\bmeds?\b|medication|\btakes\b|\bon (?:warfarin|apixaban|blood thinners)",
    "background": r"history|\bhx\b|past medical|pmh",
    "loss_of_consciousness": r"loss of consciousness|\bloc\b|knocked out|passed out|blacked out|lost consciousness",
    "eta": r"\beta\b|minutes out|out from you|arriv",
    "spo2": r"\bsats?\b|spo2|saturat|pulse ox|\bo2\b",
    "heart_rate_bpm": r"heart rate|\bhr\b|tachy|brady|pulse (?:of|is|rate|at)\b|pulse rate",  # not "weak thready pulse"
    "resp_rate_per_min": r"\bresp|respirat|breathing at|\brr\b",
    "gcs": r"\bgcs\b|glasgow",
    "bp": r"\bbp\b|blood pressure|pressure of|systolic|hypotens",
}


def numeric_facts(said):
    v, i, m = said["signs"]["vitals"], said["identification"], said["mechanism"]
    out = {"age": i["age_years"], "eta": i["eta_minutes"], "gestation": i["gestation_weeks"],
           "sbp": v["systolic_bp_mmhg"], "dbp": v["diastolic_bp_mmhg"], "hr": v["heart_rate_bpm"],
           "rr": v["resp_rate_per_min"], "spo2": v["spo2_percent"], "gcs": (said["signs"]["gcs"] or {}).get("total")}
    if m["fall"]:
        out["fall_height"] = m["fall"]["height_ft"]
    if m["vehicle"]:
        for k in ("impact_speed_mph", "extrication_minutes", "intrusion_occupant_site_in", "intrusion_any_site_in"):
            out[k] = m["vehicle"][k]
    if m["loss_of_consciousness"]:
        out["loc_minutes"] = m["loss_of_consciousness"]["duration_minutes"]
    if said["burns"]:
        out["tbsa"] = said["burns"]["tbsa_percent"]
    return {k: x for k, x in out.items() if x is not None}


def validate(row, text: str) -> list[str]:
    said = json.loads(row["said_record_json"])
    issues = []
    if not text or len(text.split()) < 25:
        return ["transcript too short or empty"]
    if re.search(r"^\s*(#|\*\*|paramedic:|medic:|hospital:|ed:)", text, re.I | re.M):
        issues.append("contains markdown or speaker labels")
    for k, val in numeric_facts(said).items():
        if not heard(text, val):
            issues.append(f"stated value missing: {k}={val}")
    low = text.lower()
    omitted = set(row["omitted"].split(";")) if row["omitted"] else set()
    absent = {"allergies": said["allergies"] is None, "medications": said["medications"] is None,
              "background": said["background"] is None,
              "loss_of_consciousness": "loss_of_consciousness" in omitted, "eta": said["identification"]["eta_minutes"] is None,
              "spo2": said["signs"]["vitals"]["spo2_percent"] is None, "heart_rate_bpm": said["signs"]["vitals"]["heart_rate_bpm"] is None,
              "resp_rate_per_min": said["signs"]["vitals"]["resp_rate_per_min"] is None, "gcs": said["signs"]["gcs"] is None,
              "bp": said["signs"]["vitals"]["systolic_bp_mmhg"] is None}
    for topic, is_absent in absent.items():
        if is_absent and re.search(TOPIC_WORDS[topic], low):
            issues.append(f"mentions a topic that was not said: {topic}")
    for r in said["retractions"]:
        wrong, right = r["retracted_value"], r["corrected_value"]
        pw = [m.start() for f in forms(wrong) for m in re.finditer(r"(?<![\w-])" + re.escape(f) + r"(?!\w)", low)]
        pr = [m.start() for f in forms(right) for m in re.finditer(r"(?<![\w-])" + re.escape(f) + r"(?!\w)", low)]
        if not pw or not pr:
            issues.append(f"self-correction missing ({r['field']}: {wrong} -> {right})")
        elif min(pw) > max(pr):
            issues.append("self-correction in the wrong order")
    return issues


# ------------------------------------------------------------------ API helpers
def client():
    import anthropic
    return anthropic.Anthropic()


def text_of(msg) -> str:
    if msg.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def call_once(cl, row, feedback=None):
    msg = cl.messages.create(**params(row, feedback))
    return text_of(msg), msg.usage


def load_rows():
    return list(csv.DictReader(SRC.open(encoding="utf-8")))


# ------------------------------------------------------------------ commands
def cmd_dryrun():
    rows = load_rows()
    chars_in = sum(len(SYSTEM) + len(user_prompt(r)) for r in rows)
    in_tok = chars_in / 3.6
    out_tok = len(rows) * 450          # transcript ~250 tokens + light thinking at effort=low
    cost = (in_tok * 2.0 + out_tok * 10.0) / 1e6
    print(f"{len(rows)} requests to {MODEL}, effort={EFFORT}")
    print(f"estimated input ~{in_tok / 1e6:.2f}M tokens, output ~{out_tok / 1e6:.2f}M tokens")
    print(f"estimated cost: ~${cost:.0f} at standard price, ~${cost / 2:.0f} through the Batch API (50% off)")
    print("\n--- example prompt (" + rows[0]["case_id"] + ") ---\n" + user_prompt(rows[0]))
    bad = sum(1 for r in rows if validate(r, r["text"]))
    print(f"\nvalidator on the v0 template texts: {len(rows) - bad}/{len(rows)} pass")


def cmd_pilot(n):
    rows, cl = load_rows(), client()
    rng = random.Random(7)
    sample = rng.sample(rows, n)
    ok = 0
    for row in sample:
        text, usage = call_once(cl, row)
        issues = validate(row, text)
        ok += not issues
        print(f"\n=== {row['case_id']}  level={row['true_level']}  {'VALID' if not issues else 'REJECTED: ' + '; '.join(issues)}")
        print(f"    tokens in={usage.input_tokens} out={usage.output_tokens}")
        print(text)
    print(f"\npilot: {ok}/{n} valid")


def cmd_submit(confirm):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if not confirm:
        sys.exit("Run `dryrun` to see the cost estimate, then `submit --yes` to spend it.")
    rows, cl = load_rows(), client()
    batch = cl.messages.batches.create(requests=[
        Request(custom_id=r["case_id"], params=MessageCreateParamsNonStreaming(**params(r))) for r in rows])
    (STATE / "batch_id.txt").write_text(batch.id)
    print(f"submitted batch {batch.id} with {len(rows)} requests")


def cmd_collect():
    cl = client()
    batch_id = (STATE / "batch_id.txt").read_text().strip()
    while True:
        b = cl.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        print(f"{b.processing_status}: {b.request_counts.processing} processing, "
              f"{b.request_counts.succeeded} done")
        time.sleep(60)

    rows = {r["case_id"]: r for r in load_rows()}
    texts = {}
    for res in cl.messages.batches.results(batch_id):
        if res.result.type == "succeeded":
            texts[res.custom_id] = text_of(res.result.message)
    (STATE / "batch_results.json").write_text(json.dumps(texts), encoding="utf-8")

    final = {}
    for cid, row in rows.items():
        text, attempts = texts.get(cid, ""), 1
        issues = validate(row, text) if text else ["no result from batch"]
        while issues and attempts < 3:               # synchronous retries with the rejection reasons
            text, _ = call_once(cl, row, "; ".join(issues))
            issues, attempts = validate(row, text), attempts + 1
        final[cid] = (text, issues, attempts)

    out_rows = []
    for cid, row in rows.items():
        text, issues, attempts = final[cid]
        valid = not issues
        out_rows.append({**row,
                         "text_template": row["text"],
                         "text": text if valid else row["text"],
                         "text_source": "llm" if valid else "template_fallback",
                         "llm_model": MODEL, "llm_effort": EFFORT, "llm_attempts": attempts,
                         "llm_issues": "; ".join(issues), "llm_style": json.dumps(style_for(cid))})
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    n_ok = sum(1 for r in out_rows if r["text_source"] == "llm")
    print(f"wrote {OUT.name}: {n_ok}/{len(out_rows)} LLM texts valid, "
          f"{len(out_rows) - n_ok} fell back to the template text")


# ------------------------------------------------------------------ interactive route (no API)
# Same facts, style briefs, rules and validator, but the transcripts are written by the
# model running Claude Code rather than through the API.
LOCAL_MODEL = "claude-opus-5-5 (Claude Code, interactive)"
PROMPTS = STATE / "prompts"
WRITTEN = STATE / "written"


def compact_facts(row) -> str:
    return json.dumps(facts_for(row), separators=(",", ":"))


def cmd_export(per_file=50):
    """Write the per-case inputs in batches for the interactive route."""
    PROMPTS.mkdir(exist_ok=True)
    WRITTEN.mkdir(exist_ok=True)
    rows = load_rows()
    for start in range(0, len(rows), per_file):
        chunk = rows[start:start + per_file]
        name = PROMPTS / f"batch_{start // per_file + 1:03d}.jsonl"
        with name.open("w", encoding="utf-8") as fh:
            for r in chunk:
                fh.write(json.dumps({"case_id": r["case_id"], "facts": compact_facts(r),
                                     "style": "; ".join(style_for(r["case_id"]).values())}) + "\n")
    print(f"wrote {len(rows) // per_file + (len(rows) % per_file > 0)} prompt files to {PROMPTS}")


def read_written() -> dict:
    texts = {}
    for f in sorted(WRITTEN.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                texts[d["case_id"]] = d["text"]
    return texts


def cmd_check(batch: str):
    """Validate one written batch file and print any rejected cases."""
    rows = {r["case_id"]: r for r in load_rows()}
    f = WRITTEN / f"batch_{int(batch):03d}.jsonl"
    bad = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        issues = validate(rows[d["case_id"]], d["text"])
        if issues:
            bad += 1
            print(d["case_id"], "; ".join(issues))
    print(f"{f.name}: {bad} rejected")


def cmd_ingest():
    """Validate everything written so far and build the v1 CSV."""
    rows, texts = load_rows(), read_written()
    out_rows, n_ok = [], 0
    for row in rows:
        text = texts.get(row["case_id"], "")
        issues = validate(row, text) if text else ["not written yet"]
        valid = not issues
        n_ok += valid
        out_rows.append({**row, "text_template": row["text"], "text": text if valid else row["text"],
                         "text_source": "llm" if valid else "template_fallback",
                         "llm_model": LOCAL_MODEL, "llm_effort": "", "llm_attempts": "",
                         "llm_issues": "; ".join(issues), "llm_style": json.dumps(style_for(row["case_id"]))})
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT.name}: {n_ok}/{len(rows)} LLM texts valid, {len(texts)} written in total")


# ------------------------------------------------------------------ v2: matched to the real recordings
# The three real recordings (examples/transcripts) run 61, 131 and 223 words, in full spoken
# sentences ("Vitals are as follows", "Do you copy?", sign-offs, read-backs). v1 texts ran
# 25-81 words in clipped shorthand. v2 rewrites the same facts to a per-case length band drawn
# to cover that spread. Length comes only from radio phrasing, protocol and hospital read-backs
# of facts already said; the no-new-facts rules and the validator are unchanged.
WRITTEN_V2 = STATE / "written_v2"
OUT_V2 = ROOT / "examples" / "synthetic_handovers_v2.csv"
BANDS = {"short": (55, 90), "medium": (90, 150), "long": (150, 230)}


def band_for(case_id) -> str:
    rng = random.Random(int(hashlib.sha256((case_id + ":len").encode()).hexdigest(), 16))
    x = rng.random()
    return "short" if x < 0.25 else "medium" if x < 0.75 else "long"


def validate_v2(row, text: str) -> list[str]:
    issues = validate(row, text)
    lo, hi = BANDS[band_for(row["case_id"])]
    n = len(text.split())
    if not lo * 0.9 <= n <= hi * 1.1:
        issues.append(f"length {n} outside {band_for(row['case_id'])} band {lo}-{hi}")
    return issues


def cmd_check_v2(batch: str):
    rows = {r["case_id"]: r for r in load_rows()}
    f = WRITTEN_V2 / f"batch_{int(batch):03d}.jsonl"
    bad = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            issues = validate_v2(rows[d["case_id"]], d["text"])
            if issues:
                bad += 1
                print(d["case_id"], "; ".join(issues))
    print(f"{f.name}: {bad} rejected")


def cmd_ingest_v2():
    rows = load_rows()
    texts = {}
    for f in sorted(WRITTEN_V2.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                texts[d["case_id"]] = d["text"]
    out_rows, n_ok = [], 0
    for row in rows:
        text = texts.get(row["case_id"], "")
        issues = validate_v2(row, text) if text else ["not written yet"]
        valid = not issues
        n_ok += valid
        out_rows.append({**row, "text_template": row["text"], "text": text if valid else row["text"],
                         "text_source": "llm" if valid else "template_fallback",
                         "llm_model": LOCAL_MODEL, "llm_effort": "", "llm_attempts": "",
                         "llm_issues": "; ".join(issues), "llm_style": json.dumps(style_for(row["case_id"])),
                         "length_band": band_for(row["case_id"])})
    with OUT_V2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_V2.name}: {n_ok}/{len(rows)} texts valid, {len(texts)} written in total")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dryrun"
    if cmd == "check2":
        cmd_check_v2(sys.argv[2])
        sys.exit()
    if cmd == "ingest2":
        cmd_ingest_v2()
        sys.exit()
    if cmd == "export":
        cmd_export()
        sys.exit()
    if cmd == "check":
        cmd_check(sys.argv[2])
        sys.exit()
    if cmd == "ingest":
        cmd_ingest()
        sys.exit()
    if cmd == "dryrun":
        cmd_dryrun()
    elif cmd == "pilot":
        cmd_pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif cmd == "submit":
        cmd_submit("--yes" in sys.argv)
    elif cmd == "collect":
        cmd_collect()
    else:
        sys.exit(__doc__)
