# synthetic_handovers_v0.csv

3,000 synthetic pre-hospital radio handovers, each with the true patient record, the record of what the crew actually said, and a trauma activation label. Regenerate with `python generator/generate.py` (seed 20261010) and check with `python generator/audit.py`.

## Columns

| Column | Meaning |
| --- | --- |
| `case_id` | H00001 to H03000 |
| `split` | `dev` (1,482) or `test` (1,518), stratified by target criterion |
| `target_criterion` | The criterion the case was built to trigger (`L1-01` ... `L3-08`, or `NONE`) |
| `in_scope` | False for the 150 patients outside ages 15-64 |
| `true_level` | 1, 2, 3, 0 (no activation) or `out_of_scope`, computed from the true record with all 37 criteria |
| `true_criteria` | Every criterion the true record meets, `;`-separated |
| `said_level_naive` | The level you get from the said record if anything unmentioned is assumed normal |
| `borderline` | Value sampled within a few units of a threshold |
| `has_retraction`, `retraction_field`, `retraction_crosses_threshold` | A planted self-correction, the field, and whether the wrong value sits on the other side of the vital-sign threshold |
| `omitted` | What the speaker left out (AMBO, ETA, a vital sign, GCS, LOC, injuries) |
| `text` | The radio handover |
| `said_record_json` | `HandoverRecord` of what was said: the **extraction target** |
| `true_record_json` | `HandoverRecord` of the patient's true state: the **triage target** |
| `generator`, `seed` | `template-v0`, 20261010 |

## Composition

| | Count |
| --- | --- |
| Level I / II / III / none / out of scope | 1,050 / 900 / 600 / 300 / 150 |
| Test set, in scope / true Level I | 1,437 / 528 |
| Criteria firing in the test set | 37 of 37, at least 33 each |
| Borderline cases | about 15% |
| Planted self-corrections | about 19% (about 9.5% cross a threshold) |
| AMBO section omitted | about 53%, matching the 47.1% AMBO adherence in Elliott et al. (2023) |

## Checks passed

Every said and true record validates against the schema. Every stated age, vital sign, GCS and ETA appears in its text, in digits, words or paramedic style ("one-oh-eight"); a deliberately corrupted value is caught.

## Limits of v0

- Phrasing is template-based, so the text is far more regular than real speech. An LLM paraphrasing pass, from a different model family than the extractor, is the planned next step.
- Case distributions are hand-set, not sampled from MIMIC-IV-ED or NEMSIS. Replace them once data access arrives.
- Labels follow the assumptions in the schema report (t-shirt/boxer boundary, wrist/ankle as distal, shock = stated or 2+ signs, boundary values at exactly 20 mph / 5 min not covered). Changing an assumption means regenerating the labels.
- `retraction_crosses_threshold` refers to the vital-sign threshold alone; whether the level would actually change also depends on the other conditions (e.g. L1-02 also needs signs of shock).
