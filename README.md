# Radio-to-Resus

**From ambulance radio handover to trauma-team activation, using a small language model and a transparent rule engine.**

MSc Data Science thesis, Princess Sumaya University for Technology (PSUT). Author: Ahmad Ali.

> ### 👉 Start here: [`notebooks/radio_to_resus_walkthrough.ipynb`](notebooks/radio_to_resus_walkthrough.ipynb)
>
> **Everything is in the notebook.** It walks through the whole project in order: the problem, the schema and the 37 criteria, the data, the rule engine, the prompt, scoring, the baseline, and every experiment run so far with its results. It runs the real project code at each step and prints the source of each function it uses, with file and line numbers. It is saved **with all outputs**, so it can be read directly on GitHub without installing anything.

> ### ⚠️ The real radio recordings are not uploaded
>
> The real ambulance radio recordings (audio) and their Whisper transcripts are **not in this repository**. They are real-world audio that is not ours to redistribute, and they are part of the frozen test set. They stay on the author's machine and are excluded in [`.gitignore`](.gitignore). Their SHA-256 hashes are kept in [`examples/splits/test_freeze.json`](examples/splits/test_freeze.json), so any change to them can be detected.
>
> In the notebook, cells that need them print only word counts, and skip cleanly when the files are absent. What *is* included is the **synthetic** data (see [Data](#data)).

---

## The problem and the idea

A paramedic radios the hospital before arrival: the *pre-notification*, or *handover*. The emergency department has to decide quickly whether to activate the trauma team, and at which level (I, II or III). The speech is fast, informal, often incomplete, and sometimes corrects itself ("BP ninety… correction, seventy-nine").

```
 Radio audio ──► Whisper ──► transcript ──► SLM extractor ──► "said" record (JSON)
                                                                   │
                                                                   ▼
                                            Rule engine (37 criteria, three-valued logic)
                                                                   │
                                     ┌─────────────────────────────┼──────────────────────┐
                                     ▼                             ▼                      ▼
                              activation level            "could be" level       questions for crew
```

1. **Speech → text:** Whisper (faster-whisper).
2. **Text → structured record:** a **small, open-weight, non-Claude language model**, small enough to run locally. It fills only what was said; anything not mentioned stays `null` ("not said").
3. **Record → activation level:** a transparent **rule engine** that implements the 37 published activation criteria with three-valued logic (true / false / unknown).
4. **What was not said:** when information is missing, the engine gives a **provisional** decision, the level it *could* reach, and the questions to ask the crew. This guards against **silent under-triage**.

The language model only extracts. The triage decision is always made by deterministic, auditable rules.

---

## Status and results so far

| Stage | Status |
|---|---|
| Record schema + 37 criteria | done |
| Synthetic data: templates (v0) → Claude rewrites (v1, v2), v2 label audit and repair | done |
| Real recordings → Whisper (3 recordings) | done (local only) |
| Train set (10,000 template cases) and frozen test set (1,518 cases) | done |
| Three-valued rule engine | done |
| Prompt schema, constrained decoding, output repair, scorer, inference harness | done |
| **E0** regex baseline | done |
| **E2** zero-shot screen (12 models, 200 dev cases) | done |
| **E2** full dev (4 finalists, 1,482 dev cases) and error analysis | done |
| **E3–E9**: few-shot / prompt ablations, fine-tuning, quantisation, final test, ASR robustness, deployment | planned (shown as labelled **mock-ups** at the end of the notebook) |

**E2 full dev: zero-shot, 1,482 dev cases.** All models run locally through Ollama, with 4-bit weights, on an RTX 3050 Ti laptop GPU (4 GB).

| Model | Field F1 | Triage accuracy | Under-triage | Over-triage | Silent under-triage | Hallucination | s / case |
|---|---|---|---|---|---|---|---|
| **Qwen3 4B** | **0.832** | **84.9%** | 7.9% | 7.1% | 0 | 4.5% | 43.8 |
| Qwen2.5 3B | 0.667 | 74.2% | 16.0% | 9.5% | 0 | 10.4% | 11.4 |
| Llama 3.2 3B | 0.653 | 70.1% | 13.0% | 14.0% | 0 | 4.7% | 25.3 |
| Qwen3 1.7B | 0.627 | 51.4% | 3.2% | 45.1% | 0 | 14.7% | 7.4 |
| *E0 regex baseline* | *0.990* | *95.3%* | *4.7%* | *0%* | | *0%* | |

- **The regex baseline is optimistic.** Its word lists come from the same generator as the data. The notebook (section 10b) shows where it breaks on real speech.
- **No model under-triaged silently.** Silent under-triage is 0 for every model: whenever a model missed a finding, the engine flagged the decision as provisional and asked a question.
- **The errors are systematic.** The error analysis (`scripts/error_analysis.py`) finds that most of Qwen3 4B's errors come from a few repeated habits:
  - a nested item is dropped when only part of it was said ("GCS fifteen", "no loss of consciousness", crash speed);
  - digit amputations are recorded on the hand or foot;
  - "normal" values are written when nobody said them;
  - empty records are returned for calls that start with the vital signs.

  Four of these patterns explain 73% of its under-triage. They set the targets for E3.

The full 12-model screen, the run history and every finding are in the notebook and in [`docs/pipeline_status.txt`](docs/pipeline_status.txt).

---

## Where to find things

| What | Where |
|---|---|
| **The full walkthrough, with results** | [`notebooks/radio_to_resus_walkthrough.ipynb`](notebooks/radio_to_resus_walkthrough.ipynb) |
| Script that builds and runs the notebook | [`notebooks/build_walkthrough.py`](notebooks/build_walkthrough.py) |
| Short status of every step, with results and run history | [`docs/pipeline_status.txt`](docs/pipeline_status.txt) |
| Experiment plan (E0–E9) | [`docs/slm_plan_full.md`](docs/slm_plan_full.md) (full) and [`docs/slm_plan.md`](docs/slm_plan.md) (laptop-limited) |
| Record schema, allowed values, the 37 criteria | [`schema/handover.py`](schema/handover.py), [`schema/vocab.py`](schema/vocab.py), [`schema/criteria.yaml`](schema/criteria.yaml) |
| Rule engine | [`triage/engine.py`](triage/engine.py) |
| Data generator (templates; Claude rewrite tooling; reference labeller) | [`generator/`](generator/) |
| Prompt, constrained decoding, inference, repair, scoring, regex baseline | [`extract/`](extract/) |
| Model sweep, error analysis, train/test split, test freeze, Whisper | [`scripts/`](scripts/) |
| Tests (5,960) | [`tests/`](tests/) |
| Model outputs and score reports | [`runs/`](runs/): `*_dev_k0.jsonl` (raw outputs), `*.report.json` (scores), `sweep_dev_k0*.csv` (tables), `error_analysis/` |

---

## Data

| File | What it is | In the repo? |
|---|---|---|
| `examples/synthetic_handovers_train_v0.csv` | 10,000 **template-only** training cases (no language model involved) | ✅ |
| `examples/synthetic_handovers_v0.csv`, `_v1.csv`, `_v2.csv` | 3,000 dev/test cases: templates (v0), Claude rewrites (v1), and rewrites matched to the real recordings' style (v2, used for evaluation). Dev 1,482, test 1,518 | ✅ |
| `examples/splits/test_freeze.json` | hashes of every test row and of the real recordings | ✅ |
| `examples/splits/v2_label_repairs.csv` | log of the 381 label repairs to v2 | ✅ |
| `examples/*.json`, `examples/jsonld/` | hand-made example records | ✅ |
| `examples/*.flac`, `*.m4a`, `*.mp4` | **real radio recordings** | ❌ not uploaded |
| `examples/transcripts/` | **Whisper transcripts of the real recordings** | ❌ not uploaded |

**About the test set.** The test split (1,518 cases) is **frozen and will be used once**, in E6. Every model choice and every prompt change is made on dev only. Please don't use the test rows for tuning. `python scripts/freeze_test.py --verify` checks that the test set is unchanged.

Each row has the transcript (`text`), the **said record** (what was actually said: the extraction target), the **true record** (the full patient state), and the true activation level.

---

## How to run it (optional)

The notebook can be read on GitHub without running anything. To reproduce the results:

```bash
git clone https://github.com/Ahmad1412K/radio-to-resus.git
cd radio-to-resus
python -m venv .venv
.venv\Scripts\activate          # Windows;  on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                   # 5,960 tests
python scripts/freeze_test.py --verify      # test set unchanged
python notebooks/build_walkthrough.py       # rebuild and execute the notebook
```

**Baseline and scoring (no model needed):**

```bash
python -m extract.regex_baseline --split dev --out runs/e0_regex_v2_dev.jsonl
python -m extract.score runs/e0_regex_v2_dev.jsonl --gold examples/synthetic_handovers_v2.csv --split dev
```

**Small language models.** Install [Ollama](https://ollama.com) first.

```bash
python -m extract.run --backend ollama --model qwen3:4b --split dev --limit 50 --think off
python scripts/sweep.py                                   # E2 screen: 12 models, 200 dev cases each
python scripts/sweep.py --limit 0 --models qwen3:4b       # full dev for one model
python scripts/error_analysis.py runs/qwen3-4b_dev_k0.jsonl
```

Runs are **resumable**: if a run stops, the same command continues from the next case. Python 3.12 was used.

---

## AI-use disclosure

The v1 and v2 synthetic transcripts used for dev and test were written by **Claude** (Anthropic; v2 by Claude Opus 5.5 via Claude Code) from structured facts produced by the template generator. The prompts and outputs are kept in [`generator/llm_state/`](generator/llm_state/). Claude Code was also used as a coding assistant on this repository.

To avoid circularity:
- the extraction model is always a **non-Claude**, open-weight small model;
- it is trained only on **template-only** text;
- it is never trained on Claude-written text.
