# Plan: small language model (SLM) extraction step

Status: draft, 2026-10-27
Hardware: laptop RTX 3050 Ti, 4 GB VRAM. That limit drives most of the choices below.

## 1. Task

- **Input:** one handover transcript.
- **Output:** the `said_record_json`, which records only what the crew actually said.
  - Anything not mentioned must be `null`, never a default "normal" value.
  - Corrected values go in the record, and the correction is logged in `retractions`.
- The existing rule engine then maps that record to an activation level.
  - The SLM only extracts; it never decides triage.
  - This keeps the safety logic deterministic and auditable, which is a strong design point for the thesis.
- Size: about 1,200 characters (~400 tokens) of output and 150–400 tokens of input. That fits 1–4B models.

## 2. Prerequisites

| # | Item | Status |
|---|---|---|
| 1 | Training split | Missing. Data has only dev (1,482) and test (1,518). |
| 2 | Freeze the test set | Decide now that v2 test is used once, at the end. |
| 3 | Compact output schema for the prompt | The full JSON Schema is long. Use a short version with field names and allowed values. |
| 4 | Scoring script | Not written. Needs field-level and downstream triage metrics. |
| 5 | Full rule engine | `triage/sketch.py` covers only part of the 37 criteria. Triage metrics depend on it. |
| 6 | Inference stack | Ollama or llama.cpp locally, both with JSON-schema-constrained output. |
| 7 | Fine-tuning compute | 4 GB is too small above about 1B parameters. Use Kaggle (free T4 GPUs, ~30 h/week), Colab, or a university GPU, with Unsloth or TRL + PEFT doing QLoRA. |

### Training data

- Generate 5–10k **template** texts with the existing generator.
  - No LLM is involved, so they cost nothing and avoid the circularity problem.
- Train on the template texts and test on the Claude-written v2 texts.
  - This also measures whether the model generalizes to a different writing style.
- v1 can be an extra training source, as long as v2 test stays untouched.

## 3. Candidate models

All candidates are open-weight and non-Claude, which preserves the circularity rule.

| Group | Models | Runs locally? | Notes |
|---|---|---|---|
| Tiny (≤1.7B) | Qwen3-0.6B / 1.7B, Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B, Gemma-3-1B | Yes, fast | Fine-tuning candidates; can almost be trained locally. |
| Small (3–4B), main group | Qwen3-4B, Qwen2.5-3B, Llama-3.2-3B, Phi-4-mini (3.8B), Gemma-3-4B, SmolLM3-3B | Yes at 4-bit (~2.5 GB) | Likely the best balance of accuracy and deployability. |
| Medical | MedGemma-4B | Yes at 4-bit | Tests whether medical pretraining helps. |
| Extraction-tuned | NuExtract (0.5–4B, built on Qwen/Phi) | Yes | Built for text-to-JSON. Strong zero-shot baseline. |
| Upper reference (7–8B) | Qwen3-8B, Llama-3.1-8B, Ministral-8B, Granite-8B | Slow (part runs on CPU) | Measures what extra size buys. Run on Kaggle. |
| Non-generative baseline | Regex/rules; DeBERTa-v3 or ClinicalBERT slot filling; GLiNER | Yes | Floor to beat. These will struggle with corrections and unsaid fields. |

Notes:
- Check Hugging Face for the newest release in each family before locking the list.
- Licences: Qwen, SmolLM and Granite are Apache-2.0; Phi is MIT; Llama and Gemma have custom licences. Cite them in the thesis.

## 4. Experiments

1. **E0, regex baseline:** about half a day of work; establishes the floor.
2. **E1, zero-shot:** about 8 models on **dev**, with schema-constrained decoding. Keep the top 3–4.
3. **E2, few-shot and prompt tuning:**
   - 2–3 examples taken from dev.
   - Explicit rules: unsaid fields → `null`, how to handle corrections, spelled-out numbers → digits.
4. **E3, QLoRA fine-tuning:** the best 1–2 small models, trained on the template split.
5. **E4, final test:** run once on v2 test plus the 3 real transcripts.
6. **E5 (optional), speech-recognition robustness:** convert some test texts to speech, run them back through Whisper, then extract. This tests resilience to transcription errors without new recordings.

## 5. Metrics

- **Validity:** percentage of outputs that parse and pass schema validation.
- **Field level:**
  - Precision, recall and F1 on non-null fields.
  - **Hallucination rate**, reported separately: a value filled in where nothing was said. Clinically this is the error that matters most.
- **Corrections:** for cases with a correction, the percentage where the final value is right and the correction is logged.
- **Triage level** (after the rule engine):
  - Accuracy.
  - **Under-triage rate**, the headline safety metric.
  - Over-triage rate.
- **Breakdowns:** by style (fillers, disorganized, Q&A), length band, borderline cases and out-of-scope cases.
- **Deployability:** time per handover and memory use on the 4 GB laptop GPU.

## 6. Open decisions

1. Is fine-tuning in scope, or only zero/few-shot? Fine-tuning needs Kaggle, Colab or university GPUs and roughly 2–3 extra weeks.
2. Is a university GPU available? If so, the 7–8B group becomes cheap to include.

## 7. Next steps

1. Generate the template training split.
2. Write the compact output schema.
3. Write the scoring script.
4. Run E0 (regex baseline) and E1 (zero-shot with Ollama) on dev.
