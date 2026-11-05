# Plan: SLM extraction step (no hardware limits)

Status: draft, 2026-10-27
Companion to `slm_plan.md`. That plan was shaped by the 4 GB laptop GPU. This one assumes we can rent or borrow whatever compute a sensible thesis budget allows. For each item it lists what is needed: data, code, GPU memory, time and approximate cost.

Hardware and cost figures are rough planning estimates. Rental prices change, so check current prices before booking.

---

## 1. Task (unchanged)

- **Input:** handover transcript.
- **Output:** `said_record_json`.
  - Unsaid fields are `null`.
  - Corrected values go in the record; the correction is logged in `retractions`.
- **Downstream:** the deterministic rule engine maps the record to an activation level. The model never decides triage.
- **Size per example:** about 150–400 input tokens and about 400 output tokens.
- **Constraint:** the extractor must not be a Claude model, because Claude generated the texts (circularity).

---

## 2. Shared prerequisites

| # | Item | What is needed | Effort |
|---|---|---|---|
| P1 | Training split | **Done (2026-10-27).** `examples/synthetic_handovers_train_v0.csv`: 10k template texts (no LLM), seed 20261027, IDs T00001–T10000, no overlap with dev/test. **Decision: templates only** — no Claude-written v1/v2 versions of the training set, so testing on v2 measures generalization to unseen phrasing. An open-model paraphrase of part of the training set (non-Claude) may be added later as a separate experiment. | Done |
| P2 status | Frozen test set | **Done (2026-10-27); re-frozen the same day after the v2 label repair (381 said-record fixes, log `examples/splits/v2_label_repairs.csv`; reason stored in the manifest history).** `examples/splits/test_freeze.json`; check with `python scripts/freeze_test.py --verify`. | Done |
| P2 | Frozen test set | v2 test (1,518) plus the 3 real transcripts. Record hashes and do not open until E6. | 1 hour |
| P3 status | Compact prompt schema | **Done (2026-10-27).** `extract/prompt_schema.py` (generated from the Pydantic model): compact schema 5.2k characters vs 12.8k for the full JSON Schema; system prompt ≈1,550 tokens; `extract/decoding_schema.json` for constrained decoding (all keys required); all 13,000 gold records validate against it. | Done |
| P4 status | Scoring script | **Done (2026-10-27).** `python -m extract.score PRED.jsonl --split dev`. Triage metrics use the interim two-valued labeller until P5. | Done |
| P5 status | Full rule engine | **Done (2026-10-27).** `triage/engine.py`: all 37 criteria, three-valued. Identical to `generator/rules_full.py` on all 13,000 complete records; on all 12,352 in-scope said records the true level lies between the certain and worst-case level (0 violations) and every under-triaged case is flagged provisional. Scorer now uses it and reports silent under-triage. Assumptions A1–A6 in the module docstring need clinician sign-off. | Done |
| P3 | Compact prompt schema | A short field list with allowed values, generated from `schema/vocab.py` so the two stay in sync. | 0.5 day |
| P4 | Scoring script | Parse and validate output; field precision, recall and F1; hallucination rate; correction accuracy; breakdowns by style, length band and borderline; triage metrics. | 2–3 days |
| P5 | Full rule engine | Finish `triage/sketch.py` for all 37 criteria from `criteria.yaml`. Unit test against `true_level` in the dataset. | 3–5 days |
| P6 status | Inference harness | **Done (2026-11-01).** `extract/run.py` (Ollama / OpenAI-compatible / gold echo; constrained JSON; resumable; latency and tokens logged) and `scripts/sweep.py`. Ollama 0.40.13 on the laptop. Output capped at 1,500 tokens (`--max-tokens`, added 2026-11-01 after runaway generations hit the 600 s timeout); sweep gate uses mean latency. Added `extract/postprocess.py`: JSON-schema decoding cannot enforce the schema's cross-field rules, so outputs that break one are repaired by nulling the contradicting detail; the scorer reports strict-valid and usable rates separately. | Done |
| P6 | Inference harness | One script driving vLLM (GPU server) or Ollama (local) with JSON-schema-constrained decoding through xgrammar or Outlines. Logs latency, tokens and memory. | 1–2 days |
| P7 | Training harness | Unsloth or TRL + PEFT, configured for LoRA, QLoRA and full fine-tuning. Fixed seeds; experiment tracking in Weights & Biases or MLflow. | 1–2 days |
| P8 | Compute account | One of the options in section 6. | Setup 0.5 day |

---

## 3. Candidate models and what each needs

### 3.1 How memory is estimated

- **Inference, weights only:**
  - bf16 ≈ 2 GB per billion parameters.
  - 8-bit ≈ 1.1 GB per billion.
  - 4-bit ≈ 0.6 GB per billion.
  - Add about 1–2 GB for cache and runtime at our short context lengths.
- **QLoRA training** (4-bit base + LoRA adapters): about 1–1.5 GB per billion parameters + 4–6 GB.
- **LoRA training** (bf16 base): about 2.5–3 GB per billion parameters.
- **Full fine-tuning** (AdamW, mixed precision): about 16–20 GB per billion parameters before memory-saving tricks such as DeepSpeed ZeRO.

### 3.2 Model groups

| Group | Example models | Params | Inference VRAM (bf16 / 4-bit) | QLoRA VRAM | Full fine-tune VRAM | Minimum GPU |
|---|---|---|---|---|---|---|
| Tiny | Qwen3-0.6B / 1.7B, Llama-3.2-1B, Gemma-3-1B, SmolLM2-1.7B | 0.6–1.7B | 2–4 GB / ≤1.5 GB | 5–7 GB | 12–35 GB | Any; full fine-tuning on one 40–80 GB GPU |
| Small | Qwen3-4B, Llama-3.2-3B, Phi-4-mini (3.8B), Gemma-3-4B, SmolLM3-3B, MedGemma-4B, NuExtract (≤4B) | 3–4B | 7–9 GB / 2.5–3 GB | 8–10 GB | 60–80 GB | 16 GB T4/L4 for QLoRA; 80 GB A100/H100 for full fine-tuning |
| Medium | Qwen3-8B, Llama-3.1-8B, Ministral-8B, Granite-8B | 7–8B | 16 GB / 5–6 GB | 12–16 GB | 130–160 GB (multi-GPU) | 24 GB for QLoRA/LoRA; 2–4 × 80 GB for full fine-tuning |
| Large-small | Qwen3-14B, Phi-4 (14B), Gemma-3-12B | 12–14B | 24–28 GB / 8–9 GB | 18–24 GB | Not planned | 24–48 GB |
| Upper SLM | Gemma-3-27B, MedGemma-27B (text), Mistral-Small-24B, Qwen3-32B, Qwen3-30B-A3B (MoE), gpt-oss-20b (MoE) | 20–32B | 40–64 GB / 13–20 GB | 28–40 GB | Not planned | 48–80 GB |
| Large open reference | Llama-3.3-70B, Qwen2.5-72B | 70B | 140 GB / 40–45 GB | 48–60 GB | Not planned | 1 × 80 GB at 4-bit; 2 × 80 GB at bf16 |
| Non-generative baselines | Regex rules; DeBERTa-v3, ClinicalBERT or BioLinkBERT slot filling; GLiNER | 0.1–0.5B | <2 GB | Full fine-tuning <16 GB | <16 GB | Any |

Notes:
- **Mixture-of-experts (MoE) models** (Qwen3-30B-A3B, gpt-oss-20b) must load all their weights into memory but run fast, because only about 3–4B parameters are active per token. That makes them interesting for latency.
- **Medical models** (MedGemma) answer the question: does medical pretraining help extraction?
- **Extraction-tuned models** (NuExtract) are a strong zero-shot reference.
- **The 70B group is not "small".** It is there to measure how much accuracy small models give up.
- Check Hugging Face for the newest release in each family before freezing the list.
- **Licences:**
  - Qwen, SmolLM, Granite, Mistral-Small and gpt-oss: Apache-2.0.
  - Phi: MIT.
  - Llama, Gemma and MedGemma: custom licences.
  - Record the licence of every model you use.

---

## 4. Experiments and what each needs

Scale assumptions:
- Dev set: 1,482 texts, about 0.6M output tokens per model per setting.
- Training set: 10k examples × ~700 tokens = 7M tokens per epoch; 3 epochs ≈ 21M tokens.

| Exp | What | Models | Needs | GPU time (approx.) |
|---|---|---|---|---|
| E0 | Regex / rule baseline | None | P3, P4 | CPU only, minutes |
| E1 | Encoder slot-filling baseline (token tagging + rules for corrections) | DeBERTa-v3, ClinicalBERT | P1, P4; BIO tags derived from the template generator | 1–2 h on any 16 GB GPU |
| E2 status | Zero-shot on dev | **Done (2026-11-02)**, 200 dev cases, 1,500-token output cap, laptop RTX 3050 Ti. Best: Qwen3 4B (F1 0.852, triage 83.0%, hallucination 3.7%, 44 s/case). Then Qwen2.5 3B 0.688 / 71.0%, Llama 3.2 3B 0.676 / 70.5%, Qwen3 1.7B 0.638 / 53.5%, Gemma 3 4B 0.622 / 74.5%, Phi-4-mini 0.586 / 65.5%, Granite 3.3 2B 0.567, Qwen3 0.6B 0.528, Qwen2.5 1.5B 0.517, Gemma 3 1B 0.442, Llama 3.2 1B 0.392; SmolLM2 1.7B dropped (usable 78%). Silent under-triage 0 for all. Finalists: Qwen3 4B, Qwen2.5 3B, Llama 3.2 3B, Qwen3 1.7B; full dev (1,482 cases, done 2026-11-04): Qwen3 4B F1 0.832 / triage 84.9% (hallucination 4.5%, corrections 14.0%), Qwen2.5 3B 0.667 / 74.2%, Llama 3.2 3B 0.653 / 70.1%, Qwen3 1.7B 0.627 / 51.4%; same ranking as the screen, 1–2 F1 points lower. Error analysis (`scripts/error_analysis.py`, 2026-11-04): errors are systematic — nested objects left null when only part was said (GCS total alone, "no LOC", crash speed, burns), digit amputations on hand/foot (L3-05 never fires), unsaid normal values (hemorrhage none, airway patent), all-null records on vitals-first calls, retraction log in words; four patterns explain 73% of Qwen3 4B's under-triage, all provisional. These set the E3 ablations. | Done |
| E2 | Zero-shot, constrained decoding, on dev | About 15 models across all groups | P3, P4, P6 | Tiny–medium: 5–20 min each on an A100 with vLLM. 14–32B: 20–60 min. 70B at 4-bit: 1–2 h. Total about 10–15 GPU hours. |
| E3 | Few-shot (2–5 examples) and prompt ablations (rules on/off, schema format) | Top 5 from E2 | E2 harness | About 5–10 GPU hours |
| E4a | QLoRA fine-tuning | 2–3 models from the small and medium groups | P1, P7 | Per model on an A100: 4B ≈ 1.5–2 h, 8B ≈ 3–4 h, 14B ≈ 6–8 h. On a T4, 4–6× longer. |
| E4b | Full fine-tuning | Best tiny and best small model | P1, P7; 1–2 × 80 GB GPUs | 1.7B ≈ 2–3 h, 4B ≈ 5–8 h on an A100 or H100 |
| E4c | Data-size curve | Best fine-tuned model at 1k / 2.5k / 5k / 10k examples | E4a | About 4–8 GPU hours |
| E5 | Quantization tradeoff: bf16 vs 8-bit vs 4-bit (GGUF/AWQ) for the fine-tuned winners | 2–3 models | llama.cpp, AutoAWQ | About 2–4 GPU hours |
| E6 | Final test, run once | About 5 finalists | P2 frozen | About 2 GPU hours |
| E7 | Speech-recognition robustness: test texts → TTS → Whisper large-v3 → extraction | Finalists | TTS (e.g. Piper or XTTS), Whisper | About 3–5 GPU hours |
| E8 (optional) | Direct audio to JSON with speech-input models (e.g. Qwen2-Audio / Qwen2.5-Omni, Phi-4-multimodal), compared with the Whisper-then-SLM pipeline | 1–2 models | E7 audio; 24–48 GB GPU | About 5–10 GPU hours |
| E9 | Deployment benchmark: latency, memory and power on realistic edge hardware (laptop GPU, Jetson Orin, CPU only) | Quantized finalists | Ollama / llama.cpp; access to devices | Your laptop counts; renting a Jetson is optional |

**Total estimate:** about 60–120 GPU hours on A100-class hardware, depending on how many models go into E4.

---

## 5. Metrics (same for every experiment)

- **Validity:** percentage of outputs that parse and pass schema validation.
- **Field level:**
  - Precision, recall and F1 on non-null fields.
  - **Hallucination rate**, reported separately: a value filled in where nothing was said.
- **Corrections:** for cases with a correction, the final value is right and the correction is logged.
- **Triage level** (after the rule engine): accuracy, **under-triage rate** (headline safety metric) and over-triage rate.
- **Breakdowns:** style (fillers, disorganized, Q&A), length band, borderline cases, out-of-scope cases.
- **Cost and deployability:** latency per handover, peak memory, model size on disk.
- **Statistics:** bootstrap 95% confidence intervals; McNemar's test for paired comparisons of triage decisions.

---

## 6. Compute options

| Option | GPU | Fits | Cost (approx.) | Notes |
|---|---|---|---|---|
| Your laptop | RTX 3050 Ti 4 GB | E0, E2 tiny, E9 | Free | Deployment reference device |
| Kaggle | 2 × T4 16 GB | E1, E2 up to 8B at 4-bit, E4a up to 8B | Free, about 30 h/week | Sessions are time-limited; save checkpoints |
| Colab Pro / Pro+ | L4 24 GB / A100 40–80 GB | E2–E4a up to 14B | About $10–50/month | Compute units run out quickly on A100 |
| Cloud rental (RunPod, Lambda, Vast.ai) | RTX 4090 24 GB / A100 80 GB / H100 80 GB | Everything | 4090 ≈ $0.3–0.7/h, A100 ≈ $1.2–2/h, H100 ≈ $2–3/h | About 60–120 A100 hours ≈ $100–250 |
| University HPC | Usually A100 / V100 | Everything | Free if you have an account | Ask the supervisor or IT; there may be a queue |

---

## 7. Suggested order and timeline

| Week | Work |
|---|---|
| 1 | P1–P4, P6; E0 |
| 2 | P5 (rule engine); E1; E2 on the tiny and small groups |
| 3 | E2 on the larger groups; E3 |
| 4–5 | E4a and E4c; E4b if compute allows |
| 6 | E5, then E6 (final test, once) |
| 7 | E7 and E9; E8 if time allows |
| 8 | Analysis, figures and the write-up of methods and results |

---

## 8. Open decisions

1. **Compute source:** university HPC, cloud rental, or Kaggle/Colab only. This decides whether E4b, E8 and the 27–70B groups are realistic.
2. **Budget ceiling** if renting, for example $250.
3. **Full fine-tuning (E4b) and audio-in models (E8):** in scope, or future work?
