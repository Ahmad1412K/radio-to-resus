"""The LLM-output validator must accept the template texts and reject fact changes."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.llm_paraphrase import load_rows, validate  # noqa: E402

ROWS = load_rows()
CASE = next(r for r in ROWS if r["case_id"] == "H00311")   # "We have a 38, I mean, 28-year-old female."
TEXT = CASE["text"]


def test_all_template_texts_pass():
    failing = [r["case_id"] for r in ROWS if validate(r, r["text"])]
    assert not failing, failing[:10]


def test_original_case_is_valid():
    assert validate(CASE, TEXT) == []


@pytest.mark.parametrize("name, text, expected", [
    ("correction removed", TEXT.replace("38, I mean, ", ""), "self-correction missing"),
    ("correction wrong order", TEXT.replace("38, I mean, 28", "28, I mean, 38"), "wrong order"),
    ("value changed", TEXT.replace("heart rate 91", "heart rate 94"), "stated value missing: hr=91"),
    ("hallucinated vital", TEXT + " Sats 98 percent.", "not said: spo2"),
    ("hallucinated history", TEXT + " History of asthma.", "not said: background"),
    ("speaker labels", "Paramedic: " + TEXT, "speaker labels"),
])
def test_fact_changes_are_rejected(name, text, expected):
    assert any(expected in issue for issue in validate(CASE, text)), name
