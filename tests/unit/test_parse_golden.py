"""Golden tests for MIRAGE-style answer parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biolit.eval.metrics import compute_run_metrics
from biolit.eval.parse import parse_answer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parse_outputs.json"


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_parse_golden(case: dict) -> None:
    options = case.get("options")
    result = parse_answer(case["raw"], options)
    assert result.unparsed is case["unparsed"]
    if case["unparsed"]:
        assert result.label is None
    else:
        assert result.label == case["expected"]
        assert result.confidence > 0


def test_unparsed_excluded_from_accuracy() -> None:
    preds = ["A", "", "B", "C"]
    golds = ["A", "A", "B", "D"]
    stats = [
        {"unparsed": False},
        {"unparsed": True},
        {"unparsed": False},
        {"unparsed": False},
    ]
    metrics = compute_run_metrics(
        predictions=preds,
        golds=golds,
        item_stats=stats,
        few_shot_k=5,
    )
    # Parsed items: A/A correct, B/B correct, C/D wrong → 2/3
    assert metrics["n_unparsed"] == 1
    assert metrics["unparsed_rate"] == 0.25
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["few_shot_k"] == 5
