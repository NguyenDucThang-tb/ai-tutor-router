from __future__ import annotations

import pytest

from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from evaluation.schemas import EvaluationStatus


@pytest.mark.parametrize(
    ("output", "reference", "extracted"),
    [
        ("FINAL ANSWER: $5", "5", "$5"),
        ("FINAL ANSWER: $18", "18", "$18"),
        ("**Final Answer:** 86", "86", "86"),
        ("**Final Answer**: 7", "7", "7"),
    ],
)
def test_kaggle_unknown_answer_formats_are_evaluated_explicitly(
    output: str, reference: str, extracted: str
) -> None:
    result = evaluate_answer(output, reference)

    assert result.status == EvaluationStatus.CORRECT
    assert result.is_correct is True
    assert result.extracted_answer == extracted
    assert result.evaluator_version == EVALUATOR_VERSION == "numeric-symbolic-v2"


def test_currency_prefix_is_normalized_but_answer_is_still_compared() -> None:
    result = evaluate_answer("FINAL ANSWER: $5", "6")

    assert result.status == EvaluationStatus.INCORRECT
    assert result.is_correct is False


@pytest.mark.parametrize(
    ("output", "reference"),
    [
        ("**Final Answer:** The price is $333,200.\nFINAL ANSWER: 333200", "333200"),
        ("**Final Answer:**\nFINAL ANSWER: 90", "90"),
        ("**Final Answer:** The milkshake cost $5.\nFINAL ANSWER: 5", "5"),
        ("**Final Answer:** The price is $13.\nFINAL ANSWER: 13", "13"),
    ],
)
def test_strict_final_answer_line_takes_precedence_over_explanatory_heading(
    output: str, reference: str
) -> None:
    result = evaluate_answer(output, reference)

    assert result.status == EvaluationStatus.CORRECT
    assert result.extracted_answer == reference
