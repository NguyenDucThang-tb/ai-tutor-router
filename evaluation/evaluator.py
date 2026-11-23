from __future__ import annotations

from evaluation.answer_extraction import extract_model_final_answer
from evaluation.numeric import NumericAnswerError, parse_numeric_answer
from evaluation.schemas import EvaluationResult, EvaluationStatus
from evaluation.symbolic import (
    MalformedExpression,
    UnsupportedExpression,
    expressions_equivalent,
)

EVALUATOR_VERSION = "numeric-symbolic-v2"


def evaluate_answer(model_output: str, reference_answer: str) -> EvaluationResult:
    extraction = extract_model_final_answer(model_output)
    if extraction.status == "ambiguous":
        return _result(EvaluationStatus.AMBIGUOUS, None, extraction.reason)
    if extraction.status != "parsed" or extraction.answer is None:
        return _result(EvaluationStatus.UNPARSEABLE, None, extraction.reason)

    try:
        model_value = parse_numeric_answer(extraction.answer)
    except NumericAnswerError as model_error:
        if str(model_error) in {"malformed_fraction", "fraction_denominator_zero"}:
            return _result(
                EvaluationStatus.UNPARSEABLE,
                extraction.answer,
                str(model_error),
                reference_answer,
            )
        try:
            is_correct = expressions_equivalent(extraction.answer, reference_answer)
        except MalformedExpression as exc:
            return _result(
                EvaluationStatus.UNPARSEABLE, extraction.answer, str(exc), reference_answer
            )
        except UnsupportedExpression as exc:
            return _result(
                EvaluationStatus.UNSUPPORTED, extraction.answer, str(exc), reference_answer
            )
    else:
        try:
            reference_value = parse_numeric_answer(reference_answer)
        except NumericAnswerError as reference_error:
            return _result(
                EvaluationStatus.UNPARSEABLE,
                extraction.answer,
                f"reference_answer_unparseable:{reference_error}",
                reference_answer,
            )
        is_correct = model_value == reference_value
    return _result(
        EvaluationStatus.CORRECT if is_correct else EvaluationStatus.INCORRECT,
        extraction.answer,
        None,
        reference_answer,
        is_correct,
    )


def _result(
    status: EvaluationStatus,
    answer: str | None,
    reason: str | None,
    reference: str | None = None,
    is_correct: bool | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        evaluator_version=EVALUATOR_VERSION,
        status=status,
        is_correct=is_correct,
        extracted_answer=answer,
        reference_answer=reference,
        reason=reason,
    )
