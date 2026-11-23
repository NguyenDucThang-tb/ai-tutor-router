from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


class NumericAnswerError(ValueError):
    pass


_DECIMAL = re.compile(r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)$")
_FRACTION = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))$")
_LATEX_FRACTION = re.compile(
    r"^\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}\s*"
    r"\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}$"
)


def parse_numeric_answer(value: str) -> Fraction:
    candidate = _strip_simple_wrappers(value)
    match = _LATEX_FRACTION.fullmatch(candidate) or _FRACTION.fullmatch(candidate)
    if match:
        numerator = _to_fraction(match.group(1))
        denominator = _to_fraction(match.group(2))
        if denominator == 0:
            raise NumericAnswerError("fraction_denominator_zero")
        return numerator / denominator
    if "/" in candidate or "\\frac" in candidate:
        raise NumericAnswerError("malformed_fraction")
    if not _DECIMAL.fullmatch(candidate):
        raise NumericAnswerError("not_a_supported_numeric_answer")
    try:
        return _to_fraction(candidate.replace(",", ""))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise NumericAnswerError("invalid_numeric_answer") from exc


def _to_fraction(value: str) -> Fraction:
    decimal = Decimal(value.replace(",", ""))
    if not decimal.is_finite():
        raise InvalidOperation
    return Fraction(decimal)


def _strip_simple_wrappers(value: str) -> str:
    candidate = value.strip()
    if candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()
    elif re.match(r"^\$\s*(?=[+-]?(?:\d|\.\d))", candidate):
        # Accept an unpaired leading dollar sign as a currency prefix, but only
        # when the remaining answer begins with a numeric literal.
        candidate = re.sub(r"^\$\s*", "", candidate, count=1)
    if candidate.startswith(r"\(") and candidate.endswith(r"\)"):
        candidate = candidate[2:-2].strip()
    boxed = re.fullmatch(r"\\boxed\s*\{(.*)\}", candidate, flags=re.DOTALL)
    if boxed:
        candidate = boxed.group(1).strip()
    return candidate
