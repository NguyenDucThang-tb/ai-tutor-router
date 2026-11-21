from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Extraction:
    answer: str | None
    status: str
    reason: str | None = None


_FINAL_LINE = re.compile(r"(?im)^\s*FINAL ANSWER\s*:\s*(.*?)\s*$")
_GSM_LINE = re.compile(r"(?m)^\s*####\s*(.*?)\s*$")


def extract_model_final_answer(output: str) -> Extraction:
    matches = list(_FINAL_LINE.finditer(output))
    if not matches:
        return Extraction(None, "unparseable", "missing_final_answer_delimiter")
    if len(matches) != 1:
        return Extraction(None, "ambiguous", "multiple_final_answer_delimiters")
    answer = matches[0].group(1).strip()
    if not answer:
        return Extraction(None, "unparseable", "empty_final_answer")
    if re.search(r"\b(?:or|alternatively)\b", answer, flags=re.IGNORECASE):
        return Extraction(answer, "ambiguous", "multiple_competing_answers")
    return Extraction(answer, "parsed")


def extract_gsm8k_reference(raw_answer: str) -> tuple[str, str]:
    matches = list(_GSM_LINE.finditer(raw_answer))
    if len(matches) != 1 or not matches[0].group(1).strip():
        raise ValueError("GSM8K reference must contain one non-empty #### final-answer line")
    marker = matches[0]
    return marker.group(1).strip(), raw_answer[: marker.start()].strip()
