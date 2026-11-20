import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryFeatures:
    char_count: int
    token_count: int
    math_symbol_count: int
    has_proof_request: bool
    has_verification_intent: bool
    requested_steps: bool
    task_type: str
    requested_step_count: int | None = None


def extract_features(question: str, task_type: str = "math") -> QueryFeatures:
    words = re.findall(r"\b[\w']+\b", question)
    proof = bool(re.search(r"\b(prove|proof|derive|demonstrate)\b", question, re.IGNORECASE))
    verify = task_type == "verification" or bool(
        re.search(r"\b(check|verify|is my solution|did I solve)\b", question, re.IGNORECASE)
    )
    step_match = re.search(r"\b(\d+)\s+steps?\b", question, re.IGNORECASE)
    steps = bool(
        re.search(r"\b(step[- ]by[- ]step|show your work|explain|hint)\b", question, re.IGNORECASE)
        or step_match
    )
    step_count = int(step_match.group(1)) if step_match else None
    math_chars = sum(ch in "=+-*/^√∫∑≤≥<>%()[]{}" for ch in question)
    return QueryFeatures(
        len(question), len(words), math_chars, proof, verify, steps, task_type, step_count
    )
