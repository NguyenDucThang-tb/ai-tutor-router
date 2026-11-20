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


def extract_features(question: str, task_type: str = "math") -> QueryFeatures:
    words = re.findall(r"\b[\w']+\b", question)
    proof = bool(re.search(r"\b(prove|proof|derive|demonstrate)\b", question, re.I))
    verify = task_type == "verification" or bool(
        re.search(r"\b(check|verify|is my solution|did I solve)\b", question, re.I)
    )
    steps = bool(re.search(r"\b(step[- ]by[- ]step|show your work|explain|hint)\b", question, re.I))
    math_chars = sum(ch in "=+-*/^√∫∑≤≥<>%()[]{}" for ch in question)
    return QueryFeatures(len(question), len(words), math_chars, proof, verify, steps, task_type)
