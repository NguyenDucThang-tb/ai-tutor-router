import hashlib

PROMPT_VERSION = "math-direct-v1"
PROMPT_TEMPLATE = (
    "You are a careful mathematics tutor. Give a concise explanation that helps a student "
    "understand the solution. Then give exactly one final answer on its own last line in "
    "this format: FINAL ANSWER: <answer>. Do not list alternative answers."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
