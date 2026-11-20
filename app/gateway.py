from dataclasses import dataclass
import os
import time


@dataclass
class ModelResult:
    text: str
    model: str
    usage: dict[str, int]
    cost_usd: float | None


class ModelGateway:
    """Provider boundary. The mock implementation is deterministic and credential-free."""
    def __init__(self, provider: str = "mock"):
        self.provider = provider

    def complete(self, tier: str, prompt: str, timeout: float = 30) -> ModelResult:
        started = time.perf_counter()
        if self.provider != "mock":
            raise RuntimeError("Provider adapter not configured; set TUTOR_PROVIDER=mock for local mode")
        model = os.getenv("TUTOR_SMALL_MODEL", "mock-small") if tier == "small" else os.getenv("TUTOR_STRONG_MODEL", "mock-strong")
        # This mock proves API plumbing only; it does not solve or evaluate mathematics.
        question = prompt.split("\nQuestion:\n", 1)[-1][:300]
        result = f"[MOCK {model}] Request received: {question}"
        _ = time.perf_counter() - started
        return ModelResult(result, model, {"prompt_tokens": len(prompt.split()), "completion_tokens": len(result.split())}, 0.0)
