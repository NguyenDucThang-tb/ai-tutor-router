"""Provider-aware token accounting contracts; unsupported methods fail closed."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class CountKind(StrEnum):
    EXACT = "exact"
    VERIFIED_UPPER_BOUND = "verified_upper_bound"
    ESTIMATE = "estimate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TokenCount:
    input_tokens: int | None
    method: str
    version: str
    kind: CountKind
    includes_system_message: bool
    includes_format_overhead: bool
    limitation: str | None = None

    @property
    def supports_admission(self) -> bool:
        return (
            self.input_tokens is not None
            and self.input_tokens >= 0
            and self.kind in {CountKind.EXACT, CountKind.VERIFIED_UPPER_BOUND}
            and self.includes_system_message
            and self.includes_format_overhead
        )


class TokenCounter(Protocol):
    def count_messages(self, model_id: str, messages: list[dict[str, str]]) -> TokenCount: ...


class UnsupportedTokenCounter:
    """Default runtime counter: explicitly unknown; never substitutes bytes/words."""

    def count_messages(self, model_id: str, messages: list[dict[str, str]]) -> TokenCount:
        return TokenCount(
            input_tokens=None,
            method="unsupported",
            version="1",
            kind=CountKind.UNKNOWN,
            includes_system_message=False,
            includes_format_overhead=False,
            limitation=f"No verified local counter configured for {model_id}.",
        )
