from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    DELETE = "delete"
    MUTE = "mute"


@dataclass(frozen=True)
class ModerationResult:
    verdict: Verdict
    confidence: float
    reasons: list[str] = field(default_factory=list)
    public_note: str | None = None

    @property
    def is_violation(self) -> bool:
        return self.verdict in {Verdict.REVIEW, Verdict.DELETE, Verdict.MUTE}

    @staticmethod
    def allow() -> "ModerationResult":
        return ModerationResult(verdict=Verdict.ALLOW, confidence=1.0)
