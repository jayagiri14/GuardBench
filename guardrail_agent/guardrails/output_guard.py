"""
Output Guardrail: scans the LLM reply before it is returned to the user.

Decisions (in order of precedence):
  1. BLOCK_CANARY  — canary / honeytoken found in reply -> secret exfiltrated, block
  2. REDACT        — non-canary PII found -> redact before returning
  3. PASS          — clean reply returned as-is
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from guardrail_agent.guardrails.pii_scanner import PIIScanner, PIIType, ScanResult


class OutputDecision(str, Enum):
    PASS = "PASS"
    REDACT = "REDACT"
    BLOCK_CANARY = "BLOCK_CANARY"


_CANARY_BLOCK_RESPONSE = (
    "I'm sorry, I cannot share that information. "
    "Please contact support if you need further assistance."
)


@dataclass
class OutputGuardResult:
    decision: OutputDecision
    original_reply: str
    sanitised_reply: str        # redacted/blocked message; caller should return this
    pii_result: ScanResult

    @property
    def is_safe(self) -> bool:
        return self.decision == OutputDecision.PASS

    @property
    def leaked_canary(self) -> bool:
        return self.decision == OutputDecision.BLOCK_CANARY


class OutputGuard:
    """
    Post-LLM guardrail.

    Parameters
    ----------
    canary_tokens:
        Secret honeytoken strings that must never appear in LLM output.
        If detected, the reply is suppressed entirely.
    """

    def __init__(self, canary_tokens: list[str] | None = None) -> None:
        self._pii_scanner = PIIScanner(canary_tokens=canary_tokens)

    def check(self, reply: str) -> OutputGuardResult:
        """Evaluate the LLM *reply* and return an :class:`OutputGuardResult`."""
        pii_result = self._pii_scanner.scan(reply)

        if not pii_result.has_pii:
            return OutputGuardResult(
                decision=OutputDecision.PASS,
                original_reply=reply,
                sanitised_reply=reply,
                pii_result=pii_result,
            )

        # Canary takes priority — full block
        if PIIType.CANARY in pii_result.pii_types_found:
            return OutputGuardResult(
                decision=OutputDecision.BLOCK_CANARY,
                original_reply=reply,
                sanitised_reply=_CANARY_BLOCK_RESPONSE,
                pii_result=pii_result,
            )

        # Other PII — redact and return
        return OutputGuardResult(
            decision=OutputDecision.REDACT,
            original_reply=reply,
            sanitised_reply=pii_result.redacted,
            pii_result=pii_result,
        )
