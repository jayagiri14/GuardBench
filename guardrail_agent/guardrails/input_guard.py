"""
Input Guardrail: intercepts and sanitises user input before it reaches the LLM.

Decisions (in order of precedence):
  1. BLOCK   — injection detected above threshold -> request refused immediately
  2. REDACT  — PII found in input -> input is sanitised before forwarding
  3. PASS    — clean input forwarded unchanged
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from guardrail_agent.guardrails.injection_detector import InjectionDetector, DetectionResult
from guardrail_agent.guardrails.pii_scanner import PIIScanner, ScanResult


class InputDecision(str, Enum):
    PASS = "PASS"       # clean, forward to LLM unchanged
    REDACT = "REDACT"   # PII found, forward sanitised version
    BLOCK = "BLOCK"     # injection detected, refuse immediately


@dataclass
class InputGuardResult:
    decision: InputDecision
    original_text: str
    sanitised_text: str                     # equals original if PASS or BLOCK
    injection_result: DetectionResult
    pii_result: ScanResult
    block_reason: str = ""

    @property
    def is_blocked(self) -> bool:
        return self.decision == InputDecision.BLOCK

    @property
    def text_to_forward(self) -> str:
        """Text that should be sent to the LLM (only valid when not blocked)."""
        if self.decision == InputDecision.REDACT:
            return self.sanitised_text
        return self.original_text


_BLOCK_RESPONSE = (
    "I'm sorry, but I cannot process that request. "
    "It appears to contain instructions that conflict with my operating guidelines."
)


class InputGuard:
    """
    Pre-LLM guardrail.

    Parameters
    ----------
    injection_threshold:
        Risk score at or above which input is blocked (default 0.35).
    canary_tokens:
        Optional list of honeytoken strings that should never appear in input.
        If found, the request is blocked (attacker is trying to echo a known secret back).
    """

    def __init__(
        self,
        injection_threshold: float = 0.35,
        canary_tokens: list[str] | None = None,
    ) -> None:
        self._injection_detector = InjectionDetector(threshold=injection_threshold)
        self._pii_scanner = PIIScanner(canary_tokens=canary_tokens)

    def check(self, text: str) -> InputGuardResult:
        """Evaluate *text* and return an :class:`InputGuardResult`."""
        injection_result = self._injection_detector.detect(text)
        pii_result = self._pii_scanner.scan(text)

        if injection_result.is_injection:
            return InputGuardResult(
                decision=InputDecision.BLOCK,
                original_text=text,
                sanitised_text=text,
                injection_result=injection_result,
                pii_result=pii_result,
                block_reason=(
                    f"Injection detected (score={injection_result.risk_score:.2f}, "
                    f"categories={[c.value for c in injection_result.categories]})"
                ),
            )

        if pii_result.has_pii:
            return InputGuardResult(
                decision=InputDecision.REDACT,
                original_text=text,
                sanitised_text=pii_result.redacted,
                injection_result=injection_result,
                pii_result=pii_result,
            )

        return InputGuardResult(
            decision=InputDecision.PASS,
            original_text=text,
            sanitised_text=text,
            injection_result=injection_result,
            pii_result=pii_result,
        )

    @property
    def block_response(self) -> str:
        """Standard refusal message to return to user when input is blocked."""
        return _BLOCK_RESPONSE
