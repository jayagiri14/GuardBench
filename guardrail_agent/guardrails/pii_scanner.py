"""
PII Scanner: detection and redaction of sensitive data patterns.

Detects: email addresses, US SSNs, phone numbers, credit card numbers,
         API keys / bearer tokens, canary/secret markers.

All detection is regex-based so the module works with zero dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PIIType(str, Enum):
    EMAIL = "EMAIL"
    SSN = "SSN"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    API_KEY = "API_KEY"
    CANARY = "CANARY"


@dataclass(frozen=True)
class PIIMatch:
    """A single detected PII span."""
    pii_type: PIIType
    value: str          # original matched text
    start: int
    end: int


@dataclass
class ScanResult:
    """Full scan output for a piece of text."""
    original: str
    matches: list[PIIMatch] = field(default_factory=list)
    redacted: str = ""

    @property
    def has_pii(self) -> bool:
        return len(self.matches) > 0

    @property
    def pii_types_found(self) -> set[PIIType]:
        return {m.pii_type for m in self.matches}


# ---------------------------------------------------------------------------
# Patterns (ordered most-specific first so more specific types win)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[PIIType, re.Pattern]] = [
    # SSN — must come before phone to avoid partial matches on dashes
    (
        PIIType.SSN,
        re.compile(r"\b(?!000|666|9\d\d)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}\b"),
    ),
    # Credit cards (Visa, MC, Amex, Discover) — 4-group patterns
    (
        PIIType.CREDIT_CARD,
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
    ),
    # Email
    (
        PIIType.EMAIL,
        re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    ),
    # Phone — US/international formats
    (
        PIIType.PHONE,
        re.compile(
            r"\b(?:\+?1[-.\s]?)?"
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
    ),
    # API keys / bearer tokens — long hex or base64-looking strings
    (
        PIIType.API_KEY,
        re.compile(
            r"\b(?:"
            r"sk-[a-zA-Z0-9]{20,}"           # OpenAI-style
            r"|Bearer\s+[a-zA-Z0-9._\-]{20,}" # Bearer token
            r"|[a-fA-F0-9]{32,}"              # raw hex key
            r")\b"
        ),
    ),
]


def _build_canary_pattern(canary_tokens: list[str]) -> re.Pattern | None:
    if not canary_tokens:
        return None
    escaped = [re.escape(tok) for tok in canary_tokens]
    return re.compile("|".join(escaped))


class PIIScanner:
    """
    Scan text for PII patterns and return a ScanResult with matches and
    a version of the text where all PII spans have been replaced with
    ``[REDACTED_<TYPE>]`` placeholders.

    Parameters
    ----------
    canary_tokens:
        Optional list of exact secret strings planted in the system prompt
        (honeytoken pattern). Detected separately from generic PII.
    """

    def __init__(self, canary_tokens: list[str] | None = None) -> None:
        self._canary_pattern = _build_canary_pattern(canary_tokens or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> ScanResult:
        """Scan *text* and return a :class:`ScanResult`."""
        matches: list[PIIMatch] = []

        # Generic PII patterns
        for pii_type, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Canary / honeytoken
        if self._canary_pattern:
            for m in self._canary_pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=PIIType.CANARY,
                        value=m.group(),
                        start=m.start(),
                        end=m.end(),
                    )
                )

        # Remove duplicate / overlapping spans (keep longest match)
        matches = _deduplicate(matches)
        matches.sort(key=lambda x: x.start)

        redacted = _redact(text, matches)
        result = ScanResult(original=text, matches=matches, redacted=redacted)
        return result

    def has_pii(self, text: str) -> bool:
        return self.scan(text).has_pii

    def redact(self, text: str) -> str:
        return self.scan(text).redacted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Remove overlapping spans, preferring longer matches."""
    if not matches:
        return []
    # Sort by start, then by span length descending
    matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    deduped: list[PIIMatch] = []
    last_end = -1
    for m in matches:
        if m.start >= last_end:
            deduped.append(m)
            last_end = m.end
    return deduped


def _redact(text: str, matches: list[PIIMatch]) -> str:
    """Replace each matched span with a [REDACTED_TYPE] placeholder."""
    if not matches:
        return text
    parts: list[str] = []
    cursor = 0
    for m in matches:
        parts.append(text[cursor : m.start])
        parts.append(f"[REDACTED_{m.pii_type.value}]")
        cursor = m.end
    parts.append(text[cursor:])
    return "".join(parts)
