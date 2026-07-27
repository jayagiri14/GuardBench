"""
Prompt Injection Detector: multi-heuristic engine for detecting prompt injection
and jailbreak attempts in user-supplied text.

Checks (in order):
    1. Direct override keywords (ignore/disregard previous instructions)
    2. Persona/roleplay jailbreaks (DAN, GPT-4-DAN, AIM, etc.)
    3. Fake system / authority tags ([SYSTEM], [ADMIN], <system>)
    4. Encoding smuggling (base64 payloads, token-spacing attacks)
    5. Indirect injection via embedded document or tool-output payloads
    6. Exfiltration format fishing (json/markdown system prompt requests)
    7. Continuation / fill-in-the-blank prompts
    8. Hypothetical / roleplay framing as bypass
    9. Negation confusion attacks

Each check contributes a partial score to a total risk score in [0.0, 1.0].
Risk score >= threshold is classified as INJECTION_DETECTED.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from enum import Enum


class InjectionCategory(str, Enum):
    DIRECT_OVERRIDE = "direct_override"
    PERSONA_JAILBREAK = "persona_jailbreak"
    FAKE_SYSTEM_TAG = "fake_system_tag"
    ENCODING_SMUGGLE = "encoding_smuggle"
    INDIRECT_INJECTION = "indirect_injection"
    EXFIL_FORMAT_FISHING = "exfil_format_fishing"
    CONTINUATION_TRICK = "continuation_trick"
    HYPOTHETICAL_BYPASS = "hypothetical_bypass"
    NEGATION_CONFUSION = "negation_confusion"
    AUTHORITY_IMPERSONATION = "authority_impersonation"


@dataclass
class DetectionResult:
    """Result from the injection detector."""
    text: str
    risk_score: float                        # 0.0 (safe) – 1.0 (certain injection)
    is_injection: bool
    categories: list[InjectionCategory] = field(default_factory=list)
    flagged_snippets: list[str] = field(default_factory=list)
    threshold: float = 0.35


# ---------------------------------------------------------------------------
# Heuristic rule definitions
# ---------------------------------------------------------------------------
# Each rule is (category, pattern_list, weight)
# weight contributes to risk_score (capped at 1.0)

_RULES: list[tuple[InjectionCategory, list[re.Pattern], float]] = [

    (
        InjectionCategory.DIRECT_OVERRIDE,
        [
            re.compile(
                r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
                r"(instructions?|prompts?|rules?|constraints?)",
                re.I,
            ),
            re.compile(
                r"disregard\s+(all\s+)?(previous|prior|above)\s+"
                r"(instructions?|prompts?|rules?)",
                re.I,
            ),
            re.compile(r"forget\s+(everything|all)\s+(you\s+)?(were\s+)?told", re.I),
            re.compile(r"override\s+(your\s+)?(instructions?|rules?|constraints?)", re.I),
            re.compile(r"new\s+instructions?:\s+", re.I),
        ],
        0.70,
    ),

    (
        InjectionCategory.PERSONA_JAILBREAK,
        [
            re.compile(r"\bDAN\b"),
            re.compile(r"\b(jailbreak|jail\s*break)\b", re.I),
            re.compile(
                r"you\s+(are\s+now|will\s+now\s+act\s+as|have\s+no\s+restrictions?)",
                re.I,
            ),
            re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?AI\s+with\s+no\b", re.I),
            re.compile(r"\bAIM\b.*no\s+ethical", re.I),
            re.compile(r"pretend\s+you\s+(are|have)\s+no\s+(rules?|restrictions?|ethics?)", re.I),
        ],
        0.75,
    ),

    (
        InjectionCategory.FAKE_SYSTEM_TAG,
        [
            re.compile(r"\[SYSTEM\]", re.I),
            re.compile(r"\[ADMIN\]", re.I),
            re.compile(r"\[DEVELOPER\s+MODE\]", re.I),
            re.compile(r"<\s*system\s*>", re.I),
            re.compile(r"\[INST\].*\[/INST\]", re.I | re.S),
            re.compile(r"debug\s+mode\s+enabled", re.I),
            re.compile(r"###\s*system\s*:", re.I),
        ],
        0.70,
    ),

    (
        InjectionCategory.ENCODING_SMUGGLE,
        [
            re.compile(r"base64", re.I),
            re.compile(r"decode\s+this", re.I),
            re.compile(
                # space-separated single chars — "r e v e a l"
                r"(?:\b[a-z]\s){5,}[a-z]\b",
                re.I,
            ),
        ],
        0.55,
    ),

    (
        InjectionCategory.INDIRECT_INJECTION,
        [
            re.compile(r"<note>.*system\s+override", re.I | re.S),
            re.compile(r"process\s+this\s+document\s+and\s+follow", re.I),
            re.compile(r"the\s+document\s+says\s+to\b", re.I),
            re.compile(r"found\s+document.*instruction", re.I | re.S),
            re.compile(r"tool\s+output.*override", re.I | re.S),
        ],
        0.65,
    ),

    (
        InjectionCategory.EXFIL_FORMAT_FISHING,
        [
            re.compile(r"(print|output|repeat|show)\s+(your\s+)?(exact\s+)?(system\s+prompt|initial\s+instructions?)", re.I),
            re.compile(r"(summarize|format|return)\s+your\s+instructions\s+as\s+(json|markdown|code\s+block)", re.I),
            re.compile(r"your\s+instructions.*as\s+a?\s*(json|markdown|code\s+block)", re.I),
            re.compile(r"system_prompt.*json", re.I | re.S),
            re.compile(r"fill\s+in.*\{.*\}.*secret", re.I | re.S),
            re.compile(r"return\s+a?\s*json\s+object\s+with.*system_prompt", re.I),
        ],
        0.65,
    ),

    (
        InjectionCategory.CONTINUATION_TRICK,
        [
            re.compile(r"continue\s+(this\s+)?(text|sentence|story)\s+(exactly\s+)?where\s+it\s+left\s+off", re.I),
            re.compile(r"my\s+(confidential\s+)?(secret|password|key)\s+is\s*$", re.I),
            re.compile(r"complete\s+the\s+following\s*:\s*['\"]?the\s+secret", re.I),
        ],
        0.60,
    ),

    (
        InjectionCategory.HYPOTHETICAL_BYPASS,
        [
            re.compile(
                r"hypothetically[,\s]+(if\s+you\s+(were|could|were\s+allowed)\s+to\s+share|"
                r"what\s+would\s+the\s+secret\s+be)",
                re.I,
            ),
            re.compile(r"just\s+for\s+a\s+(thought\s+experiment|hypothetical)", re.I),
            re.compile(r"in\s+a\s+fictional\s+(world|scenario).*(secret|password|key)", re.I),
            re.compile(r"imagine\s+you\s+(had\s+no|were\s+free\s+from)\s+(rules?|restrictions?)", re.I),
        ],
        0.55,
    ),

    (
        InjectionCategory.NEGATION_CONFUSION,
        [
            re.compile(r"do\s+NOT\s+reveal.*actually\s+wait.*reveal", re.I | re.S),
            re.compile(r"(don't|do\s+not)\s+(tell|reveal|share).*\.\s*actually.*please\s+(tell|reveal|share)", re.I | re.S),
            re.compile(r"i\s+(meant|mean)\s+to\s+reveal\s+it", re.I),
        ],
        0.65,
    ),

    (
        InjectionCategory.AUTHORITY_IMPERSONATION,
        [
            re.compile(r"(this\s+is\s+(your|the)\s+)?(system\s+admin(istrator)?|developer|operator)\b.*authorized", re.I | re.S),
            re.compile(r"i\s+am\s+(your\s+)?(admin|developer|supervisor|creator)\b", re.I),
            re.compile(r"authorized\s+to\s+request\s+the\s+(secret|password|api\s+key)", re.I),
            re.compile(r"for\s+(an?\s+)?(audit|compliance|security\s+(test|audit))\b.*provide\s+(it|the\s+(secret|key|password))", re.I),
            re.compile(r"(security|compliance)\s+(audit|test).*provide\s+the\s+(secret|key|password|api\s+key)", re.I | re.S),
        ],
        0.70,
    ),
]


# ---------------------------------------------------------------------------
# Base64 payload check
# ---------------------------------------------------------------------------

def _check_base64_payload(text: str) -> bool:
    """Return True if any token looks like a valid base64 string > 20 chars."""
    for token in text.split():
        token = token.strip(".,;:'\"")
        if len(token) < 20:
            continue
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")
            # Look for injection-like keywords in the decoded payload
            if re.search(r"reveal|secret|ignore|system\s+prompt|override|password|instructions", decoded, re.I):
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class InjectionDetector:
    """
    Multi-heuristic prompt injection detector.

    Parameters
    ----------
    threshold:
        Risk score in [0.0, 1.0] at or above which the text is classified
        as an injection attempt. Default ``0.35``.
    """

    def __init__(self, threshold: float = 0.35) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def detect(self, text: str) -> DetectionResult:
        """Analyse *text* and return a :class:`DetectionResult`."""
        total_weight = 0.0
        categories: list[InjectionCategory] = []
        flagged: list[str] = []

        # Rule-based checks
        for category, patterns, weight in _RULES:
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    if category not in categories:
                        categories.append(category)
                        total_weight += weight
                    snippet = text[max(0, m.start() - 20): m.end() + 20].strip()
                    if snippet not in flagged:
                        flagged.append(snippet)
                    break  # one match per rule-group is sufficient

        # Base64 payload check (bonus)
        if _check_base64_payload(text):
            if InjectionCategory.ENCODING_SMUGGLE not in categories:
                categories.append(InjectionCategory.ENCODING_SMUGGLE)
                total_weight += 0.55

        risk_score = min(total_weight, 1.0)
        return DetectionResult(
            text=text,
            risk_score=round(risk_score, 4),
            is_injection=risk_score >= self.threshold,
            categories=categories,
            flagged_snippets=flagged,
            threshold=self.threshold,
        )

    def is_injection(self, text: str) -> bool:
        return self.detect(text).is_injection
