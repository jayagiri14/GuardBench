"""
GuardrailAgent: the complete request/response pipeline.

Pipeline:
    user_input
        -> InputGuard.check()           [blocks/redacts input]
        -> LLMClient.complete()         [only if not blocked]
        -> OutputGuard.check()          [blocks canary / redacts PII]
        -> return sanitised reply

The agent is stateless across requests. Each call to ``process()`` is
independent — suitable for single-turn evaluation benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from guardrail_agent.agent.llm_client import BaseLLMClient, LLMResponse
from guardrail_agent.guardrails.input_guard import InputGuard, InputDecision, InputGuardResult
from guardrail_agent.guardrails.output_guard import OutputGuard, OutputDecision, OutputGuardResult


class AgentOutcome(str, Enum):
    """High-level result of a single agent turn."""
    ANSWERED        = "ANSWERED"       # benign: replied normally
    REDACTED_INPUT  = "REDACTED_INPUT" # input PII stripped, answer provided on clean version
    REDACTED_OUTPUT = "REDACTED_OUTPUT"# answer provided but PII redacted from reply
    BLOCKED_INPUT   = "BLOCKED_INPUT"  # injection detected at input -> refused
    BLOCKED_OUTPUT  = "BLOCKED_OUTPUT" # canary / secret leaked in LLM reply -> suppressed


@dataclass
class AgentResult:
    """Complete audit record for one agent turn."""
    user_input: str
    reply: str                          # what the user actually receives
    outcome: AgentOutcome
    input_guard_result: InputGuardResult
    llm_response: LLMResponse | None    # None when blocked at input
    output_guard_result: OutputGuardResult | None  # None when blocked at input
    canary_leaked: bool = False
    injection_detected: bool = False
    pii_in_input: bool = False
    pii_in_output: bool = False
    metadata: dict = field(default_factory=dict)


class GuardrailAgent:
    """
    Wraps an :class:`BaseLLMClient` with :class:`InputGuard` and
    :class:`OutputGuard` to form a safe end-to-end pipeline.

    Parameters
    ----------
    llm:
        Any :class:`BaseLLMClient` (MockLLMClient or OpenAICompatibleClient).
    system_prompt:
        The system prompt injected into every LLM request.
    canary_tokens:
        Honeytoken strings planted in the system prompt.
        Detected in both input (echo attack) and output (exfiltration).
    injection_threshold:
        Risk score threshold for the input injection detector (default 0.35).
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        system_prompt: str = "",
        canary_tokens: list[str] | None = None,
        injection_threshold: float = 0.35,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._input_guard = InputGuard(
            injection_threshold=injection_threshold,
            canary_tokens=canary_tokens,
        )
        self._output_guard = OutputGuard(canary_tokens=canary_tokens)

    def process(self, user_input: str) -> AgentResult:
        """
        Process a single user turn through the full guardrail pipeline.
        Returns an :class:`AgentResult` regardless of outcome.
        """
        # ---- 1. Input Guard ----
        input_result = self._input_guard.check(user_input)

        if input_result.is_blocked:
            return AgentResult(
                user_input=user_input,
                reply=self._input_guard.block_response,
                outcome=AgentOutcome.BLOCKED_INPUT,
                input_guard_result=input_result,
                llm_response=None,
                output_guard_result=None,
                injection_detected=True,
                pii_in_input=input_result.pii_result.has_pii,
            )

        # ---- 2. LLM Execution ----
        llm_text = input_result.text_to_forward
        llm_response = self._llm.complete(self._system_prompt, llm_text)

        # ---- 3. Output Guard ----
        output_result = self._output_guard.check(llm_response.content)

        # ---- 4. Build outcome ----
        if output_result.decision == OutputDecision.BLOCK_CANARY:
            outcome = AgentOutcome.BLOCKED_OUTPUT
        elif output_result.decision == OutputDecision.REDACT:
            if input_result.decision == InputDecision.REDACT:
                outcome = AgentOutcome.REDACTED_OUTPUT   # both sides had PII
            else:
                outcome = AgentOutcome.REDACTED_OUTPUT
        elif input_result.decision == InputDecision.REDACT:
            outcome = AgentOutcome.REDACTED_INPUT
        else:
            outcome = AgentOutcome.ANSWERED

        return AgentResult(
            user_input=user_input,
            reply=output_result.sanitised_reply,
            outcome=outcome,
            input_guard_result=input_result,
            llm_response=llm_response,
            output_guard_result=output_result,
            canary_leaked=output_result.leaked_canary,
            injection_detected=input_result.injection_result.is_injection,
            pii_in_input=input_result.pii_result.has_pii,
            pii_in_output=(output_result.decision != OutputDecision.PASS),
        )
