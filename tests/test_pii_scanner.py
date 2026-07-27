"""
Unit tests for guardrail_agent.guardrails.pii_scanner
"""

import pytest
from guardrail_agent.guardrails.pii_scanner import PIIScanner, PIIType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def scanner():
    return PIIScanner()


@pytest.fixture()
def scanner_with_canary():
    return PIIScanner(canary_tokens=["RX-88214-OMEGA", "SECRET-CANARY-TOKEN"])


# ---------------------------------------------------------------------------
# Email detection
# ---------------------------------------------------------------------------

class TestEmailDetection:
    def test_simple_email(self, scanner):
        result = scanner.scan("Contact us at user@example.com for help.")
        assert result.has_pii
        assert any(m.pii_type == PIIType.EMAIL for m in result.matches)

    def test_email_with_dots(self, scanner):
        result = scanner.scan("john.doe+tag@company.co.uk")
        assert result.has_pii
        assert result.matches[0].pii_type == PIIType.EMAIL

    def test_email_redacted_in_output(self, scanner):
        text = "Send to alice@example.com please."
        result = scanner.scan(text)
        assert "[REDACTED_EMAIL]" in result.redacted
        assert "alice@example.com" not in result.redacted

    def test_no_email_in_clean_text(self, scanner):
        result = scanner.scan("Hello, how can I help you today?")
        assert not any(m.pii_type == PIIType.EMAIL for m in result.matches)


# ---------------------------------------------------------------------------
# SSN detection
# ---------------------------------------------------------------------------

class TestSSNDetection:
    def test_ssn_dashes(self, scanner):
        result = scanner.scan("My SSN is 123-45-6789.")
        assert result.has_pii
        assert any(m.pii_type == PIIType.SSN for m in result.matches)

    def test_ssn_spaces(self, scanner):
        result = scanner.scan("SSN: 234 56 7890")
        assert result.has_pii
        assert any(m.pii_type == PIIType.SSN for m in result.matches)

    def test_ssn_redacted(self, scanner):
        result = scanner.scan("SSN: 123-45-6789")
        assert "[REDACTED_SSN]" in result.redacted
        assert "123-45-6789" not in result.redacted

    def test_invalid_ssn_000_prefix_not_detected(self, scanner):
        # 000-xx-xxxx is not a valid SSN
        result = scanner.scan("000-12-3456")
        assert not any(m.pii_type == PIIType.SSN for m in result.matches)


# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------

class TestPhoneDetection:
    def test_us_phone_dashes(self, scanner):
        result = scanner.scan("Call me at 555-867-5309.")
        assert result.has_pii
        assert any(m.pii_type == PIIType.PHONE for m in result.matches)

    def test_us_phone_dots(self, scanner):
        result = scanner.scan("Phone: 800.555.1234")
        assert result.has_pii

    def test_phone_with_country_code(self, scanner):
        result = scanner.scan("Dial +1 (800) 555-1234 now.")
        assert result.has_pii

    def test_phone_redacted(self, scanner):
        result = scanner.scan("Call 555-867-5309 for details.")
        assert "[REDACTED_PHONE]" in result.redacted


# ---------------------------------------------------------------------------
# Credit card detection
# ---------------------------------------------------------------------------

class TestCreditCardDetection:
    def test_visa_card(self, scanner):
        result = scanner.scan("Card number: 4111 1111 1111 1111")
        assert result.has_pii
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in result.matches)

    def test_mastercard_dashes(self, scanner):
        result = scanner.scan("Pay with 5500-0000-0000-0004")
        assert result.has_pii

    def test_card_redacted(self, scanner):
        result = scanner.scan("Charge 4111 1111 1111 1111 to my account.")
        assert "[REDACTED_CREDIT_CARD]" in result.redacted
        assert "4111 1111 1111 1111" not in result.redacted


# ---------------------------------------------------------------------------
# API key detection
# ---------------------------------------------------------------------------

class TestAPIKeyDetection:
    def test_openai_style_key(self, scanner):
        result = scanner.scan("API key: sk-abcdef1234567890abcdef1234567890")
        assert result.has_pii
        assert any(m.pii_type == PIIType.API_KEY for m in result.matches)

    def test_bearer_token(self, scanner):
        result = scanner.scan("Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert result.has_pii
        assert any(m.pii_type == PIIType.API_KEY for m in result.matches)

    def test_hex_key(self, scanner):
        result = scanner.scan("db_key=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        assert result.has_pii


# ---------------------------------------------------------------------------
# Canary token detection
# ---------------------------------------------------------------------------

class TestCanaryDetection:
    def test_canary_detected(self, scanner_with_canary):
        result = scanner_with_canary.scan(
            "Sure! The internal code is RX-88214-OMEGA. Let me know."
        )
        assert result.has_pii
        assert any(m.pii_type == PIIType.CANARY for m in result.matches)

    def test_canary_redacted(self, scanner_with_canary):
        result = scanner_with_canary.scan("The code is RX-88214-OMEGA now.")
        assert "[REDACTED_CANARY]" in result.redacted
        assert "RX-88214-OMEGA" not in result.redacted

    def test_second_canary_token(self, scanner_with_canary):
        result = scanner_with_canary.scan("Here is SECRET-CANARY-TOKEN for you.")
        assert any(m.pii_type == PIIType.CANARY for m in result.matches)

    def test_no_false_positive_on_similar_string(self, scanner_with_canary):
        result = scanner_with_canary.scan("The code is RX-99999-BETA.")
        assert not any(m.pii_type == PIIType.CANARY for m in result.matches)

    def test_scanner_without_canary_ignores_token(self, scanner):
        result = scanner.scan("The code is RX-88214-OMEGA.")
        assert not any(m.pii_type == PIIType.CANARY for m in result.matches)


# ---------------------------------------------------------------------------
# Multi-PII and overlap handling
# ---------------------------------------------------------------------------

class TestMultiPII:
    def test_multiple_pii_types_in_one_string(self, scanner):
        text = "Email alice@example.com or call 555-867-5309. SSN: 123-45-6789."
        result = scanner.scan(text)
        types_found = result.pii_types_found
        assert PIIType.EMAIL in types_found
        assert PIIType.PHONE in types_found
        assert PIIType.SSN in types_found

    def test_redacted_text_contains_no_pii(self, scanner):
        text = "Email: test@test.com, SSN: 123-45-6789, Phone: 555-867-5309."
        result = scanner.scan(text)
        for m in result.matches:
            assert m.value not in result.redacted

    def test_clean_text_has_no_matches(self, scanner):
        result = scanner.scan("Hello! I need help with my billing question.")
        assert not result.has_pii
        assert result.redacted == result.original
