"""Unit tests for log redaction.

Redaction has to be precise in both directions. Too loose and a key leaks into a
log that gets pasted into an issue; too broad and it destroys the token counts
that make quota problems diagnosable, which is what happened before this test
existed.
"""

from __future__ import annotations

import pytest

from emrag.observability.logging import is_secret_key


class TestRedactedKeys:
    """Values of these keys must never appear in a log line."""

    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "apikey",
            "api-key",
            "groq_api_key",
            "GEMINI_API_KEY",
            "qdrant_api_key",
            "access_token",
            "refresh_token",
            "auth_token",
            "id_token",
            "bearer_token",
            "authorization",
            "secret",
            "client_secret",
            "hf_webhook_secret",
            "password",
            "passwd",
            "private_key",
            "jwt_hs256_secret",
            "token",
            "bearer",
        ],
    )
    def test_is_redacted(self, key: str) -> None:
        assert is_secret_key(key)


class TestPreservedKeys:
    """These are counters and identifiers, and must stay readable."""

    @pytest.mark.parametrize(
        "key",
        [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "token_count",
            "max_tokens",
            "est_tokens",
            "tokens",
            "tokens_used",
            "tpm_used",
            "tpd_limit",
            "request_id",
            "tenant_id",
            "doc_id",
            "chunk_id",
            "model",
            "provider",
            "latency_ms",
            "status",
            "gate_score",
        ],
    )
    def test_is_not_redacted(self, key: str) -> None:
        assert not is_secret_key(key)


class TestRedactionProcessor:
    """The processor applies the rule, recursively, in place."""

    def test_redacts_and_preserves_in_one_event(self) -> None:
        from emrag.observability.logging import _redact

        event = {
            "event": "llm.call",
            "api_key": "emk_secret_value",
            "prompt_tokens": 120,
            "completion_tokens": 60,
            "model": "llama-3.3-70b-versatile",
        }
        result = _redact(None, "info", dict(event))
        assert result["api_key"] == "***redacted***"
        assert result["prompt_tokens"] == 120
        assert result["completion_tokens"] == 60
        assert result["model"] == "llama-3.3-70b-versatile"

    def test_recurses_into_nested_mappings(self) -> None:
        from emrag.observability.logging import _redact

        result = _redact(None, "info", {"config": {"groq_api_key": "abc", "tpm_limit": 12000}})
        assert result["config"]["groq_api_key"] == "***redacted***"
        assert result["config"]["tpm_limit"] == 12000
