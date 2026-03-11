"""Tests for LocalProvider — dual backend (openai SDK / httpx) + Ollama native.

15 tests across 4 classes:
  TestUrlDetection       — 5 tests (Ollama native vs OpenAI-compat heuristic)
  TestBackendSelection   — 2 tests (SDK preferred, httpx fallback)
  TestHttpxOpenAICompat  — 4 tests (llama.cpp, vLLM, LM Studio path)
  TestHttpxOllamaNative  — 4 tests (Ollama /api/chat path)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna.llm_bridge.bridge import LLMBridgeError
from luna.llm_bridge.providers.local import LocalProvider, _is_ollama_native


# ═══════════════════════════════════════════════════════════════════════════
#  I. URL Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestUrlDetection:
    """Heuristic: Ollama native vs OpenAI-compatible from base_url."""

    def test_ollama_bare_port(self):
        assert _is_ollama_native("http://localhost:11434") is True

    def test_ollama_with_v1_is_compat(self):
        assert _is_ollama_native("http://localhost:11434/v1") is False

    def test_ollama_api_chat(self):
        assert _is_ollama_native("http://localhost:11434/api/chat") is True

    def test_llamacpp_v1(self):
        assert _is_ollama_native("http://localhost:8080/v1") is False

    def test_vllm_default(self):
        assert _is_ollama_native("http://localhost:8000/v1") is False

    def test_custom_port_no_v1(self):
        """Unknown port without /v1 → NOT native (only 11434 triggers)."""
        assert _is_ollama_native("http://localhost:9999") is False

    def test_ollama_api_prefix(self):
        assert _is_ollama_native("http://myhost:11434/api/generate") is True


# ═══════════════════════════════════════════════════════════════════════════
#  II. Backend Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendSelection:
    """Verify openai SDK preferred, httpx fallback."""

    def test_prefers_openai_sdk(self):
        """With openai installed, backend should be 'openai_sdk'."""
        provider = LocalProvider(model="test", base_url="http://localhost:8080/v1")
        assert provider._backend == "openai_sdk"
        assert provider._client is not None
        assert provider._http is None

    def test_fallback_to_httpx(self):
        """Without openai, falls back to httpx."""
        with patch.dict("sys.modules", {"openai": None}):
            # Force ImportError on `import openai`
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "openai":
                    raise ImportError("mocked")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                provider = LocalProvider(
                    model="test",
                    base_url="http://localhost:8080/v1",
                )
                assert provider._backend == "httpx"
                assert provider._http is not None
                assert provider._client is None


# ═══════════════════════════════════════════════════════════════════════════
#  III. httpx — OpenAI-compatible backend
# ═══════════════════════════════════════════════════════════════════════════

class TestHttpxOpenAICompat:
    """Mock httpx responses for OpenAI-compatible servers."""

    def _make_httpx_provider(self, base_url: str = "http://localhost:8080/v1") -> LocalProvider:
        """Create a provider forced into httpx backend."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            return LocalProvider(model="llama3", base_url=base_url)

    @pytest.mark.asyncio
    async def test_complete_with_usage(self):
        """Full OpenAI-compat response with usage field."""
        provider = self._make_httpx_provider()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Bonjour!"}}],
            "model": "llama3",
            "usage": {"prompt_tokens": 42, "completion_tokens": 10},
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            [{"role": "user", "content": "Hello"}],
            system_prompt="Tu es Luna.",
        )

        assert result.content == "Bonjour!"
        assert result.model == "llama3"
        assert result.input_tokens == 42
        assert result.output_tokens == 10

    @pytest.mark.asyncio
    async def test_complete_without_usage(self):
        """Server returns no usage field — tokens default to 0."""
        provider = self._make_httpx_provider()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "model": "test-model",
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_server_error_raises_bridge_error(self):
        """HTTP 500 → LLMBridgeError."""
        provider = self._make_httpx_provider()

        import httpx
        provider._http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(),
            ),
        )

        with pytest.raises(LLMBridgeError, match="Local LLM error"):
            await provider.complete([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_connection_error_raises_bridge_error(self):
        """Connection refused → LLMBridgeError."""
        provider = self._make_httpx_provider()

        import httpx
        provider._http.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with pytest.raises(LLMBridgeError, match="Local LLM error"):
            await provider.complete([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_url_construction_without_v1(self):
        """base_url without /v1 gets /v1/chat/completions appended."""
        provider = self._make_httpx_provider("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        await provider.complete([{"role": "user", "content": "test"}])

        call_url = provider._http.post.call_args[0][0]
        assert "/v1/chat/completions" in call_url

    @pytest.mark.asyncio
    async def test_auth_header_when_api_key_set(self):
        """Non-'ollama' api_key → Authorization header."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            provider = LocalProvider(
                model="test", base_url="http://localhost:8080/v1",
                api_key="sk-my-secret-key",
            )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }
        provider._http.post = AsyncMock(return_value=mock_response)

        await provider.complete([{"role": "user", "content": "test"}])

        call_headers = provider._http.post.call_args[1]["headers"]
        assert "Authorization" in call_headers
        assert call_headers["Authorization"] == "Bearer sk-my-secret-key"


# ═══════════════════════════════════════════════════════════════════════════
#  IV. httpx — Ollama native backend
# ═══════════════════════════════════════════════════════════════════════════

class TestHttpxOllamaNative:
    """Mock httpx responses for Ollama native /api/chat."""

    def _make_ollama_provider(self) -> LocalProvider:
        """Create a provider forced into httpx + Ollama native."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            return LocalProvider(
                model="llama3",
                base_url="http://localhost:11434",
            )

    @pytest.mark.asyncio
    async def test_complete_ollama_native(self):
        """Standard Ollama /api/chat response."""
        provider = self._make_ollama_provider()
        assert provider._ollama_native is True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Salut!"},
            "prompt_eval_count": 28,
            "eval_count": 15,
            "done": True,
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            [{"role": "user", "content": "Hello"}],
            system_prompt="Tu es Luna.",
        )

        assert result.content == "Salut!"
        assert result.model == "llama3"
        assert result.input_tokens == 28
        assert result.output_tokens == 15

        # Verify URL is /api/chat
        call_url = provider._http.post.call_args[0][0]
        assert call_url.endswith("/api/chat")

    @pytest.mark.asyncio
    async def test_ollama_native_no_eval_counts(self):
        """Ollama response without eval counts → tokens = 0."""
        provider = self._make_ollama_provider()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        result = await provider.complete([{"role": "user", "content": "Hi"}])
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_ollama_native_payload_format(self):
        """Ollama native uses 'options' dict, not flat 'temperature'."""
        provider = self._make_ollama_provider()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "OK"},
            "done": True,
        }

        provider._http.post = AsyncMock(return_value=mock_response)

        await provider.complete(
            [{"role": "user", "content": "test"}],
            max_tokens=2048,
            temperature=0.3,
        )

        call_payload = provider._http.post.call_args[1]["json"]
        assert "options" in call_payload
        assert call_payload["options"]["temperature"] == 0.3
        assert call_payload["options"]["num_predict"] == 2048
        assert call_payload["stream"] is False

    @pytest.mark.asyncio
    async def test_ollama_error_raises_bridge_error(self):
        """Ollama connection failure → LLMBridgeError."""
        provider = self._make_ollama_provider()

        import httpx
        provider._http.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with pytest.raises(LLMBridgeError, match="Ollama error"):
            await provider.complete([{"role": "user", "content": "Hi"}])


# ═══════════════════════════════════════════════════════════════════════════
#  V. close()
# ═══════════════════════════════════════════════════════════════════════════

class TestClose:
    """Resource cleanup."""

    @pytest.mark.asyncio
    async def test_close_httpx_backend(self):
        """close() calls aclose() on httpx client."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            provider = LocalProvider(model="test", base_url="http://localhost:8080/v1")

        provider._http.aclose = AsyncMock()
        await provider.close()
        provider._http.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_openai_backend_noop(self):
        """close() is safe no-op for SDK backend."""
        provider = LocalProvider(model="test", base_url="http://localhost:8080/v1")
        assert provider._http is None
        await provider.close()  # should not raise
