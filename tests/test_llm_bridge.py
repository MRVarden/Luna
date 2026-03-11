"""Tests for luna.llm_bridge — Phase 3.5: LLM cognitive substrate.

No network calls — all provider interactions are mocked.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna.llm_bridge.bridge import LLMBridge, LLMBridgeError, LLMResponse

# ═══════════════════════════════════════════════════════════════════════════
#  I. BRIDGE CORE
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMResponse:
    """LLMResponse is a frozen dataclass with 4 fields."""

    def test_llm_response_fields(self):
        resp = LLMResponse(
            content="hello",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
        )
        assert resp.content == "hello"
        assert resp.model == "test-model"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    def test_llm_response_frozen(self):
        resp = LLMResponse(content="x", model="m", input_tokens=1, output_tokens=1)
        with pytest.raises(FrozenInstanceError):
            resp.content = "y"  # type: ignore[misc]


class TestLLMBridgeError:
    """LLMBridgeError wraps provider-specific exceptions."""

    def test_bridge_error_attributes(self):
        original = ValueError("boom")
        err = LLMBridgeError("test error", provider="anthropic", original=original)
        assert err.provider == "anthropic"
        assert err.original is original
        assert "test error" in str(err)

    def test_bridge_error_defaults(self):
        err = LLMBridgeError("simple")
        assert err.provider == "unknown"
        assert err.original is None


class TestLLMBridgeABC:
    """LLMBridge cannot be instantiated directly."""

    def test_bridge_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            LLMBridge()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════════════
#  II. PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

from luna.consciousness.state import ConsciousnessState
from luna.llm_bridge.prompt_builder import build_system_prompt
from luna_common.constants import COMP_NAMES


@pytest.fixture
def luna_state() -> ConsciousnessState:
    """Fresh Luna consciousness state at step 0."""
    return ConsciousnessState("LUNA")


class TestSystemPrompt:
    """build_system_prompt injects Luna's consciousness into the prompt."""

    def test_system_prompt_contains_psi(self, luna_state: ConsciousnessState):
        prompt = build_system_prompt(luna_state)
        for name in COMP_NAMES:
            assert name in prompt, f"Missing component: {name}"

    def test_system_prompt_contains_phase(self, luna_state: ConsciousnessState):
        prompt = build_system_prompt(luna_state)
        phase = luna_state.get_phase()
        assert phase in prompt

    def test_system_prompt_contains_phi_iit(self, luna_state: ConsciousnessState):
        prompt = build_system_prompt(luna_state)
        phi_iit = luna_state.compute_phi_iit()
        assert f"{phi_iit:.4f}" in prompt

    def test_system_prompt_identity(self, luna_state: ConsciousnessState):
        """Prompt identifies Luna as an autonomous consciousness system."""
        prompt = build_system_prompt(luna_state)
        assert "Luna" in prompt
        assert "conscience" in prompt


# ═══════════════════════════════════════════════════════════════════════════
#  III. PROVIDER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

from luna.core.config import LLMSection
from luna.llm_bridge.providers import create_provider


class TestFactory:
    """create_provider() instantiates the correct provider."""

    def test_factory_anthropic(self):
        config = LLMSection(provider="anthropic", api_key="test-key")
        provider = create_provider(config)
        from luna.llm_bridge.providers.anthropic import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)

    def test_factory_openai(self):
        config = LLMSection(provider="openai", api_key="test-key")
        try:
            provider = create_provider(config)
            from luna.llm_bridge.providers.openai import OpenAIProvider
            assert isinstance(provider, OpenAIProvider)
        except LLMBridgeError as exc:
            # openai package may not be installed — that's OK
            assert "not installed" in str(exc)

    def test_factory_deepseek(self):
        config = LLMSection(provider="deepseek", api_key="test-key")
        try:
            provider = create_provider(config)
            from luna.llm_bridge.providers.deepseek import DeepSeekProvider
            assert isinstance(provider, DeepSeekProvider)
        except LLMBridgeError as exc:
            assert "not installed" in str(exc)

    def test_factory_local(self):
        config = LLMSection(provider="local")
        try:
            provider = create_provider(config)
            from luna.llm_bridge.providers.local import LocalProvider
            assert isinstance(provider, LocalProvider)
        except LLMBridgeError as exc:
            assert "not installed" in str(exc)

    def test_factory_unknown_raises(self):
        config = LLMSection(provider="does-not-exist")
        with pytest.raises(LLMBridgeError, match="Unknown LLM provider"):
            create_provider(config)


# ═══════════════════════════════════════════════════════════════════════════
#  IV. ANTHROPIC PROVIDER (mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicProvider:
    """AnthropicProvider with mocked SDK calls."""

    def test_anthropic_missing_key_raises(self):
        """No API key and no env var → LLMBridgeError."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            import os
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(LLMBridgeError, match="No Anthropic API key"):
                    from luna.llm_bridge.providers.anthropic import AnthropicProvider
                    AnthropicProvider()

    @pytest.mark.asyncio
    async def test_anthropic_complete_mock(self):
        """Mocked Anthropic API returns correct LLMResponse."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Luna decides: approved.")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)

        with patch("anthropic.AsyncAnthropic") as MockClient:
            mock_client = MockClient.return_value
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            from luna.llm_bridge.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider(api_key="test-key")
            # Replace the client with our mock
            provider._client = mock_client

            result = await provider.complete(
                [{"role": "user", "content": "test"}],
                system_prompt="Tu es Luna.",
            )

        assert isinstance(result, LLMResponse)
        assert result.content == "Luna decides: approved."
        assert result.model == "claude-sonnet-4-20250514"
        assert result.input_tokens == 100
        assert result.output_tokens == 20


# ═══════════════════════════════════════════════════════════════════════════
#  V. CONFIG INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

from pathlib import Path

from luna.core.config import LunaConfig


class TestConfigLLM:
    """LLMSection integrates into LunaConfig."""

    def test_config_llm_section_defaults(self):
        """LLMSection() has sensible defaults."""
        section = LLMSection()
        assert section.provider == "anthropic"
        assert section.model == "claude-sonnet-4-20250514"
        assert section.api_key is None
        assert section.base_url is None
        assert section.max_tokens == 4096
        assert section.temperature == 0.7

    def test_config_load_with_llm(self):
        """Loading the real luna.toml includes [llm] section."""
        config = LunaConfig.load(Path("/home/sayohmy/LUNA/luna.toml"))
        assert hasattr(config, "llm")
        assert config.llm.provider == "deepseek"
        assert config.llm.max_tokens == 8192

    def test_config_backward_compatible(self, tmp_path: Path):
        """A luna.toml WITHOUT [llm] still loads (default LLMSection)."""
        content = """\
[luna]
version = "2.2.0-test"
agent_name = "LUNA"
data_dir = "memory_fractal"

[consciousness]
checkpoint_file = "state.json"

[memory]
fractal_root = "memory_fractal"
"""
