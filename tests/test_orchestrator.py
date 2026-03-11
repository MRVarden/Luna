"""Tests for luna.orchestrator — retry policy and config sections.

The LunaOrchestrator (pipeline 4-agents) has been replaced by CognitiveLoop.
This file retains tests for retry.py and OrchestratorSection config, which
are still active modules.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from luna_common.constants import PHI

from luna.core.config import LunaConfig, OrchestratorSection
from luna.llm_bridge.bridge import LLMBridgeError
from luna.orchestrator.retry import RetryPolicy, retry_async

# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

LUNA_TOML_PATH = Path("/home/sayohmy/LUNA/luna.toml")


@pytest.fixture
def config() -> LunaConfig:
    return LunaConfig.load(LUNA_TOML_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  I. RETRY POLICY
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    """RetryPolicy is frozen with PHI-derived defaults."""

    def test_retry_policy_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.backoff_factor == PHI

    def test_retry_policy_frozen(self):
        policy = RetryPolicy()
        with pytest.raises(FrozenInstanceError):
            policy.max_retries = 5  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
#  II. RETRY ASYNC
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryAsync:
    """retry_async executes a callable with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self):
        fn = AsyncMock(return_value=42)
        result = await retry_async(fn)
        assert result == 42
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self):
        fn = AsyncMock(
            side_effect=[
                LLMBridgeError("fail1"),
                42,
            ]
        )
        policy = RetryPolicy(base_delay=0.0)  # No delay in tests
        result = await retry_async(fn, policy=policy)
        assert result == 42
        assert fn.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_async_all_fail(self):
        fn = AsyncMock(side_effect=LLMBridgeError("always fails"))
        policy = RetryPolicy(max_retries=2, base_delay=0.0)
        with pytest.raises(LLMBridgeError, match="always fails"):
            await retry_async(fn, policy=policy)
        assert fn.await_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_async_calls_on_retry(self):
        fn = AsyncMock(
            side_effect=[
                LLMBridgeError("fail1"),
                LLMBridgeError("fail2"),
                99,
            ]
        )
        on_retry = MagicMock()
        policy = RetryPolicy(base_delay=0.0)
        result = await retry_async(fn, policy=policy, on_retry=on_retry)
        assert result == 99
        assert on_retry.call_count == 2
        # First retry callback: attempt=1
        assert on_retry.call_args_list[0][0][0] == 1
        # Second retry callback: attempt=2
        assert on_retry.call_args_list[1][0][0] == 2

    @pytest.mark.asyncio
    async def test_retry_async_does_not_catch_non_llm_errors(self):
        """Only LLMBridgeError is retried — other exceptions propagate immediately."""
        fn = AsyncMock(side_effect=ValueError("logic error"))
        policy = RetryPolicy(base_delay=0.0)
        with pytest.raises(ValueError, match="logic error"):
            await retry_async(fn, policy=policy)
        fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_async_forwards_args_kwargs(self):
        fn = AsyncMock(return_value="ok")
        await retry_async(fn, "arg1", "arg2", policy=RetryPolicy(), key="val")
        fn.assert_awaited_once_with("arg1", "arg2", key="val")


# ═══════════════════════════════════════════════════════════════════════════
#  III. CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestratorConfig:
    """OrchestratorSection defaults and TOML loading."""

    def test_config_orchestrator_defaults(self):
        section = OrchestratorSection()
        assert section.llm_augment is True
        assert section.max_cycles == 0
        assert section.checkpoint_interval == 1
        assert section.cycle_timeout == 600.0
        assert section.retry_max == 3
        assert section.retry_base_delay == 1.0

    def test_config_orchestrator_frozen(self):
        section = OrchestratorSection()
        with pytest.raises(FrozenInstanceError):
            section.llm_augment = False  # type: ignore[misc]

    def test_config_load_with_orchestrator(self, config: LunaConfig):
        """luna.toml includes [orchestrator] section."""
        assert hasattr(config, "orchestrator")
        assert isinstance(config.orchestrator, OrchestratorSection)
        assert config.orchestrator.llm_augment is True
        assert config.orchestrator.max_cycles == 0

    def test_config_load_without_orchestrator_section(self, tmp_path: Path):
        """Config loads with defaults if [orchestrator] is missing from TOML."""
        toml_content = """\
[luna]
version = "2.2.0-test"
agent_name = "LUNA"
data_dir = "data"

[consciousness]
checkpoint_file = "ckpt.json"

[memory]
fractal_root = "memory"
"""
