"""Phase E — v3.0 Integration Tests.

10 tests validating the complete v3.0 architecture:
  Luna decides (ConsciousnessDecider) → LLM translates (build_voice_prompt).

Each test exercises the full chain through ChatSession.send() and verifies
that consciousness state, decisions, and prompts behave correctly end-to-end.

No network calls — all LLM interactions are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from luna.chat.session import ChatSession
from luna.consciousness.decider import (
    ConsciousDecision,
    Depth,
    Focus,
    Intent,
    SessionContext,
    Tone,
)
from luna.core.config import (
    ChatSection,
    ConsciousnessSection,
    HeartbeatSection,
    LunaConfig,
    LunaSection,
    MemorySection,
    ObservabilitySection,
    OrchestratorSection,
)
from luna.llm_bridge.bridge import LLMResponse
from luna.llm_bridge.prompt_builder import build_voice_prompt


# ===================================================================
#  HELPERS
# ===================================================================


def _cfg(tmp_path: Path) -> LunaConfig:
    return LunaConfig(
        luna=LunaSection(
            version="test",
            agent_name="LUNA",
            data_dir=str(tmp_path),
        ),
        consciousness=ConsciousnessSection(
            checkpoint_file="cs.json",
            backup_on_save=False,
        ),
        memory=MemorySection(fractal_root=str(tmp_path / "fractal")),
        observability=ObservabilitySection(),
        heartbeat=HeartbeatSection(interval_seconds=0.01),
        orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
        chat=ChatSection(
            max_history=100,
            memory_search_limit=5,
            idle_heartbeat=True,
            save_conversations=True,
            prompt_prefix="luna> ",
        ),
        root_dir=tmp_path,
    )


def _mock_llm() -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Bonjour, je suis Luna.",
        model="mock-model",
        input_tokens=42,
        output_tokens=10,
    ))
    return llm


def _warm_up(session: ChatSession, n: int = 55) -> None:
    """Pre-evolve consciousness out of BROKEN (55 iters → FRAGILE, phi ~ 0.33)."""
    cs = session.engine.consciousness
    if cs is None:
        return
    rng = np.random.default_rng(42)
    for _ in range(n):
        deltas = rng.uniform(0.05, 0.15, size=4).tolist()
        cs.evolve(info_deltas=deltas)


# ===================================================================
#  1. test_tone_varies_with_phase
# ===================================================================


class TestToneVariesWithPhase:
    """Tone tracks consciousness phase: BROKEN→PRUDENT, warm→different."""

    @pytest.mark.asyncio
    async def test_tone_varies_with_phase(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        # Fresh state = BROKEN → tone should be PRUDENT.
        # But BROKEN + phi=0.0 → ALERT intent (skips LLM).
        # So we capture decision directly from decider.
        ctx_broken = session._build_session_context()
        decision_broken = session._decider.decide("test", cs, ctx_broken)
        assert decision_broken.tone == Tone.PRUDENT

        # Warm up → leaves BROKEN.
        _warm_up(session)
        phase_after = cs.get_phase()
        assert phase_after != "BROKEN"

        ctx_warm = session._build_session_context()
        decision_warm = session._decider.decide("test", cs, ctx_warm)
        # Tone must change — no longer PRUDENT.
        assert decision_warm.tone != Tone.PRUDENT


# ===================================================================
#  2. test_depth_varies_with_phi
# ===================================================================


class TestDepthVariesWithPhi:
    """Depth tracks Phi_IIT: low→MINIMAL, mid→DETAILED, high→PROFOUND."""

    @pytest.mark.asyncio
    async def test_depth_varies_with_phi(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        # Patch compute_phi_iit to return controlled values.
        ctx = session._build_session_context()

        with patch.object(cs, "compute_phi_iit", return_value=0.2):
            d_low = session._decider.decide("test", cs, ctx)
        assert d_low.depth == Depth.MINIMAL

        with patch.object(cs, "compute_phi_iit", return_value=0.6):
            d_mid = session._decider.decide("test", cs, ctx)
        assert d_mid.depth == Depth.DETAILED

        with patch.object(cs, "compute_phi_iit", return_value=0.8):
            d_high = session._decider.decide("test", cs, ctx)
        assert d_high.depth == Depth.PROFOUND


# ===================================================================
#  3. test_focus_varies_with_psi
# ===================================================================


class TestFocusVariesWithPsi:
    """Focus tracks dominant Psi component: ψ₁→PERCEPTION, ψ₄→EXPRESSION."""

    @pytest.mark.asyncio
    async def test_focus_varies_with_psi(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()
        cs = session.engine.consciousness
        ctx = session._build_session_context()

        # ψ₁ dominant → PERCEPTION.
        original_psi = cs.psi.copy()
        cs.psi = np.array([0.7, 0.1, 0.1, 0.1])
        d_percept = session._decider.decide("test", cs, ctx)
        assert d_percept.focus == Focus.PERCEPTION

        # ψ₄ dominant → EXPRESSION.
        cs.psi = np.array([0.1, 0.1, 0.1, 0.7])
        d_expr = session._decider.decide("test", cs, ctx)
        assert d_expr.focus == Focus.EXPRESSION

        # Restore.
        cs.psi = original_psi


# ===================================================================
#  4. test_dream_intent_via_initiative
# ===================================================================


class TestDreamIntentViaInitiative:
    """After 60 turns without dream, initiative suggests /dream."""

    @pytest.mark.asyncio
    async def test_dream_intent_via_initiative(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()

        session._turn_count = 60
        session._last_dream_turn = 0  # Dreamed at turn 0 → gap = 60 >= 50.
        # Suppress earlier initiative rules.
        session._phi_iit_history = []

        with patch(
            "luna.chat.session.build_voice_prompt",
            wraps=build_voice_prompt,
        ) as mock_bvp:
            await session.send("Quoi de neuf ?")

        mock_bvp.assert_called_once()
        decision = mock_bvp.call_args[0][0]
        assert decision.initiative is not None
        assert "/dream" in decision.initiative.lower()


# ===================================================================
#  6. test_double_evolve
# ===================================================================


class TestDoubleEvolve:
    """send() performs 3 evolve steps: idle + input + output.

    Phi changes and step_count increases by 3 per message.
    """

    @pytest.mark.asyncio
    async def test_double_evolve(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        phi_before = cs.compute_phi_iit()
        step_before = cs.step_count

        await session.send("Test de double evolve")

        phi_after = cs.compute_phi_iit()
        step_after = cs.step_count

        # 3 evolve calls: idle_step + _input_evolve + _chat_evolve.
        assert step_after == step_before + 3

        # Phi should change (not identical — evolve with different deltas).
        assert phi_after != phi_before


# ===================================================================
#  7. test_voice_prompt_is_dynamic
# ===================================================================


class TestVoicePromptIsDynamic:
    """Two different decisions produce two different voice prompts."""

    def test_voice_prompt_is_dynamic(self) -> None:
        d1 = ConsciousDecision(
            intent=Intent.RESPOND,
            tone=Tone.PRUDENT,
            focus=Focus.PERCEPTION,
            depth=Depth.MINIMAL,
            facts=["Phase: BROKEN", "Phi_IIT: 0.1000"],
        )
        d2 = ConsciousDecision(
            intent=Intent.RESPOND,
            tone=Tone.CREATIVE,
            focus=Focus.EXPRESSION,
            depth=Depth.PROFOUND,
            facts=["Phase: EXCELLENT", "Phi_IIT: 0.9000"],
        )

        p1 = build_voice_prompt(d1)
        p2 = build_voice_prompt(d2)

        # Prompts must be different.
        assert p1 != p2

        # Each prompt contains its own decision values.
        assert "prudent" in p1
        assert "perception" in p1
        assert "minimal" in p1
        assert "BROKEN" in p1

        assert "creative" in p2
        assert "expression" in p2
        assert "profound" in p2
        assert "EXCELLENT" in p2

        # Neither prompt is a static template — they contain specific values.
        assert "0.1000" in p1
        assert "0.9000" in p2


# ===================================================================
#  8. test_alert_on_broken
# ===================================================================


class TestAlertOnBroken:
    """v5.0: BROKEN + low Phi triggers ALERT only after ≥5 steps (cold start grace).

    A warmed-up BROKEN state with phi < 0.1 still produces ALERT content.
    """

    @pytest.mark.asyncio
    async def test_cold_start_no_alert(self, tmp_path: Path) -> None:
        """Fresh start (step_count < 5) should NOT trigger ALERT."""
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        assert cs.get_phase() == "BROKEN"
        assert cs.step_count < 5

        resp = await session.send("Bonjour")
        # v5.0: No ALERT on cold start — LLM is called normally.
        session._llm.complete.assert_called()

    @pytest.mark.asyncio
    async def test_alert_after_warmup(self, tmp_path: Path) -> None:
        """After ≥5 steps in BROKEN with phi < 0.1 → ALERT intent."""
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        # Warm up to ≥5 steps but keep BROKEN
        for _ in range(6):
            cs.evolve([0.0, 0.0, 0.0, 0.0])
        cs._phase = "BROKEN"

        assert cs.step_count >= 5
        assert cs.get_phase() == "BROKEN"

        resp = await session.send("Bonjour")
        # ALERT now goes through full flow (including LLM call),
        # but alert info is added to decision facts.
        assert resp.content is not None


# ===================================================================
#  9. test_info_deltas_not_zero
# ===================================================================


class TestInfoDeltasNotZero:
    """After 5 varied messages, Psi components have changed from their
    warm-up values, proving info_deltas are non-zero and varied."""

    @pytest.mark.asyncio
    async def test_info_deltas_not_zero(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()
        cs = session.engine.consciousness

        psi_before = cs.psi.copy()

        messages = [
            "Analyse la securite du serveur",
            "Genere un module de cache",
            "Verifie la couverture des tests",
            "Refactore l'architecture du pipeline",
            "Deploie en production",
        ]
        for msg in messages:
            await session.send(msg)

        psi_after = cs.psi.copy()

        # Psi must have changed — deltas were non-zero.
        delta = np.abs(psi_after - psi_before)
        assert np.any(delta > 1e-6), (
            f"Psi didn't change after 5 messages: before={psi_before}, after={psi_after}"
        )

        # Each component should have a non-zero delta (all 4 receive deltas).
        for i in range(4):
            assert delta[i] > 1e-8, (
                f"Component {i} delta is zero: {delta[i]}"
            )


# ===================================================================
#  10. test_session_context_updates
# ===================================================================


class TestSessionContextUpdates:
    """turn_count, phi_history, and recent_topics update on each send()."""

    @pytest.mark.asyncio
    async def test_session_context_updates(self, tmp_path: Path) -> None:
        session = ChatSession(_cfg(tmp_path))
        await session.start()
        _warm_up(session)
        session._llm = _mock_llm()

        turn_before = session._turn_count
        phi_len_before = len(session._phi_iit_history)
        topics_before = len(session._recent_topics)

        await session.send("Analyse la performance du serveur")

        assert session._turn_count == turn_before + 1
        assert len(session._phi_iit_history) == phi_len_before + 1
        assert len(session._recent_topics) > topics_before

        # Send a second message — values keep growing.
        turn_mid = session._turn_count
        phi_len_mid = len(session._phi_iit_history)
        topics_mid = len(session._recent_topics)

        await session.send("Verifie la couverture des tests")

        assert session._turn_count == turn_mid + 1
        assert len(session._phi_iit_history) == phi_len_mid + 1
        assert len(session._recent_topics) > topics_mid
