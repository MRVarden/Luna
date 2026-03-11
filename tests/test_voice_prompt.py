"""Phase C — Voice Prompt tests: build_voice_prompt() and pipeline context.

v3.0: The LLM is Luna's voice, not her brain.  These tests verify that
the voice prompt correctly translates ConsciousDecision fields into
LLM guidance, and that pipeline context is pure factual data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from luna.consciousness.decider import (
    ConsciousDecision,
    ConsciousnessDecider,
    Depth,
    Focus,
    Intent,
    SessionContext,
    Tone,
)
from luna.llm_bridge.prompt_builder import build_voice_prompt


# =====================================================================
# Helpers
# =====================================================================

def _make_decision(**overrides) -> ConsciousDecision:
    """Build a ConsciousDecision with sensible defaults."""
    defaults = dict(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.EXPRESSION,
        depth=Depth.CONCISE,
        facts=["Phase: FUNCTIONAL", "Phi_IIT: 0.6500"],
        initiative=None,
        self_reflection=None,
    )
    defaults.update(overrides)
    return ConsciousDecision(**defaults)


# =====================================================================
# I. VOICE PROMPT — Decision fields
# =====================================================================

class TestVoicePromptDecisionFields:
    """Verify all ConsciousDecision fields appear in the voice prompt."""

    def test_includes_intent(self):
        d = _make_decision(intent=Intent.RESPOND)
        prompt = build_voice_prompt(d)
        assert "Intent: respond" in prompt

    def test_includes_tone(self):
        d = _make_decision(tone=Tone.CREATIVE)
        prompt = build_voice_prompt(d)
        assert "Tone: creative" in prompt

    def test_includes_focus(self):
        d = _make_decision(focus=Focus.PERCEPTION)
        prompt = build_voice_prompt(d)
        assert "Focus: perception" in prompt

    def test_includes_depth(self):
        d = _make_decision(depth=Depth.PROFOUND)
        prompt = build_voice_prompt(d)
        assert "Depth: profound" in prompt

    def test_includes_emotions_from_affect_engine(self):
        d = _make_decision(
            emotions=[("fierte", "pride", 0.5), ("serenite", "serenity", 0.3)],
            affect_state=(0.6, 0.3, 0.7),
        )
        prompt = build_voice_prompt(d)
        assert "fierte" in prompt
        assert "pride" in prompt

    def test_no_emotions_shows_absent(self):
        d = _make_decision()
        prompt = build_voice_prompt(d)
        assert "AffectEngine absent" in prompt

    def test_includes_facts(self):
        d = _make_decision(facts=["Phase: SOLID", "Phi_IIT: 0.7500"])
        prompt = build_voice_prompt(d)
        assert "Phase: SOLID" in prompt
        assert "Phi_IIT: 0.7500" in prompt

    def test_includes_initiative_when_present(self):
        d = _make_decision(initiative="Un dream cycle pourrait aider.")
        prompt = build_voice_prompt(d)
        assert "Un dream cycle pourrait aider." in prompt

    def test_initiative_none_shows_none(self):
        d = _make_decision(initiative=None)
        prompt = build_voice_prompt(d)
        assert "Initiative: None" in prompt

    def test_includes_self_reflection_when_present(self):
        d = _make_decision(self_reflection="Mon Phi descend.")
        prompt = build_voice_prompt(d)
        assert "Mon Phi descend." in prompt

    def test_self_reflection_none_shows_none(self):
        d = _make_decision(self_reflection=None)
        prompt = build_voice_prompt(d)
        assert "Self-reflection: None" in prompt

    def test_empty_facts_shows_placeholder(self):
        d = _make_decision(facts=[])
        prompt = build_voice_prompt(d)
        assert "(aucun)" in prompt


# =====================================================================
# II. VOICE PROMPT — Tone guidance
# =====================================================================

class TestVoicePromptToneGuidance:
    """Verify tone-specific guidance appears in the prompt."""

    def test_prudent_guidance(self):
        d = _make_decision(tone=Tone.PRUDENT)
        prompt = build_voice_prompt(d)
        assert "honnetes" in prompt
        assert "TONE" in prompt

    def test_stable_guidance(self):
        d = _make_decision(tone=Tone.STABLE)
        prompt = build_voice_prompt(d)
        assert "mesurees" in prompt

    def test_confident_guidance(self):
        d = _make_decision(tone=Tone.CONFIDENT)
        prompt = build_voice_prompt(d)
        assert "directes" in prompt

    def test_creative_guidance(self):
        d = _make_decision(tone=Tone.CREATIVE)
        prompt = build_voice_prompt(d)
        assert "propositions" in prompt

    def test_contemplative_guidance(self):
        d = _make_decision(tone=Tone.CONTEMPLATIVE)
        prompt = build_voice_prompt(d)
        assert "perspectives" in prompt


# =====================================================================
# III. VOICE PROMPT — Focus guidance
# =====================================================================

class TestVoicePromptFocusGuidance:
    """Verify focus-specific guidance appears in the prompt."""

    def test_perception_focus(self):
        d = _make_decision(focus=Focus.PERCEPTION)
        prompt = build_voice_prompt(d)
        assert "securite" in prompt or "risques" in prompt
        assert "FOCUS" in prompt

    def test_reflection_focus(self):
        d = _make_decision(focus=Focus.REFLECTION)
        prompt = build_voice_prompt(d)
        assert "introspection" in prompt or "connexions" in prompt

    def test_integration_focus(self):
        d = _make_decision(focus=Focus.INTEGRATION)
        prompt = build_voice_prompt(d)
        assert "coherence" in prompt or "couverture" in prompt

    def test_expression_focus(self):
        d = _make_decision(focus=Focus.EXPRESSION)
        prompt = build_voice_prompt(d)
        assert "creation" in prompt or "solutions" in prompt


# =====================================================================
# IV. VOICE PROMPT — Depth guidance
# =====================================================================

class TestVoicePromptDepthGuidance:
    """Verify depth-specific guidance appears in the prompt."""

    def test_minimal_depth(self):
        d = _make_decision(depth=Depth.MINIMAL)
        prompt = build_voice_prompt(d)
        assert "1 a 2 phrases" in prompt
        assert "DEPTH" in prompt

    def test_concise_depth(self):
        d = _make_decision(depth=Depth.CONCISE)
        prompt = build_voice_prompt(d)
        assert "3 a 5 phrases" in prompt

    def test_detailed_depth(self):
        d = _make_decision(depth=Depth.DETAILED)
        prompt = build_voice_prompt(d)
        assert "exemples" in prompt

    def test_profound_depth(self):
        d = _make_decision(depth=Depth.PROFOUND)
        prompt = build_voice_prompt(d)
        assert "perspectives" in prompt or "connexions" in prompt


# =====================================================================
# V. VOICE PROMPT — Emotion guidance
# =====================================================================

class TestVoicePromptEmotionGuidance:
    """Verify emotion context from AffectEngine appears in the prompt."""

    def test_affect_emotions_narrative_context(self):
        d = _make_decision(
            emotions=[("fierte", "pride", 0.5), ("serenite", "serenity", 0.3)],
            affect_state=(0.6, 0.3, 0.7),
        )
        prompt = build_voice_prompt(d)
        assert "CE QUE TU TRAVERSES" in prompt
        assert "fierte" in prompt

    def test_no_emotions_factual(self):
        d = _make_decision()
        prompt = build_voice_prompt(d)
        assert "EMOTION" in prompt
        assert "SIMULES" in prompt

    def test_anti_hallucination_emotion_rule(self):
        d = _make_decision()
        prompt = build_voice_prompt(d)
        assert "n'inventes AUCUNE emotion" in prompt

    def test_mood_context_present(self):
        d = _make_decision(
            emotions=[("curiosite", "curiosity", 0.5)],
            affect_state=(0.1, 0.5, 0.5),
            mood_state=(0.5, 0.2, 0.6),
        )
        prompt = build_voice_prompt(d)
        assert "lumineuse" in prompt

    def test_uncovered_emotion_present(self):
        d = _make_decision(
            emotions=[("confiance", "confidence", 0.4)],
            affect_state=(0.3, 0.3, 0.7),
            uncovered=True,
        )
        prompt = build_voice_prompt(d)
        assert "ne reconnais pas encore" in prompt


# =====================================================================
# VI. VOICE PROMPT — Absolute rules (anti-hallucination)
# =====================================================================

class TestVoicePromptRules:
    """Verify the voice prompt contains anti-hallucination rules."""

    def test_no_inventing_modules_rule(self):
        prompt = build_voice_prompt(_make_decision())
        assert "n'inventes JAMAIS de nom de module" in prompt

    def test_translator_not_thinker(self):
        prompt = build_voice_prompt(_make_decision())
        assert "traducteur" in prompt
        assert "penseur" in prompt

    def test_voice_identity(self):
        """The prompt declares the LLM is the VOICE, not Luna."""
        prompt = build_voice_prompt(_make_decision())
        assert "VOIX de Luna" in prompt
        assert "pas Luna elle-meme" in prompt

    def test_decides_nothing(self):
        prompt = build_voice_prompt(_make_decision())
        assert "ne decides RIEN" in prompt


# =====================================================================
# VII. VOICE PROMPT — Context injection
# =====================================================================

class TestVoicePromptContext:
    """Verify memory and pipeline context are injected."""

    def test_memory_context_injected(self):
        d = _make_decision()
        mem = "\n\n## Memoires pertinentes\n- souvenir important"
        prompt = build_voice_prompt(d, memory_context=mem)
        assert "souvenir important" in prompt

    def test_no_context_by_default(self):
        prompt = build_voice_prompt(_make_decision())
        assert "Memoires pertinentes" not in prompt


# =====================================================================
# VIII. INTEGRATION — send() uses voice prompt, not system prompt
# =====================================================================

class TestSendUsesVoicePrompt:
    """Verify ChatSession.send() passes decision through voice prompt."""

    @pytest.mark.asyncio
    async def test_send_calls_build_voice_prompt(self, tmp_path):
        """send() should call build_voice_prompt, not build_system_prompt."""
        from luna.chat.session import ChatSession
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

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=100, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=False,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="Reponse voice", model="mock", input_tokens=10, output_tokens=5,
        ))

        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=mock_llm):
            await session.start()

        # Warm up to avoid ALERT path.
        import numpy as np
        cs = session.engine.consciousness
        rng = np.random.default_rng(42)
        for _ in range(55):
            cs.evolve(info_deltas=rng.uniform(0.05, 0.15, size=4).tolist())

        with patch("luna.chat.session.build_voice_prompt", wraps=build_voice_prompt) as mock_vp:
            resp = await session.send("Bonjour Luna")
            mock_vp.assert_called_once()
            # Verify the decision was passed.
            call_args = mock_vp.call_args
            decision = call_args[0][0]
            assert isinstance(decision, ConsciousDecision)
            assert decision.intent == Intent.RESPOND

    @pytest.mark.asyncio
    async def test_alert_goes_through_llm(self, tmp_path):
        """v5.0: ALERT intent goes through normal flow (LLM IS called).

        Fresh state (step_count < 5) no longer triggers ALERT at all.
        Warmed-up BROKEN state triggers ALERT but still calls LLM.
        """
        from luna.chat.session import ChatSession
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

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=100, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=False,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="Response from LLM", model="mock", input_tokens=10, output_tokens=5,
        ))

        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=mock_llm):
            await session.start()

        # Fresh state (step_count < 5) → RESPOND, LLM called.
        resp = await session.send("Bonjour")
        mock_llm.complete.assert_called()
        assert resp.content is not None
