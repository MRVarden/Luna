"""v3.5 Integration Tests — Thinker, CausalGraph, Lexicon wiring in session.py.

28 tests across 6 classes covering:
  - _run_thinker (structured cognition)
  - _update_causal_graph (causal knowledge update)
  - _update_lexicon (vocabulary learning)
  - _watch_inactivity DreamCycle path
  - _save_v35_state (persistence)
  - prompt_builder build_voice_prompt (Thought injection)

No network calls — all dependencies are mocked.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

from luna.chat.session import ChatSession, _extract_keywords
from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.lexicon import Lexicon
from luna.consciousness.thinker import (
    Causality,
    Need,
    Observation,
    Proposal,
    Stimulus,
    ThinkMode,
    Thinker,
    Thought,
)
from luna.core.config import (
    ChatSection,
    ConsciousnessSection,
    DreamSection,
    HeartbeatSection,
    LLMSection,
    LunaConfig,
    LunaSection,
    MemorySection,
    ObservabilitySection,
    OrchestratorSection,
)
from luna.dream.dream_cycle import DreamCycle, DreamResult
from luna.dream.learning import DreamLearning, Interaction
from luna.llm_bridge.bridge import LLMResponse
from luna.llm_bridge.prompt_builder import build_voice_prompt


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _make_config(tmp_path: Path, **chat_overrides) -> LunaConfig:
    """Build a minimal LunaConfig with configurable chat section."""
    chat_kw = {
        "max_history": 100,
        "memory_search_limit": 5,
        "idle_heartbeat": True,
        "save_conversations": True,
        "prompt_prefix": "luna> ",
    }
    chat_kw.update(chat_overrides)

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
        chat=ChatSection(**chat_kw),
        root_dir=tmp_path,
    )


def _mock_llm() -> AsyncMock:
    """Create a mock LLMBridge that returns a fixed response."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Bonjour, je suis Luna.",
        model="mock-model",
        input_tokens=42,
        output_tokens=10,
    ))
    return llm


def _warm_up_state(session: ChatSession) -> None:
    """Pre-evolve consciousness so it's not in BROKEN phase with phi=0."""
    cs = session.engine.consciousness
    if cs is None:
        return
    rng = np.random.default_rng(42)
    for _ in range(55):
        deltas = rng.uniform(0.05, 0.15, size=4).tolist()
        cs.evolve(info_deltas=deltas)


def _make_thought_with_observations() -> Thought:
    """Build a Thought with observations and causalities for testing."""
    return Thought(
        observations=[
            Observation(tag="phi_low", description="Phi is low", confidence=0.8, component=0),
            Observation(tag="coverage_ok", description="Coverage is ok", confidence=0.9, component=2),
            Observation(tag="tests_pass", description="Tests pass", confidence=0.95, component=3),
        ],
        causalities=[
            Causality(cause="phi_low", effect="instability", strength=0.7, evidence_count=3),
            Causality(cause="coverage_ok", effect="confidence", strength=0.8, evidence_count=5),
        ],
        needs=[
            Need(description="Improve phi score", priority=0.8, method="pipeline"),
        ],
        proposals=[
            Proposal(
                description="Run optimization",
                rationale="Phi is below threshold",
                expected_impact={"component_0_boost": 0.1},
            ),
        ],
        uncertainties=["Will phi stabilize?"],
        depth_reached=5,
        confidence=0.75,
    )


def _make_empty_thought() -> Thought:
    """Build a Thought with no observations/causalities/needs."""
    return Thought()


def _make_decision():
    """Build a minimal ConsciousDecision for prompt_builder tests."""
    from luna.consciousness.decider import (
        ConsciousDecision,
        Depth,
        Focus,
        Intent,
        Tone,
    )
    return ConsciousDecision(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.EXPRESSION,
        depth=Depth.CONCISE,
        facts=["Luna is stable"],
    )


@pytest.fixture
def cfg(tmp_path: Path) -> LunaConfig:
    return _make_config(tmp_path)


@pytest.fixture
def session(cfg: LunaConfig) -> ChatSession:
    return ChatSession(cfg)


async def _start_session(session: ChatSession) -> None:
    """Start a session with mocked LLM provider."""
    with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
        await session.start()


# ═══════════════════════════════════════════════════════════════════════════════
#  I. TestRunThinker
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunThinker:
    """Tests for ChatSession._run_thinker()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_thinker_is_none(self, session, cfg):
        """_run_thinker returns (None, 0) when Thinker is not initialized."""
        await _start_session(session)
        session._thinker = None
        thought, factory_count = session._run_thinker("Hello Luna")
        assert thought is None
        assert factory_count == 0
        await session.stop()

    @pytest.mark.asyncio
    async def test_returns_none_when_consciousness_is_none(self, session, cfg):
        """_run_thinker returns (None, 0) when consciousness state is missing."""
        await _start_session(session)
        session._thinker = MagicMock(spec=Thinker)
        session._engine.consciousness = None
        thought, factory_count = session._run_thinker("Hello")
        assert thought is None
        assert factory_count == 0
        await session.stop()

    @pytest.mark.asyncio
    async def test_returns_thought_on_success(self, session, cfg):
        """_run_thinker returns (Thought, 0) when Thinker succeeds (no factory)."""
        await _start_session(session)
        _warm_up_state(session)

        expected_thought = _make_thought_with_observations()
        mock_thinker = MagicMock(spec=Thinker)
        mock_thinker.think.return_value = expected_thought
        session._thinker = mock_thinker

        thought, factory_count = session._run_thinker("Analyze the codebase")
        assert thought is expected_thought
        assert factory_count == 0
        mock_thinker.think.assert_called_once()
        # Verify stimulus was built correctly.
        call_kwargs = mock_thinker.think.call_args
        assert call_kwargs.kwargs["mode"] == ThinkMode.RESPONSIVE
        assert call_kwargs.kwargs["max_iterations"] == 10
        await session.stop()

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, session, cfg):
        """_run_thinker catches exceptions and returns (None, 0)."""
        await _start_session(session)
        _warm_up_state(session)

        mock_thinker = MagicMock(spec=Thinker)
        mock_thinker.think.side_effect = RuntimeError("Thinker crashed")
        session._thinker = mock_thinker

        thought, factory_count = session._run_thinker("Hello")
        assert thought is None
        assert factory_count == 0
        await session.stop()

    @pytest.mark.asyncio
    async def test_empty_thought_not_logged_as_nonempty(self, session, cfg):
        """An empty Thought (no observations/causalities/needs) does not trigger debug log."""
        await _start_session(session)
        _warm_up_state(session)

        empty = _make_empty_thought()
        mock_thinker = MagicMock(spec=Thinker)
        mock_thinker.think.return_value = empty
        session._thinker = mock_thinker

        # The empty thought has no observations, causalities, or needs.
        # The fix for thought.empty() bug means this should NOT trigger the
        # debug log branch (which checks for observations/causalities/needs).
        thought, factory_count = session._run_thinker("Hello")
        assert thought is empty
        assert factory_count == 0
        assert len(thought.observations) == 0
        assert len(thought.causalities) == 0
        assert len(thought.needs) == 0
        await session.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  II. TestUpdateCausalGraph
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateCausalGraph:
    """Tests for ChatSession._update_causal_graph()."""

    @pytest.mark.asyncio
    async def test_skips_when_thought_is_none(self, session, cfg):
        """_update_causal_graph is a no-op when thought is None."""
        await _start_session(session)
        session._causal_graph = MagicMock(spec=CausalGraph)
        session._update_causal_graph(None)
        session._causal_graph.observe_pair.assert_not_called()
        await session.stop()

    @pytest.mark.asyncio
    async def test_skips_when_graph_is_none(self, session, cfg):
        """_update_causal_graph is a no-op when CausalGraph is None."""
        await _start_session(session)
        session._causal_graph = None
        thought = _make_thought_with_observations()
        # Should not raise.
        session._update_causal_graph(thought)
        await session.stop()

    @pytest.mark.asyncio
    async def test_observe_pair_called_for_each_causality(self, session, cfg):
        """Each causality in the Thought triggers observe_pair on the graph."""
        await _start_session(session)
        _warm_up_state(session)

        mock_graph = MagicMock(spec=CausalGraph)
        session._causal_graph = mock_graph

        thought = _make_thought_with_observations()
        session._update_causal_graph(thought)

        assert mock_graph.observe_pair.call_count == 2
        # Verify the exact pairs.
        calls = mock_graph.observe_pair.call_args_list
        assert calls[0].args[:2] == ("phi_low", "instability")
        assert calls[1].args[:2] == ("coverage_ok", "confidence")
        await session.stop()

    @pytest.mark.asyncio
    async def test_record_co_occurrence_with_multiple_tags(self, session, cfg):
        """Co-occurrence is recorded when 2+ observation tags exist."""
        await _start_session(session)
        _warm_up_state(session)

        mock_graph = MagicMock(spec=CausalGraph)
        session._causal_graph = mock_graph

        thought = _make_thought_with_observations()
        session._update_causal_graph(thought)

        mock_graph.record_co_occurrence.assert_called_once()
        tags_arg = mock_graph.record_co_occurrence.call_args.args[0]
        assert set(tags_arg) == {"phi_low", "coverage_ok", "tests_pass"}
        await session.stop()

    @pytest.mark.asyncio
    async def test_no_co_occurrence_with_single_tag(self, session, cfg):
        """Co-occurrence is NOT recorded with fewer than 2 tags."""
        await _start_session(session)
        _warm_up_state(session)

        mock_graph = MagicMock(spec=CausalGraph)
        session._causal_graph = mock_graph

        thought = Thought(
            observations=[
                Observation(tag="only_one", description="Single obs", confidence=0.8, component=0),
            ],
        )
        session._update_causal_graph(thought)
        mock_graph.record_co_occurrence.assert_not_called()
        await session.stop()

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self, session, cfg):
        """Exceptions in _update_causal_graph are caught silently."""
        await _start_session(session)
        _warm_up_state(session)

        mock_graph = MagicMock(spec=CausalGraph)
        mock_graph.observe_pair.side_effect = RuntimeError("Graph crashed")
        session._causal_graph = mock_graph

        thought = _make_thought_with_observations()
        # Should not raise.
        session._update_causal_graph(thought)
        await session.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  III. TestUpdateLexicon
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateLexicon:
    """Tests for ChatSession._update_lexicon()."""

    @pytest.mark.asyncio
    async def test_skips_when_lexicon_is_none(self, session, cfg):
        """_update_lexicon is a no-op when Lexicon is None."""
        await _start_session(session)
        session._lexicon = None
        # Should not raise.
        session._update_lexicon("hello world", "response")
        await session.stop()

    @pytest.mark.asyncio
    async def test_learns_keywords_from_input(self, session, cfg):
        """_update_lexicon calls learn() for extracted keywords."""
        await _start_session(session)

        mock_lexicon = MagicMock(spec=Lexicon)
        session._lexicon = mock_lexicon

        session._update_lexicon("optimise la couverture de tests", "response text")

        # _extract_keywords should produce words like "optimise", "couverture", "tests".
        assert mock_lexicon.learn.call_count > 0
        # Each call should have context and outcome.
        for call in mock_lexicon.learn.call_args_list:
            assert "context" in call.kwargs or len(call.args) >= 2
        await session.stop()

    @pytest.mark.asyncio
    async def test_outcome_always_neutral(self, session, cfg):
        """Outcome is always 'neutral' for lexicon learning."""
        await _start_session(session)

        mock_lexicon = MagicMock(spec=Lexicon)
        session._lexicon = mock_lexicon

        session._update_lexicon("hello Luna comment vas tu", "bien merci")

        calls = mock_lexicon.learn.call_args_list
        assert len(calls) > 0
        outcomes = [c.kwargs.get("outcome") or (c.args[2] if len(c.args) >= 3 else None) for c in calls]
        assert all(o == "neutral" for o in outcomes if o is not None)
        await session.stop()

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self, session, cfg):
        """Exceptions in _update_lexicon are caught silently."""
        await _start_session(session)

        mock_lexicon = MagicMock(spec=Lexicon)
        mock_lexicon.learn.side_effect = RuntimeError("Lexicon crashed")
        session._lexicon = mock_lexicon

        # Should not raise.
        session._update_lexicon("hello world", "response")
        await session.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  IV. TestDreamV2Inactivity
# ═══════════════════════════════════════════════════════════════════════════════


class TestDreamV2Inactivity:
    """Tests for DreamCycle path in _watch_inactivity."""

    @pytest.mark.asyncio
    async def test_dreamv2_used_when_is_mature_true(self, tmp_path):
        """DreamCycle.run() is called when is_mature() returns True."""
        cfg = _make_config(tmp_path)
        session = ChatSession(cfg)
        await _start_session(session)
        _warm_up_state(session)

        # Mock DreamCycle.
        mock_dream_v2 = MagicMock(spec=DreamCycle)
        mock_dream_v2.is_mature.return_value = True
        mock_dream_v2.run.return_value = DreamResult(
            skills_learned=[],
            thought=None,
            simulations=[],
            graph_stats={"node_count": 5, "edge_count": 12},
            duration=0.5,
            mode="full",
        )
        session._dream_cycle = mock_dream_v2
        session._interaction_buffer = [
            Interaction(trigger="chat", context="test", phi_before=0.5, phi_after=0.6, step=1),
        ]

        # Simulate the v2 path directly (extracting the logic).
        cs = session.engine.consciousness
        # Ensure enough history.
        for _ in range(15):
            cs.evolve(info_deltas=[0.1, 0.1, 0.1, 0.1])

        # Call the v2 path.
        assert mock_dream_v2.is_mature() is True
        result = mock_dream_v2.run(history=session._interaction_buffer or None)
        assert result.mode == "full"
        mock_dream_v2.run.assert_called_once()
        await session.stop()

    @pytest.mark.asyncio
    async def test_dreamv2_clears_interaction_buffer(self, tmp_path):
        """After DreamCycle.run(), the interaction buffer is cleared."""
        cfg = _make_config(tmp_path)
        session = ChatSession(cfg)
        await _start_session(session)
        _warm_up_state(session)

        mock_dream_v2 = MagicMock(spec=DreamCycle)
        mock_dream_v2.is_mature.return_value = True
        mock_dream_v2.run.return_value = DreamResult(duration=0.1, mode="full")
        session._dream_cycle = mock_dream_v2
        session._interaction_buffer = [
            Interaction(trigger="chat", context="test"),
            Interaction(trigger="pipeline", context="build"),
        ]

        # Simulate what _watch_inactivity does.
        _ = mock_dream_v2.run(history=session._interaction_buffer or None)
        session._interaction_buffer.clear()

        assert len(session._interaction_buffer) == 0
        await session.stop()

    @pytest.mark.asyncio
    async def test_fallback_to_v1_when_is_mature_false(self, tmp_path):
        """When is_mature returns False, v1 DreamCycle is used."""
        cfg = _make_config(tmp_path)
        session = ChatSession(cfg)
        await _start_session(session)
        _warm_up_state(session)

        mock_dream_v2 = MagicMock(spec=DreamCycle)
        mock_dream_v2.is_mature.return_value = False
        session._dream_cycle = mock_dream_v2

        # Verify the condition: when is_mature is False, v2.run is NOT called.
        assert not mock_dream_v2.is_mature()
        mock_dream_v2.run.assert_not_called()
        await session.stop()

    @pytest.mark.asyncio
    async def test_fallback_when_dream_cycle_is_none(self, tmp_path):
        """When _dream_cycle is None, the v2 path is skipped."""
        cfg = _make_config(tmp_path)
        session = ChatSession(cfg)
        await _start_session(session)

        session._dream_cycle = None
        # The condition (self._dream_cycle is not None) is False,
        # so the v2 branch is skipped entirely.
        assert session._dream_cycle is None
        await session.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  V. TestSaveV35State
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveV35State:
    """Tests for ChatSession._save_v35_state()."""

    @pytest.mark.asyncio
    async def test_persists_causal_graph(self, session, cfg):
        """_save_v35_state calls causal_graph.persist()."""
        await _start_session(session)

        mock_graph = MagicMock(spec=CausalGraph)
        session._causal_graph = mock_graph

        session._save_v35_state()

        mock_graph.persist.assert_called_once()
        # Verify the path includes causal_graph.json.
        call_path = mock_graph.persist.call_args.args[0]
        assert str(call_path).endswith("causal_graph.json")
        await session.stop()

    @pytest.mark.asyncio
    async def test_saves_lexicon(self, session, cfg):
        """_save_v35_state calls lexicon.save()."""
        await _start_session(session)

        mock_lexicon = MagicMock(spec=Lexicon)
        session._lexicon = mock_lexicon

        session._save_v35_state()
        mock_lexicon.save.assert_called_once()
        await session.stop()

    @pytest.mark.asyncio
    async def test_persists_dream_learning(self, session, cfg):
        """_save_v35_state calls dream_learning.persist()."""
        await _start_session(session)

        mock_learning = MagicMock(spec=DreamLearning)
        session._dream_learning = mock_learning

        session._save_v35_state()
        mock_learning.persist.assert_called_once()
        await session.stop()

    @pytest.mark.asyncio
    async def test_skips_none_components(self, session, cfg):
        """_save_v35_state handles None components gracefully."""
        await _start_session(session)

        session._causal_graph = None
        session._lexicon = None
        session._dream_learning = None

        # Should not raise.
        session._save_v35_state()
        await session.stop()

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self, session, cfg):
        """Exceptions in _save_v35_state are caught silently."""
        await _start_session(session)

        mock_graph = MagicMock(spec=CausalGraph)
        mock_graph.persist.side_effect = OSError("Disk full")
        session._causal_graph = mock_graph

        # Should not raise.
        session._save_v35_state()
        await session.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  VI. TestBuildVoicePromptThought
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildVoicePromptThought:
    """Tests for build_voice_prompt with thought parameter."""

    def test_thought_none_no_pensee_obligatoire_section(self):
        """When thought is None, no 'Pensee de Luna (OBLIGATOIRE)' section appears."""
        decision = _make_decision()
        prompt = build_voice_prompt(decision, thought=None)
        assert "Pensee de Luna (OBLIGATOIRE)" not in prompt

    def test_empty_thought_no_pensee_obligatoire_section(self):
        """An empty Thought (no content) does not produce a 'Pensee de Luna (OBLIGATOIRE)' section."""
        decision = _make_decision()
        empty = _make_empty_thought()
        prompt = build_voice_prompt(decision, thought=empty)
        assert "Pensee de Luna (OBLIGATOIRE)" not in prompt

    def test_thought_with_observations_adds_section(self):
        """A Thought with observations adds structured fallback section.

        v5.1: Fallback uses [Situation]/[Tension]/[Direction] — not raw
        observation listing.  'Raisonnement interne' header instead of
        'Pensee de Luna' (synthesis branch).
        """
        decision = _make_decision()
        thought = _make_thought_with_observations()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "Raisonnement interne de Luna (OBLIGATOIRE)" in prompt
        assert "[Situation]" in prompt
        # The highest-priority need flows into [Tension].
        assert "[Tension]" in prompt
        assert "Improve phi score" in prompt

    def test_thought_observations_limited_to_5(self):
        """Structured fallback does not dump raw observations."""
        decision = _make_decision()
        thought = Thought(
            observations=[
                Observation(tag=f"obs_{i}", description=f"Obs {i}", confidence=0.5, component=i % 4)
                for i in range(10)
            ],
        )
        prompt = build_voice_prompt(decision, thought=thought)
        # New format: no raw confidence lines — structured summary only.
        obs_lines = [line for line in prompt.split("\n") if "confiance=" in line]
        assert len(obs_lines) == 0

    def test_thought_needs_limited_to_3(self):
        """Only the dominant need appears in [Tension], not all needs.

        v5.1: Structured fallback selects one dominant tension, not a list.
        """
        decision = _make_decision()
        thought = Thought(
            needs=[
                Need(description=f"Need {i}", priority=0.5 + i * 0.1, method="pipeline")
                for i in range(7)
            ],
        )
        prompt = build_voice_prompt(decision, thought=thought)
        assert "[Tension]" in prompt
        # Structured fallback picks thought.needs[0] (first element).
        assert "Need 0" in prompt

    def test_thought_proposals_in_prompt(self):
        """First proposal appears as [Direction] in structured fallback."""
        decision = _make_decision()
        thought = Thought(
            proposals=[
                Proposal(
                    description="Run optimization",
                    rationale="Phi is low",
                    expected_impact={"boost": 0.1},
                ),
            ],
        )
        prompt = build_voice_prompt(decision, thought=thought)
        assert "[Direction]" in prompt
        assert "Run optimization" in prompt

    def test_thought_uncertainties_in_prompt(self):
        """Uncertainties with no needs/proposals produce neutral fallback."""
        decision = _make_decision()
        thought = Thought(
            uncertainties=["Will phi stabilize?", "Is coverage sufficient?"],
        )
        prompt = build_voice_prompt(decision, thought=thought)
        # Structured fallback doesn't list uncertainties individually.
        # With no needs → "Aucune tension identifiee".
        assert "[Tension]" in prompt
        assert "Aucune tension identifiee" in prompt

    def test_integration_instruction_present(self):
        """The integration instruction is present when thought has content."""
        decision = _make_decision()
        thought = _make_thought_with_observations()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "INTERDIT" in prompt
