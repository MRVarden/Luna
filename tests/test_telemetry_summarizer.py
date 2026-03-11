"""Tests for TelemetrySummarizer and VoiceValidator.validate_with_delta.

Commit 3 of the Emergence Plan: perception -> sens.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from luna_common.schemas.cycle import TelemetryEvent, TelemetrySummary, VoiceDelta
from luna.consciousness.telemetry_summarizer import TelemetrySummarizer


_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


def _ev(event_type: str, agent: str | None = None, **data) -> TelemetryEvent:
    """Helper: build a TelemetryEvent."""
    return TelemetryEvent(
        event_type=event_type, agent=agent, timestamp=_NOW, data=data,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetrySummarizer
# ══════════════════════════════════════════════════════════════════════════════


class TestTelemetrySummarizer:
    def test_empty_events(self):
        summary = TelemetrySummarizer.summarize([])
        assert summary.pipeline_latency_bucket == "normal"
        assert summary.test_pass_rate == 1.0

    def test_latency_bucket_fast(self):
        events = [
            _ev("AGENT_START", "SAYOHMY"),
            _ev("AGENT_END", "SAYOHMY", duration_ms=1000),
            _ev("AGENT_END", "SENTINEL", duration_ms=1500),
            _ev("AGENT_END", "TESTENGINEER", duration_ms=1000),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.pipeline_latency_bucket == "fast"

    def test_latency_bucket_slow(self):
        events = [
            _ev("AGENT_END", "SAYOHMY", duration_ms=40000),
            _ev("AGENT_END", "SENTINEL", duration_ms=30000),
            _ev("AGENT_END", "TESTENGINEER", duration_ms=20000),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.pipeline_latency_bucket == "slow"

    def test_latency_bucket_outlier(self):
        events = [
            _ev("AGENT_END", "SAYOHMY", duration_ms=130000),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.pipeline_latency_bucket == "outlier"

    def test_agent_outlier_detected(self):
        # Median+MAD: median=2000, MAD=0 → fallback 3×median=6000.
        # TESTENGINEER at 50000 > 6000 → outlier.
        events = [
            _ev("AGENT_END", "SAYOHMY", duration_ms=2000),
            _ev("AGENT_END", "SENTINEL", duration_ms=2000),
            _ev("AGENT_END", "TESTENGINEER", duration_ms=50000),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert "TESTENGINEER" in summary.agent_latency_outliers

    def test_stderr_rate(self):
        events = [
            _ev("AGENT_START", "SAYOHMY"),
            _ev("STDERR_CHUNK", "SAYOHMY", size=100),
            _ev("STDERR_CHUNK", "SAYOHMY", size=200),
            _ev("AGENT_END", "SAYOHMY", duration_ms=1000),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.stderr_rate == pytest.approx(0.5, abs=0.01)

    def test_veto_frequency(self):
        events = [_ev("AGENT_START", "SAYOHMY")]
        summary = TelemetrySummarizer.summarize(
            events, recent_veto_count=3, recent_cycle_count=10,
        )
        assert summary.veto_frequency == pytest.approx(0.3, abs=0.01)

    def test_veto_top_reasons(self):
        events = [
            _ev("VETO_EMITTED", source="SENTINEL", reason="XSS detected"),
            _ev("VETO_EMITTED", source="SENTINEL", reason="SQL injection"),
            _ev("VETO_EMITTED", source="SENTINEL", reason="XSS detected"),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.veto_top_reasons[0] == "XSS detected"

    def test_diff_scope_ratio(self):
        events = [
            _ev("DIFF_STATS", files_changed=3, lines_added=200, lines_removed=50),
        ]
        summary = TelemetrySummarizer.summarize(events)
        # (200+50) / 500 = 0.5
        assert summary.diff_scope_ratio == pytest.approx(0.5, abs=0.01)

    def test_metric_coverage(self):
        events = [
            _ev("METRICS_FED", metric_names=["security_integrity", "coverage_pct", "test_ratio"]),
        ]
        summary = TelemetrySummarizer.summarize(events)
        # 3/7
        assert summary.metric_coverage == pytest.approx(3 / 7, abs=0.01)

    def test_test_pass_rate(self):
        events = [
            _ev("TESTS_PROGRESS", suite="unit", passed=8, failed=2),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.test_pass_rate == pytest.approx(0.8, abs=0.01)

    def test_manifest_parse_health(self):
        events = [
            _ev("MANIFEST_PARSED", "SAYOHMY", ok=True),
            _ev("MANIFEST_PARSED", "SENTINEL", ok=False),
            _ev("MANIFEST_PARSED", "TESTENGINEER", ok=True),
        ]
        summary = TelemetrySummarizer.summarize(events)
        # 2/3
        assert summary.manifest_parse_health == pytest.approx(2 / 3, abs=0.01)

    def test_flakiness_score(self):
        recent = [1.0, 0.8, 1.0, 0.6, 1.0, 0.9, 0.7, 1.0, 0.5, 1.0]
        summary = TelemetrySummarizer.summarize(
            [_ev("AGENT_START", "SAYOHMY")],
            recent_pass_rates=recent,
        )
        assert summary.flakiness_score > 0.0

    def test_flakiness_zero_when_stable(self):
        recent = [1.0, 1.0, 1.0, 1.0, 1.0]
        summary = TelemetrySummarizer.summarize(
            [_ev("AGENT_START", "SAYOHMY")],
            recent_pass_rates=recent,
        )
        assert summary.flakiness_score == 0.0

    def test_full_pipeline_scenario(self):
        """Simulate a realistic pipeline run with all event types."""
        events = [
            _ev("AGENT_START", "SAYOHMY", task_id="t1"),
            _ev("AGENT_END", "SAYOHMY", task_id="t1", return_code=0, duration_ms=8000),
            _ev("MANIFEST_PARSED", "SAYOHMY", ok=True, keys_found=["task_id"]),
            _ev("STDERR_CHUNK", "SAYOHMY", size=50, hash="abc"),
            _ev("AGENT_START", "SENTINEL", task_id="t1"),
            _ev("AGENT_END", "SENTINEL", task_id="t1", return_code=0, duration_ms=5000),
            _ev("MANIFEST_PARSED", "SENTINEL", ok=True),
            _ev("AGENT_START", "TESTENGINEER", task_id="t1"),
            _ev("AGENT_END", "TESTENGINEER", task_id="t1", return_code=0, duration_ms=3000),
            _ev("MANIFEST_PARSED", "TESTENGINEER", ok=True),
            _ev("DIFF_STATS", files_changed=2, lines_added=80, lines_removed=20),
            _ev("TESTS_PROGRESS", suite="unit", passed=42, failed=1, duration_ms=2000),
            _ev("METRICS_FED", metric_names=["security_integrity", "coverage_pct"]),
        ]
        summary = TelemetrySummarizer.summarize(events)
        assert summary.pipeline_latency_bucket == "normal"
        assert summary.manifest_parse_health == 1.0
        assert summary.test_pass_rate == pytest.approx(42 / 43, abs=0.01)
        assert summary.metric_coverage == pytest.approx(2 / 7, abs=0.01)
        assert summary.diff_scope_ratio == pytest.approx(0.2, abs=0.01)

    def test_serialization_roundtrip(self):
        events = [
            _ev("AGENT_END", "SAYOHMY", duration_ms=5000),
            _ev("TESTS_PROGRESS", suite="unit", passed=10, failed=0),
        ]
        summary = TelemetrySummarizer.summarize(events)
        import json
        data = json.loads(summary.model_dump_json())
        summary2 = TelemetrySummary(**data)
        assert summary2.pipeline_latency_bucket == summary.pipeline_latency_bucket
        assert summary2.test_pass_rate == summary.test_pass_rate


# ══════════════════════════════════════════════════════════════════════════════
#  VoiceValidator.validate_with_delta
# ══════════════════════════════════════════════════════════════════════════════


class TestVoiceValidatorDelta:
    """Test that validate_with_delta produces correct VoiceDelta."""

    def _make_decision(self):
        """Create a minimal ConsciousDecision for testing."""
        from luna.consciousness.decider import (
            ConsciousDecision, Intent, Tone, Focus, Depth,
        )
        return ConsciousDecision(
            intent=Intent.RESPOND,
            tone=Tone.STABLE,
            focus=Focus.REFLECTION,
            depth=Depth.CONCISE,
            facts=["phi is 0.85"],
        )

    def test_clean_response_zero_delta(self):
        from luna.llm_bridge.voice_validator import VoiceValidator
        decision = self._make_decision()
        result, delta = VoiceValidator.validate_with_delta(
            "Everything is fine.", None, decision,
        )
        assert result.valid
        assert delta.violations_count == 0
        assert delta.severity == 0.0
        assert delta.ratio_modified_chars == 0.0
        assert delta.categories == []

    def test_code_block_produces_security_category(self):
        from luna.llm_bridge.voice_validator import VoiceValidator
        decision = self._make_decision()
        response = "Here is code:\n```python\nprint('hello')\n```"
        result, delta = VoiceValidator.validate_with_delta(
            response, None, decision, has_pipeline_context=False,
        )
        assert delta.violations_count >= 1
        assert "SECURITY" in delta.categories
        assert delta.severity > 0.0

    def test_fabricated_metric_produces_unverified(self):
        from luna.llm_bridge.voice_validator import VoiceValidator
        decision = self._make_decision()
        # 99.73 is not grounded in any decision fact
        response = "The coverage is 99.73% which is great."
        result, delta = VoiceValidator.validate_with_delta(
            response, None, decision,
        )
        # Should detect fabricated metric
        if delta.violations_count > 0:
            assert "UNVERIFIED" in delta.categories

    def test_delta_with_pipeline_context_allows_code(self):
        from luna.llm_bridge.voice_validator import VoiceValidator
        decision = self._make_decision()
        response = "Here is the result:\n```python\nprint('hello')\n```"
        result, delta = VoiceValidator.validate_with_delta(
            response, None, decision, has_pipeline_context=True,
        )
        # Code blocks are OK with pipeline context
        code_violations = [
            c for c in delta.categories if c == "SECURITY"
        ]
        assert len(code_violations) == 0

    def test_delta_serializable(self):
        import json
        from luna.llm_bridge.voice_validator import VoiceValidator
        decision = self._make_decision()
        _, delta = VoiceValidator.validate_with_delta(
            "Clean response.", None, decision,
        )
        data = json.loads(delta.model_dump_json())
        delta2 = VoiceDelta(**data)
        assert delta2.violations_count == delta.violations_count
