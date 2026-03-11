"""TelemetrySummarizer — converts raw pipeline telemetry into Thinker-readable signals.

Luna logs the "film" (TelemetryEvent timeline). She *thinks* with the summary.
This module bridges the gap: raw events -> TelemetrySummary.

Usage:
    summary = TelemetrySummarizer.summarize(events, recent_pass_rates)
"""

from __future__ import annotations

import statistics
from collections import Counter

from luna_common.schemas.cycle import TelemetryEvent, TelemetrySummary


class TelemetrySummarizer:
    """Stateless converter: list[TelemetryEvent] -> TelemetrySummary."""

    @staticmethod
    def summarize(
        events: list[TelemetryEvent],
        recent_pass_rates: list[float] | None = None,
        recent_veto_count: int = 0,
        recent_cycle_count: int = 1,
    ) -> TelemetrySummary:
        """Produce a TelemetrySummary from a timeline of events.

        Args:
            events: TelemetryEvent list from one pipeline run.
            recent_pass_rates: Test pass rates from last ~10 cycles (for flakiness).
            recent_veto_count: Number of vetos in recent cycles (for veto_frequency).
            recent_cycle_count: Number of recent cycles (denominator for veto_frequency).
        """
        if not events:
            return TelemetrySummary()

        # -- Agent durations ---------------------------------------------------
        agent_durations: dict[str, float] = {}
        for ev in events:
            if ev.event_type == "AGENT_END" and ev.agent:
                duration_ms = ev.data.get("duration_ms", 0)
                agent_durations[ev.agent] = duration_ms

        total_duration_ms = sum(agent_durations.values())

        # Latency bucket (based on total pipeline duration)
        if total_duration_ms < 5_000:
            latency_bucket = "fast"
        elif total_duration_ms < 30_000:
            latency_bucket = "normal"
        elif total_duration_ms < 120_000:
            latency_bucket = "slow"
        else:
            latency_bucket = "outlier"

        # Agent outliers via Median + MAD (robust to the outlier itself).
        # MAD = median(|xᵢ - median|), scaled by 1.4826 for normal consistency.
        # Outlier if xᵢ > median + k * 1.4826 * MAD, with k=3.
        # Fallback: if MAD == 0 (identical durations), use 3× median.
        _MAD_SCALE: float = 1.4826
        _MAD_K: float = 3.0
        _FALLBACK_FACTOR: float = 3.0

        agent_outliers: list[str] = []
        if len(agent_durations) >= 2:
            durations = list(agent_durations.values())
            median_d = statistics.median(durations)
            mad = statistics.median([abs(d - median_d) for d in durations])

            if mad > 0:
                threshold = median_d + _MAD_K * _MAD_SCALE * mad
            elif median_d > 0:
                threshold = _FALLBACK_FACTOR * median_d
            else:
                threshold = float("inf")

            for agent, dur in agent_durations.items():
                if dur > threshold:
                    agent_outliers.append(agent)

        # -- Stderr rate -------------------------------------------------------
        stderr_events = [e for e in events if e.event_type == "STDERR_CHUNK"]
        stderr_rate = len(stderr_events) / max(len(events), 1)

        # -- Veto frequency (from recent history) ------------------------------
        veto_frequency = 0.0
        if recent_cycle_count > 0:
            veto_frequency = min(1.0, recent_veto_count / recent_cycle_count)

        # Veto reasons from this run
        veto_events = [e for e in events if e.event_type == "VETO_EMITTED"]
        reason_counter: Counter[str] = Counter()
        for ve in veto_events:
            reason = ve.data.get("reason", "unknown")
            reason_counter[reason[:100]] += 1
        veto_top_reasons = [r for r, _ in reason_counter.most_common(3)]

        # -- Diff scope ratio --------------------------------------------------
        diff_events = [e for e in events if e.event_type == "DIFF_STATS"]
        diff_scope_ratio = 0.0
        if diff_events:
            last_diff = diff_events[-1]
            lines_changed = (
                last_diff.data.get("lines_added", 0)
                + last_diff.data.get("lines_removed", 0)
            )
            # Use a sensible baseline (500 lines) if no budget known
            diff_scope_ratio = lines_changed / 500.0

        # -- Metric coverage ---------------------------------------------------
        metrics_events = [e for e in events if e.event_type == "METRICS_FED"]
        metric_coverage = 0.0
        if metrics_events:
            last_metrics = metrics_events[-1]
            fed_count = len(last_metrics.data.get("metric_names", []))
            metric_coverage = min(1.0, fed_count / 7.0)  # 7 canonical metrics

        # -- Test pass rate (from this run) ------------------------------------
        test_events = [e for e in events if e.event_type == "TESTS_PROGRESS"]
        test_pass_rate = 1.0
        if test_events:
            total_pass = sum(e.data.get("passed", 0) for e in test_events)
            total_fail = sum(e.data.get("failed", 0) for e in test_events)
            total = total_pass + total_fail
            if total > 0:
                test_pass_rate = total_pass / total

        # -- Manifest parse health ---------------------------------------------
        manifest_events = [e for e in events if e.event_type == "MANIFEST_PARSED"]
        manifest_parse_health = 1.0
        if manifest_events:
            ok_count = sum(1 for e in manifest_events if e.data.get("ok"))
            manifest_parse_health = ok_count / len(manifest_events)

        # -- Flakiness (variance of pass rates over recent cycles) -------------
        flakiness_score = 0.0
        if recent_pass_rates and len(recent_pass_rates) >= 2:
            flakiness_score = min(1.0, statistics.variance(recent_pass_rates))

        return TelemetrySummary(
            pipeline_latency_bucket=latency_bucket,
            agent_latency_outliers=agent_outliers[:4],
            stderr_rate=round(min(1.0, stderr_rate), 4),
            veto_frequency=round(veto_frequency, 4),
            veto_top_reasons=veto_top_reasons,
            diff_scope_ratio=round(diff_scope_ratio, 4),
            metric_coverage=round(metric_coverage, 4),
            test_pass_rate=round(test_pass_rate, 4),
            manifest_parse_health=round(manifest_parse_health, 4),
            flakiness_score=round(flakiness_score, 4),
        )
