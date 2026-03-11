"""Verdict — final validation protocol.

Runs the complete benchmark suite comparing cognition-guided
performance vs baseline. Applies 5 criteria to determine if the
cognitive system is VALIDATED or DECORATIVE.

Criteria:
  1. Performance improvement (mean cognitive score > mean baseline)
  2. No catastrophic regression (worst-case delta > -1/phi^3)
  3. Coherence (PHI_IIT > 0.618 for 60% of steps)
  4. Adaptability (improvement across different task types)
  5. Effect size (positive improvement / total change > 1/phi)

4/5 criteria met → VALIDATED
< 3/5 criteria met → DECORATIVE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from luna_common.constants import INV_PHI, INV_PHI3

from luna.validation.comparator import Comparator

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VerdictCriterion:
    """A single verdict criterion evaluation."""

    name: str
    description: str
    passed: bool
    value: float
    threshold: float


@dataclass(frozen=True, slots=True)
class Verdict:
    """Final verdict from the validation protocol."""

    result: str  # "VALIDATED" or "DECORATIVE"
    criteria_met: int
    total_criteria: int
    criteria: list[VerdictCriterion]
    baseline_mean: float
    consciousness_mean: float
    improvement_pct: float

    def to_dict(self) -> dict:
        return {
            "result": self.result,
            "criteria_met": self.criteria_met,
            "total_criteria": self.total_criteria,
            "criteria": [
                {
                    "name": c.name,
                    "description": c.description,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                }
                for c in self.criteria
            ],
            "baseline_mean": self.baseline_mean,
            "consciousness_mean": self.consciousness_mean,
            "improvement_pct": self.improvement_pct,
        }


class VerdictRunner:
    """Runs the final verdict protocol.

    Evaluates 5 criteria to determine if the cognitive system
    provides measurable, non-decorative value.

    All thresholds are phi-derived.
    """

    def __init__(
        self,
        coherence_threshold: float = INV_PHI,  # 0.618
        coherence_pct: float = 0.80,
        max_regression: float = INV_PHI3,  # 0.236
        effect_size_threshold: float = INV_PHI,  # 0.618
    ) -> None:
        self._coherence_threshold = coherence_threshold
        self._coherence_pct = coherence_pct
        self._max_regression = max_regression
        self._effect_size_threshold = effect_size_threshold
        self._comparator = Comparator()

    def evaluate(
        self,
        baseline_scores: list[float],
        consciousness_scores: list[float],
        phi_iit_history: list[float],
        task_categories: dict[str, list[tuple[float, float]]] | None = None,
    ) -> Verdict:
        """Evaluate all 5 criteria and produce a verdict.

        Args:
            baseline_scores: Scores from baseline runs.
            consciousness_scores: Scores from cognition-guided runs.
            phi_iit_history: History of PHI_IIT values during cognitive runs.
            task_categories: Optional per-category (baseline, conscious) scores.

        Returns:
            Verdict with all criteria evaluations.
        """
        comparison = self._comparator.compare(baseline_scores, consciousness_scores)
        criteria: list[VerdictCriterion] = []

        # Compute per-task deltas
        deltas = [c - b for b, c in zip(baseline_scores, consciousness_scores)]

        # ── Criterion 1: Performance improvement ──
        # Mean cognitive score is higher than mean baseline score.
        perf = VerdictCriterion(
            name="performance",
            description="Cognition-guided scores are higher on average",
            passed=comparison.delta > 0,
            value=comparison.delta,
            threshold=0.0,
        )
        criteria.append(perf)

        # ── Criterion 2: No catastrophic regression ──
        # Worst-case per-task delta > -INV_PHI3 (-0.236).
        # The cognitive system must not make any task significantly worse.
        worst_delta = min(deltas) if deltas else 0.0
        no_regression = VerdictCriterion(
            name="no_catastrophic_regression",
            description=f"Worst-case delta > -{self._max_regression:.3f}",
            passed=worst_delta > -self._max_regression,
            value=worst_delta,
            threshold=-self._max_regression,
        )
        criteria.append(no_regression)

        # ── Criterion 3: Coherence (PHI_IIT > threshold for X% of steps) ──
        if phi_iit_history:
            above = sum(1 for p in phi_iit_history if p > self._coherence_threshold)
            coherence_ratio = above / len(phi_iit_history)
        else:
            coherence_ratio = 0.0

        coherence = VerdictCriterion(
            name="coherence",
            description=f"PHI_IIT > {self._coherence_threshold:.3f} for {self._coherence_pct*100:.0f}% of steps",
            passed=coherence_ratio >= self._coherence_pct,
            value=coherence_ratio,
            threshold=self._coherence_pct,
        )
        criteria.append(coherence)

        # ── Criterion 4: Adaptability (improvement across categories) ──
        if task_categories and len(task_categories) > 1:
            categories_improved = 0
            for cat_name, pairs in task_categories.items():
                cat_baseline = [p[0] for p in pairs]
                cat_conscious = [p[1] for p in pairs]
                cat_mean_b = sum(cat_baseline) / len(cat_baseline)
                cat_mean_c = sum(cat_conscious) / len(cat_conscious)
                if cat_mean_c > cat_mean_b:
                    categories_improved += 1
            adaptability_ratio = categories_improved / len(task_categories)
        else:
            adaptability_ratio = 1.0 if comparison.delta > 0 else 0.0

        adaptability = VerdictCriterion(
            name="adaptability",
            description="Improvement across different task types",
            passed=adaptability_ratio > 0.5,
            value=adaptability_ratio,
            threshold=0.5,
        )
        criteria.append(adaptability)

        # ── Criterion 5: Effect size ──
        # The magnitude of positive improvement overwhelms negative change.
        # Ratio = sum(positive deltas) / sum(|all deltas|) > INV_PHI (0.618).
        positive_sum = sum(d for d in deltas if d > 0)
        total_abs = sum(abs(d) for d in deltas)
        effect_ratio = positive_sum / total_abs if total_abs > 1e-12 else 0.0

        effect = VerdictCriterion(
            name="effect_size",
            description=f"Positive improvement / total change > {self._effect_size_threshold:.3f}",
            passed=effect_ratio > self._effect_size_threshold,
            value=effect_ratio,
            threshold=self._effect_size_threshold,
        )
        criteria.append(effect)

        # Final verdict
        criteria_met = sum(1 for c in criteria if c.passed)
        result = "VALIDATED" if criteria_met >= 4 else "DECORATIVE"

        verdict = Verdict(
            result=result,
            criteria_met=criteria_met,
            total_criteria=len(criteria),
            criteria=criteria,
            baseline_mean=comparison.baseline_mean,
            consciousness_mean=comparison.consciousness_mean,
            improvement_pct=comparison.improvement_pct,
        )

        log.info(
            "VERDICT: %s (%d/%d criteria met, improvement=%.1f%%)",
            result,
            criteria_met,
            len(criteria),
            comparison.improvement_pct,
        )
        return verdict
