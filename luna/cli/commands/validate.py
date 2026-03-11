"""Validate command — run validation benchmarks.

Compares baseline (pure physics) vs cognitive (Thinker + Reactor) performance
to determine if the cognitive system is VALIDATED or DECORATIVE.

Baseline: evolve([0,0,0,0]) — only kappa restoring force.
Cognitive: Thinker observations → Reactor deltas → evolve(deltas).
"""

from __future__ import annotations

import asyncio
import json

import typer

from luna.core.config import LunaConfig
from luna.core.luna import LunaEngine
from luna.validation import BenchmarkHarness, VerdictRunner
from luna.validation.verdict_tasks import get_all_tasks, register_all_tasks

# Both engines get this many idle steps before benchmarking.
# Brings the system from BROKEN/phi=0 to a stabilized state
# so we test cognitive VALUE, not cold-start recovery.
WARMUP_STEPS: int = 10


def validate(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str = typer.Option("luna.toml", help="Path to config file"),
) -> None:
    """Run the validation benchmark suite.

    Baseline: pure equation of state (idle_step, zero info_deltas).
    Cognitive: full pipeline (Thinker -> Reactor -> evolve with real deltas).
    """
    cfg = LunaConfig.load(config)

    # ── Baseline run (pure physics — no cognitive pipeline) ──
    typer.echo("Running baseline benchmarks (pure physics)...")
    baseline_engine = LunaEngine(cfg)
    baseline_engine.initialize()
    for _ in range(WARMUP_STEPS):
        baseline_engine.idle_step()
    baseline_harness = BenchmarkHarness(timeout_seconds=30.0)
    baseline_tasks = register_all_tasks(baseline_harness, baseline_engine, cognitive=False)
    baseline_report = asyncio.run(baseline_harness.run_all())
    baseline_scores = [r.score for r in baseline_report.results]

    # ── Cognitive run (Thinker + Reactor pipeline) ──
    typer.echo("Running cognitive benchmarks (Thinker + Reactor)...")
    cognitive_engine = LunaEngine(cfg)
    cognitive_engine.initialize()
    for _ in range(WARMUP_STEPS):
        cognitive_engine.idle_step()
    cognitive_harness = BenchmarkHarness(timeout_seconds=60.0)
    cognitive_tasks = register_all_tasks(cognitive_harness, cognitive_engine, cognitive=True)
    cognitive_report = asyncio.run(cognitive_harness.run_all())
    cognitive_scores = [r.score for r in cognitive_report.results]

    # ── Collect phi_iit history from the cognitive engine ──
    # phi_iit is sampled during every cognitive_step() call in the tasks.
    phi_history = cognitive_engine.phi_iit_samples

    # ── Build per-category score pairs for adaptability criterion ──
    task_categories: dict[str, list[tuple[float, float]]] = {}
    for bt, bs, cs in zip(baseline_tasks, baseline_scores, cognitive_scores):
        cat = bt.category
        if cat not in task_categories:
            task_categories[cat] = []
        task_categories[cat].append((bs, cs))

    # ── Run verdict ──
    runner = VerdictRunner()
    verdict = runner.evaluate(
        baseline_scores,
        cognitive_scores,
        phi_history,
        task_categories=task_categories,
    )

    typer.echo(f"\n{'='*50}")
    typer.echo(f"VERDICT: {verdict.result}")
    typer.echo(f"Criteria met: {verdict.criteria_met}/{verdict.total_criteria}")
    typer.echo(f"Improvement: {verdict.improvement_pct:.1f}%")
    typer.echo(f"{'='*50}")

    if verbose:
        typer.echo()
        for c in verdict.criteria:
            status_str = "PASS" if c.passed else "FAIL"
            typer.echo(f"  [{status_str}] {c.name}: {c.value:.4f} (threshold: {c.threshold})")
        typer.echo()

        # Show per-task comparison
        typer.echo("Per-task scores:")
        for bt, bs, cs in zip(baseline_tasks, baseline_scores, cognitive_scores):
            delta = cs - bs
            arrow = "+" if delta > 0 else ""
            typer.echo(f"  {bt.name:<25s}  baseline={bs:.4f}  cognitive={cs:.4f}  delta={arrow}{delta:.4f}")
        typer.echo()
        typer.echo(json.dumps(verdict.to_dict(), indent=2, default=str))
