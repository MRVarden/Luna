"""Dashboard command — live status display."""

from __future__ import annotations

import time

import typer

from luna_common.constants import PHI

from luna.core.config import LunaConfig
from luna.core.luna import LunaEngine


def dashboard(
    refresh: float = typer.Option(PHI, help="Refresh interval in seconds"),
    config: str = typer.Option("luna.toml", help="Path to config file"),
) -> None:
    """Launch the live dashboard (Phi-modulated refresh)."""
    cfg = LunaConfig.load(config)
    engine = LunaEngine(cfg)
    engine.initialize()

    typer.echo("Luna Dashboard — Press Ctrl+C to exit.")
    typer.echo(f"Refresh: {refresh:.3f}s\n")

    try:
        while True:
            engine.idle_step()
            status_data = engine.get_status()
            # Clear and redisplay
            typer.echo(f"\r--- Step {status_data['step_count']} ---")
            typer.echo(f"Phase: {status_data.get('phase', '?')}")
            typer.echo(f"Quality: {status_data.get('quality_score', 0):.4f}")
            typer.echo(f"PHI IIT: {status_data.get('phi_iit', 0):.4f}")
            typer.echo(f"Identity: {'OK' if status_data.get('identity_preserved') else 'DRIFTED'}")

            # Emotions from AffectEngine (if available via engine).
            affect_engine = getattr(engine, "_affect_engine", None)
            if affect_engine is not None:
                aff = affect_engine.affect
                typer.echo(
                    f"Affect:  V={aff.valence:+.2f}  A={aff.arousal:.2f}  "
                    f"D={aff.dominance:.2f}"
                )
                try:
                    from luna.consciousness.emotion_repertoire import interpret
                    ec = getattr(affect_engine, "event_count", -1)
                    raw = interpret(
                        aff.as_tuple(), affect_engine.mood.as_tuple(),
                        affect_engine._repertoire, event_count=ec,
                    )
                    if raw:
                        emo = ", ".join(f"{ew.fr}" for ew, _ in raw[:3])
                        typer.echo(f"Ressenti: {emo}")
                except Exception:
                    pass

            typer.echo("")
            time.sleep(refresh)
    except KeyboardInterrupt:
        typer.echo("\nDashboard stopped.")
