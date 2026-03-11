"""Start command — launch the Luna engine."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from luna.core.config import LunaConfig


def start(
    config: str = typer.Option("luna.toml", help="Path to config file"),
    api: bool = typer.Option(False, help="Start the API server"),
    daemon: bool = typer.Option(False, help="Run in background"),
) -> None:
    """Start the Luna cognitive engine."""
    logging.basicConfig(level=logging.INFO)
    cfg = LunaConfig.load(config)

    from luna.orchestrator.cognitive_loop import CognitiveLoop

    if api:
        import uvicorn
        from luna.api.app import create_app

        async def run_with_api() -> None:
            loop = CognitiveLoop(cfg)
            await loop.start()
            app = create_app(loop)
            server_config = uvicorn.Config(
                app, host=cfg.api.host, port=cfg.api.port, log_level="info",
            )
            server = uvicorn.Server(server_config)
            try:
                await server.serve()
            finally:
                await loop.stop()

        asyncio.run(run_with_api())
    else:
        # Non-API mode: start loop and keep running.
        async def run_loop() -> None:
            loop = CognitiveLoop(cfg)
            await loop.start()
            try:
                # Keep alive until interrupted.
                while loop.is_running:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await loop.stop()

        asyncio.run(run_loop())
