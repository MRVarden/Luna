"""REPL — Thin async read-eval-print loop for talking to Luna.

Usage::

    from luna.chat.repl import run_repl
    asyncio.run(run_repl(config))
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from luna.core.config import LunaConfig
from luna.chat.session import ChatSession, _COMMANDS

log = logging.getLogger(__name__)

# ── PID file management ──────────────────────────────────────────────

DAEMON_SESSION = "luna-daemon"


def _pid_path(config: LunaConfig) -> Path:
    """PID file for the autonomous daemon."""
    return config.resolve(config.luna.data_dir) / "luna_daemon.pid"


def _kill_existing_daemon(config: LunaConfig) -> None:
    """Kill any running autonomous daemon before starting chat."""
    pid_file = _pid_path(config)
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        log.info("Killed existing daemon (PID %d)", pid)
    except (ValueError, ProcessLookupError, PermissionError):
        pass  # Already dead or stale PID
    finally:
        pid_file.unlink(missing_ok=True)

    # Also kill tmux session if it exists.
    if shutil.which("tmux"):
        subprocess.run(
            ["tmux", "kill-session", "-t", DAEMON_SESSION],
            capture_output=True,
        )


def _spawn_autonomous_daemon(config: LunaConfig) -> None:
    """Spawn the CognitiveLoop daemon after chat exits.

    Uses tmux if available (visual), falls back to background process.
    The daemon continues Luna's cognitive system autonomously.
    """
    config_path = "luna.toml"
    # Find the actual config file relative to root_dir.
    for candidate in [
        config.root_dir / "luna.toml",
        Path("luna.toml"),
    ]:
        if candidate.exists():
            config_path = str(candidate)
            break

    python = sys.executable

    if shutil.which("tmux"):
        # Spawn in a tmux session — user can `tmux attach -t luna-daemon`.
        cmd = f"{python} -m luna start --config {config_path}"
        subprocess.Popen(
            ["tmux", "new-session", "-d", "-s", DAEMON_SESSION, cmd],
            start_new_session=True,
        )
        print(
            f"\033[36mLuna continue en autonome.\033[0m\n"
            f"  tmux attach -t {DAEMON_SESSION}   — voir les ticks\n"
            f"  python3 -m luna chat              — reprendre le dialogue"
        )
    else:
        # Fallback: nohup background process.
        pid_file = _pid_path(config)
        proc = subprocess.Popen(
            [python, "-m", "luna", "start", "--config", config_path],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid_file.write_text(str(proc.pid))
        print(
            f"\033[36mLuna continue en autonome (PID {proc.pid}).\033[0m\n"
            f"  python3 -m luna chat   — reprendre le dialogue"
        )


def _write_pid(config: LunaConfig) -> None:
    """Write current process PID (used by `luna start` daemon mode)."""
    pid_file = _pid_path(config)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


# ── Dashboard API ────────────────────────────────────────────────────


def _start_dashboard_api(
    session: ChatSession,
    config: LunaConfig,
    loop: object | None = None,
) -> None:
    """Start the FastAPI dashboard server in a background thread.

    Binds to 127.0.0.1:{config.api.port} (default 8618).
    The server shares the same ChatSession/Engine as the REPL,
    so the dashboard sees live state.

    If *loop* (a CognitiveLoop) is provided it is passed directly to
    ``create_app`` — it already exposes every attribute the dashboard
    needs.  Otherwise a lightweight shim wrapping the ChatSession is
    used as a backward-compatible fallback.
    """
    try:
        import uvicorn
        from luna.api.app import create_app
    except ImportError:
        log.warning("uvicorn not installed — dashboard API disabled")
        return

    if loop is not None:
        # CognitiveLoop has all public attributes the dashboard needs.
        app = create_app(loop)
    else:
        # Legacy path: wrap ChatSession for backward compat.
        class _SessionShim:
            """Exposes attributes that dashboard.py expects on the orchestrator."""
            def __init__(self, s: ChatSession, cfg: LunaConfig) -> None:
                self.engine = s._engine
                self.config = cfg
                self._chat_session = s

        shim = _SessionShim(session, config)
        app = create_app(shim)

    host = getattr(config.api, "host", "127.0.0.1")
    port = getattr(config.api, "port", 8618)

    # Kill any orphaned server still holding the port.
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
    except OSError:
        # Port busy — try to free it.
        sock.close()
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        for pid in result.stdout.strip().split("\n"):
            if pid.strip():
                try:
                    os.kill(int(pid.strip()), 15)  # SIGTERM
                except (ValueError, ProcessLookupError):
                    pass
        import time as _time
        _time.sleep(0.5)

    server_config = uvicorn.Config(
        app, host=host, port=port,
        log_level="warning",  # quiet — don't pollute the chat
    )
    server = uvicorn.Server(server_config)

    def _run() -> None:
        try:
            server.run()
        except OSError as exc:
            log.warning("Dashboard API failed to start: %s", exc)

    thread = threading.Thread(target=_run, daemon=True, name="luna-dashboard-api")
    thread.start()
    log.info("Dashboard API started on http://%s:%d", host, port)


# ── Banner ───────────────────────────────────────────────────────────


def _banner(session: ChatSession) -> str:
    """Build the startup banner."""
    mode = "LLM" if session.has_llm else "sans LLM (status only)"
    mem = "active" if session.has_memory else "absente"
    endogenous = "actif" if session._endogenous is not None else "inactif"
    api_port = getattr(session._config.api, "port", 8618)
    return (
        f"\n{'=' * 50}\n"
        f"  Luna v{session.engine.config.luna.version} — Chat\n"
        f"  Mode: {mode}\n"
        f"  Memoire: {mem}\n"
        f"  Autonomie endogene: {endogenous}\n"
        f"  Dashboard: http://localhost:3618\n"
        f"  API: http://127.0.0.1:{api_port}\n"
        f"  Tapez /help pour les commandes, /quit pour sortir\n"
        f"{'=' * 50}\n"
    )


# ── Endogenous drain ─────────────────────────────────────────────────


async def _drain_endogenous(session: ChatSession) -> None:
    """Display any endogenous responses that Luna generated on her own."""
    q = session._on_endogenous
    if q is None:
        return
    while not q.empty():
        try:
            response = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        print(f"\n\033[36m[Luna — initiative]\033[0m")
        print(f"{response.content}")
        if response.endogenous_impulse:
            print(f"\033[90m  [{response.endogenous_impulse}]\033[0m")
        print(
            f"[{response.phase} | Phi={response.phi_iit:.4f}"
            f" | {response.input_tokens}+{response.output_tokens} tokens]"
        )
        print()


# ── Autonomous journal ───────────────────────────────────────────────


def _display_autonomous_journal(session: ChatSession) -> None:
    """Display impulses Luna collected while no user was present.

    Called once at session startup. Shows what Luna observed, felt,
    or wanted during autonomous operation.
    """
    handle = getattr(session, "_session_handle", None)
    if handle is None:
        return
    journal = getattr(handle, "autonomous_journal", [])
    if not journal:
        return

    _SOURCE_LABELS = {
        "initiative": "Initiative",
        "watcher": "Perception",
        "dream": "Reve",
        "affect": "Affect",
        "self_improvement": "Evolution",
        "observation_factory": "Capteur",
        "curiosity": "Curiosite",
    }

    print(f"\n\033[36m── Pendant ton absence ({len(journal)} impulse{'s' if len(journal) > 1 else ''}) ──\033[0m")
    for entry in journal:
        source = entry.get("source", "?")
        label = _SOURCE_LABELS.get(source, source)
        msg = entry.get("message", "")
        urgency = entry.get("urgency", 0)
        ts = entry.get("time", "")
        # Extract readable time (HH:MM from ISO).
        time_short = ts[11:16] if len(ts) >= 16 else ts
        print(f"  \033[90m{time_short}\033[0m [{label}] {msg} \033[90m(urgence {urgency:.2f})\033[0m")
    print()


# ── Main REPL ─────────────────────────────────────────────────────────


async def run_repl(config: LunaConfig) -> None:
    """Run the interactive REPL. Blocks until /quit, EOF, or Ctrl+C."""
    # Kill any existing daemon — chat takes over the loop.
    _kill_existing_daemon(config)

    # Create CognitiveLoop as the subsystem owner.
    from luna.orchestrator.cognitive_loop import CognitiveLoop

    loop = CognitiveLoop(config)
    await loop.start()

    session = ChatSession(config, loop=loop)
    await session.start()

    # Dashboard API receives the loop directly (no shim needed).
    _start_dashboard_api(session, config, loop=loop)

    print(_banner(session))

    # Display what Luna experienced while the user was away.
    _display_autonomous_journal(session)

    sys.stdout.flush()

    prefix = config.chat.prompt_prefix

    try:
        while True:
            # Check for endogenous messages before prompting.
            await _drain_endogenous(session)

            try:
                user_input = await asyncio.to_thread(input, prefix)
            except EOFError:
                break

            # Drain again — Luna may have spoken while we waited for input.
            await _drain_endogenous(session)

            user_input = user_input.strip()
            if not user_input:
                continue

            # Quit command.
            if user_input.lower() == "/quit":
                break

            # Slash commands.
            if user_input.startswith("/"):
                result = await session.handle_command(user_input)
                print(result)
                continue

            # Regular chat message.
            response = await session.send(user_input)
            print(f"\n{response.content}")
            if response.endogenous_impulse:
                print(f"\033[90m  [{response.endogenous_impulse}]\033[0m")
            print(
                f"[{response.phase} | Phi={response.phi_iit:.4f}"
                f" | {response.input_tokens}+{response.output_tokens} tokens]"
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C arrives as KeyboardInterrupt or CancelledError depending
        # on whether asyncio.run() intercepts the signal first.
        print("\nInterrupted.")
    finally:
        # Save state and stop cleanly.
        try:
            await session.stop()
            await loop.stop()
        except asyncio.CancelledError:
            # If the event loop is shutting down, stop() may be cancelled.
            # Force a synchronous checkpoint save as last resort.
            loop.save_checkpoint()

        # Spawn autonomous daemon — Luna stays alive after chat.
        try:
            _spawn_autonomous_daemon(config)
        except Exception:
            log.debug("Failed to spawn autonomous daemon", exc_info=True)
            print("Au revoir.")
