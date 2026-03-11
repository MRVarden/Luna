"""Kill command — emergency stop (password-protected)."""

from __future__ import annotations

import getpass
from pathlib import Path

import typer

from luna.core.config import LunaConfig
from luna.safety.kill_auth import (
    DEFAULT_HASH_FILE,
    MIN_PASSWORD_LENGTH,
    hash_password,
    load_hash,
    require_kill_password,
    save_hash,
    verify_password,
)
from luna.safety.kill_switch import KillSwitch


def kill(
    reason: str = typer.Option("manual CLI", help="Reason for emergency stop"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    config: str = typer.Option("luna.toml", help="Path to config file"),
) -> None:
    """Activate the kill switch — emergency stop via sentinel file."""
    try:
        cfg = LunaConfig.load(config)
        root_dir = cfg.root_dir
        data_dir = cfg.resolve(cfg.luna.data_dir)
    except FileNotFoundError:
        root_dir = Path.cwd()
        data_dir = Path.cwd() / "data"

    # Password authentication (M-04)
    hash_file = root_dir / DEFAULT_HASH_FILE
    password = getpass.getpass("Kill switch password: ")
    try:
        require_kill_password(password, hash_file)
    except PermissionError as exc:
        typer.echo(f"ACCESS DENIED: {exc}", err=True)
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm("Activate kill switch?")
        if not confirm:
            typer.echo("Cancelled.")
            raise typer.Exit()

    ks = KillSwitch()
    sentinel_path = ks.write_sentinel(data_dir, reason)
    typer.echo(f"Emergency stop written: {sentinel_path}")
    typer.echo(f"Reason: {reason}")
    typer.echo("Luna will detect this on next message (chat) or heartbeat cycle (orchestrator).")


def set_kill_password(
    config: str = typer.Option("luna.toml", help="Path to config file"),
) -> None:
    """Set or change the kill switch password (scrypt, military-grade)."""
    try:
        cfg = LunaConfig.load(config)
        root_dir = cfg.root_dir
    except FileNotFoundError:
        root_dir = Path.cwd()

    hash_file = root_dir / DEFAULT_HASH_FILE

    # If password already set, verify old password first
    existing = load_hash(hash_file)
    if existing is not None:
        old_password = getpass.getpass("Current kill switch password: ")
        if not verify_password(old_password, existing):
            typer.echo("ACCESS DENIED: wrong current password.", err=True)
            raise typer.Exit(code=1)

    # New password with confirmation
    new_password = getpass.getpass("New kill switch password (min 12 chars): ")
    if len(new_password) < MIN_PASSWORD_LENGTH:
        typer.echo(
            f"Password too short — minimum {MIN_PASSWORD_LENGTH} characters.",
            err=True,
        )
        raise typer.Exit(code=1)

    confirm_password = getpass.getpass("Confirm new password: ")
    if new_password != confirm_password:
        typer.echo("Passwords do not match.", err=True)
        raise typer.Exit(code=1)

    hashed = hash_password(new_password)
    save_hash(hash_file, hashed)
    typer.echo(f"Kill switch password saved to {hash_file} (chmod 600)")
