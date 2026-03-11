"""Tests for CLI main entry point — wired commands.

These tests exercise the real Luna logic behind each CLI command.
Commands that start infinite loops (start, dashboard) are tested only
at the import / registration level.

v5.1: Tests are fully isolated — each test gets its own temporary
memory_fractal/ with a minimal checkpoint generated on-the-fly.
No dependency on the real memory_fractal/ state.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luna.cli.main import app

runner = CliRunner()

# Minimal luna.toml template — all paths relative to tmp dir.
_TOML_TEMPLATE = """\
[luna]
version = "3.5.2"
agent_name = "LUNA"
data_dir = "{mem}"

[consciousness]
checkpoint_file = "{mem}/consciousness_state_v2.json"
backup_on_save = false

[memory]
fractal_root = "{mem}"
levels = ["seeds", "roots", "branches", "leaves"]
max_memories_per_level = 500

[observability]
log_level = "WARNING"
log_file = "{tmp}/logs/luna.log"

[heartbeat]
interval_seconds = 30
fingerprint_enabled = false
checkpoint_interval = 100

[llm]
provider = "deepseek"
model = "deepseek-reasoner"
base_url = "https://api.deepseek.com/v1"
max_tokens = 4096
temperature = 0.7

[dream]
inactivity_threshold = 7200
consolidation_window = 100
max_dream_duration = 300
report_dir = "{mem}/dreams"
enabled = true

[orchestrator]
llm_augment = false
max_cycles = 0
checkpoint_interval = 1
cycle_timeout = 600
retry_max = 3
retry_base_delay = 1.0

[chat]
max_history = 30
memory_search_limit = 5
idle_heartbeat = false
save_conversations = false

[metrics]
enabled = false
cache_enabled = false
cache_dir = "{tmp}/data/metrics_cache"
timeout_seconds = 60
python_enabled = false

[fingerprint]
enabled = true
secret_file = "{tmp}/config/fingerprint.key"
ledger_file = "{tmp}/data/fingerprints.jsonl"
watermark_enabled = false

[safety]
enabled = false
snapshot_dir = "{tmp}/data/snapshots"
max_snapshots = 10
retention_days = 7
max_generations_per_hour = 100
max_commits_per_hour = 20
watchdog_threshold = 3

[api]
host = "127.0.0.1"
port = 8618
auth_enabled = false
"""


def _make_minimal_checkpoint(ckpt_path: Path) -> None:
    """Create a minimal valid checkpoint from Ψ₀."""
    from luna_common.constants import AGENT_PROFILES

    psi0 = list(AGENT_PROFILES.get("LUNA", (0.260, 0.322, 0.250, 0.168)))
    state = {
        "psi": psi0,
        "psi0": psi0,
        "phi_metrics": {},
        "history": [],
        "step_count": 0,
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(state))


@pytest.fixture
def cli_env(tmp_path: Path):
    """Create an isolated CLI environment and return the config path.

    Suppresses logging to avoid Click/Typer CliRunner stream bug
    triggered by identity recovery CRITICAL logs.
    """
    mem = tmp_path / "memory_fractal"
    mem.mkdir()
    for d in ("seeds", "roots", "branches", "leaves", "cycles", "dreams", "snapshots"):
        (mem / d).mkdir()

    # Minimal checkpoint.
    _make_minimal_checkpoint(mem / "consciousness_state_v2.json")

    # Config file.
    toml_content = _TOML_TEMPLATE.format(
        mem=str(mem), tmp=str(tmp_path),
    )
    config_path = tmp_path / "luna.toml"
    config_path.write_text(toml_content)

    # Identity documents (fingerprint/validate may look for these).
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)

    # Kill switch password hash (M-04 security fix).
    from luna.safety.kill_auth import DEFAULT_HASH_FILE, hash_password, save_hash

    save_hash(tmp_path / DEFAULT_HASH_FILE, hash_password("test_password_12"))

    # Suppress logging — identity recovery logs at CRITICAL level which
    # causes Click's CliRunner to close its output stream prematurely.
    logging.disable(logging.CRITICAL)
    yield str(config_path)
    logging.disable(logging.NOTSET)


class TestCLIMain:
    """Tests for the CLI main app."""

    def test_no_args_shows_help(self):
        """No arguments shows help text."""
        result = runner.invoke(app)
        assert result.exit_code in (0, 2)
        assert "luna" in result.output.lower() or "Usage" in result.output

    def test_start_command_registered(self):
        """Start command is registered in the app."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_status_command(self, cli_env):
        """Status command runs and displays engine state."""
        result = runner.invoke(app, ["status", "--config", cli_env])
        assert result.exit_code == 0
        assert "Agent:" in result.output
        assert "Phase:" in result.output
        assert "Step count:" in result.output

    def test_status_json(self, cli_env):
        """Status with --json flag returns valid JSON."""
        result = runner.invoke(app, ["status", "--json", "--config", cli_env])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "agent_name" in data
        assert "phase" in data
        assert "phi_iit" in data

    def test_evolve_command(self, cli_env):
        """Evolve command runs N idle steps."""
        result = runner.invoke(app, ["evolve", "5", "--config", cli_env])
        assert result.exit_code == 0
        assert "5 step(s)" in result.output
        assert "Phase:" in result.output
        assert "PHI_IIT:" in result.output
        assert "Checkpoint saved:" in result.output

    def test_evolve_verbose(self, cli_env):
        """Evolve command with --verbose shows per-step output."""
        result = runner.invoke(app, ["evolve", "3", "--verbose", "--config", cli_env])
        assert result.exit_code == 0
        assert "Step 1/3:" in result.output
        assert "Step 3/3:" in result.output

    def test_score_command(self):
        """Score command runs on current directory."""
        result = runner.invoke(app, ["score", "."])
        assert result.exit_code == 0
        assert "Metrics for:" in result.output or "No metrics collected" in result.output

    def test_score_nonexistent_path(self):
        """Score command errors on nonexistent path."""
        result = runner.invoke(app, ["score", "/nonexistent/path/xyz"])
        assert result.exit_code == 1
        assert "Error: path does not exist" in result.output

    def test_fingerprint_command(self, cli_env):
        """Fingerprint command generates a fingerprint."""
        result = runner.invoke(app, ["fingerprint", "--config", cli_env])
        assert result.exit_code == 0
        assert "Fingerprint:" in result.output

    def test_fingerprint_verify(self, cli_env):
        """Fingerprint with --verify confirms integrity."""
        result = runner.invoke(app, ["fingerprint", "--verify", "--config", cli_env])
        assert result.exit_code == 0
        assert "Fingerprint:" in result.output
        assert "Verification:" in result.output

    def test_fingerprint_history_empty(self, cli_env):
        """Fingerprint with --history on fresh system."""
        result = runner.invoke(app, ["fingerprint", "--history", "5", "--config", cli_env])
        assert result.exit_code == 0

    def test_validate_command(self, cli_env):
        """Validate command runs benchmarks and produces a verdict."""
        result = runner.invoke(app, ["validate", "--config", cli_env])
        assert result.exit_code == 0
        assert "VERDICT:" in result.output
        assert "Criteria met:" in result.output
        assert "Improvement:" in result.output

    def test_validate_verbose(self, cli_env):
        """Validate --verbose shows individual criteria."""
        result = runner.invoke(app, ["validate", "--verbose", "--config", cli_env])
        assert result.exit_code == 0
        assert "VERDICT:" in result.output
        assert "PASS" in result.output or "FAIL" in result.output

    def test_dashboard_command_registered(self):
        """Dashboard command is registered (cannot test infinite loop)."""
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "refresh" in result.output.lower()

    def test_heartbeat_command(self, cli_env):
        """Heartbeat command shows vital signs."""
        result = runner.invoke(app, ["heartbeat", "--config", cli_env])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "phi_iit" in data
        assert "overall_vitality" in data

    def test_dream_command(self, cli_env):
        """Dream command shows status."""
        result = runner.invoke(app, ["dream", "--config", cli_env])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_dream_status(self, cli_env):
        """Dream --status shows status JSON."""
        result = runner.invoke(app, ["dream", "--status", "--config", cli_env])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_memory_command(self, cli_env):
        """Memory command shows recent memories or empty message."""
        result = runner.invoke(app, ["memory", "--config", cli_env])
        assert result.exit_code == 0
        assert "No memories stored yet" in result.output or "Recent memories" in result.output

    def test_memory_search(self, cli_env):
        """Memory search flag."""
        result = runner.invoke(app, ["memory", "--search", "test", "--config", cli_env])
        assert result.exit_code == 0
        assert "No memories found" in result.output or "Found" in result.output

    def test_memory_stats(self, cli_env):
        """Memory --stats shows statistics."""
        result = runner.invoke(app, ["memory", "--stats", "--config", cli_env])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_memories" in data

    def test_kill_with_force(self, cli_env):
        """Kill command with --force."""
        result = runner.invoke(
            app, ["kill", "--force", "--config", cli_env],
            input="test_password_12\n",
        )
        assert result.exit_code == 0
        assert "Emergency stop written" in result.output
        assert "Reason: manual CLI" in result.output

    def test_kill_cancel(self, cli_env):
        """Kill command without force, user cancels."""
        result = runner.invoke(
            app, ["kill", "--config", cli_env],
            input="test_password_12\nn\n",
        )
        assert "Cancelled" in result.output

    def test_rollback_help(self):
        """Rollback command shows help."""
        result = runner.invoke(app, ["rollback", "--help"])
        assert result.exit_code == 0
        assert "snapshot" in result.output.lower()
