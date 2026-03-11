"""Tests for IdentityBundle and IdentityLedger (PlanManifest Phase A)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from luna.identity.bundle import (
    IdentityBundle,
    canonicalize,
    compute_bundle,
    hash_bytes,
)
from luna.identity.ledger import IdentityLedger


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_docs(tmp_path: Path) -> dict[str, Path]:
    """Create temporary founding documents."""
    docs = {
        "FOUNDERS_MEMO": tmp_path / "FOUNDERS_MEMO.md",
        "LUNA_CONSTITUTION": tmp_path / "LUNA_CONSTITUTION.md",
        "FOUNDING_EPISODES": tmp_path / "FOUNDING_EPISODES.md",
    }
    docs["FOUNDERS_MEMO"].write_text("# Memo\nVarden.\n", encoding="utf-8")
    docs["LUNA_CONSTITUTION"].write_text("# Constitution\nArticle 1.\n", encoding="utf-8")
    docs["FOUNDING_EPISODES"].write_text("# Episodes\nEpisode 01.\n", encoding="utf-8")
    return docs


@pytest.fixture
def sample_bundle(tmp_docs: dict[str, Path]) -> IdentityBundle:
    """Compute a bundle from temp docs."""
    return compute_bundle(tmp_docs)


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> IdentityLedger:
    """Create a ledger in a temp directory."""
    return IdentityLedger(path=tmp_path / "test_ledger.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanonicalize:
    """Tests for canonicalization — deterministic hashing input."""

    def test_deterministic(self) -> None:
        """Same content = same bytes."""
        text = "Hello\nWorld\n"
        assert canonicalize(text) == canonicalize(text)

    def test_crlf_normalized(self) -> None:
        """CRLF is converted to LF before hashing."""
        unix = "Hello\nWorld\n"
        windows = "Hello\r\nWorld\r\n"
        assert canonicalize(unix) == canonicalize(windows)

    def test_trailing_whitespace_stripped(self) -> None:
        """Trailing spaces/tabs per line are removed."""
        dirty = "Hello   \nWorld\t\n"
        clean = "Hello\nWorld\n"
        assert canonicalize(dirty) == canonicalize(clean)

    def test_trailing_newlines_stripped(self) -> None:
        """Extra trailing newlines are removed, exactly 1 is added."""
        extra = "Hello\nWorld\n\n\n\n"
        clean = "Hello\nWorld\n"
        assert canonicalize(extra) == canonicalize(clean)

    def test_exactly_one_final_newline(self) -> None:
        """Output always ends with exactly one newline."""
        no_newline = "Hello\nWorld"
        result = canonicalize(no_newline)
        assert result.endswith(b"\n")
        assert not result.endswith(b"\n\n")

    def test_utf8_encoding(self) -> None:
        """Non-ASCII characters are preserved in UTF-8."""
        text = "φ = 1.618\népisode fondateur\n"
        result = canonicalize(text)
        assert "φ".encode("utf-8") in result
        assert "é".encode("utf-8") in result


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY BUNDLE
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentityBundle:
    """Tests for IdentityBundle dataclass."""

    def test_round_trip(self, sample_bundle: IdentityBundle) -> None:
        """to_dict / from_dict preserves all fields."""
        restored = IdentityBundle.from_dict(sample_bundle.to_dict())
        assert restored == sample_bundle

    def test_json_round_trip(self, sample_bundle: IdentityBundle) -> None:
        """to_json / from_dict(json.loads) preserves all fields."""
        json_str = sample_bundle.to_json()
        data = json.loads(json_str)
        restored = IdentityBundle.from_dict(data)
        assert restored == sample_bundle

    def test_frozen(self, sample_bundle: IdentityBundle) -> None:
        """Bundle is immutable."""
        with pytest.raises(AttributeError):
            sample_bundle.version = "2.0"  # type: ignore[misc]

    def test_doc_hashes_present(self, sample_bundle: IdentityBundle) -> None:
        """All 3 document hashes are present."""
        assert "FOUNDERS_MEMO" in sample_bundle.doc_hashes
        assert "LUNA_CONSTITUTION" in sample_bundle.doc_hashes
        assert "FOUNDING_EPISODES" in sample_bundle.doc_hashes

    def test_hashes_are_sha256(self, sample_bundle: IdentityBundle) -> None:
        """All hashes have sha256: prefix and 64 hex chars."""
        for h in sample_bundle.doc_hashes.values():
            assert h.startswith("sha256:")
            assert len(h) == 7 + 64  # "sha256:" + 64 hex
        assert sample_bundle.bundle_hash.startswith("sha256:")

    def test_bundle_hash_deterministic(self, tmp_docs: dict[str, Path]) -> None:
        """Same documents = same bundle hash."""
        b1 = compute_bundle(tmp_docs)
        b2 = compute_bundle(tmp_docs)
        assert b1.bundle_hash == b2.bundle_hash
        assert b1.doc_hashes == b2.doc_hashes

    def test_content_change_changes_hash(self, tmp_docs: dict[str, Path]) -> None:
        """Changing document content changes the hash."""
        b1 = compute_bundle(tmp_docs)
        tmp_docs["FOUNDERS_MEMO"].write_text("# Memo v2\nAmended.\n", encoding="utf-8")
        b2 = compute_bundle(tmp_docs)
        assert b1.bundle_hash != b2.bundle_hash
        assert b1.doc_hashes["FOUNDERS_MEMO"] != b2.doc_hashes["FOUNDERS_MEMO"]
        # Other docs unchanged
        assert b1.doc_hashes["LUNA_CONSTITUTION"] == b2.doc_hashes["LUNA_CONSTITUTION"]

    def test_missing_doc_raises(self, tmp_path: Path) -> None:
        """Missing document raises FileNotFoundError."""
        docs = {"FOUNDERS_MEMO": tmp_path / "missing.md"}
        with pytest.raises(FileNotFoundError):
            compute_bundle(docs)


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY LEDGER
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentityLedger:
    """Tests for append-only JSONL ledger."""

    def test_empty_ledger(self, tmp_ledger: IdentityLedger) -> None:
        """Fresh ledger has no history and no latest."""
        assert tmp_ledger.history() == []
        assert tmp_ledger.latest() is None
        assert not tmp_ledger.exists()

    def test_append_and_latest(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """Appending a bundle makes it retrievable."""
        tmp_ledger.append(sample_bundle)
        assert tmp_ledger.exists()
        assert tmp_ledger.latest() == sample_bundle

    def test_verify_match(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """Verify returns True for a recorded bundle."""
        tmp_ledger.append(sample_bundle)
        assert tmp_ledger.verify(sample_bundle) is True

    def test_verify_mismatch(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """Verify returns False for an unrecorded bundle."""
        tmp_ledger.append(sample_bundle)
        fake = IdentityBundle(
            version="1.0",
            timestamp="2026-01-01T00:00:00+00:00",
            repo_commit="unknown",
            doc_hashes={"FOUNDERS_MEMO": "sha256:0000"},
            bundle_hash="sha256:ffff",
            intent="fake",
        )
        assert tmp_ledger.verify(fake) is False

    def test_verify_empty_ledger(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """Verify returns False on empty ledger."""
        assert tmp_ledger.verify(sample_bundle) is False

    def test_append_only(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """Multiple appends accumulate — nothing is overwritten."""
        tmp_ledger.append(sample_bundle)
        # Create a second bundle (different intent)
        b2 = IdentityBundle(
            version="1.1",
            timestamp="2026-03-07T00:00:00+00:00",
            repo_commit="unknown",
            doc_hashes=sample_bundle.doc_hashes,
            bundle_hash=sample_bundle.bundle_hash,
            intent="amendment-v1.1",
        )
        tmp_ledger.append(b2)

        history = tmp_ledger.history()
        assert len(history) == 2
        assert history[0].intent == "founding"
        assert history[1].intent == "amendment-v1.1"

    def test_history_preserved(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """History is append-only and chronological."""
        for i in range(5):
            b = IdentityBundle(
                version=f"1.{i}",
                timestamp=f"2026-03-0{i+1}T00:00:00+00:00",
                repo_commit="unknown",
                doc_hashes=sample_bundle.doc_hashes,
                bundle_hash=sample_bundle.bundle_hash,
                intent=f"test-{i}",
            )
            tmp_ledger.append(b)

        history = tmp_ledger.history()
        assert len(history) == 5
        assert [h.version for h in history] == ["1.0", "1.1", "1.2", "1.3", "1.4"]

    def test_malformed_line_skipped(self, tmp_ledger: IdentityLedger) -> None:
        """Malformed JSONL lines are skipped with warning."""
        tmp_ledger._path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_ledger._path, "w", encoding="utf-8") as f:
            f.write("not valid json\n")
            f.write('{"version":"1.0","timestamp":"t","repo_commit":"u",'
                    '"doc_hashes":{},"bundle_hash":"sha256:abc","intent":"ok"}\n')

        history = tmp_ledger.history()
        assert len(history) == 1
        assert history[0].intent == "ok"

    def test_bundle_missing_detectable(
        self, tmp_ledger: IdentityLedger, sample_bundle: IdentityBundle
    ) -> None:
        """A bundle not in ledger is detectable via verify."""
        # Append one bundle
        tmp_ledger.append(sample_bundle)
        # Check a different hash
        different = IdentityBundle(
            version="1.0",
            timestamp=sample_bundle.timestamp,
            repo_commit=sample_bundle.repo_commit,
            doc_hashes=sample_bundle.doc_hashes,
            bundle_hash="sha256:" + "a" * 64,
            intent="different",
        )
        assert tmp_ledger.verify(different) is False
        assert tmp_ledger.verify(sample_bundle) is True
