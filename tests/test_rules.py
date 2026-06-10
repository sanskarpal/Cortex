"""Tests for RuleEngine (M3) and PreferenceStore (M4).

Offline, hermetic: no network, no models, no real config file. RuleEngine cases
are driven by inline category dicts shaped exactly like AppConfig.categories;
PreferenceStore cases use tmp_path so each run starts from a clean JSONL log.

Refs: ARCHITECTURE.md §4.2(1) rule layer, §4.2(4) user adaptation;
ARCHITECTURE-EXTENSION.md §2 (RuleEngine / PreferenceStore rows), §3
(DM-FeedbackEvent), G7.
"""

from __future__ import annotations

from pathlib import Path

from organizer.preferences import FeedbackEvent, PreferenceStore
from organizer.rules import RuleEngine, RuleVerdict
from organizer.types import FileRecord


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Shaped like AppConfig.categories (ConfigLoader output): cat_id + optional rules.
CATEGORIES: list[dict] = [
    {
        "cat_id": "documents/invoices",
        "rules": {
            "extensions": [".pdf", ".docx"],
            "path_keywords": ["invoice", "billing"],
        },
    },
    {
        "cat_id": "code/python",
        "rules": {
            "extensions": [".py"],
            "path_keywords": ["python", "script"],
        },
    },
    # Ruleless category: empty rules dict -> must be ignored cleanly.
    {"cat_id": "media/music", "rules": {}},
    # No 'rules' key at all -> must be ignored cleanly.
    {"cat_id": "downloads/uncategorized"},
]


def make_rec(path: str, extension: str, mime: str = "application/octet-stream") -> FileRecord:
    """Build a minimal FileRecord. `extension` is lowercase, no leading dot."""
    return FileRecord(
        path=Path(path),
        size=123,
        mtime=0.0,
        extension=extension,
        mime=mime,
    )


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

def test_extension_match_wins_with_high_confidence_and_provenance():
    engine = RuleEngine(CATEGORIES)
    rec = make_rec("/home/u/random/report.pdf", "pdf")
    verdict = engine.apply(rec)
    assert isinstance(verdict, RuleVerdict)
    assert verdict.cat_id == "documents/invoices"
    assert verdict.confidence == 0.95
    assert verdict.reason == "extension:pdf"


def test_path_keyword_match_medium_confidence():
    engine = RuleEngine(CATEGORIES)
    # Extension 'xyz' is unknown; the directory name carries the keyword.
    rec = make_rec("/home/u/invoices/scan_2024.xyz", "xyz")
    verdict = engine.apply(rec)
    assert isinstance(verdict, RuleVerdict)
    assert verdict.cat_id == "documents/invoices"
    assert verdict.confidence == 0.85
    assert verdict.reason == "path_keyword:invoice"


def test_extension_beats_path_keyword():
    engine = RuleEngine(CATEGORIES)
    # Path contains "invoice" (would map to documents/invoices via keyword) but
    # the .py extension must win and route to code/python at 0.95.
    rec = make_rec("/home/u/invoice_tools/run.py", "py")
    verdict = engine.apply(rec)
    assert isinstance(verdict, RuleVerdict)
    assert verdict.cat_id == "code/python"
    assert verdict.confidence == 0.95
    assert verdict.reason == "extension:py"


def test_no_match_returns_none():
    engine = RuleEngine(CATEGORIES)
    rec = make_rec("/home/u/misc/notes.xyz", "xyz")
    assert engine.apply(rec) is None


def test_ruleless_categories_ignored_cleanly():
    # Only ruleless / rule-free categories -> empty lookup tables, never crashes.
    engine = RuleEngine([
        {"cat_id": "media/music", "rules": {}},
        {"cat_id": "downloads/uncategorized"},
        {"cat_id": "broken", "rules": "not-a-dict"},
    ])
    rec = make_rec("/home/u/song.mp3", "mp3")
    assert engine.apply(rec) is None


def test_keyword_match_is_case_insensitive():
    engine = RuleEngine(CATEGORIES)
    rec = make_rec("/home/u/Billing/Q1.dat", "dat")
    verdict = engine.apply(rec)
    assert verdict is not None
    assert verdict.cat_id == "documents/invoices"
    assert verdict.reason == "path_keyword:billing"


# ---------------------------------------------------------------------------
# PreferenceStore
# ---------------------------------------------------------------------------

def test_record_and_reload_round_trip(tmp_path: Path):
    store_path = tmp_path / "prefs.jsonl"
    store = PreferenceStore(store_path)
    events = [
        FeedbackEvent("/a.pdf", None, "documents/invoices", "accept", 1.0),
        FeedbackEvent("/b.pdf", "documents/invoices", "documents/receipts", "override", 2.0),
        FeedbackEvent("/c.pdf", None, "documents/tax", "reject", 3.0),
    ]
    for e in events:
        store.record(e)

    # Reload from disk via a fresh instance -> durable, append-only.
    reloaded = PreferenceStore(store_path).events()
    assert reloaded == events


def test_empty_store_bias_is_zero(tmp_path: Path):
    store = PreferenceStore(tmp_path / "empty.jsonl")
    assert store.events() == []
    assert store.bias("documents/invoices") == 0.0


def test_bias_accept_and_reject_arithmetic(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.jsonl")
    store.record(FeedbackEvent("/1", None, "documents/invoices", "accept", 1.0))
    store.record(FeedbackEvent("/2", None, "documents/invoices", "accept", 2.0))
    store.record(FeedbackEvent("/3", None, "documents/invoices", "reject", 3.0))
    # +0.01 +0.01 -0.01 = +0.01
    assert abs(store.bias("documents/invoices") - 0.01) < 1e-9
    # An unrelated category is unaffected.
    assert store.bias("code/python") == 0.0


def test_bias_override_into_and_away(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.jsonl")
    # Override moves a file from code/python into documents/invoices.
    store.record(
        FeedbackEvent("/x", "code/python", "documents/invoices", "override", 1.0)
    )
    # Destination gets +0.02, source gets -0.01.
    assert abs(store.bias("documents/invoices") - 0.02) < 1e-9
    assert abs(store.bias("code/python") - (-0.01)) < 1e-9


def test_bias_clamps_positive(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.jsonl")
    for i in range(20):
        store.record(FeedbackEvent(f"/f{i}", None, "documents/invoices", "accept", float(i)))
    # 20 * 0.01 = 0.20, clamped to +0.10.
    assert store.bias("documents/invoices") == 0.10


def test_bias_clamps_negative(tmp_path: Path):
    store = PreferenceStore(tmp_path / "prefs.jsonl")
    for i in range(20):
        store.record(FeedbackEvent(f"/f{i}", None, "documents/invoices", "reject", float(i)))
    # 20 * -0.01 = -0.20, clamped to -0.10.
    assert store.bias("documents/invoices") == -0.10
