"""Tests for the M5 TUI — pure data layer + a headless Textual pilot smoke.

gather_stats / render_* are dependency-free (no textual import needed), so
most coverage runs without a terminal. One pilot test drives the real App
in Textual's test harness to assert the widgets mount and populate.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from organizer.database import Database
from organizer.tui import (
    WorkspaceStats,
    gather_stats,
    render_distribution,
    render_health,
)
from organizer.types import FileRecord, Tier


def _seed_db(db_path: Path) -> None:
    """Index three files: two classified, one needs_review (category None)."""
    files = [
        ("a.py", "code/python", 0.9, "classified"),
        ("b.txt", "documents/personal", 0.7, "classified"),
        ("c.bin", None, None, "needs_review"),
    ]
    with Database(db_path) as db:
        db.migrate()
        for name, cat, conf, status in files:
            p = db_path.parent / name
            p.write_text("x")
            rec = FileRecord(
                path=p, size=1, mtime=0.0, extension=p.suffix.lstrip("."),
                mime="text/plain", tier=Tier.TEXT, status=status,
            )
            db.upsert(rec, category=cat, confidence=conf)


class TestGatherStats:
    def test_missing_db_yields_empty_stats(self, tmp_path):
        stats = gather_stats(tmp_path / "nope.db")
        assert stats.total == 0
        assert stats.by_category == {}
        assert stats.avg_confidence is None

    def test_aggregates(self, tmp_path):
        db_path = tmp_path / "index.db"
        _seed_db(db_path)
        stats = gather_stats(db_path)
        assert stats.total == 3
        assert stats.by_category == {"code/python": 1, "documents/personal": 1}
        assert stats.avg_confidence == (0.9 + 0.7) / 2

    def test_log_line_counts(self, tmp_path):
        db_path = tmp_path / "index.db"
        _seed_db(db_path)
        oplog = tmp_path / "oplog.jsonl"
        oplog.write_text('{"op": 1}\n{"op": 2}\n')
        prefs = tmp_path / "prefs.jsonl"
        prefs.write_text('{"e": 1}\n')
        stats = gather_stats(db_path, oplog, prefs)
        assert stats.oplog_entries == 2
        assert stats.feedback_events == 1


class TestRenderers:
    def test_render_health_contains_counts(self):
        stats = WorkspaceStats(
            db_path="x.db", total=5,
            by_status={"classified": 4, "needs_review": 1},
            avg_confidence=0.8, oplog_entries=2, feedback_events=3,
        )
        out = render_health(stats)
        assert "files:     5" in out
        assert "needs_review" in out and "1" in out
        assert "avg conf:  0.80" in out

    def test_render_distribution_bars_scale(self):
        stats = WorkspaceStats(
            db_path="x.db",
            by_category={"code/python": 10, "documents/personal": 5},
        )
        out = render_distribution(stats)
        lines = out.splitlines()
        assert lines[0].startswith("code/python")  # sorted desc
        top_bar = lines[0].count("█")
        half_bar = lines[1].count("█")
        assert top_bar > half_bar >= 1

    def test_render_distribution_empty(self):
        out = render_distribution(WorkspaceStats(db_path="x.db"))
        assert "no classified files" in out


class TestPilotSmoke:
    def test_app_mounts_and_populates(self, tmp_path):
        """Headless Textual pilot: widgets mount and show the seeded data."""
        pytest.importorskip("textual", reason="textual extra not installed")
        from organizer.tui import build_app
        from textual.widgets import Static

        db_path = tmp_path / "index.db"
        _seed_db(db_path)
        app = build_app(db_path, tmp_path / "oplog.jsonl", tmp_path / "prefs.jsonl")

        async def drive():
            async with app.run_test() as pilot:
                await pilot.pause()
                health = app.query_one("#health", Static)
                dist = app.query_one("#dist", Static)
                assert "files:     3" in str(health.render())
                assert "code/python" in str(dist.render())

        asyncio.run(drive())
