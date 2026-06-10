"""M5 — minimal TUI: workspace health + category distribution.

ARCHITECTURE.md §13 M5: "Minimal TUI or tray app showing workspace health +
category distribution." Per §10, the UI is a thin shell over the headless
core: it only READS the SQLite index and the operation/preference logs —
no classification, no filesystem mutation from the UI.

Data gathering (gather_stats) is pure and separated from rendering so it is
testable without a terminal. Run via:

    python src/organizer/cli.py tui [--db PATH]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from organizer.database import Database

# Bar glyph used for the distribution chart.
_BAR = "█"
_BAR_WIDTH = 30  # max bar length in cells


# --------------------------------------------------------------------------- #
# Data layer — pure, headless, testable
# --------------------------------------------------------------------------- #
@dataclass
class WorkspaceStats:
    """Aggregated workspace health snapshot (read-only view of the index)."""

    db_path: str
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    avg_confidence: float | None = None
    oplog_entries: int = 0
    feedback_events: int = 0


def gather_stats(
    db_path: Path,
    oplog_path: Path | None = None,
    prefs_path: Path | None = None,
) -> WorkspaceStats:
    """Read the index + logs and aggregate workspace health.

    Missing files are reported as empty stats rather than errors: the TUI must
    be safe to open before any scan has run.
    """
    stats = WorkspaceStats(db_path=str(db_path))

    if Path(db_path).exists():
        with Database(db_path) as db:
            db.migrate()
            rows = db.all_rows()
        stats.total = len(rows)
        confidences: list[float] = []
        for r in rows:
            status = r.get("status") or "unknown"
            stats.by_status[status] = stats.by_status.get(status, 0) + 1
            cat = r.get("category")
            if cat:
                stats.by_category[cat] = stats.by_category.get(cat, 0) + 1
            if r.get("confidence") is not None:
                confidences.append(float(r["confidence"]))
        if confidences:
            stats.avg_confidence = sum(confidences) / len(confidences)

    for attr, p in (("oplog_entries", oplog_path), ("feedback_events", prefs_path)):
        if p is not None and Path(p).exists():
            with open(p, "r", encoding="utf-8") as fh:
                setattr(stats, attr, sum(1 for line in fh if line.strip()))

    return stats


def render_health(stats: WorkspaceStats) -> str:
    """Plain-text health panel (pure; reused by the TUI widget)."""
    lines = [
        f"index:     {stats.db_path}",
        f"files:     {stats.total}",
    ]
    for status in ("classified", "pending", "needs_review", "confirmed"):
        if status in stats.by_status:
            lines.append(f"  {status:13s} {stats.by_status[status]}")
    for status, n in sorted(stats.by_status.items()):
        if status not in ("classified", "pending", "needs_review", "confirmed"):
            lines.append(f"  {status:13s} {n}")
    if stats.avg_confidence is not None:
        lines.append(f"avg conf:  {stats.avg_confidence:.2f}")
    lines.append(f"op log:    {stats.oplog_entries} entries")
    lines.append(f"feedback:  {stats.feedback_events} events")
    return "\n".join(lines)


def render_distribution(stats: WorkspaceStats) -> str:
    """Plain-text category distribution bar chart (pure)."""
    if not stats.by_category:
        return "(no classified files yet — run `scan` + `classify`)"
    peak = max(stats.by_category.values())
    lines = []
    for cat, n in sorted(stats.by_category.items(), key=lambda kv: -kv[1]):
        bar = _BAR * max(1, round(n / peak * _BAR_WIDTH))
        lines.append(f"{cat:28s} {bar} {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering layer — Textual app (thin shell over the pure renderers)
# --------------------------------------------------------------------------- #
def build_app(db_path: Path, oplog_path: Path, prefs_path: Path):
    """Construct the Textual App class lazily so importing this module never
    requires textual (the data layer must stay dependency-free for tests)."""
    from textual.app import App, ComposeResult
    from textual.containers import Vertical
    from textual.widgets import Footer, Header, Static

    class OrganizerTui(App):
        """Workspace health + category distribution (read-only)."""

        TITLE = "AI File Organizer — workspace"
        BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]
        CSS = """
        #health { border: round $accent; padding: 1; height: auto; }
        #dist   { border: round $accent; padding: 1; height: 1fr; }
        """

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                yield Static(id="health")
                yield Static(id="dist")
            yield Footer()

        def on_mount(self) -> None:
            self.action_refresh()

        def action_refresh(self) -> None:
            stats = gather_stats(db_path, oplog_path, prefs_path)
            self.query_one("#health", Static).update(
                "[b]Workspace health[/b]\n" + render_health(stats)
            )
            self.query_one("#dist", Static).update(
                "[b]Category distribution[/b]\n" + render_distribution(stats)
            )

    return OrganizerTui()


def run_tui(db_path: Path, oplog_path: Path, prefs_path: Path) -> None:
    """Entry point used by the CLI `tui` subcommand."""
    build_app(db_path, oplog_path, prefs_path).run()
