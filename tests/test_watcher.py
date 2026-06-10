"""Tests for organizer.watcher — polling Watcher (ARCHITECTURE-EXTENSION.md §2).

Hermetic (tmp_path only), offline, stdlib-only.  Each test uses a short poll
interval and bounded waits so total runtime stays well under 5 s per test.
"""

from __future__ import annotations

import pathlib
import queue
import time

from organizer.watcher import Watcher, WorkItem

# Short interval for fast, bounded tests (spec: 0.05–0.1 s).
INTERVAL = 0.05
# Generous upper bound for an event to arrive (many poll cycles, < 5 s).
EVENT_TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# Helpers — bounded queue polling, never unbounded blocking
# ---------------------------------------------------------------------------

def wait_for_event(
    q: "queue.Queue[WorkItem]",
    event: str,
    path: pathlib.Path | None = None,
    timeout: float = EVENT_TIMEOUT,
) -> WorkItem | None:
    """Poll *q* until a WorkItem with *event* (and *path*, if given) arrives.

    Returns the matching item, or None if the deadline passes.  Non-matching
    items are consumed and discarded.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item.event == event and (path is None or item.path == str(path)):
            return item
    return None


def drain(q: "queue.Queue[WorkItem]") -> list[WorkItem]:
    """Return every item currently in *q* without blocking."""
    items: list[WorkItem] = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            return items


# ---------------------------------------------------------------------------
# 1. Baseline emits nothing for pre-existing files
# ---------------------------------------------------------------------------

def test_baseline_emits_no_events_for_preexisting_files(tmp_path: pathlib.Path) -> None:
    (tmp_path / "old_a.txt").write_text("already here", encoding="utf-8")
    (tmp_path / "old_b.md").write_text("# also here", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "old_c.py").write_text("x = 1\n", encoding="utf-8")

    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)
    try:
        # ~3+ poll intervals with margin; baseline files must stay silent.
        time.sleep(INTERVAL * 6)
        assert drain(q) == []
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 2. Created file -> WorkItem(event='created')
# ---------------------------------------------------------------------------

def test_created_file_emits_created_event(tmp_path: pathlib.Path) -> None:
    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)
    try:
        new_file = tmp_path / "fresh.txt"
        new_file.write_text("hello", encoding="utf-8")

        item = wait_for_event(q, "created", path=new_file)
        assert item is not None, "no 'created' event arrived in time"
        assert item.path == str(new_file)
        assert item.event == "created"
        assert isinstance(item.ts, float) and item.ts > 0
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 3. Modified file -> WorkItem(event='modified')
# ---------------------------------------------------------------------------

def test_modified_file_emits_modified_event(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "doc.txt"
    target.write_text("v1", encoding="utf-8")

    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)  # baseline includes doc.txt
    try:
        # Let at least one poll pass so the diff is against a settled state.
        time.sleep(INTERVAL * 2)
        # Different length guarantees the (size, mtime) key changes even on
        # filesystems with coarse mtime resolution.
        target.write_text("v2 -- now with much longer content", encoding="utf-8")

        item = wait_for_event(q, "modified", path=target)
        assert item is not None, "no 'modified' event arrived in time"
        assert item.event == "modified"
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 4. Deleted file -> WorkItem(event='deleted')
# ---------------------------------------------------------------------------

def test_deleted_file_emits_deleted_event(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "doomed.txt"
    target.write_text("goodbye", encoding="utf-8")

    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)  # baseline includes doomed.txt
    try:
        time.sleep(INTERVAL * 2)
        target.unlink()

        item = wait_for_event(q, "deleted", path=target)
        assert item is not None, "no 'deleted' event arrived in time"
        assert item.event == "deleted"
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 5. Excluded dirs: files under .git/ produce NO events
# ---------------------------------------------------------------------------

def test_excluded_git_dir_produces_no_events(tmp_path: pathlib.Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)
    try:
        # File inside the excluded dir must be invisible to the watcher.
        (git_dir / "secret.txt").write_text("hidden", encoding="utf-8")
        # Sentinel outside it bounds the wait: once its event arrives, the
        # watcher has definitely completed polls covering the .git write.
        sentinel = tmp_path / "sentinel.txt"
        sentinel.write_text("visible", encoding="utf-8")

        item = wait_for_event(q, "created", path=sentinel)
        assert item is not None, "sentinel 'created' event never arrived"

        # One more full poll cycle, then assert nothing referenced .git.
        time.sleep(INTERVAL * 3)
        leftovers = drain(q)
        assert all(".git" not in it.path for it in leftovers), (
            f"events leaked from excluded dir: {leftovers}"
        )
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 6. stop(): no further events; idempotent; safe before start
# ---------------------------------------------------------------------------

def test_stop_halts_events_and_is_idempotent(tmp_path: pathlib.Path) -> None:
    q: "queue.Queue[WorkItem]" = queue.Queue()
    w = Watcher([tmp_path], interval=INTERVAL)
    w.start(q)
    w.stop()  # joins the worker thread

    # Files created after stop must produce no events.
    (tmp_path / "after_stop.txt").write_text("too late", encoding="utf-8")
    time.sleep(INTERVAL * 6)
    assert drain(q) == []

    # Calling stop twice is safe.
    w.stop()


def test_stop_before_start_is_safe(tmp_path: pathlib.Path) -> None:
    w = Watcher([tmp_path], interval=INTERVAL)
    w.stop()   # never started — must not raise
    w.stop()   # and again
