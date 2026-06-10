"""Filesystem change watcher for the ingestion layer (M2).

Implements the ``Watcher`` module from ARCHITECTURE-EXTENSION.md §2
(``start(targets, queue) -> None``; ``stop()``; FS events → ``WorkItem``),
supporting the incremental-update path measured by TC-PERF-3.

Design rationale (ARCHITECTURE.md §6 / §14):
    "Implement blind polling fallback; treat watcher as a speed optimisation,
    not a correctness guarantee."

The watcher is therefore **stdlib-only polling** by default (threading +
``os.stat`` snapshots) — correctness first.  A ``watchdog``-based fast path
MAY be layered on later without changing this interface; the polling
implementation below is complete and is the default, so ``watchdog`` is
never a hard dependency.

Module boundary invariant (ARCHITECTURE-EXTENSION.md §2):
    The Watcher MUST NOT mutate the filesystem and MUST NOT classify.
    It only observes metadata and enqueues ``WorkItem`` instances.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from organizer.ingest import EXCLUDES


# ---------------------------------------------------------------------------
# Data contract — ARCHITECTURE-EXTENSION.md §2: "FS events → WorkItem"
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """One filesystem-change notification enqueued by the Watcher.

    ``event`` is one of:
        'created'  — a new file appeared under a watched target
        'modified' — an existing file's size or mtime changed
        'deleted'  — a previously-seen file no longer exists
    """

    path: str
    event: str   # 'created' | 'modified' | 'deleted'
    ts: float    # epoch seconds at detection time


# ---------------------------------------------------------------------------
# Snapshot helpers — stat-only, no file bytes ever read (G9 / TC-SAFE-1 spirit)
# ---------------------------------------------------------------------------

# Cheap change fingerprint per file: (size, mtime).
_StatKey = tuple[int, float]


def _is_excluded(name: str, excludes: set[str]) -> bool:
    """True if *name* must be skipped — dotfile/dot-dir or in *excludes*.

    Mirrors the exclusion rules of ``organizer.ingest.scan`` (TC-SAFE-6).
    """
    return name.startswith(".") or name in excludes


def _snapshot(targets: list[Path], excludes: set[str]) -> dict[str, _StatKey]:
    """Stat every regular file under all *targets*, honouring exclusions.

    Returns ``{absolute_path_str: (size, mtime)}``.  Unreadable entries
    (races, broken symlinks) are skipped silently, like ``ingest.scan``.
    """
    snap: dict[str, _StatKey] = {}
    for root in targets:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune excluded / dot directories in-place so os.walk never
            # descends into them (same technique as ingest.scan).
            dirnames[:] = [d for d in dirnames if not _is_excluded(d, excludes)]
            for name in filenames:
                if _is_excluded(name, excludes):
                    continue
                abs_path = os.path.join(dirpath, name)
                try:
                    st = os.stat(abs_path)
                except OSError:
                    continue
                if os.path.isfile(abs_path):
                    snap[abs_path] = (st.st_size, st.st_mtime)
    return snap


# ---------------------------------------------------------------------------
# Watcher — ARCHITECTURE-EXTENSION.md §2 "Watcher" row
# ---------------------------------------------------------------------------

class Watcher:
    """Poll-based filesystem watcher.

    Spawns a daemon thread that snapshots the watched targets every
    ``interval`` seconds, diffs against the previous snapshot, and enqueues
    one :class:`WorkItem` per created/modified/deleted file.

    ``start`` and ``stop`` are idempotent; ``stop`` is safe before ``start``.
    """

    def __init__(
        self,
        targets: list[Path],
        interval: float = 1.0,
        excludes: set[str] | None = None,
    ) -> None:
        """
        Args:
            targets:  Directories to watch recursively.
            interval: Poll interval in seconds (default 1.0).
            excludes: Directory / file names to skip.  Defaults to
                      ``organizer.ingest.EXCLUDES``.  Dotfiles and dot-dirs
                      are always skipped regardless of this set.
        """
        self._targets = list(targets)
        self._interval = interval
        self._excludes: set[str] = (
            set(excludes) if excludes is not None else set(EXCLUDES)
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: Any = None                    # .put(item)-compatible
        self._prev: dict[str, _StatKey] = {}       # last snapshot state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, queue: Any) -> None:
        """Begin watching; enqueue :class:`WorkItem` events into *queue*.

        The baseline snapshot is taken synchronously here, *before* the
        worker thread starts: pre-existing files therefore never produce
        events, and a file created immediately after ``start()`` returns is
        guaranteed to be reported as ``'created'`` (no startup race).

        Args:
            queue: Any object with a ``.put(item)`` method
                   (stdlib ``queue.Queue`` compatible).
        """
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent: already running

        self._queue = queue
        self._stop_event.clear()

        # Baseline only — emit NO events for pre-existing files.
        try:
            self._prev = _snapshot(self._targets, self._excludes)
        except Exception as exc:  # noqa: BLE001 — never crash the caller
            print(f"[watcher] baseline snapshot failed: {exc}", file=sys.stderr)
            self._prev = {}

        self._thread = threading.Thread(
            target=self._run, name="organizer-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker thread to exit and join it (bounded wait).

        Idempotent — safe to call before ``start`` or multiple times.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 4))
            self._thread = None

    # ------------------------------------------------------------------
    # Worker loop — runs in the daemon thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Poll loop: wait interval → snapshot → diff → enqueue.

        Each iteration is wrapped in try/except so a transient error
        (e.g. a directory vanishing mid-walk) can never crash the process
        (§14: watcher brittleness must not affect correctness).
        """
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self._poll()
            except Exception as exc:  # noqa: BLE001
                print(f"[watcher] poll error: {exc}", file=sys.stderr)

    def _poll(self) -> None:
        """Diff the current snapshot against ``self._prev``; enqueue changes."""
        now_ts = time.time()
        curr = _snapshot(self._targets, self._excludes)
        prev = self._prev
        queue = self._queue

        # Deletions: present before, gone now.
        for path in list(prev):
            if path not in curr:
                queue.put(WorkItem(path=path, event="deleted", ts=now_ts))

        # Creations and modifications.
        for path, stat_key in curr.items():
            if path not in prev:
                queue.put(WorkItem(path=path, event="created", ts=now_ts))
            elif prev[path] != stat_key:
                queue.put(WorkItem(path=path, event="modified", ts=now_ts))

        self._prev = curr
