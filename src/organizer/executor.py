"""Organization layer — Executor (M2).

The ONLY module in the organizer package that mutates the filesystem
(§2 invariant: "Executor is the only module that mutates the filesystem").

Refs: ARCHITECTURE.md §7 (Safety Model), ARCHITECTURE-EXTENSION.md §2
(Executor row), §3 (DM-OpLogEntry), G1 (trash default / no-silent-clobber),
G10 (hash_before for undo verification), TC-SAFE-1 (dry-run default).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from send2trash import send2trash

from organizer.types import MoveMode, MoveOp, OpLogEntry, Plan

if TYPE_CHECKING:
    from organizer.history import History


def _sha256(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or None on read failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class Executor:
    """Apply approved MoveOps from a Plan to the filesystem.

    Safety guarantees enforced here:

    - **Dry-run default** (TC-SAFE-1): ``dry_run=True`` by default; no
      filesystem mutation occurs unless the caller explicitly passes
      ``dry_run=False``.
    - **Approval gate** (§7.1): even with ``dry_run=False``, only ops whose
      ``op.approved is True`` are enacted.
    - **Trash default / no-silent-clobber** (G1): for TRASH mode, if the
      destination already exists it is sent to the system trash via
      ``send2trash`` before the source is moved in, so no file is silently
      overwritten.
    - **Per-op atomicity** (§7.4): each op is wrapped in its own
      ``try/except``; a failure skips that op and the batch continues.
    - **Privilege guard** (§7.6): no privilege escalation; purely operates
      within user-owned paths.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def apply(
        self,
        plan: Plan,
        *,
        mode: MoveMode = MoveMode.TRASH,
        history: "History | None" = None,
        dry_run: bool = True,
    ) -> list[OpLogEntry]:
        """Execute approved MoveOps in *plan* and return a log of completed ops.

        Args:
            plan:     The Plan produced by Planner (all ops start unapproved).
            mode:     Override the per-op mode for ops that have not been
                      explicitly assigned a different mode.
            history:  If provided, each successful OpLogEntry is appended to
                      the History log automatically.
            dry_run:  When True (the default), return [] immediately without
                      touching the filesystem (TC-SAFE-1).

        Returns:
            List of OpLogEntry for every op that was **successfully enacted**.
            Empty list on dry_run=True or when no op is both approved and
            successfully executed.
        """
        # TC-SAFE-1: dry-run default — mutate nothing, return empty.
        if dry_run:
            return []

        log: list[OpLogEntry] = []

        for op in plan.moves:
            # §7.1: only act on explicitly approved ops.
            if not op.approved:
                continue

            # Resolve effective mode (per-op wins, then caller-level override).
            effective_mode = op.mode if op.mode != MoveMode.TRASH else mode

            entry = self._execute_op(op, effective_mode)
            if entry is None:
                continue  # op failed; already logged to stderr inside helper

            log.append(entry)
            if history is not None:
                history.append(entry)

        return log

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_op(
        self,
        op: MoveOp,
        mode: MoveMode,
    ) -> OpLogEntry | None:
        """Enact a single MoveOp.  Returns None on failure (per-op atomicity).

        Computes hash_before for G10 undo verification before touching disk.
        """
        src = Path(op.src)
        dst = Path(op.dst)

        # Validate source still exists before doing anything.
        if not src.exists():
            _warn(f"[executor] source missing, skipping: {src}")
            return None

        # Capture pre-move hash for G10 before any filesystem change.
        hash_before = _sha256(src)

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)

            if mode == MoveMode.TRASH:
                self._enact_trash(src, dst)
                reversible = True

            elif mode == MoveMode.COPY:
                shutil.copy2(str(src), str(dst))
                reversible = True  # original untouched; dst can be removed

            elif mode == MoveMode.SYMLINK:
                os.symlink(str(src), str(dst))
                reversible = True  # original untouched; symlink can be removed

            elif mode == MoveMode.HARD:
                # Unrecoverable move (G1 explicit opt-in path).
                shutil.move(str(src), str(dst))
                reversible = False

            else:
                _warn(f"[executor] unknown mode {mode!r}, skipping op")
                return None

        except Exception as exc:  # noqa: BLE001
            # §7.4 per-op atomicity: log failure, do not abort the batch.
            _warn(f"[executor] failed to enact op {op.src!r} -> {op.dst!r}: {exc}")
            return None

        return OpLogEntry(
            op_id=0,            # assigned by History.append if history is given
            ts=time.time(),
            src_before=op.src,
            dst_after=str(dst),
            mode=mode,
            hash_before=hash_before,
            reversible=reversible,
        )

    @staticmethod
    def _enact_trash(src: Path, dst: Path) -> None:
        """TRASH mode move: no-silent-clobber guarantee (G1).

        If *dst* already exists, send it to the system trash first so the
        existing file is recoverable, then move *src* into *dst*'s place.
        """
        if dst.exists():
            # Send the existing destination to trash before overwriting (G1).
            send2trash(str(dst))

        shutil.move(str(src), str(dst))


def _warn(msg: str) -> None:
    """Emit a warning to stderr without importing logging."""
    import sys
    print(msg, file=sys.stderr)
