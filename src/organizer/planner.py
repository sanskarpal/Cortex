"""Organization layer — Planner (M2).

Turns a list of (FileRecord, Classification) pairs into a pure, side-effect-free
Plan of proposed MoveOps.  Nothing in this module may touch the filesystem
(§7.1 dry-run; §2 invariant: only Executor mutates).

Refs: ARCHITECTURE.md §7, ARCHITECTURE-EXTENSION.md §2 (Planner row), §3
(DM-MoveOp / DM-Plan), G1 (trash default), G5 (leaf categories).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from organizer.types import Classification, FileRecord, MoveMode, MoveOp, Plan


class Planner:
    """Produce a Plan from classifier output.

    Pure: no filesystem I/O, no side effects.  Every MoveOp emitted has
    ``approved=False`` and ``mode=MoveMode.TRASH`` (G1 trash default).

    Args:
        dest_root: Root directory under which organised files will be placed.
                   Sub-directories are derived from each op's cat_id.
        min_confidence: If provided, files whose ``Classification.confidence``
                        falls below this threshold are routed to
                        ``Plan.needs_review`` instead of being proposed for a
                        move.  When None, confidence gating is assumed to have
                        been applied upstream (§7.2).
    """

    def __init__(
        self,
        dest_root: Path,
        min_confidence: Optional[float] = None,
    ) -> None:
        if not isinstance(dest_root, Path):
            raise TypeError(f"dest_root must be a Path, got {type(dest_root)}")
        if min_confidence is not None and not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {min_confidence}"
            )
        self._dest_root = dest_root
        self._min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(
        self,
        items: list[tuple[FileRecord, Classification]],
    ) -> Plan:
        """Build a Plan from classified file records.

        For each item:
        - If ``classification.cat_id is None`` (needs_review) OR
          ``min_confidence`` is set and ``confidence < min_confidence``:
          route ``rec.path`` to ``Plan.needs_review``.
        - Otherwise: build a MoveOp with a deterministic collision-safe dst
          and append to ``Plan.moves``.

        All MoveOps are emitted with ``approved=False`` and
        ``mode=MoveMode.TRASH`` (G1 default).  No filesystem operations are
        performed here (TC-SAFE-1).

        Returns:
            A fully populated Plan.
        """
        moves: list[MoveOp] = []
        needs_review: list[str] = []

        # Track names already allocated in this planning pass to resolve
        # collisions deterministically without any disk access.
        # key: lowercase target path string -> count of times seen.
        _dst_counts: dict[str, int] = {}

        for rec, cls in items:
            src_str = str(rec.path)

            # Gate 1: unclassifiable file (Tier 4 / classifier abstained)
            if cls.cat_id is None:
                needs_review.append(src_str)
                continue

            # Gate 2: optional confidence threshold (§7.2)
            if (
                self._min_confidence is not None
                and cls.confidence < self._min_confidence
            ):
                needs_review.append(src_str)
                continue

            # Build destination path: <dest_root>/<cat_id>/<filename>
            dst = self._compute_dst(rec.path, cls.cat_id, _dst_counts)

            moves.append(
                MoveOp(
                    src=src_str,
                    dst=dst,
                    cat_id=cls.cat_id,
                    confidence=cls.confidence,
                    mode=MoveMode.TRASH,   # G1: trash is the safe default
                    approved=False,        # §7.1: dry-run; user must confirm
                )
            )

        return Plan(moves=moves, needs_review=needs_review)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_dst(
        self,
        src: Path,
        cat_id: str,
        seen: dict[str, int],
    ) -> str:
        """Return a deterministic, collision-free destination path string.

        Collisions within the same planning pass are resolved by appending
        ``_<n>`` before the suffix (no disk access required; purely
        deterministic based on the items seen so far in this call).
        """
        # cat_id uses forward slashes (G5 leaf ids, e.g. "documents/invoices")
        cat_subdir = Path(*cat_id.split("/")) if "/" in cat_id else Path(cat_id)
        base_dir = self._dest_root / cat_subdir

        stem = src.stem
        suffix = src.suffix  # includes the dot, e.g. ".pdf"

        candidate = str(base_dir / f"{stem}{suffix}")
        key = candidate.lower()  # case-insensitive collision tracking

        if key not in seen:
            seen[key] = 0
            return candidate

        # Increment and keep trying until we find an unseen slot.
        while True:
            seen[key] += 1
            n = seen[key]
            numbered = str(base_dir / f"{stem}_{n}{suffix}")
            nkey = numbered.lower()
            if nkey not in seen:
                seen[nkey] = 0
                return numbered
