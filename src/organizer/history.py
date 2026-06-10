"""Organization layer — History (M2).

Durable, append-only JSONL operation log with verified undo (G10).

Refs: ARCHITECTURE.md §7.4, ARCHITECTURE-EXTENSION.md §2 (History row),
§3 (DM-OpLogEntry), G10 (undo vs. watcher race), TC-SAFE-5.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from organizer.types import MoveMode, OpLogEntry


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _entry_to_dict(e: OpLogEntry) -> dict:
    return {"op_id": e.op_id, "ts": e.ts, "src_before": e.src_before,
            "dst_after": e.dst_after, "mode": e.mode.value,
            "hash_before": e.hash_before, "reversible": e.reversible}


def _dict_to_entry(d: dict) -> OpLogEntry:
    return OpLogEntry(op_id=d["op_id"], ts=d["ts"],
                      src_before=d["src_before"], dst_after=d["dst_after"],
                      mode=MoveMode(d["mode"]),
                      hash_before=d.get("hash_before"),
                      reversible=d.get("reversible", True))


def _sha256(path: Path) -> str | None:
    """SHA-256 hex digest of *path*, or None on read failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# History class
# ---------------------------------------------------------------------------

class History:
    """Append-only JSONL op log supporting verified undo (G10).

    Existing lines are never rewritten. ``undo`` performs inverse filesystem
    moves directly (§2: History is permitted to make inverse moves for undo).

    Args:
        log_path: Path to JSONL log; created on first append.
    """

    def __init__(self, log_path: Path) -> None:
        if not isinstance(log_path, Path):
            raise TypeError(f"log_path must be a Path, got {type(log_path)}")
        self._log_path = log_path
        self._next_op_id: int = self._read_max_op_id() + 1

    def append(self, entry: OpLogEntry) -> None:
        """Append one entry as a JSON line; assigns monotonic op_id if unset."""
        if entry.op_id == 0:
            entry.op_id = self._next_op_id
            self._next_op_id += 1
        line = json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n"
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def entries(self) -> list[OpLogEntry]:
        """Return all recorded entries in append order."""
        if not self._log_path.exists():
            return []
        out: list[OpLogEntry] = []
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(_dict_to_entry(json.loads(raw)))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[history] bad log line {lineno}: {exc}", file=sys.stderr)
        return out

    def undo(self, n: int) -> dict:
        """Reverse the last *n* reversible ops with per-op state verification (G10).

        For each op (reverse chronological): verify dst_after exists, src_before
        absent, and hash matches. Abort only the mismatched op; batch continues.

        Returns ``{"reversed": int, "skipped": int, "details": list[dict]}``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        candidates = list(reversed(
            [e for e in self.entries() if e.reversible][-n:]
        ))
        reversed_count = skipped_count = 0
        details: list[dict] = []
        for entry in candidates:
            result = self._undo_one(entry)
            details.append(result)
            if result["status"] == "reversed":
                reversed_count += 1
            else:
                skipped_count += 1
        return {"reversed": reversed_count, "skipped": skipped_count,
                "details": details}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _undo_one(self, entry: OpLogEntry) -> dict:
        """Attempt to reverse one op; returns a status detail dict."""
        dst, src = Path(entry.dst_after), Path(entry.src_before)
        base = {"op_id": entry.op_id, "src_before": entry.src_before,
                "dst_after": entry.dst_after}

        # G10 check 1: file must be at dst_after.
        if not dst.exists():
            return {**base, "status": "skipped",
                    "reason": "dst_after not found on filesystem"}
        # G10 check 2: src_before must be absent (no clobber).
        if src.exists():
            return {**base, "status": "skipped",
                    "reason": "src_before already exists; would clobber"}
        # G10 check 3: hash integrity if recorded.
        if entry.hash_before is not None:
            cur = _sha256(dst)
            if cur != entry.hash_before:
                return {**base, "status": "skipped",
                        "reason": (f"hash mismatch (expected "
                                   f"{entry.hash_before[:8]}…, "
                                   f"got {(cur or 'none')[:8]}…)")}
        # All checks pass — inverse move.
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
        except Exception as exc:  # noqa: BLE001
            return {**base, "status": "skipped", "reason": f"move failed: {exc}"}
        return {**base, "status": "reversed", "ts": time.time()}

    def _read_max_op_id(self) -> int:
        """Highest op_id already in the log (0 if absent/empty)."""
        if not self._log_path.exists():
            return 0
        max_id = 0
        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        max_id = max(max_id, int(json.loads(raw).get("op_id", 0)))
                    except (json.JSONDecodeError, ValueError):
                        pass
        except OSError:
            pass
        return max_id
