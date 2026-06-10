"""Classification layer — PreferenceStore (M4).

Durable, append-only JSONL feedback log feeding the user-adaptation layer
(ARCHITECTURE.md §4.2(4)). Implements the PreferenceStore module contract
(ARCHITECTURE-EXTENSION.md §2) and persists DM-FeedbackEvent rows (§3). It is
also the resolution sink for needs_review files (gap G7): a manual label emits
a FeedbackEvent recorded here.

Storage mirrors history.py: one JSON object per line, existing lines are never
rewritten.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedbackEvent:
    """One user feedback signal (DM-FeedbackEvent, ARCHITECTURE-EXTENSION.md §3).

    file_path: Source file the feedback is about.
    from_cat:  Previously-assigned/suggested leaf category, or None.
    to_cat:    The category the user confirmed (accept), rejected (reject), or
               moved the file into (override).
    action:    One of 'accept' | 'reject' | 'override'.
    ts:        Epoch seconds.
    """

    file_path: str
    from_cat: str | None
    to_cat: str
    action: str
    ts: float


def _event_to_dict(e: FeedbackEvent) -> dict:
    return {
        "file_path": e.file_path,
        "from_cat": e.from_cat,
        "to_cat": e.to_cat,
        "action": e.action,
        "ts": e.ts,
    }


def _dict_to_event(d: dict) -> FeedbackEvent:
    return FeedbackEvent(
        file_path=d["file_path"],
        from_cat=d.get("from_cat"),
        to_cat=d["to_cat"],
        action=d["action"],
        ts=d.get("ts", 0.0),
    )


class PreferenceStore:
    """Append-only JSONL feedback log with a simple rule-weight bias (§4.2(4)).

    Args:
        store_path: Path to the JSONL log; created on first record().
    """

    # Per-signal weight deltas (§4.2(4) "rule weights" starting point).
    _ACCEPT_DELTA = 0.01        # accept of a cat nudges it up
    _REJECT_DELTA = -0.01       # reject of a cat / override away from it nudges down
    _OVERRIDE_IN_DELTA = 0.02   # override INTO a cat is the strongest positive signal
    _OVERRIDE_AWAY_DELTA = -0.01
    _CLAMP = 0.10               # bias is clamped to [-0.10, +0.10]

    def __init__(self, store_path: Path) -> None:
        if not isinstance(store_path, Path):
            raise TypeError(f"store_path must be a Path, got {type(store_path)}")
        self._store_path = store_path

    def record(self, event: FeedbackEvent) -> None:
        """Append one FeedbackEvent as a JSON line (durable, append-only)."""
        line = json.dumps(_event_to_dict(event), ensure_ascii=False) + "\n"
        with open(self._store_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def events(self) -> list[FeedbackEvent]:
        """Return all recorded events in append order (empty if no log yet)."""
        if not self._store_path.exists():
            return []
        out: list[FeedbackEvent] = []
        with open(self._store_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(_dict_to_event(json.loads(raw)))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[preferences] bad log line {lineno}: {exc}",
                          file=sys.stderr)
        return out

    def bias(self, cat_id: str) -> float:
        """Return the accumulated rule-weight bias for *cat_id*, clamped to ±0.10.

        Simple linear adaptation (§4.2(4) — the M4 "rule weights" starting point;
        embedding-space adaptation is deferred per §12.5). Starting from 0.0, each
        feedback event contributes relative to *cat_id*:

            accept   where to_cat == cat_id   -> +0.01
            reject   where to_cat == cat_id   -> -0.01
            override where to_cat == cat_id   -> +0.02  (override INTO the cat)
            override where from_cat == cat_id -> -0.01  (override AWAY from the cat)

        (An override both nudges its destination up and its source down.) The
        running total is clamped to the inclusive range [-0.10, +0.10], so e.g.
        20 accepts of the same category still yield exactly +0.10.
        """
        delta = 0.0
        for e in self.events():
            if e.action == "accept" and e.to_cat == cat_id:
                delta += self._ACCEPT_DELTA
            elif e.action == "reject" and e.to_cat == cat_id:
                delta += self._REJECT_DELTA
            elif e.action == "override":
                if e.to_cat == cat_id:
                    delta += self._OVERRIDE_IN_DELTA
                if e.from_cat == cat_id:
                    delta += self._OVERRIDE_AWAY_DELTA
        return max(-self._CLAMP, min(self._CLAMP, delta))
