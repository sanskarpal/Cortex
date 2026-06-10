"""Persistence layer for the AI File Organizer — M2 milestone.

Wraps stdlib ``sqlite3`` with the schema described in ARCHITECTURE-EXTENSION.md
§2 (module contract for ``Database``) and §3 (DM-FileRecord M2 subset,
DM-CacheKey).  No third-party ORM: stdlib only, as mandated by the spec.

Cache-key semantics (G6 / DM-CacheKey):
  - **Content-cache** (by_content=True): key = (content_hash,).
    Falls back silently to (path, size, mtime) when content_hash is None and
    logs a weaker-guarantee comment; see ``cache_key``.
  - **Stat-cache** (by_content=False): key = (path, size, mtime).
    A row is considered cached (TC-CACHE-1) iff its stored key matches *and*
    ``category`` is non-null, so a pending row is never considered skippable.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from organizer.types import FileRecord, Tier

# Increment when the schema changes; migrate() is idempotent at each version.
_SCHEMA_VERSION = 1


class Database:
    """SQLite-backed store for FileRecord rows (ARCHITECTURE-EXTENSION.md §2).

    Usage::

        with Database(":memory:") as db:
            db.migrate()
            row_id = db.upsert(rec)
    """

    # ------------------------------------------------------------------
    # Construction / lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the SQLite database at *db_path*.

        ``:memory:`` is fully supported for tests.
        """
        self._path = str(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        # Enable WAL for concurrent readers during long classify passes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema migration (idempotent, versioned via `meta` table)
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """Create schema if absent and advance to the current version.

        Safe to call multiple times (idempotent).  Uses a ``meta`` key/value
        table to track ``schema_version`` so future migrations can gate on it.

        Schema corresponds to DM-FileRecord M2 subset
        (ARCHITECTURE-EXTENSION.md §3) plus the columns needed for M2 cache
        and classification output.  Columns deferred to later milestones
        (``text_snippet``, ``embedding``, ``embedding_space``, ``ctime``) are
        intentionally absent here; they will be added in a future migration step.
        """
        cur = self._conn.cursor()

        # --- meta table (schema version tracking) ---
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # Read current stored version (None == fresh database).
        cur.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        stored_version = int(row["value"]) if row else 0

        if stored_version >= _SCHEMA_VERSION:
            self._conn.commit()
            return  # already up to date

        # --- version 0 → 1: create `files` table ---
        if stored_version < 1:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    path         TEXT    NOT NULL UNIQUE,
                    size         INTEGER NOT NULL,
                    mtime        REAL    NOT NULL,
                    content_hash TEXT    DEFAULT NULL,
                    extension    TEXT    NOT NULL,
                    mime         TEXT    NOT NULL,
                    tier         INTEGER NOT NULL,
                    category     TEXT    DEFAULT NULL,
                    confidence   REAL    DEFAULT NULL,
                    status       TEXT    NOT NULL DEFAULT 'pending'
                )
                """
            )
            # §6 cache lookups: index on path alone and on the stat-cache key.
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_path "
                "ON files (path)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_files_path_size_mtime "
                "ON files (path, size, mtime)"
            )

        # Stamp the new version.
        cur.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(_SCHEMA_VERSION),),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(
        self,
        rec: FileRecord,
        *,
        category: str | None = None,
        confidence: float | None = None,
        content_hash: str | None = None,
        status: str | None = None,
    ) -> int:
        """Insert or update the row for *rec*; return its row ``id``.

        ``content_hash`` is accepted as a separate argument because
        ``FileRecord`` does not (yet) carry that field — see NOTE in task
        brief and G6.  Callers that have already computed a hash pass it here;
        it is also read defensively from ``rec`` via ``getattr`` in case a
        future version of the type adds the field.

        ``tier`` is stored as its integer value (``rec.tier.value``) when set,
        or 0 as a sentinel when unset (should not occur in normal usage but is
        tolerated so that upsert never raises on a partially-populated record).
        """
        # Defensively read content_hash from rec if present in a future version.
        effective_hash: str | None = content_hash or getattr(rec, "content_hash", None)

        # Status lifecycle (DM-FileRecord): an explicit `status` argument wins;
        # otherwise a classify-pass (category kwarg used) derives it, and a
        # plain scan upsert keeps the record's own status.
        if status is None:
            if category is not None:
                status = "classified"
            elif confidence is not None:
                # classify ran but produced no category -> needs_review
                status = "needs_review"
            else:
                status = rec.status

        tier_int: int = rec.tier.value if isinstance(rec.tier, Tier) else (
            int(rec.tier) if rec.tier is not None else 0
        )

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO files
                (path, size, mtime, content_hash, extension, mime,
                 tier, category, confidence, status)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size         = excluded.size,
                mtime        = excluded.mtime,
                content_hash = excluded.content_hash,
                extension    = excluded.extension,
                mime         = excluded.mime,
                tier         = excluded.tier,
                category     = excluded.category,
                confidence   = excluded.confidence,
                status       = excluded.status
            """,
            (
                str(rec.path),
                rec.size,
                rec.mtime,
                effective_hash,
                rec.extension,
                rec.mime,
                tier_int,
                category,
                confidence,
                status,
            ),
        )
        self._conn.commit()
        # Return the rowid of the affected row (inserted or updated).
        if cur.lastrowid and cur.lastrowid != 0:
            return cur.lastrowid
        # On conflict-update SQLite may return 0; resolve via a SELECT.
        cur.execute("SELECT id FROM files WHERE path = ?", (str(rec.path),))
        return int(cur.fetchone()["id"])

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        """Return the stored row for *path* as a plain dict, or ``None``.

        Parameterized query only — no string interpolation (CLAUDE.md rule).
        """
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM files WHERE path = ?", (path,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def all_rows(self) -> list[dict[str, Any]]:
        """Return every stored file row as a list of dicts (for `status`)."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM files")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Cache-key support (G6 / DM-CacheKey, TC-CACHE-1)
    # ------------------------------------------------------------------

    def cache_key(
        self,
        rec_or_dict: FileRecord | dict[str, Any],
        *,
        by_content: bool,
    ) -> tuple[Any, ...]:
        """Return the cache key for *rec_or_dict* (DM-CacheKey).

        Two modes (G6):

        ``by_content=True``
            Key = ``(content_hash,)``.  This is the **strong** guarantee:
            two files are identical iff they share a SHA-256 hash regardless
            of path or mtime.

            **Weaker-guarantee fallback (G6):** if ``content_hash`` is
            ``None`` (hash not yet computed, or hash disabled), the key falls
            back to ``(path, size, mtime)``; the caller is responsible for
            ensuring that mode is only used when content-cache is truly
            enabled.  The fallback is documented here rather than silently
            promoted to strong-cache semantics.

        ``by_content=False``
            Key = ``(path, size, mtime)``.  Matches the §6 stat-only cache:
            suitable when content-hashing is disabled, as required by G6.
        """
        if isinstance(rec_or_dict, dict):
            path = rec_or_dict["path"]
            size = rec_or_dict["size"]
            mtime = rec_or_dict["mtime"]
            # getattr-style fallback for dict; missing key → None.
            h = rec_or_dict.get("content_hash")
        else:
            path = str(rec_or_dict.path)
            size = rec_or_dict.size
            mtime = rec_or_dict.mtime
            h = getattr(rec_or_dict, "content_hash", None)

        if by_content:
            if h is not None:
                return (h,)
            # G6: content_hash is None — fall back to stat key and surface the
            # weaker guarantee (the caller should warn the user per G6 contract).
            return (path, size, mtime)

        return (path, size, mtime)

    def is_cached(self, rec: FileRecord, *, by_content: bool) -> bool:
        """Return ``True`` iff this file can be safely skipped (TC-CACHE-1).

        A row is considered **cached** (and re-classification skippable) when:

        1. A stored row exists whose cache key matches the current file's key.
        2. That row already has a non-null ``category`` — a row that is still
           ``pending`` with no category must *not* be skipped.

        When ``by_content=True`` but ``content_hash`` is ``None`` on *rec*,
        the stat-key fallback is used (G6 weaker guarantee).

        This satisfies TC-CACHE-1: "second pass performs zero embedding calls
        if size/mtime/hash unchanged."
        """
        key = self.cache_key(rec, by_content=by_content)
        cur = self._conn.cursor()

        if by_content and len(key) == 1:
            # Strong content-hash lookup.
            cur.execute(
                "SELECT category FROM files WHERE content_hash = ? "
                "AND category IS NOT NULL LIMIT 1",
                (key[0],),
            )
        else:
            # Stat-key lookup: (path, size, mtime) — covers both
            # by_content=False and the G6 fallback when hash is None.
            cur.execute(
                "SELECT category FROM files "
                "WHERE path = ? AND size = ? AND mtime = ? "
                "AND category IS NOT NULL LIMIT 1",
                (key[0], key[1], key[2]),
            )

        return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(path: str | Path) -> str:
        """Return the SHA-256 hex digest of the file at *path*.

        Streamed in 1 MiB chunks to avoid loading large files into memory.
        Only called when content-cache is enabled (G6 opt-in).
        """
        h = hashlib.sha256()
        chunk_size = 1 << 20  # 1 MiB
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
