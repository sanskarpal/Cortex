"""Ingestion layer for the M1 prototype — Scanner + Tierer.

Implements the ``Scanner`` and ``Tierer`` responsibilities from
ARCHITECTURE-EXTENSION.md §2.  Both operate stat-only; no file bytes are
ever opened or read (TC-SAFE-1, M1 done-criterion 7, G9 — crawl and classify
are distinct phases with independent budgets).

Importable as::

    from organizer.ingest import scan, tier_of, EXCLUDES

Callers must ensure ``src/`` is on ``sys.path``.
"""

from __future__ import annotations

import mimetypes
import os
import pathlib
from typing import Iterator

from organizer.types import FileRecord, Tier

# ---------------------------------------------------------------------------
# §5 / TC-SAFE-6 — default-excluded directory names.
# Any path component that matches a name in this set (case-sensitive) is
# silently skipped.  Dotfiles and dot-dirs are excluded by a separate rule
# in scan() so that any new dot-prefixed name is caught automatically.
# ---------------------------------------------------------------------------
EXCLUDES: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}

# ---------------------------------------------------------------------------
# §3 — extension sets, one per tier.
# Each constant is a frozenset of lowercased extensions *without* a leading
# dot so that lookups after `os.path.splitext` need only one `.lstrip(".")`.
# ---------------------------------------------------------------------------

# Tier 1 — Metadata / heuristic only (ARCHITECTURE.md §3 Tier 1)
_TIER1_EXT: frozenset[str] = frozenset(
    {
        # archives
        "zip", "tar", "gz", "tgz", "bz2", "xz", "7z", "rar",
        # binaries / executables / disk images
        "app", "dmg", "exe", "dll", "so", "dylib", "iso", "img",
        # fonts
        "ttf", "otf", "woff", "woff2",
    }
)

# Tier 2 — Lightweight content extraction / text (ARCHITECTURE.md §3 Tier 2)
_TIER2_EXT: frozenset[str] = frozenset(
    {
        # plain text / markup
        "txt", "md", "rst", "rtf",
        # source code
        "py", "js", "ts", "tsx", "jsx",
        "rs", "go", "java", "c", "cpp", "h", "hpp",
        "sh", "bash", "zsh", "fish",
        "rb", "php", "swift", "kt", "scala",
        # data / config
        "json", "yaml", "yml", "toml", "ini", "cfg", "env",
        "csv", "tsv",
        # documents
        "pdf",
        # web
        "html", "htm", "css", "xml", "svg",
    }
)

# Tier 3 — Vision / deep content extraction (ARCHITECTURE.md §3 Tier 3)
_TIER3_EXT: frozenset[str] = frozenset(
    {
        "jpg", "jpeg", "png", "heic", "heif",
        "webp", "gif", "bmp", "tiff", "tif",
        "raw", "cr2", "nef", "arw", "dng",
    }
)

# 2 GiB threshold — very large files go to Tier 4 regardless of extension
# (ARCHITECTURE.md §3 Tier 4: "very large files")
_VERY_LARGE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GiB


# ---------------------------------------------------------------------------
# Public API — Tierer (ARCHITECTURE-EXTENSION.md §2, "Tierer" row)
# ---------------------------------------------------------------------------

def tier_of(extension: str, mime: str, size: int) -> Tier:
    """Map a file's extension/MIME/size to a processing Tier.

    Pure function — no I/O.  Implements TC-TIER-1 exactly:
        .zip  → Tier.METADATA
        .py   → Tier.TEXT
        .jpg  → Tier.VISION
        (none)→ Tier.REVIEW

    Args:
        extension: lowercased extension without leading dot (e.g. ``"py"``).
                   Empty string means no extension.
        mime:      MIME type string, e.g. ``"text/plain"`` or ``""`` if
                   unknown.  Currently used only as a future hook; tier
                   assignment is extension-primary per §3.
        size:      File size in bytes (from ``os.stat``).

    Returns:
        The appropriate :class:`~organizer.types.Tier` value.
    """
    # Very large files always need human review regardless of type.
    if size >= _VERY_LARGE_BYTES:
        return Tier.REVIEW

    # No extension → Tier.REVIEW (TC-TIER-1: no-ext → Tier4)
    if not extension:
        return Tier.REVIEW

    if extension in _TIER1_EXT:
        return Tier.METADATA

    if extension in _TIER2_EXT:
        return Tier.TEXT

    if extension in _TIER3_EXT:
        return Tier.VISION

    # Unknown extension — needs human review (§3 Tier 4)
    return Tier.REVIEW


# Markers that identify a directory as a self-contained project/checkout which
# must NOT be reorganized from the inside (safety guard, see scan()).
_PROJECT_MARKERS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",            # version control
    "pyproject.toml", "setup.py",     # python project
    "package.json", "Cargo.toml",     # node / rust
    "go.mod", "pom.xml", "build.gradle",  # go / java
})


def _is_project_root(dirpath: str) -> bool:
    """True if *dirpath* contains a marker identifying it as a project/repo."""
    try:
        entries = set(os.listdir(dirpath))
    except OSError:
        return False
    return bool(entries & _PROJECT_MARKERS)


# ---------------------------------------------------------------------------
# Public API — Scanner (ARCHITECTURE-EXTENSION.md §2, "Scanner" row)
# ---------------------------------------------------------------------------

def scan(root: pathlib.Path) -> Iterator[FileRecord]:
    """Recursively walk *root* and yield one :class:`~organizer.types.FileRecord`
    per regular file.

    **Stat-only**: this function never opens or reads file bytes.  All
    information is obtained from ``os.stat`` and ``mimetypes.guess_type``
    (no network, no python-magic).  This satisfies TC-SAFE-1 and M1
    done-criterion 7.

    Exclusion rules (TC-SAFE-6, ARCHITECTURE-EXTENSION.md §5):
    - Directory (or file) names in :data:`EXCLUDES` are skipped entirely.
    - Names that begin with ``"."`` (dotfiles / dot-dirs) are skipped.

    Args:
        root: The directory to walk.  Must be an existing directory; if it
              is not, the generator simply yields nothing.

    Yields:
        :class:`~organizer.types.FileRecord` instances with ``tier`` and
        ``status`` already populated.  ``status`` is ``"needs_review"``
        when ``tier == Tier.REVIEW``; ``"pending"`` otherwise.
    """
    if not root.is_dir():
        return

    # os.walk with topdown=True lets us prune excluded dirs in-place.
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # SAFETY GUARD: never rip files out of a project/repo. A directory
        # that is the root of a version-control checkout or a packaged project
        # is treated as an atomic unit — we do not descend into it. This stops
        # the footgun where organizing a parent folder (e.g. ~/Desktop) scatters
        # a repo's source files into category folders. The repo's own root is
        # still scanned at the top level only if `root` IS that repo.
        if pathlib.Path(dirpath) != root and _is_project_root(dirpath):
            dirnames[:] = []  # do not recurse into this project
            continue

        # Prune excluded and dot-dirs to avoid descending into them.
        # Mutating `dirnames` in-place prevents os.walk from recursing.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in EXCLUDES
        ]

        for name in filenames:
            # Skip dotfiles.
            if name.startswith("."):
                continue

            file_path = pathlib.Path(dirpath) / name

            # stat — no bytes read; only metadata (TC-SAFE-1, G9).
            try:
                st = os.stat(file_path)
            except OSError:
                # Unreadable / broken symlink / race condition — skip silently.
                continue

            # Regular files only; skip symlinks, devices, sockets, etc.
            if not os.path.isfile(file_path):
                continue

            # Extension: lowercased, without leading dot.
            _, raw_ext = os.path.splitext(name)
            extension = raw_ext.lstrip(".").lower()

            # MIME: stdlib guess — no network, no file read.
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime = mime_type or ""

            size = st.st_size
            mtime = st.st_mtime

            tier = tier_of(extension, mime, size)
            status = "needs_review" if tier is Tier.REVIEW else "pending"

            yield FileRecord(
                path=file_path,
                size=size,
                mtime=mtime,
                extension=extension,
                mime=mime,
                tier=tier,
                status=status,
            )
