"""pytest configuration for the M1 test suite.

Inserts the repo's ``src/`` directory onto sys.path so that
``import organizer.*`` works when pytest is run from the repository root
without a pip-installed package.  Path is computed relative to *this* file
so it works regardless of where the user has cloned the repository.
"""

from __future__ import annotations

import pathlib
import sys
import struct
import zipfile

import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap — must happen before any test file imports organizer.*
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent          # .../tests/
_SRC = (_HERE / ".." / "src").resolve()        # .../src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Minimal valid 1×1 PNG blob (stdlib only, no Pillow).
#
# A PNG file is: 8-byte signature + IHDR chunk + IDAT chunk + IEND chunk.
# Each chunk is: 4-byte length + 4-byte type + data + 4-byte CRC.
# ---------------------------------------------------------------------------

def _png_1x1_rgba(r: int = 0, g: int = 0, b: int = 0, a: int = 255) -> bytes:
    """Return the raw bytes of a minimal 1×1 RGBA PNG."""
    import zlib

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: width=1, height=1, bit_depth=8, color_type=6 (RGBA), rest 0
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: single pixel, filter byte 0 then RGBA
    raw_row = bytes([0, r, g, b, a])          # filter-none row
    compressed = zlib.compress(raw_row)
    idat = _chunk(b"IDAT", compressed)

    # IEND
    iend = _chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Fixture tree factory
#
# Built in tmp_path so every test run gets a clean, hermetic tree.
# Layout (relative to the root tmp dir):
#
#   fixture_root/
#   ├── sample.txt          # Tier 2 TEXT
#   ├── sample.md           # Tier 2 TEXT
#   ├── sample.py           # Tier 2 TEXT
#   ├── sample.png          # Tier 3 VISION  (valid 1×1 PNG)
#   ├── sample.zip          # Tier 1 METADATA (stdlib zipfile)
#   ├── no_extension        # Tier 4 REVIEW  (no extension)
#   └── .git/
#       └── HEAD            # MUST be excluded by scan()
# ---------------------------------------------------------------------------

@pytest.fixture()
def fixture_root(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build the M1 fixture tree inside *tmp_path* and return the root dir."""

    root = tmp_path / "fixture_root"
    root.mkdir()

    # --- Tier 2 text files ---
    (root / "sample.txt").write_text(
        "This is a plain text document with some words.", encoding="utf-8"
    )
    (root / "sample.md").write_text(
        "# Markdown Heading\n\nSome **markdown** content here.", encoding="utf-8"
    )
    (root / "sample.py").write_text(
        "def hello():\n    \"\"\"A simple Python function.\"\"\"\n    return 'hello'\n",
        encoding="utf-8",
    )

    # --- Tier 3 vision file ---
    (root / "sample.png").write_bytes(_png_1x1_rgba(100, 150, 200))

    # --- Tier 1 archive ---
    zip_path = root / "sample.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hello.txt", "hello from inside the zip")

    # --- Tier 4 no-extension file ---
    (root / "no_extension").write_text(
        "file with no extension for review tier", encoding="utf-8"
    )

    # --- .git dir that MUST be excluded ---
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    return root


@pytest.fixture()
def zero_byte_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A zero-byte .txt file (Tier 2) whose extraction should return needs_review."""
    p = tmp_path / "empty.txt"
    p.touch()
    return p


@pytest.fixture()
def corrupt_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A file named .png (Tier 3) but containing garbage bytes."""
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"\x00\xff\xfe\xfd" * 8)
    return p
