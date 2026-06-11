"""Feature extraction for the M1 prototype classifier.

This module implements FeatureExtractor (ARCHITECTURE-EXTENSION.md §2,
"Feature Extraction | FeatureExtractor" row) for the M1 scope only.

M1 scope constraints (ARCHITECTURE-EXTENSION.md §5):
- Tier 2 (TEXT): read at most `text_cap_bytes` of raw bytes — the *first N KB*
  cap from ARCHITECTURE.md §3 — decode as UTF-8 with errors="replace".
- Tier 3 (VISION): do NOT load pixels (CLIP load is deferred to M2+). Record
  image_path and lightweight stat metadata for the downstream EmbeddingService.
- Tier 1 (METADATA) / Tier 4 (REVIEW): no content read; modality=NONE.
- All file I/O is wrapped in try/except.  A corrupt, unreadable, or zero-byte
  file ALWAYS returns FileFeatures(error=...) with modality=NONE so the caller
  routes it to needs_review (done-criterion 8).

No ML imports, no OCR, no third-party deps — stdlib only (pathlib, os, typing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from organizer.types import FileFeatures, FileRecord, Modality, Tier


def extract(rec: FileRecord, text_cap_bytes: int = 4096) -> FileFeatures:
    """Extract lightweight features from *rec* appropriate to its tier.

    Parameters
    ----------
    rec:
        A ``FileRecord`` produced by the Scanner.  ``rec.tier`` must be set by
        the Tierer before this function is called.
    text_cap_bytes:
        Maximum bytes to read from a Tier-2 text file.  The caller controls
        this cap so tests can exercise the boundary without large fixtures.
        NEVER reads more than this value (ARCHITECTURE.md §3 "first N KB").

    Returns
    -------
    FileFeatures
        Always returns a value; never raises.  On any I/O or decode failure,
        the returned object has ``error`` set and ``modality=Modality.NONE``
        (done-criterion 8: "No crash on malformed input").
    """

    tier = rec.tier

    # ------------------------------------------------------------------ Tier 2
    if tier is Tier.TEXT:
        # PDFs carry compressed binary streams; raw-byte reads embed garbage.
        # Extract the real text layer via pymupdf when available (§4.3 "Text
        # extraction: pymupdf"). Scanned PDFs without a text layer come back
        # empty -> needs_review rather than a garbage classification.
        if rec.extension == "pdf":
            return _extract_pdf(rec, text_cap_bytes)
        return _extract_text(rec, text_cap_bytes)

    # ------------------------------------------------------------------ Tier 3
    if tier is Tier.VISION:
        return _extract_vision(rec)

    # ------------------------------------------------------------------ Tier 1
    if tier is Tier.METADATA:
        # No content read; extension/MIME-based classification only (§3 Tier 1).
        return FileFeatures(
            modality=Modality.NONE,
            metadata={"reason": "tier1_metadata_only"},
        )

    # ------------------------------------------------------------------ Tier 4
    if tier is Tier.REVIEW:
        # Ambiguous / risky files are never auto-moved (§3 Tier 4).
        return FileFeatures(
            modality=Modality.NONE,
            metadata={"reason": "tier4_needs_review"},
        )

    # ----------------------------------------------------------------- Unknown
    # Defensive fallback: an unrecognised tier value is treated as needs_review
    # rather than an exception, preserving the no-crash invariant.
    return FileFeatures(
        modality=Modality.NONE,
        error=f"unknown_tier:{tier!r}",
        metadata={"reason": "unknown_tier"},
    )


# ---------------------------------------------------------------------------
# Internal helpers — each returns FileFeatures and never raises
# ---------------------------------------------------------------------------


def _extract_pdf(rec: FileRecord, text_cap_bytes: int) -> FileFeatures:
    """Extract the text layer of a PDF, capped at *text_cap_bytes* characters.

    pymupdf is imported lazily so the stdlib-only paths stay import-light.
    Failure modes all route to needs_review (never raise):
      - pymupdf not installed        -> error="pdf_support_not_installed"
      - corrupt / encrypted PDF      -> error="pdf_unreadable:..."
      - no text layer (scanned doc)  -> error="pdf_no_text_layer" (OCR is out
        of scope; §14 "leave these in needs_review rather than guessing")
    """
    try:
        import fitz  # pymupdf — lazy
    except ImportError:
        return FileFeatures(
            modality=Modality.NONE,
            error="pdf_support_not_installed",
            metadata={"hint": "pip install pymupdf"},
        )

    try:
        text_parts: list[str] = []
        remaining = text_cap_bytes
        with fitz.open(rec.path) as doc:
            if doc.needs_pass:
                return FileFeatures(
                    modality=Modality.NONE,
                    error="pdf_encrypted",
                    metadata={"pages": doc.page_count},
                )
            for page in doc:
                if remaining <= 0:
                    break
                chunk = page.get_text("text")[:remaining]
                text_parts.append(chunk)
                remaining -= len(chunk)
            pages = doc.page_count
        text = "".join(text_parts).strip()
    except Exception as exc:  # noqa: BLE001 — never raise out of extract()
        return FileFeatures(
            modality=Modality.NONE,
            error=f"pdf_unreadable:{type(exc).__name__}",
            metadata={},
        )

    if not text:
        # Scanned document with no text layer; OCR deferred (§12.4).
        return FileFeatures(
            modality=Modality.NONE,
            error="pdf_no_text_layer",
            metadata={"pages": pages},
        )

    return FileFeatures(
        modality=Modality.TEXT,
        text=text,
        metadata={"pages": pages, "chars_extracted": len(text)},
    )


def _extract_text(rec: FileRecord, text_cap_bytes: int) -> FileFeatures:
    """Read at most *text_cap_bytes* from a Tier-2 file and decode as UTF-8."""

    path: Path = rec.path

    try:
        # Guard: zero-byte files carry no useful signal (done-criterion 8).
        stat = path.stat()
        file_size = stat.st_size
        if file_size == 0:
            return FileFeatures(
                modality=Modality.NONE,
                error="zero_byte_file",
                metadata={"path": str(path), "bytes_read": 0},
            )

        # Read exactly the cap — no more (ARCHITECTURE.md §3 "first N KB").
        with path.open("rb") as fh:
            raw = fh.read(text_cap_bytes)

        bytes_read = len(raw)
        # Decode with errors="replace" so corrupt sequences never raise.
        snippet = raw.decode("utf-8", errors="replace")

        return FileFeatures(
            modality=Modality.TEXT,
            text=snippet,
            metadata={"bytes_read": bytes_read},
        )

    except (OSError, PermissionError, UnicodeError) as exc:
        # Surface as a non-fatal error so the caller routes to needs_review.
        return FileFeatures(
            modality=Modality.NONE,
            error=f"text_read_error:{type(exc).__name__}",
            metadata={"path": str(path)},
        )


def _extract_vision(rec: FileRecord) -> FileFeatures:
    """Prepare a Tier-3 image record without reading pixel data (M1 scope-out).

    Real CLIP inference is deferred to M2+
    (ARCHITECTURE-EXTENSION.md §5 "Scope out — OCR / pixel read").
    We record lightweight stat metadata so EmbeddingService can locate the file.
    """

    path: Path = rec.path

    try:
        stat = path.stat()
        file_size = stat.st_size

        if file_size == 0:
            return FileFeatures(
                modality=Modality.NONE,
                error="zero_byte_file",
                metadata={"path": str(path), "bytes_read": 0},
            )

        metadata: dict[str, Any] = {
            "size_bytes": file_size,
            "extension": rec.extension,
            "mtime": stat.st_mtime,
        }

        return FileFeatures(
            modality=Modality.IMAGE,
            image_path=path,
            metadata=metadata,
        )

    except (OSError, PermissionError) as exc:
        return FileFeatures(
            modality=Modality.NONE,
            error=f"vision_stat_error:{type(exc).__name__}",
            metadata={"path": str(path)},
        )
