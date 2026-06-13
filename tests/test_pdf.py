"""Tests for PDF text-layer extraction (features._extract_pdf).

Real PDFs are compressed binary; raw-byte reads embed garbage. These verify
the pymupdf path: text PDFs yield clean TEXT features, scanned (no text
layer) and corrupt PDFs route to needs_review via the error field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="pymupdf not installed")

from organizer.features import extract
from organizer.types import FileRecord, Modality, Tier


def _pdf_rec(p: Path) -> FileRecord:
    st = p.stat()
    return FileRecord(
        path=p, size=st.st_size, mtime=st.st_mtime,
        extension="pdf", mime="application/pdf", tier=Tier.TEXT,
    )


def _make_text_pdf(path: Path, text: str, pages: int = 1) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestPdfExtraction:
    def test_text_pdf_extracts_real_text(self, tmp_path):
        p = tmp_path / "invoice.pdf"
        _make_text_pdf(p, "INVOICE total amount due $1,200 payment net 30")
        feats = extract(_pdf_rec(p))
        assert feats.error is None
        assert feats.modality is Modality.TEXT
        assert "total amount due" in feats.text
        assert feats.metadata["pages"] == 1

    def test_text_capped(self, tmp_path):
        p = tmp_path / "long.pdf"
        _make_text_pdf(p, "word " * 500, pages=10)
        feats = extract(_pdf_rec(p), text_cap_bytes=100)
        assert feats.modality is Modality.TEXT
        assert len(feats.text) <= 100

    def test_scanned_pdf_no_text_layer_needs_review(self, tmp_path):
        # A page with no text simulates a scan without OCR layer.
        p = tmp_path / "scan.pdf"
        doc = fitz.open(); doc.new_page(); doc.save(p); doc.close()
        feats = extract(_pdf_rec(p))
        assert feats.modality is Modality.NONE
        assert feats.error == "pdf_no_text_layer"

    def test_corrupt_pdf_needs_review_not_crash(self, tmp_path):
        p = tmp_path / "broken.pdf"
        p.write_bytes(b"%PDF-1.7 garbage \x00\x01\x02 not really a pdf")
        feats = extract(_pdf_rec(p))
        assert feats.modality is Modality.NONE
        assert feats.error is not None  # pdf_unreadable:* or no_text_layer


class TestPdfOcr:
    """OCR fallback for scanned PDFs (optional [ocr] extra + tesseract binary)."""

    def _tesseract_available(self) -> bool:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def test_scanned_pdf_with_rendered_text_is_ocrd(self, tmp_path):
        if not self._tesseract_available():
            pytest.skip("tesseract not installed")
        # Build a "scan": render a text page to an image, embed image in a
        # fresh PDF so there is no text layer — only pixels.
        src = fitz.open(); page = src.new_page()
        page.insert_text((72, 144), "INVOICE TOTAL AMOUNT DUE 4500 DOLLARS",
                         fontsize=24)
        pix = page.get_pixmap(dpi=200)
        src.close()
        scan = fitz.open(); spage = scan.new_page()
        spage.insert_image(spage.rect, pixmap=pix)
        p = tmp_path / "scan.pdf"
        scan.save(p); scan.close()

        feats = extract(_pdf_rec(p))
        assert feats.modality is Modality.TEXT
        assert feats.metadata.get("source") == "ocr"
        assert "INVOICE" in feats.text.upper()

    def test_blank_scan_still_needs_review(self, tmp_path):
        # Blank page has nothing to OCR -> needs_review regardless of tesseract.
        doc = fitz.open(); doc.new_page(); p = tmp_path / "blank.pdf"
        doc.save(p); doc.close()
        feats = extract(_pdf_rec(p))
        assert feats.modality is Modality.NONE
        assert feats.error == "pdf_no_text_layer"
