"""M1 acceptance suite (part 2) — needs_review routing + structural contracts.

Split from test_m1.py to honour the 500-line file guideline. Covers
ARCHITECTURE-EXTENSION.md §5 done-criteria 3 + 8 and the structural type /
FakeEmbeddingService / taxonomy contracts. All offline (TC-PRIV-1).
"""

from __future__ import annotations

import pathlib

import pytest

from organizer.types import (
    Classification,
    EmbeddingSpace,
    FileRecord,
    Modality,
    SPACE_FOR_MODALITY,
    Tier,
)
from organizer.embedding import FakeEmbeddingService, cosine
from organizer.features import extract
from organizer.taxonomy import build_taxonomy, DEFAULT_TAXONOMY
from organizer.classify import classify

from conftest import RecordingEmbeddingService


# ---------------------------------------------------------------------------
# Done-criterion 3 + 8 — needs_review for Tier 4 and corrupt/zero-byte files
# (ARCHITECTURE-EXTENSION.md §5 done-criteria #3 and #8)
# ---------------------------------------------------------------------------

class TestNeedsReviewAndNoEmbedding:
    """Done-criteria 3 + 8: Tier-4, zero-byte, and corrupt files → needs_review + no embedding."""

    def _no_ext_record(self, tmp_path: pathlib.Path) -> FileRecord:
        p = tmp_path / "no_extension"
        p.write_text("some content", encoding="utf-8")
        return FileRecord(
            path=p,
            size=p.stat().st_size,
            mtime=p.stat().st_mtime,
            extension="",
            mime="",
            tier=Tier.REVIEW,
            status="needs_review",
        )

    def _zero_byte_record(self, tmp_path: pathlib.Path) -> FileRecord:
        p = tmp_path / "empty.txt"
        p.touch()
        return FileRecord(
            path=p,
            size=0,
            mtime=p.stat().st_mtime,
            extension="txt",
            mime="text/plain",
            tier=Tier.TEXT,
            status="pending",
        )

    def _classify_via_pipeline(
        self, rec: FileRecord, recorder: RecordingEmbeddingService
    ) -> Classification:
        """Run extract → classify with the given recorder."""
        taxonomy = build_taxonomy(FakeEmbeddingService())  # pre-built, not recorded
        features = extract(rec)
        return classify(features, taxonomy, recorder)

    def test_done_criterion_3_no_ext_classifies_needs_review(self, tmp_path):
        """Done-criterion 3: no-extension file → source='needs_review', cat_id=None."""
        recorder = RecordingEmbeddingService()
        rec = self._no_ext_record(tmp_path)
        result = self._classify_via_pipeline(rec, recorder)

        assert result.cat_id is None, (
            f"No-extension file must have cat_id=None; got {result.cat_id}"
        )
        assert result.source == "needs_review", (
            f"source must be 'needs_review'; got {result.source!r}"
        )

    def test_done_criterion_3_no_ext_not_embedded(self, tmp_path):
        """Done-criterion 3: no-extension file triggers zero embed calls."""
        recorder = RecordingEmbeddingService()
        rec = self._no_ext_record(tmp_path)
        self._classify_via_pipeline(rec, recorder)

        assert recorder.total_calls == 0, (
            f"Tier-4 file must not be embedded; got {recorder.total_calls} embed calls"
        )

    def test_done_criterion_8_zero_byte_classifies_needs_review(self, tmp_path):
        """Done-criterion 8: zero-byte file → source='needs_review', cat_id=None."""
        recorder = RecordingEmbeddingService()
        rec = self._zero_byte_record(tmp_path)
        result = self._classify_via_pipeline(rec, recorder)

        assert result.cat_id is None, (
            f"Zero-byte file must have cat_id=None; got {result.cat_id}"
        )
        assert result.source == "needs_review", (
            f"source must be 'needs_review'; got {result.source!r}"
        )

    def test_done_criterion_8_zero_byte_not_embedded(self, tmp_path):
        """Done-criterion 8: zero-byte file triggers zero embed calls."""
        recorder = RecordingEmbeddingService()
        rec = self._zero_byte_record(tmp_path)
        self._classify_via_pipeline(rec, recorder)

        assert recorder.total_calls == 0, (
            f"Zero-byte file must not be embedded; got {recorder.total_calls} embed calls"
        )

    def test_done_criterion_8_corrupt_file_classifies_needs_review(self, tmp_path):
        """Done-criterion 8: corrupt/unreadable file → source='needs_review', cat_id=None."""
        recorder = RecordingEmbeddingService()
        # A genuinely corrupt file: tier=TEXT with zero-byte content, so
        # extraction hits the error path.
        p = tmp_path / "corrupt_text.txt"
        p.write_bytes(b"")  # zero-byte txt
        corrupt_rec = FileRecord(
            path=p,
            size=0,
            mtime=p.stat().st_mtime,
            extension="txt",
            mime="text/plain",
            tier=Tier.TEXT,
            status="pending",
        )
        result = self._classify_via_pipeline(corrupt_rec, recorder)
        assert result.cat_id is None
        assert result.source == "needs_review"
        assert recorder.total_calls == 0

    def test_done_criterion_8_no_crash_on_tier4(self, tmp_path):
        """Done-criterion 8: classifying a Tier-4 file never raises an exception."""
        recorder = RecordingEmbeddingService()
        rec = self._no_ext_record(tmp_path)
        # Should not raise
        try:
            self._classify_via_pipeline(rec, recorder)
        except Exception as exc:
            pytest.fail(f"classify raised an unexpected exception: {exc}")


# ---------------------------------------------------------------------------
# Structural / type-contract sanity checks (always run; no sibling deps)
# ---------------------------------------------------------------------------

class TestTypesContract:
    """Verify the stable type contracts in organizer.types are consistent."""

    def test_tier_enum_has_four_values(self):
        """types.py: Tier enum has exactly four members."""
        assert len(list(Tier)) == 4

    def test_modality_enum_members(self):
        """types.py: Modality has TEXT, IMAGE, NONE."""
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.NONE.value == "none"

    def test_embedding_space_values(self):
        """types.py: EmbeddingSpace has BGE and CLIP."""
        assert EmbeddingSpace.BGE.value == "bge"
        assert EmbeddingSpace.CLIP.value == "clip"

    def test_space_for_modality_covers_all_modalities(self):
        """types.py: SPACE_FOR_MODALITY has an entry for every Modality."""
        for m in Modality:
            assert m in SPACE_FOR_MODALITY, f"Modality.{m.name} missing from SPACE_FOR_MODALITY"

    def test_file_record_defaults(self, tmp_path):
        """types.py: FileRecord tier defaults to None, status defaults to 'pending'."""
        p = tmp_path / "f.txt"
        p.touch()
        rec = FileRecord(
            path=p, size=0, mtime=0.0, extension="txt", mime="text/plain"
        )
        assert rec.tier is None
        assert rec.status == "pending"

    def test_classification_fields(self):
        """types.py: Classification has cat_id, cosine, source."""
        c = Classification(cat_id="code/python", cosine=0.85, source="embedding")
        assert c.cat_id == "code/python"
        assert c.cosine == 0.85
        assert c.source == "embedding"

    def test_classification_none_cat_id(self):
        """types.py: Classification accepts cat_id=None (needs_review path)."""
        c = Classification(cat_id=None, cosine=0.0, source="needs_review")
        assert c.cat_id is None


class TestFakeEmbeddingService:
    """Verify FakeEmbeddingService properties relied upon by all other tests."""

    def test_embed_text_returns_correct_length(self):
        """FakeEmbeddingService: embed_text returns a Vec of length dim."""
        svc = FakeEmbeddingService(dim=64)
        v = svc.embed_text("hello world")
        assert len(v) == 64

    def test_embed_image_returns_correct_length(self, tmp_path):
        """FakeEmbeddingService: embed_image returns a Vec of length dim."""
        p = tmp_path / "img.png"
        p.write_bytes(b"\x00" * 16)
        svc = FakeEmbeddingService(dim=64)
        v = svc.embed_image(p)
        assert len(v) == 64

    def test_embed_text_is_deterministic(self):
        """FakeEmbeddingService: same text → same vector (TC-PRIV-1, done-criterion 5)."""
        svc = FakeEmbeddingService()
        v1 = svc.embed_text("hello world")
        v2 = svc.embed_text("hello world")
        assert v1 == v2

    def test_embed_image_is_deterministic(self, tmp_path):
        """FakeEmbeddingService: same path stem → same vector."""
        p = tmp_path / "photo.png"
        p.write_bytes(b"\x00" * 16)
        svc = FakeEmbeddingService()
        v1 = svc.embed_image(p)
        v2 = svc.embed_image(p)
        assert v1 == v2

    def test_embed_prompt_bge_differs_from_clip(self):
        """FakeEmbeddingService: same text embedded in BGE vs CLIP gives different vecs."""
        svc = FakeEmbeddingService()
        vbge = svc.embed_prompt("invoice", EmbeddingSpace.BGE)
        vclip = svc.embed_prompt("invoice", EmbeddingSpace.CLIP)
        # Seeds differ by space prefix so vectors must be different.
        assert vbge != vclip

    def test_cosine_self_similarity_is_one(self):
        """embedding.cosine: v · v / (|v| |v|) == 1.0."""
        svc = FakeEmbeddingService()
        v = svc.embed_text("some text")
        sim = cosine(v, v)
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_orthogonal_near_zero(self):
        """embedding.cosine: orthogonal vectors have cosine ≈ 0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine(a, b)) < 1e-9

    def test_ensure_models_is_noop(self):
        """FakeEmbeddingService: ensure_models() accepts any consent value silently."""
        svc = FakeEmbeddingService()
        svc.ensure_models(consent=False)
        svc.ensure_models(consent=True)


class TestTaxonomyContract:
    """Verify build_taxonomy structure without needing organizer.classify."""

    def test_build_taxonomy_returns_all_categories(self):
        """taxonomy: build_taxonomy returns one CategoryPrompt per DEFAULT_TAXONOMY entry."""
        svc = FakeEmbeddingService()
        taxonomy = build_taxonomy(svc)
        assert len(taxonomy) == len(DEFAULT_TAXONOMY)

    def test_build_taxonomy_populates_both_spaces(self):
        """taxonomy: each CategoryPrompt has prompt_vecs for BGE and CLIP spaces."""
        svc = FakeEmbeddingService()
        taxonomy = build_taxonomy(svc)
        for cat in taxonomy:
            assert EmbeddingSpace.BGE in cat.prompt_vecs, (
                f"{cat.cat_id} missing BGE prompt_vecs"
            )
            assert EmbeddingSpace.CLIP in cat.prompt_vecs, (
                f"{cat.cat_id} missing CLIP prompt_vecs"
            )

    def test_build_taxonomy_leaf_ids_are_hierarchical(self):
        """taxonomy: every cat_id is a leaf (contains a '/' separator per G5)."""
        svc = FakeEmbeddingService()
        taxonomy = build_taxonomy(svc)
        for cat in taxonomy:
            assert "/" in cat.cat_id, (
                f"cat_id '{cat.cat_id}' is not a leaf path (expected 'parent/leaf')"
            )

    def test_build_taxonomy_prompt_vecs_have_correct_dim(self):
        """taxonomy: all embedded prompt vectors have the expected dimension."""
        dim = 64
        svc = FakeEmbeddingService(dim=dim)
        taxonomy = build_taxonomy(svc)
        for cat in taxonomy:
            for space, vecs in cat.prompt_vecs.items():
                for v in vecs:
                    assert len(v) == dim, (
                        f"{cat.cat_id}/{space}: expected dim={dim}, got {len(v)}"
                    )
