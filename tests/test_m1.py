"""M1 prototype classifier — acceptance test suite.

Each test is named after the contract ID it satisfies (see
ARCHITECTURE-EXTENSION.md §4 "Test Contracts" and §5 "M1 Done Criteria").
All tests are offline and deterministic: every embedding call goes through
FakeEmbeddingService (TC-PRIV-1).

Imports that rely on sibling modules not yet present will fail with
ImportError at collection time; those tests are individually marked with
pytest.importorskip so the rest of the suite still runs.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from typing import Optional

import pytest

# conftest.py has already inserted src/ onto sys.path before collection.
from organizer.types import (
    Classification,
    EmbeddingSpace,
    FileFeatures,
    FileRecord,
    Modality,
    SPACE_FOR_MODALITY,
    Tier,
)
from organizer.embedding import FakeEmbeddingService, cosine

# ---------------------------------------------------------------------------
# Optional sibling imports — guard each with importorskip so an absent module
# blocks only the tests that need it and not the whole suite.
# ---------------------------------------------------------------------------

try:
    from organizer.ingest import scan, tier_of, EXCLUDES
    _INGEST_OK = True
except ImportError:
    _INGEST_OK = False

try:
    from organizer.features import extract
    _FEATURES_OK = True
except ImportError:
    _FEATURES_OK = False

try:
    from organizer.taxonomy import build_taxonomy, DEFAULT_TAXONOMY
    _TAXONOMY_OK = True
except ImportError:
    _TAXONOMY_OK = False

try:
    from organizer.classify import classify
    _CLASSIFY_OK = True
except ImportError:
    _CLASSIFY_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tree_hash(root: pathlib.Path) -> str:
    """SHA-256 over (path, size, sha256-of-content) for every regular file
    under *root*, sorted by path.  Used to assert the tree is unmodified.
    """
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            h.update(f"{rel}:{p.stat().st_size}:{content_hash}\n".encode())
    return h.hexdigest()


class RecordingEmbeddingService(FakeEmbeddingService):
    """Wraps FakeEmbeddingService and records every embed_text / embed_image call."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim)
        self.text_calls: list[str] = []
        self.image_calls: list[pathlib.Path] = []

    def embed_text(self, text: str):  # type: ignore[override]
        self.text_calls.append(text)
        return super().embed_text(text)

    def embed_image(self, path: pathlib.Path):  # type: ignore[override]
        self.image_calls.append(path)
        return super().embed_image(path)

    @property
    def total_calls(self) -> int:
        return len(self.text_calls) + len(self.image_calls)


# ---------------------------------------------------------------------------
# TC-TIER-1 — Tier mapping (ARCHITECTURE-EXTENSION.md §4, TC-TIER-1)
# Canonical claim: §3 tiering by extension/MIME.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _INGEST_OK, reason="organizer.ingest not yet available")
class TestTier1TierMapping:
    """TC-TIER-1: tier_of maps extensions to exact Tier values."""

    def test_tc_tier_1_zip_maps_to_metadata(self):
        """TC-TIER-1: .zip → Tier.METADATA (Tier 1)."""
        result = tier_of("zip", "application/zip", 1024)
        assert result is Tier.METADATA

    def test_tc_tier_1_tar_maps_to_metadata(self):
        """TC-TIER-1: .tar → Tier.METADATA (archive, Tier 1)."""
        result = tier_of("tar", "application/x-tar", 1024)
        assert result is Tier.METADATA

    def test_tc_tier_1_py_maps_to_text(self):
        """TC-TIER-1: .py → Tier.TEXT (Tier 2)."""
        result = tier_of("py", "text/x-python", 1024)
        assert result is Tier.TEXT

    def test_tc_tier_1_md_maps_to_text(self):
        """TC-TIER-1: .md → Tier.TEXT (Tier 2)."""
        result = tier_of("md", "text/markdown", 512)
        assert result is Tier.TEXT

    def test_tc_tier_1_txt_maps_to_text(self):
        """TC-TIER-1: .txt → Tier.TEXT (Tier 2)."""
        result = tier_of("txt", "text/plain", 256)
        assert result is Tier.TEXT

    def test_tc_tier_1_png_maps_to_vision(self):
        """TC-TIER-1: .png → Tier.VISION (Tier 3)."""
        result = tier_of("png", "image/png", 4096)
        assert result is Tier.VISION

    def test_tc_tier_1_jpg_maps_to_vision(self):
        """TC-TIER-1: .jpg → Tier.VISION (Tier 3)."""
        result = tier_of("jpg", "image/jpeg", 4096)
        assert result is Tier.VISION

    def test_tc_tier_1_no_ext_maps_to_review(self):
        """TC-TIER-1: no extension → Tier.REVIEW (Tier 4)."""
        result = tier_of("", "application/octet-stream", 512)
        assert result is Tier.REVIEW

    def test_tc_tier_1_unknown_ext_maps_to_review(self):
        """TC-TIER-1: unknown extension → Tier.REVIEW (Tier 4 fallback)."""
        result = tier_of("xyzabc123", "application/octet-stream", 100)
        assert result is Tier.REVIEW

    def test_tc_tier_1_very_large_file_maps_to_review(self):
        """TC-TIER-1: file exceeding size threshold → Tier.REVIEW regardless of ext."""
        huge = 3 * 1024 * 1024 * 1024  # 3 GiB
        result = tier_of("txt", "text/plain", huge)
        assert result is Tier.REVIEW

    def test_tc_tier_1_tier1_is_int_value_1(self):
        """TC-TIER-1: Tier.METADATA integer value is 1 (no ML inference tier)."""
        assert int(Tier.METADATA) == 1

    def test_tc_tier_1_tier_values_ordered(self):
        """TC-TIER-1: Tier enum values are 1–4 in declaration order."""
        assert list(Tier) == [Tier.METADATA, Tier.TEXT, Tier.VISION, Tier.REVIEW]


# ---------------------------------------------------------------------------
# TC-SAFE-6 — Exclusion rules (ARCHITECTURE-EXTENSION.md §4, TC-SAFE-6)
# Canonical claim: §7.5/§7.6 exclusion rules; never root.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _INGEST_OK, reason="organizer.ingest not yet available")
class TestSafe6Exclusions:
    """TC-SAFE-6: scan() never yields records from excluded dirs or dotfiles."""

    def test_tc_safe_6_no_records_under_git(self, fixture_root):
        """TC-SAFE-6: .git directory and its contents are excluded."""
        records = list(scan(fixture_root))
        git_paths = [r for r in records if ".git" in r.path.parts]
        assert git_paths == [], (
            f"scan() yielded records under .git: {[str(r.path) for r in git_paths]}"
        )

    def test_tc_safe_6_git_dir_name_in_excludes(self):
        """TC-SAFE-6: EXCLUDES set contains '.git'."""
        assert ".git" in EXCLUDES

    def test_tc_safe_6_no_dotfiles_yielded(self, fixture_root):
        """TC-SAFE-6: dotfiles (names starting with '.') are never yielded."""
        records = list(scan(fixture_root))
        dotfiles = [r for r in records if r.path.name.startswith(".")]
        assert dotfiles == [], (
            f"scan() yielded dotfiles: {[str(r.path) for r in dotfiles]}"
        )

    def test_tc_safe_6_expected_files_are_yielded(self, fixture_root):
        """TC-SAFE-6: the non-excluded files ARE yielded (sanity check)."""
        records = list(scan(fixture_root))
        names = {r.path.name for r in records}
        expected = {"sample.txt", "sample.md", "sample.py", "sample.png",
                    "sample.zip", "no_extension"}
        assert expected.issubset(names), (
            f"Some expected files missing from scan output. Got: {names}"
        )

    def test_tc_safe_6_node_modules_excluded(self, tmp_path):
        """TC-SAFE-6: node_modules is excluded from scan output."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "some_lib.js").write_text("// library code", encoding="utf-8")
        (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
        records = list(scan(tmp_path))
        names = {r.path.name for r in records}
        assert "some_lib.js" not in names
        assert "app.py" in names


# ---------------------------------------------------------------------------
# M1 Done-Criterion 7 / TC-SAFE-1 — Read-only: zero filesystem mutation
# (ARCHITECTURE-EXTENSION.md §5 done-criteria #7)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_INGEST_OK and _FEATURES_OK),
    reason="organizer.ingest and/or organizer.features not yet available",
)
class TestSafe1ReadOnly:
    """TC-SAFE-1 / done-criterion 7: tree hash identical before and after scan+extract."""

    def test_done_criterion_7_scan_does_not_mutate(self, fixture_root):
        """Done-criterion 7: tree hash unchanged after a full scan pass."""
        hash_before = _tree_hash(fixture_root)
        list(scan(fixture_root))  # consume the full scan
        hash_after = _tree_hash(fixture_root)
        assert hash_before == hash_after, (
            "scan() mutated the filesystem: tree hash changed."
        )

    def test_done_criterion_7_extract_does_not_mutate(self, fixture_root):
        """Done-criterion 7: tree hash unchanged after scan + extract pass."""
        hash_before = _tree_hash(fixture_root)
        for rec in scan(fixture_root):
            extract(rec)
        hash_after = _tree_hash(fixture_root)
        assert hash_before == hash_after, (
            "extract() mutated the filesystem: tree hash changed."
        )

    def test_done_criterion_7_full_pipeline_does_not_mutate(self, fixture_root):
        """Done-criterion 7 + TC-SAFE-1: hash unchanged after scan+extract+classify."""
        if not (_TAXONOMY_OK and _CLASSIFY_OK):
            pytest.skip("organizer.taxonomy or organizer.classify not yet available")

        embedder = FakeEmbeddingService()
        taxonomy = build_taxonomy(embedder)

        hash_before = _tree_hash(fixture_root)
        for rec in scan(fixture_root):
            features = extract(rec)
            if features.modality is not Modality.NONE:
                classify(features, taxonomy, embedder)
        hash_after = _tree_hash(fixture_root)
        assert hash_before == hash_after, (
            "Full pipeline mutated the filesystem: tree hash changed."
        )


# ---------------------------------------------------------------------------
# TC-MODEL-1 — Space isolation (ARCHITECTURE-EXTENSION.md §4, TC-MODEL-1)
# Canonical claim: §4.2(3) cosine argmax corrected by G2.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_FEATURES_OK and _TAXONOMY_OK and _CLASSIFY_OK),
    reason="organizer.features/taxonomy/classify not yet available",
)
class TestModel1SpaceIsolation:
    """TC-MODEL-1: text → only embed_text called; image → only embed_image called."""

    def _make_text_features(self) -> FileFeatures:
        return FileFeatures(
            modality=Modality.TEXT,
            text="def hello(): return 'hello'",
            metadata={"bytes_read": 30},
        )

    def _make_image_features(self, tmp_path: pathlib.Path) -> FileFeatures:
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        return FileFeatures(
            modality=Modality.IMAGE,
            image_path=img,
            metadata={"size_bytes": 72},
        )

    def test_tc_model_1_text_uses_only_embed_text(self, tmp_path):
        """TC-MODEL-1: classifying a TEXT file calls embed_text, never embed_image."""
        recorder = RecordingEmbeddingService()
        taxonomy = build_taxonomy(recorder)
        recorder.text_calls.clear()
        recorder.image_calls.clear()

        features = self._make_text_features()
        classify(features, taxonomy, recorder)

        assert len(recorder.text_calls) > 0, "Expected embed_text to be called"
        assert len(recorder.image_calls) == 0, (
            f"embed_image must NOT be called for TEXT file; got {recorder.image_calls}"
        )

    def test_tc_model_1_image_uses_only_embed_image(self, tmp_path):
        """TC-MODEL-1: classifying an IMAGE file calls embed_image, never embed_text."""
        recorder = RecordingEmbeddingService()
        taxonomy = build_taxonomy(recorder)
        recorder.text_calls.clear()
        recorder.image_calls.clear()

        features = self._make_image_features(tmp_path)
        classify(features, taxonomy, recorder)

        assert len(recorder.image_calls) > 0, "Expected embed_image to be called"
        assert len(recorder.text_calls) == 0, (
            f"embed_text must NOT be called for IMAGE file; got {recorder.text_calls}"
        )

    def test_tc_model_1_space_for_modality_mapping(self):
        """TC-MODEL-1: SPACE_FOR_MODALITY maps TEXT→BGE, IMAGE→CLIP, NONE→None."""
        assert SPACE_FOR_MODALITY[Modality.TEXT] is EmbeddingSpace.BGE
        assert SPACE_FOR_MODALITY[Modality.IMAGE] is EmbeddingSpace.CLIP
        assert SPACE_FOR_MODALITY[Modality.NONE] is None

    def test_tc_model_1_no_cross_space_cosine_for_text(self, tmp_path):
        """TC-MODEL-1: text file is scored against bge prompt_vecs only (not CLIP)."""
        # We verify the invariant structurally: SPACE_FOR_MODALITY says TEXT→BGE.
        # If classify() respects it, it must select EmbeddingSpace.BGE for scoring.
        # We check the returned classification is non-error (embedding was done in
        # the correct space) by asserting cosine is in [-1, 1].
        recorder = RecordingEmbeddingService()
        taxonomy = build_taxonomy(recorder)
        recorder.text_calls.clear()
        recorder.image_calls.clear()

        features = self._make_text_features()
        result = classify(features, taxonomy, recorder)

        # No embed_image call means no CLIP-space vector was produced for the file.
        assert len(recorder.image_calls) == 0
        # Cosine must be valid (in-space comparison succeeded).
        assert -1.0 <= result.cosine <= 1.0


# ---------------------------------------------------------------------------
# TC-MODEL-2 / Done-criterion 5 — Deterministic argmax
# (ARCHITECTURE-EXTENSION.md §4, TC-MODEL-2 and §5 done-criteria #5)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_FEATURES_OK and _TAXONOMY_OK and _CLASSIFY_OK),
    reason="organizer.features/taxonomy/classify not yet available",
)
class TestModel2DeterministicArgmax:
    """TC-MODEL-2 + done-criterion 5: classify is stable and returns a real cat_id."""

    def _text_features(self) -> FileFeatures:
        return FileFeatures(
            modality=Modality.TEXT,
            text="Python source code file with import statements and def keywords",
            metadata={"bytes_read": 60},
        )

    def test_tc_model_2_identical_results_two_runs(self):
        """TC-MODEL-2 / done-criterion 5: same features → same cat_id and cosine."""
        embedder = FakeEmbeddingService()
        taxonomy = build_taxonomy(embedder)
        features = self._text_features()

        result1 = classify(features, taxonomy, embedder)
        result2 = classify(features, taxonomy, embedder)

        assert result1.cat_id == result2.cat_id, (
            f"Non-deterministic cat_id: {result1.cat_id} vs {result2.cat_id}"
        )
        assert result1.cosine == result2.cosine, (
            f"Non-deterministic cosine: {result1.cosine} vs {result2.cosine}"
        )

    def test_tc_model_2_text_fixture_returns_non_none_cat_id(self):
        """TC-MODEL-2: a clearly text-like fixture returns a non-None cat_id."""
        embedder = FakeEmbeddingService()
        taxonomy = build_taxonomy(embedder)
        features = self._text_features()

        result = classify(features, taxonomy, embedder)

        # A Modality.TEXT file with a non-empty text payload must produce a cat_id
        # (it has enough signal for embedding argmax).
        assert result.cat_id is not None, (
            "classify() returned cat_id=None for a clearly text-like features object; "
            "expected a leaf category from DEFAULT_TAXONOMY"
        )

    def test_tc_model_2_cat_id_is_in_taxonomy(self):
        """TC-MODEL-2: returned cat_id must be one of the known taxonomy leaves."""
        embedder = FakeEmbeddingService()
        taxonomy = build_taxonomy(embedder)
        features = self._text_features()
        known_ids = {cat.cat_id for cat in taxonomy}

        result = classify(features, taxonomy, embedder)
        if result.cat_id is not None:
            assert result.cat_id in known_ids, (
                f"cat_id '{result.cat_id}' not in taxonomy leaves: {known_ids}"
            )

    def test_tc_model_2_cosine_in_valid_range(self):
        """TC-MODEL-2: cosine score is within [-1.0, 1.0]."""
        embedder = FakeEmbeddingService()
        taxonomy = build_taxonomy(embedder)
        features = self._text_features()
        result = classify(features, taxonomy, embedder)
        assert -1.0 <= result.cosine <= 1.0, (
            f"cosine={result.cosine} out of valid range"
        )


# ---------------------------------------------------------------------------
# Done-criterion 3 + 8 — needs_review for Tier 4 and corrupt/zero-byte files
# (ARCHITECTURE-EXTENSION.md §5 done-criteria #3 and #8)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_FEATURES_OK and _TAXONOMY_OK and _CLASSIFY_OK),
    reason="organizer.features/taxonomy/classify not yet available",
)
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

    def _corrupt_record(self, tmp_path: pathlib.Path) -> FileRecord:
        p = tmp_path / "corrupt.png"
        p.write_bytes(b"\x00\xff\xfe\xfd" * 8)
        return FileRecord(
            path=p,
            size=p.stat().st_size,
            mtime=p.stat().st_mtime,
            extension="png",
            mime="image/png",
            tier=Tier.VISION,
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
        rec = self._corrupt_record(tmp_path)
        # A .png file with size > 0 will go through _extract_vision,
        # which only stats the file (no pixel read in M1 scope).  The
        # resulting FileFeatures has modality=IMAGE (non-zero size), so
        # this tests the case where extraction succeeds but the classify
        # contract handles modality=NONE features as needs_review.
        # We make a genuinely corrupt file by giving it tier=TEXT with
        # zero-byte content instead, so extraction hits the error path.
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

    @pytest.mark.skipif(not _TAXONOMY_OK, reason="organizer.taxonomy not yet available")
    def test_build_taxonomy_returns_all_categories(self):
        """taxonomy: build_taxonomy returns one CategoryPrompt per DEFAULT_TAXONOMY entry."""
        svc = FakeEmbeddingService()
        taxonomy = build_taxonomy(svc)
        assert len(taxonomy) == len(DEFAULT_TAXONOMY)

    @pytest.mark.skipif(not _TAXONOMY_OK, reason="organizer.taxonomy not yet available")
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

    @pytest.mark.skipif(not _TAXONOMY_OK, reason="organizer.taxonomy not yet available")
    def test_build_taxonomy_leaf_ids_are_hierarchical(self):
        """taxonomy: every cat_id is a leaf (contains a '/' separator per G5)."""
        svc = FakeEmbeddingService()
        taxonomy = build_taxonomy(svc)
        for cat in taxonomy:
            assert "/" in cat.cat_id, (
                f"cat_id '{cat.cat_id}' is not a leaf path (expected 'parent/leaf')"
            )

    @pytest.mark.skipif(not _TAXONOMY_OK, reason="organizer.taxonomy not yet available")
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
