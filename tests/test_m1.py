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


# Shared recorder lives in conftest.py (also used by test_m1_contracts.py).
from conftest import RecordingEmbeddingService  # noqa: E402


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
