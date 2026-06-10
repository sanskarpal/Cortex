"""Tests for confidence calibration (G4) — ARCHITECTURE-EXTENSION.md §1, TC-CONF-1.

These verify the calibration math and the gating behaviour in classify(),
using the offline FakeEmbeddingService only (no models, no network).
"""

from __future__ import annotations

import math

import pytest

from organizer.calibrate import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_TEMPERATURE,
    min_confidence_for,
    softmax_confidence,
)
from organizer.classify import classify
from organizer.embedding import FakeEmbeddingService
from organizer.taxonomy import build_taxonomy
from organizer.types import (
    CategoryPrompt,
    EmbeddingSpace,
    FileFeatures,
    Modality,
)


class TestSoftmaxConfidence:
    def test_empty_scores_is_zero(self):
        assert softmax_confidence([], EmbeddingSpace.BGE) == 0.0

    def test_single_score_is_one(self):
        # Only one category -> all probability mass on it.
        assert softmax_confidence([0.5], EmbeddingSpace.BGE) == pytest.approx(1.0)

    def test_in_unit_interval(self):
        c = softmax_confidence([0.3, 0.2, 0.1], EmbeddingSpace.CLIP)
        assert 0.0 <= c <= 1.0

    def test_clear_winner_high_confidence(self):
        # A large gap -> confidence near 1.
        c = softmax_confidence([0.9, 0.1, 0.1], EmbeddingSpace.BGE)
        assert c > 0.9

    def test_tie_low_confidence(self):
        # A near-tie across k categories -> confidence near 1/k.
        c = softmax_confidence([0.30, 0.30, 0.30], EmbeddingSpace.CLIP)
        assert c == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_monotonic_in_margin(self):
        # Bigger top-vs-rest margin => higher confidence.
        small = softmax_confidence([0.31, 0.30, 0.30], EmbeddingSpace.CLIP)
        big = softmax_confidence([0.50, 0.30, 0.30], EmbeddingSpace.CLIP)
        assert big > small

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax_confidence([0.5, 0.1], EmbeddingSpace.BGE, {EmbeddingSpace.BGE: 0.0})


class TestThresholds:
    def test_min_confidence_defaults_present(self):
        assert EmbeddingSpace.BGE in DEFAULT_MIN_CONFIDENCE
        assert EmbeddingSpace.CLIP in DEFAULT_MIN_CONFIDENCE

    def test_temperatures_positive(self):
        assert all(t > 0 for t in DEFAULT_TEMPERATURE.values())

    def test_min_confidence_for_override(self):
        assert min_confidence_for(EmbeddingSpace.BGE, {EmbeddingSpace.BGE: 0.99}) == 0.99


class TestClassifyGating:
    """Gating must be opt-in and must never emit a gated category."""

    def _text_features(self, text: str) -> FileFeatures:
        return FileFeatures(modality=Modality.TEXT, text=text)

    def test_gate_off_by_default_returns_category(self):
        emb = FakeEmbeddingService()
        tax = build_taxonomy(emb)
        res = classify(self._text_features("hello world"), tax, emb)
        assert res.cat_id is not None
        assert res.source == "embedding"

    def test_confidence_is_populated_even_without_gate(self):
        emb = FakeEmbeddingService()
        tax = build_taxonomy(emb)
        res = classify(self._text_features("hello world"), tax, emb)
        assert 0.0 <= res.confidence <= 1.0

    def test_gate_with_impossible_threshold_forces_review(self):
        # A threshold of 1.0 (impossible to meet across >1 categories) must
        # route every file to needs_review when gating is on.
        emb = FakeEmbeddingService()
        tax = build_taxonomy(emb)
        # Patch the BGE threshold to 1.0 via a custom taxonomy is awkward; instead
        # assert the structural property: with gate on and a near-tie, cat_id None.
        # Build a degenerate taxonomy where all prompts are identical -> tie ->
        # confidence ~ 1/k < threshold -> needs_review.
        ident = [
            CategoryPrompt(cat_id=f"c{i}", match=["same"]) for i in range(4)
        ]
        for cp in ident:
            cp.prompt_vecs[EmbeddingSpace.BGE] = [emb.embed_prompt("same", EmbeddingSpace.BGE)]
        res = classify(self._text_features("anything"), ident, emb, gate=True)
        assert res.source == "needs_review"
        assert res.cat_id is None
        assert res.confidence == pytest.approx(1.0 / 4.0, abs=1e-6)
