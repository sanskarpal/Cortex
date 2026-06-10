"""Embedding-based file classifier for the M1 prototype.

Implements the Classifier responsibility from ARCHITECTURE-EXTENSION.md §2:
combine FileFeatures with a pre-built taxonomy to produce a Classification
by argmax of cosine similarity over the correct embedding space (G2, G5).

Key invariants enforced here:
  - Files with errors or Modality.NONE are routed to needs_review and NEVER
    embedded (§3 Tier 4 / DM-FileFeatures.error -> needs_review).
  - Scoring uses ONLY the prompt_vecs[space] matching the file's modality
    (TC-MODEL-1: no cross-space cosine).
  - argmax is taken over per-category max-prompt cosine (TC-MODEL-2).
  - Confidence is calibrated per-space (G4, calibrate.py). With gate=True a
    sub-threshold confidence routes the file to needs_review (§7.2); gate
    defaults to False so plain argmax behaviour is unchanged.
"""

from __future__ import annotations

from organizer.calibrate import min_confidence_for, softmax_confidence
from organizer.embedding import EmbeddingService, cosine
from organizer.types import (
    CategoryPrompt,
    Classification,
    EmbeddingSpace,
    FileFeatures,
    Modality,
    SPACE_FOR_MODALITY,
)

# Sentinel returned for files that cannot be embedded (§3 Tier 4, error state).
_NEEDS_REVIEW = Classification(cat_id=None, cosine=0.0, source="needs_review")


def classify(
    features: FileFeatures,
    taxonomy: list[CategoryPrompt],
    embedder: EmbeddingService,
    gate: bool = False,
) -> Classification:
    """Classify one file against the pre-built taxonomy.

    Algorithm (ARCHITECTURE.md §4.2, corrected by G2):
      1. If the file has an extraction error or Modality.NONE, return
         needs_review immediately without any embedding call.
      2. Determine the embedding space for this modality via SPACE_FOR_MODALITY.
      3. Embed the file content into that space:
           Modality.TEXT  -> embedder.embed_text(features.text or "")  -> BGE vec
           Modality.IMAGE -> embedder.embed_image(features.image_path)  -> CLIP vec
      4. For each CategoryPrompt, score ONLY against prompt_vecs[space]
         (TC-MODEL-1: never compare across spaces).
         Per-category score = max cosine over that category's prompt vectors.
      5. Return the argmax category (TC-MODEL-2) with its raw cosine.
         No confidence calibration — G4 is deferred to M2+ (§5 "Scope out").

    Args:
        features:  Transient features for the file being classified.
        taxonomy:  List of CategoryPrompt built by taxonomy.build_taxonomy(),
                   with prompt_vecs already populated for both spaces.
        embedder:  An EmbeddingService with models already loaded.

    Returns:
        Classification — cat_id is None iff the file is routed to needs_review.
    """
    # ------------------------------------------------------------------ #
    # Step 1: needs_review fast-path — do NOT embed errors or Modality.NONE
    # ------------------------------------------------------------------ #
    if features.error is not None or features.modality is Modality.NONE:
        return _NEEDS_REVIEW

    # ------------------------------------------------------------------ #
    # Step 2: resolve embedding space for this modality (single source of
    # truth: SPACE_FOR_MODALITY in types.py — TC-MODEL-1).
    # ------------------------------------------------------------------ #
    space: EmbeddingSpace | None = SPACE_FOR_MODALITY[features.modality]

    # SPACE_FOR_MODALITY maps Modality.NONE to None; we already handled that
    # branch above, so space is always a concrete EmbeddingSpace here.
    assert space is not None, (  # pragma: no cover — defensive only
        f"Unexpected None space for modality {features.modality!r}; "
        "Modality.NONE should have been caught above."
    )

    # ------------------------------------------------------------------ #
    # Step 3: embed the file content into the correct space only.
    # ------------------------------------------------------------------ #
    if features.modality is Modality.TEXT:
        # BGE space — embed the extracted text snippet (or empty string if
        # extraction yielded nothing; cosine against prompts will be low but
        # deterministic rather than crashing).
        file_vec = embedder.embed_text(features.text or "")
    else:
        # Modality.IMAGE -> CLIP space (G3: clip-ViT-B-32, not CurrencyCLIP).
        file_vec = embedder.embed_image(features.image_path)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Step 4: score file_vec against ONLY this space's prompt vectors.
    # Per-category score = max cosine over its prompts (§4.2, TC-MODEL-1).
    # ------------------------------------------------------------------ #
    best_cat_id: str | None = None
    best_cosine: float = -1.0  # cosine ∈ [-1, 1]; initialise below min
    scores: list[float] = []   # all per-category scores, for calibration (G4)

    for category in taxonomy:
        space_vecs = category.prompt_vecs.get(space)
        if not space_vecs:
            # Prompt vectors not available for this space — skip rather than
            # risk an empty-max or cross-space comparison.
            continue

        # Category score = max cosine across the category's prompt vectors.
        cat_score = max(cosine(file_vec, pvec) for pvec in space_vecs)
        scores.append(cat_score)

        if cat_score > best_cosine:
            best_cosine = cat_score
            best_cat_id = category.cat_id

    # ------------------------------------------------------------------ #
    # Step 5: calibrate confidence (G4) and return.
    # Confidence = per-space temperature-softmax probability on the argmax,
    # comparable across the bge/clip spaces (calibrate.py). When `gate` is on,
    # a sub-threshold confidence routes the file to needs_review (§7.2) instead
    # of emitting a confidently-wrong category.
    # ------------------------------------------------------------------ #
    if best_cat_id is None:
        # Taxonomy was empty or had no vectors for this space.
        return _NEEDS_REVIEW

    confidence = softmax_confidence(scores, space)

    if gate and confidence < min_confidence_for(space):
        return Classification(
            cat_id=None,
            cosine=best_cosine,
            source="needs_review",
            confidence=confidence,
        )

    return Classification(
        cat_id=best_cat_id,
        cosine=best_cosine,
        source="embedding",
        confidence=confidence,
    )
