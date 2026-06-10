"""Confidence calibration — resolves gap G4 (ARCHITECTURE-EXTENSION.md §1).

Raw cosine is not a calibrated probability and is NOT comparable across the two
embedding spaces (G2/G4): bge text↔text cosines sit high (~0.5-0.7) while CLIP
image↔text cosines sit low (~0.2-0.3) because of the cross-modal gap. A single
`min_confidence` threshold on raw cosine is therefore meaningless across spaces.

Fix: map the per-category cosine scores to a probability with a temperature
softmax computed *within one modality space*. `confidence` = probability mass on
the argmax category. Temperature is per-space (CLIP's compressed cosine range
needs a smaller temperature to separate). The result is in [0,1] and a single
threshold means roughly the same thing in both spaces (TC-CONF-1).

This makes the score depend on the *separation* between the top category and the
rest, not the absolute cosine — so a low, ambiguous match (top ≈ runner-up) gets
a low confidence and is routed to needs_review even if its raw cosine is similar
to a confident match's.
"""

from __future__ import annotations

import math

from organizer.types import EmbeddingSpace

# Per-space softmax temperature. CLIP cosines occupy a narrow band, so a smaller
# temperature is needed to turn small cosine gaps into meaningful probability
# separation. bge cosines are already well spread.
# NOTE: these defaults were tuned on a 14-file public-sample eval (real photos,
# screenshots, source code, prose) against the 13-category config/categories.yaml
# taxonomy — illustrative, NOT production-calibrated. A real calibration set
# (§12 benchmarking) should refit them. Confidence spreads thinner as the
# category count grows, so retune whenever the taxonomy changes materially.
DEFAULT_TEMPERATURE: dict[EmbeddingSpace, float] = {
    EmbeddingSpace.BGE: 0.015,
    EmbeddingSpace.CLIP: 0.015,
}

# Per-space auto-move gate. Below this calibrated confidence -> needs_review.
# Conservative by design (safety > recall, §7): on the 13-cat eval this gave
# bge 5 correct / 0 wrong / 0 review and clip 2 correct / 0 wrong / 6 review —
# zero confidently-wrong auto-moves in both spaces.
DEFAULT_MIN_CONFIDENCE: dict[EmbeddingSpace, float] = {
    EmbeddingSpace.BGE: 0.40,
    EmbeddingSpace.CLIP: 0.60,
}


def softmax_confidence(
    scores: list[float],
    space: EmbeddingSpace,
    temperature: dict[EmbeddingSpace, float] | None = None,
) -> float:
    """Probability mass on the top category after a per-space temperature softmax.

    Args:
        scores: per-category cosine scores (one per candidate category), all in
                the SAME embedding space (G2 — never mix spaces).
        space:  the embedding space these scores came from (selects temperature).
        temperature: optional override of DEFAULT_TEMPERATURE.

    Returns:
        Confidence in [0,1]: the softmax probability of the argmax category.
        Returns 0.0 for an empty score list.
    """
    if not scores:
        return 0.0
    temps = temperature or DEFAULT_TEMPERATURE
    t = temps.get(space, 0.1)
    if t <= 0:
        raise ValueError(f"temperature must be > 0, got {t}")

    # Numerically stable softmax.
    top = max(scores)
    exps = [math.exp((s - top) / t) for s in scores]
    total = sum(exps)
    if total == 0.0:  # pragma: no cover — defensive
        return 0.0
    # exp((top-top)/t) == 1, so the argmax's probability is 1/total.
    return 1.0 / total


def min_confidence_for(
    space: EmbeddingSpace,
    overrides: dict[EmbeddingSpace, float] | None = None,
) -> float:
    """Per-space auto-move threshold (DEFAULT_MIN_CONFIDENCE, with overrides)."""
    table = overrides or DEFAULT_MIN_CONFIDENCE
    return table.get(space, 0.5)
