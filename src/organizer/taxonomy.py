"""Hardcoded leaf taxonomy for the M1 prototype classifier.

Implements the TaxonomyStore responsibility from ARCHITECTURE-EXTENSION.md §2:
load a set of leaf categories (G5: leaves only), embed each category's prompts
in BOTH embedding spaces once (G2), and cache the vectors on CategoryPrompt.

M1 scope: taxonomy is hardcoded here rather than loaded from categories.yaml
(yaml config is deferred to M2, per §5/ARCHITECTURE-EXTENSION.md §5 "Scope out").
"""

from __future__ import annotations

from organizer.embedding import EmbeddingService
from organizer.types import CategoryPrompt, EmbeddingSpace


# ---------------------------------------------------------------------------
# DEFAULT_TAXONOMY — hardcoded leaf categories (G5: no parent nodes here).
# Covers the six representative leaves called out in the task brief, drawn
# from the §5 taxonomy tree.  2–4 natural-language prompt strings per entry
# gives the argmax room to score robustly across varied file content (§4.2).
# ---------------------------------------------------------------------------
DEFAULT_TAXONOMY: list[dict] = [
    {
        "cat_id": "documents/invoices",
        "match": [
            "invoice for services rendered",
            "billing statement with total amount due",
            "receipt from a vendor or supplier",
            "purchase order or payment request",
        ],
    },
    {
        "cat_id": "documents/personal",
        "match": [
            "personal letter or note written to someone",
            "private journal entry or diary",
            "personal identification document such as a passport or ID card",
            "personal correspondence or private message",
        ],
    },
    {
        "cat_id": "code/python",
        "match": [
            "Python source code file with functions and classes",
            "Python script with import statements",
            "Jupyter notebook or Python module",
            "Python program with def and class keywords",
        ],
    },
    {
        "cat_id": "photos/screenshots",
        "match": [
            "screenshot of a computer screen or application window",
            "screen capture showing a user interface",
            "desktop screenshot with icons and taskbar",
            "screenshot of a website or software",
        ],
    },
    {
        "cat_id": "photos/personal",
        "match": [
            "personal photograph of a person or family",
            "photo taken at a social gathering or event",
            "portrait or selfie photo",
            "candid personal photo",
        ],
    },
    {
        "cat_id": "downloads/archives",
        "match": [
            "compressed archive file such as zip or tar",
            "downloaded archive containing multiple files",
            "software package or installer archive",
        ],
    },
]


def build_taxonomy(
    embedder: EmbeddingService,
    spaces: tuple[EmbeddingSpace, ...] = (EmbeddingSpace.BGE, EmbeddingSpace.CLIP),
) -> list[CategoryPrompt]:
    """Build and return the taxonomy with prompt vectors cached in all spaces.

    For each leaf entry in DEFAULT_TAXONOMY, constructs a CategoryPrompt and
    populates prompt_vecs[space] for every requested space by calling
    embedder.embed_prompt(prompt, space) once per (prompt, space) pair (G2).

    The returned list is the live Taxonomy consumed by classify(). Vectors are
    cached on the CategoryPrompt objects so each prompt is embedded at most
    once per build_taxonomy call — callers should hold the list and not rebuild
    on every file (§4.2 "embed each prompt once … and cache them").

    Args:
        embedder: An EmbeddingService whose ensure_models() has already been
                  called before this function is invoked.
        spaces:   Which embedding spaces to pre-populate; defaults to both
                  (BGE for text files, CLIP for image files) per G2.

    Returns:
        list[CategoryPrompt] — one entry per DEFAULT_TAXONOMY row, with
        prompt_vecs filled for each requested space.
    """
    taxonomy: list[CategoryPrompt] = []

    for entry in DEFAULT_TAXONOMY:
        cat_id: str = entry["cat_id"]
        match: list[str] = entry["match"]

        category = CategoryPrompt(cat_id=cat_id, match=match)

        # Embed each prompt string in every requested space and cache the
        # resulting vectors.  This satisfies G2: each category is represented
        # in both BGE space (for scoring text-modality files) and CLIP space
        # (for scoring image-modality files) so no cross-space comparison
        # can occur at classification time.
        for space in spaces:
            vecs = [embedder.embed_prompt(prompt, space) for prompt in match]
            category.prompt_vecs[space] = vecs

        taxonomy.append(category)

    return taxonomy
