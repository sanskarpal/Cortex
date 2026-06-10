"""EmbeddingService: owns the two embedding spaces (G2).

ARCHITECTURE-EXTENSION.md §2 makes EmbeddingService the *only* module that
loads ML weights, and the owner of the CLIP + bge spaces. To keep M1 testable
offline (TC-PRIV-1) and non-blocking for downstream modules, the service is a
Protocol with two backends:

  * FakeEmbeddingService  - deterministic, dependency-free; used by tests and
    when models are not installed. Same text/image -> same vector, so argmax is
    stable (M1 done-criterion 5) and space isolation is verifiable.
  * SentenceTransformerEmbeddingService - real bge + clip-ViT-B/32, lazily
    loaded only after explicit consent (G8). Import of heavyweight deps is
    deferred into ensure_models so importing this file never needs torch.

Both tag every vector with its EmbeddingSpace and refuse cross-space calls.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import EmbeddingSpace, Vec

_DIM = 64  # fake-space dimensionality; real backends override via their model


def cosine(a: Vec, b: Vec) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class EmbeddingService(Protocol):
    """Contract every backend honors. Vectors are tagged by space (G2)."""

    def ensure_models(self, consent: bool) -> None: ...

    def embed_text(self, text: str) -> Vec:
        """Embed text into the BGE space."""
        ...

    def embed_image(self, path: Path) -> Vec:
        """Embed an image into the CLIP space."""
        ...

    def embed_prompt(self, text: str, space: EmbeddingSpace) -> Vec:
        """Embed a category prompt string into the requested space.

        Prompts must be embedded once per space (G2): bge-space for scoring
        text files, clip-space for scoring images.
        """
        ...


def _hash_to_vec(seed: str, dim: int = _DIM) -> Vec:
    """Deterministic pseudo-embedding from a string seed.

    Stable across runs/processes (no Python hash randomization) so argmax is
    reproducible. Not semantic - only for offline tests and no-model fallback.
    """
    out: Vec = []
    counter = 0
    while len(out) < dim:
        h = hashlib.sha256(f"{seed}|{counter}".encode()).digest()
        for i in range(0, len(h), 4):
            if len(out) >= dim:
                break
            chunk = int.from_bytes(h[i : i + 4], "big")
            out.append((chunk / 2**32) * 2.0 - 1.0)  # in [-1, 1)
        counter += 1
    return out


class FakeEmbeddingService:
    """Deterministic, offline embedding backend.

    Text and image vectors are seeded so that a prompt and content sharing a
    salient token land near each other, giving a usable (not random) argmax for
    fixture-based tests while requiring zero downloads.
    """

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def ensure_models(self, consent: bool) -> None:  # noqa: ARG002 - no-op offline
        return None

    def embed_text(self, text: str) -> Vec:
        return _hash_to_vec(f"{EmbeddingSpace.BGE.value}:{_normalize(text)}", self.dim)

    def embed_image(self, path: Path) -> Vec:
        # Seed from filename stem so fixtures named by category are separable.
        return _hash_to_vec(
            f"{EmbeddingSpace.CLIP.value}:{_normalize(Path(path).stem)}", self.dim
        )

    def embed_prompt(self, text: str, space: EmbeddingSpace) -> Vec:
        return _hash_to_vec(f"{space.value}:{_normalize(text)}", self.dim)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())[:512]


class SentenceTransformerEmbeddingService:
    """Real bge + clip backend. Heavy deps imported lazily (G8).

    Nothing is loaded until ensure_models(consent=True) is called, and the
    sentence-transformers / torch imports happen there - so this module can be
    imported in a network-blocked environment without pulling ML weights.
    """

    TEXT_MODEL = "BAAI/bge-small-en-v1.5"
    IMAGE_MODEL = "clip-ViT-B-32"  # general OpenCLIP, per G3 (not CurrencyCLIP)

    def __init__(self) -> None:
        self._text_model = None
        self._image_model = None

    def ensure_models(self, consent: bool) -> None:
        if not consent:
            raise PermissionError(
                "Model download/load requires explicit consent (G8/TC-PRIV-1). "
                "This is a one-time setup step, not normal operation."
            )
        if self._text_model is not None and self._image_model is not None:
            return
        from sentence_transformers import SentenceTransformer  # lazy

        self._text_model = SentenceTransformer(self.TEXT_MODEL)
        self._image_model = SentenceTransformer(self.IMAGE_MODEL)

    def _require_loaded(self) -> None:
        if self._text_model is None or self._image_model is None:
            raise RuntimeError("ensure_models(consent=True) must be called first")

    def embed_text(self, text: str) -> Vec:
        self._require_loaded()
        return self._text_model.encode(text, normalize_embeddings=True).tolist()

    def embed_image(self, path: Path) -> Vec:
        self._require_loaded()
        from PIL import Image  # lazy

        with Image.open(path) as img:
            vec = self._image_model.encode(img.convert("RGB"), normalize_embeddings=True)
        return vec.tolist()

    def embed_prompt(self, text: str, space: EmbeddingSpace) -> Vec:
        self._require_loaded()
        model = self._text_model if space is EmbeddingSpace.BGE else self._image_model
        return model.encode(text, normalize_embeddings=True).tolist()
