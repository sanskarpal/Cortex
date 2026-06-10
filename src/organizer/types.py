"""Shared data contracts for the M1 prototype classifier.

These types are the *coordination contract* between modules. Every module in
the M1 pipeline communicates only through the structures defined here (see
ARCHITECTURE-EXTENSION.md §2 "Interface invariants" and §3 "Data Models").

M1 scope only: no DB, no watcher, no moves. Fields not needed for the throwaway
prototype script are intentionally omitted from the durable models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# Vec is a plain list of floats tagged by the space that produced it (G2).
Vec = list[float]


class Tier(int, Enum):
    """Processing tier from ARCHITECTURE.md §3."""

    METADATA = 1  # archives, binaries, disk images, fonts, system files
    TEXT = 2      # text / markdown / source / csv / json / yaml / text-PDF
    VISION = 3    # images, screenshots, scanned docs
    REVIEW = 4    # no-extension, unknown MIME, encrypted, very large -> needs_review


class Modality(str, Enum):
    """Which embedding space a file's content belongs to (G2)."""

    TEXT = "text"
    IMAGE = "image"
    NONE = "none"  # Tier 1 / Tier 4: no embedding


class EmbeddingSpace(str, Enum):
    """The two non-comparable embedding spaces (G2)."""

    BGE = "bge"    # text space  (bge-small-en-v1.5)
    CLIP = "clip"  # image space (clip-ViT-B-32)


# Modality -> the embedding space its vectors live in. The single source of
# truth that guarantees no cross-space cosine is ever computed (TC-MODEL-1).
SPACE_FOR_MODALITY: dict[Modality, Optional[EmbeddingSpace]] = {
    Modality.TEXT: EmbeddingSpace.BGE,
    Modality.IMAGE: EmbeddingSpace.CLIP,
    Modality.NONE: None,
}


@dataclass
class FileRecord:
    """Per-file row produced by the Scanner (DM-FileRecord, M1 subset).

    M1 is stat-only at ingestion; content fields are filled later by the
    FeatureExtractor / EmbeddingService.
    """

    path: Path
    size: int
    mtime: float
    extension: str          # lowercased, no leading dot
    mime: str
    tier: Optional[Tier] = None
    status: str = "pending"  # pending | classified | needs_review


@dataclass
class FileFeatures:
    """Transient features extracted per classify pass (DM-FileFeatures)."""

    modality: Modality
    metadata: dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None          # extracted/OCR'd text, capped
    image_path: Optional[Path] = None   # path fed to the CLIP encoder
    error: Optional[str] = None         # set when extraction failed -> needs_review


@dataclass
class CategoryPrompt:
    """A leaf taxonomy category with prompts embedded in BOTH spaces (G2, G5).

    `cat_id` is a leaf id such as "documents/invoices". Parent path segments are
    pure prefixes derived from `cat_id`; classification targets leaves only.
    """

    cat_id: str
    match: list[str]                                  # prompt strings (§5)
    prompt_vecs: dict[EmbeddingSpace, list[Vec]] = field(default_factory=dict)


@dataclass
class Classification:
    """Result of classifying one file (DM-Classification)."""

    cat_id: Optional[str]   # argmax leaf, or None -> needs_review
    cosine: float           # raw top-1 cosine similarity within its own space
    source: str             # 'rule' | 'embedding' | 'needs_review'
    confidence: float = 0.0  # calibrated f(cosine) in [0,1] (G4); 0.0 if uncalibrated


class MoveMode(str, Enum):
    """Reorganization semantics (ARCHITECTURE.md §7.3, resolved by G1).

    Default is TRASH: the original is recoverable from the system trash. A true
    unrecoverable move requires explicit HARD (G1).
    """

    TRASH = "trash"      # default — recoverable (G1)
    COPY = "copy"        # leave original, organize a copy
    SYMLINK = "symlink"  # organized symlink, original untouched
    HARD = "hard"        # unrecoverable move; requires explicit opt-in


@dataclass
class MoveOp:
    """A single proposed reorganization action (DM-MoveOp).

    Pure proposal — produced by Planner, only enacted by Executor (§2 invariant).
    """

    src: str                       # current absolute path
    dst: str                       # computed from destination + leaf
    cat_id: str                    # leaf category that motivated the move
    confidence: float              # calibrated confidence behind the decision
    mode: MoveMode = MoveMode.TRASH  # default trash (G1)
    approved: bool = False         # False until the user confirms (§7.1)


@dataclass
class Plan:
    """An ordered, side-effect-free proposal (DM-Plan).

    `moves` are auto-movable files; `needs_review` are paths held back (Tier 4,
    sub-threshold confidence, or exclusions). Counts summarize the dry-run.
    """

    moves: list[MoveOp] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, int]:
        return {"moves": len(self.moves), "needs_review": len(self.needs_review)}


@dataclass
class OpLogEntry:
    """A durable, append-only record of one executed move (DM-OpLogEntry).

    Carries enough state for `undo` to verify and reverse the op (G10).
    """

    op_id: int
    ts: float
    src_before: str
    dst_after: str
    mode: MoveMode
    hash_before: Optional[str] = None  # verified before undo (G10)
    reversible: bool = True            # False for HARD without trash
