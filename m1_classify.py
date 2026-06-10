#!/usr/bin/env python3
"""M1 prototype classifier — throwaway script (ARCHITECTURE.md §13 / EXT §5).

Usage:
    python m1_classify.py <dir> [--real] [--consent]

Walks <dir> stat-only, tiers each file, extracts capped features, embeds text
with bge / images with clip (two isolated spaces, G2), and prints one line per
file: `path -> top1_category (cosine=...)`. Tier-1/Tier-4/unsupported/unreadable
files print `needs_review` and are never embedded.

Backends:
    default      FakeEmbeddingService (offline, deterministic, no downloads).
    --real       SentenceTransformerEmbeddingService (bge + clip-ViT-B-32).
                 Requires --consent: model download/load is an explicit one-time
                 setup step, not normal operation (G8 / TC-PRIV-1).

This script mutates nothing on disk (M1 done-criterion 7).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# M1 is a throwaway script; put the package dir on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from organizer.classify import classify
from organizer.embedding import (
    FakeEmbeddingService,
    SentenceTransformerEmbeddingService,
)
from organizer.features import extract
from organizer.ingest import scan
from organizer.taxonomy import build_taxonomy


def build_embedder(real: bool, consent: bool):
    if not real:
        return FakeEmbeddingService()
    svc = SentenceTransformerEmbeddingService()
    svc.ensure_models(consent=consent)  # raises without consent (G8)
    return svc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 prototype file classifier")
    ap.add_argument("directory", type=Path, help="directory to walk (read-only)")
    ap.add_argument(
        "--real",
        action="store_true",
        help="use real bge+clip models instead of the offline fake backend",
    )
    ap.add_argument(
        "--consent",
        action="store_true",
        help="consent to one-time model download/load (required with --real)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="apply calibrated confidence gating (G4): low-confidence files "
        "are routed to needs_review instead of a confidently-wrong category",
    )
    args = ap.parse_args(argv)

    root = args.directory
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    embedder = build_embedder(args.real, args.consent)
    taxonomy = build_taxonomy(embedder)

    for rec in scan(root):
        features = extract(rec)
        result = classify(features, taxonomy, embedder, gate=args.gate)
        if result.cat_id is None:
            print(f"{rec.path} -> needs_review")
        else:
            print(f"{rec.path} -> {result.cat_id} (cosine={result.cosine:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
