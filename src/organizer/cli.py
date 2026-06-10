"""CLI composition root for the M2 file organizer (ARCHITECTURE-EXTENSION §2 Cli).

This is the ONLY place that wires the modules together; no module imports this
one (§2 invariant). Subcommands (ARCHITECTURE.md §13 M2):

    status   - show DB / config health and category distribution
    scan     - walk targets, persist FileRecords (stat-only, cache-first §6)
    classify - classify pending files (gated, G4), persist category/confidence
    preview  - dry-run reorganization plan (§7.1 dry-run default)
    apply    - execute an approved plan (default trash mode, G1)
    undo     - reverse the last N moves (state-verified, G10)
    review   - manually label a needs_review file (G7)

Uses argparse (stdlib) rather than typer to avoid an extra dependency; the
spec's `typer` suggestion (§11) is non-binding for the prototype.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# M2 entrypoint also runnable directly; ensure the package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from organizer.classify import classify
from organizer.config import ConfigLoader
from organizer.database import Database
from organizer.rules import RuleEngine
from organizer.embedding import (
    FakeEmbeddingService,
    SentenceTransformerEmbeddingService,
)
from organizer.executor import Executor
from organizer.features import extract
from organizer.history import History
from organizer.ingest import scan
from organizer.planner import Planner
from organizer.types import MoveMode, Tier

DEFAULT_DB = ".organizer/index.db"
DEFAULT_LOG = ".organizer/oplog.jsonl"
DEFAULT_CONFIG = "config/categories.yaml"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _build_embedder(real: bool, consent: bool):
    if not real:
        return FakeEmbeddingService()
    svc = SentenceTransformerEmbeddingService()
    svc.ensure_models(consent=consent)
    return svc


def _load_taxonomy(config_path: Path, embedder):
    loader = ConfigLoader(config_path)
    cfg = loader.load()
    for w in cfg.warnings:
        print(f"warning: {w}", file=sys.stderr)
    return cfg, loader.build_category_prompts(cfg, embedder)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_status(args) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print("no index yet — run `scan` first")
        return 0
    with Database(db_path) as db:
        db.migrate()
        rows = db.all_rows() if hasattr(db, "all_rows") else []
    print(f"index: {db_path}  ({len(rows)} files)")
    dist: dict[str, int] = {}
    for r in rows:
        dist[r.get("category") or r.get("status", "?")] = (
            dist.get(r.get("category") or r.get("status", "?"), 0) + 1
        )
    for k, v in sorted(dist.items()):
        print(f"  {k:28s} {v}")
    return 0


def cmd_scan(args) -> int:
    db_path = Path(args.db)
    _ensure_parent(db_path)
    n = 0
    with Database(db_path) as db:
        db.migrate()
        for rec in scan(Path(args.directory)):
            db.upsert(rec)
            n += 1
    print(f"scanned {n} files -> {db_path}")
    return 0


def _classify_hybrid(rec, rule_engine, taxonomy, embedder, gate: bool):
    """Rule layer first (§4.2 layer 1, M3 hybrid), then embedding classifier."""
    from organizer.types import Classification

    verdict = rule_engine.apply(rec) if rule_engine is not None else None
    if verdict is not None:
        return Classification(
            cat_id=verdict.cat_id,
            cosine=0.0,  # not embedding-derived
            source="rule",
            confidence=verdict.confidence,
        )
    return classify(extract(rec), taxonomy, embedder, gate=gate)


def cmd_classify(args) -> int:
    db_path = Path(args.db)
    _ensure_parent(db_path)
    embedder = _build_embedder(args.real, args.consent)
    cfg, taxonomy = _load_taxonomy(Path(args.config), embedder)
    rule_engine = RuleEngine(cfg.categories)
    n = classified = 0
    with Database(db_path) as db:
        db.migrate()
        for rec in scan(Path(args.directory)):
            n += 1
            if args.cache and db.is_cached(rec, by_content=False):
                continue  # cache-first skip (§6, TC-CACHE-1)
            res = _classify_hybrid(rec, rule_engine, taxonomy, embedder, args.gate)
            db.upsert(rec, category=res.cat_id, confidence=res.confidence)
            if res.cat_id is not None:
                classified += 1
    print(f"classified {classified}/{n} files (rest needs_review)")
    return 0


def _plan_from_scan(args, embedder, taxonomy, rule_engine=None):
    planner = Planner(Path(args.dest))
    items = []
    for rec in scan(Path(args.directory)):
        if rec.tier is Tier.REVIEW:
            items.append((rec, _needs_review_cls()))
            continue
        res = _classify_hybrid(rec, rule_engine, taxonomy, embedder, args.gate)
        items.append((rec, res))
    return planner.plan(items)


def _needs_review_cls():
    from organizer.types import Classification

    return Classification(cat_id=None, cosine=0.0, source="needs_review")


def cmd_preview(args) -> int:
    embedder = _build_embedder(args.real, args.consent)
    cfg, taxonomy = _load_taxonomy(Path(args.config), embedder)
    plan = _plan_from_scan(args, embedder, taxonomy, RuleEngine(cfg.categories))
    for op in plan.moves:
        print(f"  MOVE  {op.src}\n     -> {op.dst}  [{op.mode.value} conf={op.confidence:.2f}]")
    for p in plan.needs_review:
        print(f"  REVIEW {p}")
    print(f"\nplan: {plan.summary['moves']} moves, {plan.summary['needs_review']} needs_review (dry-run)")
    return 0


def cmd_apply(args) -> int:
    embedder = _build_embedder(args.real, args.consent)
    cfg, taxonomy = _load_taxonomy(Path(args.config), embedder)
    plan = _plan_from_scan(args, embedder, taxonomy, RuleEngine(cfg.categories))
    for op in plan.moves:  # explicit user invocation == approval (§7.1)
        op.approved = True
    log_path = Path(args.log)
    _ensure_parent(log_path)
    history = History(log_path)
    executor = Executor()
    mode = MoveMode(args.mode)
    entries = executor.apply(plan, mode=mode, history=history, dry_run=False)
    print(f"applied {len(entries)} moves in {mode.value} mode; log -> {log_path}")
    return 0


def cmd_undo(args) -> int:
    history = History(Path(args.log))
    report = history.undo(args.n)
    print(f"undo: reversed {report.get('reversed', 0)}, skipped {report.get('skipped', 0)}")
    return 0


def cmd_review(args) -> int:
    """Manually label a needs_review file and move it (G7)."""
    embedder = _build_embedder(args.real, args.consent)
    _, _ = _load_taxonomy(Path(args.config), embedder)
    from organizer.types import Classification, MoveOp, Plan

    src = Path(args.file).resolve()
    op = MoveOp(
        src=str(src),
        dst=str(Path(args.dest) / args.label / src.name),
        cat_id=args.label,
        confidence=1.0,  # human label is ground truth
        mode=MoveMode(args.mode),
        approved=True,
    )
    log_path = Path(args.log)
    _ensure_parent(log_path)
    entries = Executor().apply(
        Plan(moves=[op]), mode=MoveMode(args.mode), history=History(log_path), dry_run=False
    )
    # G7: manual label exits needs_review AND feeds the preference store (§4.4).
    import time

    from organizer.preferences import FeedbackEvent, PreferenceStore

    prefs_path = log_path.parent / "preferences.jsonl"
    PreferenceStore(prefs_path).record(
        FeedbackEvent(
            file_path=str(src), from_cat=None, to_cat=args.label,
            action="override", ts=time.time(),
        )
    )
    print(f"reviewed {src} -> {args.label} ({len(entries)} moved; feedback recorded)")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="organizer", description="AI File Organizer (M2)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, *, need_dir=True):
        if need_dir:
            sp.add_argument("directory", type=str, help="target directory")
        sp.add_argument("--db", default=DEFAULT_DB)
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--real", action="store_true", help="use real bge+clip models")
        sp.add_argument("--consent", action="store_true", help="consent to model load (G8)")
        sp.add_argument("--gate", action="store_true", help="confidence gating (G4)")

    sp = sub.add_parser("status"); sp.add_argument("--db", default=DEFAULT_DB); sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("scan"); sp.add_argument("directory"); sp.add_argument("--db", default=DEFAULT_DB); sp.set_defaults(func=cmd_scan)
    sp = sub.add_parser("classify"); add_common(sp); sp.add_argument("--cache", action="store_true", help="skip cached files (§6)"); sp.set_defaults(func=cmd_classify)
    sp = sub.add_parser("preview"); add_common(sp); sp.add_argument("--dest", default="organized"); sp.set_defaults(func=cmd_preview)
    sp = sub.add_parser("apply"); add_common(sp); sp.add_argument("--dest", default="organized"); sp.add_argument("--log", default=DEFAULT_LOG); sp.add_argument("--mode", default="trash", choices=[m.value for m in MoveMode]); sp.set_defaults(func=cmd_apply)
    sp = sub.add_parser("undo"); sp.add_argument("--log", default=DEFAULT_LOG); sp.add_argument("-n", type=int, default=1); sp.set_defaults(func=cmd_undo)
    sp = sub.add_parser("review"); sp.add_argument("file"); sp.add_argument("label"); add_common(sp, need_dir=False); sp.add_argument("--dest", default="organized"); sp.add_argument("--log", default=DEFAULT_LOG); sp.add_argument("--mode", default="trash", choices=[m.value for m in MoveMode]); sp.set_defaults(func=cmd_review)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
