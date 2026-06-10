"""M2 integration tests — persistence, planning, execution, undo, config.

Mapped to ARCHITECTURE-EXTENSION.md §4 test contracts. Offline only
(FakeEmbeddingService); no models, no network. Filesystem effects happen inside
pytest tmp_path so they are hermetic.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from organizer.classify import classify
from organizer.config import ConfigLoader
from organizer.database import Database
from organizer.embedding import FakeEmbeddingService
from organizer.executor import Executor
from organizer.features import extract
from organizer.history import History
from organizer.ingest import scan
from organizer.planner import Planner
from organizer.types import (
    Classification,
    FileRecord,
    MoveMode,
    MoveOp,
    Plan,
    Tier,
)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "categories.yaml"


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rec(p: Path) -> FileRecord:
    st = p.stat()
    return FileRecord(
        path=p, size=st.st_size, mtime=st.st_mtime,
        extension=p.suffix.lstrip(".").lower(), mime="text/plain", tier=Tier.TEXT,
    )


# --------------------------------------------------------------------------- #
# Database / cache
# --------------------------------------------------------------------------- #
class TestDatabase:
    def test_migrate_idempotent(self):
        with Database(":memory:") as db:
            db.migrate()
            db.migrate()  # second call must not error

    def test_upsert_and_get(self, tmp_path):
        f = tmp_path / "a.txt"; f.write_text("hi")
        with Database(":memory:") as db:
            db.migrate()
            rid = db.upsert(_rec(f), category="documents/personal", confidence=0.9)
            assert rid >= 1
            row = db.get_by_path(str(f))
            assert row["category"] == "documents/personal"

    def test_status_lifecycle_on_upsert(self, tmp_path):
        """Classify-pass upserts derive status; scan upserts keep rec.status."""
        f = tmp_path / "a.txt"; f.write_text("hi")
        with Database(":memory:") as db:
            db.migrate()
            db.upsert(_rec(f))  # scan: no category kwarg
            assert db.get_by_path(str(f))["status"] == "pending"
            db.upsert(_rec(f), category="code/python", confidence=0.9)
            assert db.get_by_path(str(f))["status"] == "classified"
            db.upsert(_rec(f), category=None, confidence=0.1)  # gated out
            assert db.get_by_path(str(f))["status"] == "needs_review"
            db.upsert(_rec(f), category="code/python", confidence=0.9,
                      status="confirmed")  # explicit override wins
            assert db.get_by_path(str(f))["status"] == "confirmed"

    def test_tc_cache_1_skip_unchanged(self, tmp_path):
        """TC-CACHE-1: an unchanged, classified file is reported cached."""
        f = tmp_path / "a.txt"; f.write_text("invoice total due")
        with Database(":memory:") as db:
            db.migrate()
            rec = _rec(f)
            assert db.is_cached(rec, by_content=False) is False  # not yet classified
            db.upsert(rec, category="documents/invoices", confidence=0.8)
            assert db.is_cached(_rec(f), by_content=False) is True  # now cached


# --------------------------------------------------------------------------- #
# Planner — dry-run / safety
# --------------------------------------------------------------------------- #
class TestPlanner:
    def test_tc_safe_2_threshold_gates_move(self, tmp_path):
        """TC-SAFE-2: needs_review (cat_id None) yields no MoveOp."""
        f1 = tmp_path / "ok.txt"; f1.write_text("x")
        f2 = tmp_path / "rev.txt"; f2.write_text("y")
        items = [
            (_rec(f1), Classification(cat_id="code/python", cosine=0.7, source="embedding", confidence=0.9)),
            (_rec(f2), Classification(cat_id=None, cosine=0.1, source="needs_review", confidence=0.2)),
        ]
        plan = Planner(tmp_path / "organized").plan(items)
        assert len(plan.moves) == 1
        assert plan.moves[0].src == str(f1.resolve()) or plan.moves[0].src == str(f1)
        assert str(f2) in plan.needs_review

    def test_tc_safe_4_default_mode_is_trash(self, tmp_path):
        """TC-SAFE-4: default MoveOp.mode == trash (G1)."""
        f = tmp_path / "a.txt"; f.write_text("x")
        items = [(_rec(f), Classification(cat_id="code/python", cosine=0.7, source="embedding", confidence=0.9))]
        plan = Planner(tmp_path / "organized").plan(items)
        assert plan.moves[0].mode is MoveMode.TRASH


# --------------------------------------------------------------------------- #
# Executor — dry-run default / execution
# --------------------------------------------------------------------------- #
class TestExecutor:
    def test_tc_safe_1_dry_run_default_no_mutation(self, tmp_path):
        """TC-SAFE-1: apply() with default dry_run mutates nothing."""
        f = tmp_path / "a.txt"; f.write_text("x")
        before = _hash(f)
        op = MoveOp(src=str(f), dst=str(tmp_path / "organized" / "a.txt"),
                    cat_id="code/python", confidence=0.9, approved=True)
        entries = Executor().apply(Plan(moves=[op]))  # dry_run defaults True
        assert entries == []
        assert f.exists() and _hash(f) == before
        assert not (tmp_path / "organized").exists()

    def test_apply_copy_keeps_original(self, tmp_path):
        f = tmp_path / "a.txt"; f.write_text("payload")
        dst = tmp_path / "organized" / "a.txt"
        op = MoveOp(src=str(f), dst=str(dst), cat_id="code/python", confidence=0.9,
                    mode=MoveMode.COPY, approved=True)
        entries = Executor().apply(Plan(moves=[op]), mode=MoveMode.COPY, dry_run=False)
        assert len(entries) == 1
        assert f.exists()  # original kept (copy)
        assert dst.exists() and dst.read_text() == "payload"

    def test_unapproved_op_not_executed(self, tmp_path):
        f = tmp_path / "a.txt"; f.write_text("x")
        op = MoveOp(src=str(f), dst=str(tmp_path / "o" / "a.txt"),
                    cat_id="code/python", confidence=0.9, approved=False)
        entries = Executor().apply(Plan(moves=[op]), dry_run=False)
        assert entries == []
        assert f.exists()


# --------------------------------------------------------------------------- #
# History — undo with state verification (G10)
# --------------------------------------------------------------------------- #
class TestHistoryUndo:
    def test_tc_safe_5_undo_restores(self, tmp_path):
        """TC-SAFE-5: undo(N) restores moved files to src_before."""
        f = tmp_path / "a.txt"; f.write_text("data")
        src = str(f); dst = str(tmp_path / "organized" / "a.txt")
        log = History(tmp_path / "oplog.jsonl")
        op = MoveOp(src=src, dst=dst, cat_id="code/python", confidence=0.9,
                    mode=MoveMode.TRASH, approved=True)
        Executor().apply(Plan(moves=[op]), mode=MoveMode.TRASH, history=log, dry_run=False)
        assert not Path(src).exists() and Path(dst).exists()
        report = log.undo(1)
        assert report["reversed"] == 1
        assert Path(src).exists() and not Path(dst).exists()

    def test_g10_undo_aborts_op_on_state_mismatch(self, tmp_path):
        """G10: if dst no longer matches, that op is skipped, not crashed."""
        f = tmp_path / "a.txt"; f.write_text("data")
        src = str(f); dst = str(tmp_path / "organized" / "a.txt")
        log = History(tmp_path / "oplog.jsonl")
        op = MoveOp(src=src, dst=dst, cat_id="code/python", confidence=0.9,
                    mode=MoveMode.TRASH, approved=True)
        Executor().apply(Plan(moves=[op]), mode=MoveMode.TRASH, history=log, dry_run=False)
        # Simulate a concurrent change: delete the moved file.
        Path(dst).unlink()
        report = log.undo(1)
        assert report["skipped"] == 1
        assert report["reversed"] == 0


# --------------------------------------------------------------------------- #
# Config loader
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_load_yaml_and_warnings(self):
        cfg = ConfigLoader(CONFIG).load()
        assert len(cfg.categories) >= 6
        ids = {c["cat_id"] for c in cfg.categories}
        assert "documents/invoices" in ids and "code/python" in ids
        # G6 warning present when content_cache disabled (default).
        assert any("G6" in w for w in cfg.warnings)

    def test_build_category_prompts_both_spaces(self):
        from organizer.types import EmbeddingSpace
        emb = FakeEmbeddingService()
        loader = ConfigLoader(CONFIG)
        cps = loader.build_category_prompts(loader.load(), emb)
        assert cps
        for cp in cps:
            assert cp.prompt_vecs[EmbeddingSpace.BGE]
            assert cp.prompt_vecs[EmbeddingSpace.CLIP]


# --------------------------------------------------------------------------- #
# End-to-end pipeline (offline) — TC-REVIEW-1 style + full flow
# --------------------------------------------------------------------------- #
class TestPipelineE2E:
    def test_scan_classify_preview_no_mutation(self, tmp_path):
        # build a small tree
        (tmp_path / "inv.txt").write_text("invoice total amount due payment")
        (tmp_path / "code.py").write_text("def f():\n    return 1\n")
        before = {p: _hash(p) for p in tmp_path.rglob("*") if p.is_file()}

        emb = FakeEmbeddingService()
        loader = ConfigLoader(CONFIG)
        taxonomy = loader.build_category_prompts(loader.load(), emb)
        planner = Planner(tmp_path / "organized")
        items = []
        for rec in scan(tmp_path):
            res = classify(extract(rec), taxonomy, emb, gate=False)
            items.append((rec, res))
        plan = planner.plan(items)
        assert plan.summary["moves"] >= 1
        # preview/plan must not have mutated anything
        after = {p: _hash(p) for p in tmp_path.rglob("*") if p.is_file()}
        assert before == after
