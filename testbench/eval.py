"""Large accuracy evaluation on the public-file test bench (no personal files).

Pass A: full system — RuleEngine -> embedding -> calibration gate (user path).
Pass B: embedding-only, gated and ungated (isolates model quality).
Safety metric throughout: confidently-wrong count (would-be bad auto-moves).
"""
import sys, hashlib, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from organizer.ingest import scan
from organizer.features import extract
from organizer.classify import classify
from organizer.config import ConfigLoader
from organizer.rules import RuleEngine
from organizer.embedding import SentenceTransformerEmbeddingService
from organizer.types import Classification, Tier

D = Path(__file__).resolve().parent / "files"

def expected(name: str):
    for prefix, cat in [
        ("py_", "code/python"), ("js_", "code/javascript"),
        ("doc_", "documents/personal"), ("invoice_", "documents/invoices"),
        ("receipt_", "documents/receipts"), ("photo_", "photos/personal"),
        ("screenshot_", "photos/screenshots"), ("backup_", "downloads/archives"),
    ]:
        if name.startswith(prefix):
            return cat
    return "needs_review"  # mystery_blob, empty_file, corrupt_image, weird.qz9

def tree_hash():
    h = hashlib.sha256()
    for p in sorted(D.rglob("*")):
        if p.is_file():
            h.update(p.name.encode()); h.update(str(p.stat().st_size).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()

def score(results):
    """results: list of (name, expected, got). Returns metrics dict."""
    correct = wrong = held = held_should_classify = 0
    wrong_list = []
    for name, exp, got in results:
        if got == "needs_review":
            if exp == "needs_review": correct += 1
            else: held += 1; held_should_classify += 1
        elif got == exp: correct += 1
        else: wrong += 1; wrong_list.append((name, exp, got))
    total = len(results)
    auto = sum(1 for _, e, g in results if g != "needs_review")
    auto_ok = sum(1 for _, e, g in results if g != "needs_review" and g == e)
    return dict(total=total, correct=correct, wrong=wrong, held=held_should_classify,
                auto=auto, auto_ok=auto_ok, wrong_list=wrong_list)

def report(tag, m):
    prec = 100*m["auto_ok"]/m["auto"] if m["auto"] else 0
    acc = 100*m["correct"]/m["total"]
    print(f"\n[{tag}] files={m['total']}  overall-correct={m['correct']} ({acc:.0f}%)  "
          f"auto-moved={m['auto']} (precision {prec:.0f}%)  "
          f"WRONG-MOVES={m['wrong']}  held-for-review={m['held']}")
    for n, e, g in m["wrong_list"]:
        print(f"   XX {n}: expected {e}, got {g}")

h0 = tree_hash()
emb = SentenceTransformerEmbeddingService(); emb.ensure_models(consent=True)
loader = ConfigLoader(ROOT / "config" / "categories.yaml")
cfg = loader.load()
tax = loader.build_category_prompts(cfg, emb)
rules = RuleEngine(cfg.categories)

recs = list(scan(D))
print(f"scanned {len(recs)} files")

t0 = time.perf_counter()
passA, passB_gated, passB_open = [], [], []
rule_hits = 0
for rec in recs:
    exp = expected(rec.path.name)
    feats = extract(rec)

    # ---- Pass A: full system (rules first, then gated embedding) ----
    if rec.tier is Tier.REVIEW:
        a = Classification(cat_id=None, cosine=0.0, source="needs_review")
    else:
        v = rules.apply(rec)
        if v is not None:
            rule_hits += 1
            a = Classification(cat_id=v.cat_id, cosine=0.0, source="rule", confidence=v.confidence)
        else:
            a = classify(feats, tax, emb, gate=True)
    passA.append((rec.path.name, exp, a.cat_id or "needs_review"))

    # ---- Pass B: embedding only (no rules) ----
    if rec.tier is Tier.REVIEW or rec.tier is Tier.METADATA:
        bg = bo = "needs_review"
    else:
        bg = (classify(feats, tax, emb, gate=True).cat_id or "needs_review")
        bo = (classify(feats, tax, emb, gate=False).cat_id or "needs_review")
    passB_gated.append((rec.path.name, exp, bg))
    passB_open.append((rec.path.name, exp, bo))

dt = time.perf_counter() - t0
print(f"classification wall time: {dt:.1f}s  ({dt/len(recs)*1000:.0f} ms/file avg, incl. 3 passes)")
print(f"rule-layer decisions in pass A: {rule_hits}")

report("A: FULL SYSTEM (rules + embedding + gate)", score(passA))
report("B1: EMBEDDING ONLY, gated", score(passB_gated))
report("B2: EMBEDDING ONLY, ungated (raw argmax)", score(passB_open))

print("\nmutation check:", "NONE" if tree_hash() == h0 else "MUTATED!")
