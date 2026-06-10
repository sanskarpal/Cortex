# ARCHITECTURE-EXTENSION.md
## AI File Organizer — Engineering Contract Extension

Version: 0.1 — Companion to `ARCHITECTURE.md` (treated as canonical contract).

This document does **not** modify `ARCHITECTURE.md`. It records gaps found in it,
pins down the contracts the architecture leaves implicit, and defines verifiable
acceptance criteria. Where this document and `ARCHITECTURE.md` disagree,
`ARCHITECTURE.md` wins and the disagreement is logged in Section 1 as a gap to be
resolved by the author of the canonical doc.

---

## 1. Architectural Gaps

Problems, contradictions, and underspecifications found in `ARCHITECTURE.md`.
Each is tagged with a severity and the section it originates from.

### G1 — Move-default contradiction (BLOCKER) — §7.3
The header reads **"Move semantics: soft by default"**, but the body states
**"By default, files are MOVED."** A hard move is not soft. These cannot both
hold. The safest reading consistent with §7.1 ("Dry-run by default") and §14
("soft-move options") is: **default action = `trash`-backed move** (original
recoverable from system trash), not an in-place hard move. Until the canonical
doc resolves this, the contract in this document assumes **`trash` semantics as
the default**, and a true unrecoverable move requires explicit `--hard`.

### G2 — Cross-model embedding-space mismatch (BLOCKER) — §4.2 / §4.3 / M1
§4.3 specifies **two different embedding models**: CLIP for images and
`bge-small-en-v1.5` for text. §4.2(3) says "Classification = argmax of cosine
similarities between file embedding and cached category prompt embeddings." But
cosine similarity is only meaningful **within one embedding space**. A `bge`
text-document vector and a CLIP text-prompt vector are not comparable. Therefore
each category's prompt set must be embedded **twice** — once in CLIP space (for
image files) and once in `bge` space (for text files) — and the file is scored
only against the prompt embeddings from its own modality's model. The canonical
doc never states this; it is a required invariant (see C-MODEL, DM-CategoryPrompt).

### G3 — `BAAI/CurrencyCLIP` is not a general vision model (MAJOR) — §4.3 table
The "Recommended (Initial)" vision embedding lists `BAAI/CurrencyCLIP`. That name
denotes a currency/finance-specialized CLIP, not a general-purpose image encoder,
and contradicts §4.4 ("zero-shot… across file types") and the milestone text in
§13/M1 which simply says "CLIP." Treat the general open CLIP `ViT-B/32`
(`clip-ViT-B-32` via sentence-transformers) as the canonical M1 vision model;
`CurrencyCLIP` is assumed to be a typo for `OpenCLIP`.

### G4 — Confidence is uncalibrated (MAJOR) — §4.2 / §5 / §7.2
§7.2 gates auto-moves on `confidence >= min_confidence`, and §5 stores
`min_confidence` per category. But the only defined score is **raw cosine
similarity** (§4.2). Raw cosine is not a calibrated probability, is not
comparable across categories with differing prompt-set sizes, and shifts between
the CLIP and `bge` spaces (G2). The contract requires a defined, documented
mapping `confidence = f(cosine)` (e.g. softmax over per-modality category scores
with a fixed temperature) so that a single `min_confidence` threshold means the
same thing everywhere. This function is currently undefined.

### G5 — Hierarchical taxonomy vs. flat `destination` (MAJOR) — §5
§5 shows a **nested** taxonomy (`documents/invoices/`, `photos/screenshots/`)
but the per-category schema exposes a single flat `destination` and a flat
`match`/`rules`/`min_confidence`. It is unspecified whether classification
targets **leaf** categories only, parent categories, or both; how a parent's
prompts relate to its children's; and how argmax behaves across a two-level
tree. Contract decision (this doc): **classification operates on leaf categories
only**; parents are pure path prefixes derived from `destination`.

### G6 — Dedupe depends on an "optional" hash (MEDIUM) — §2 / §6
The ingestion box lists "dedupe" as a first-class step, but §6 marks the content
hash **optional**. Without the hash, dedupe degrades to (size, mtime, name),
which is unreliable. Either dedupe is best-effort (and must be labelled so), or
the hash is mandatory for any run that enables dedupe. Contract: **hash is
required iff dedupe or cache-by-content is enabled**; otherwise the cache key
falls back to (path, size, mtime) and this weaker guarantee is surfaced to the user.

### G7 — `needs_review` has no exit path (MEDIUM) — §3 Tier 4 / §7.2
Tier-4 and sub-threshold files are routed to `needs_review` and "never
auto-moved," but no mechanism is defined for a user to **resolve** that state
(manual label → move → and whether that label feeds §4.4 preference learning).
Without a resolution path, `needs_review` is a terminal sink. Contract requires a
`classify --review` / manual-label operation feeding the operation log and the
preference store.

### G8 — "No network calls" vs. lazy model download (MINOR) — §9 vs §14
§9 promises "No network calls during normal operation," while §14 mitigates disk
footprint via "lazy-download on first run." These are reconcilable only if
**first-run model fetch is explicitly classified as setup, not normal
operation**, and is gated behind explicit user consent. The contract pins that
distinction (see TC-PRIV-1).

### G9 — Tier-1 throughput vs. crawl target (MINOR) — §8
§8 targets full crawl of 1M files in `< 2 min`, and Tier-1 classification at
`< 1 ms/file`. Tier-1 alone at 1 ms × 1M = ~16.6 min, far exceeding the crawl
budget. The two targets are only consistent if Tier-1 classification is **not**
on the crawl's critical path (i.e. crawl = stat-only inventory; classification is
a separate, batched, incremental pass). Contract pins crawl and classify as
**distinct phases** with independent budgets.

### G10 — Undo vs. live watcher race (MINOR) — §6 / §7.4
§7.4 defines `undo` as replaying inverse moves for the last N actions, while §6
runs a change watcher. A watcher reacting to the organizer's own moves (or to a
concurrent user edit) can interleave operations such that an inverse replay no
longer matches the current filesystem state. Contract requires the operation log
to be **append-only with per-op pre/post path + hash**, and `undo` to **verify
state before each inverse move** and abort the op (not the batch) on mismatch.

---

## 2. Module Contracts

One module per row. Each is a single class with one responsibility and a typed
interface. Modules communicate only through the data models in Section 3; no
shared mutable global state.

| Module | Class | Responsibility | Key methods (signature sketch) | Consumes → Produces |
|---|---|---|---|---|
| Ingestion | `Scanner` | Walk targets, apply exclusions, dedupe, emit candidate file records. Stat-only; no content read. | `scan(targets: list[Path], cfg: ScanConfig) -> Iterator[FileRecord]` | `ScanConfig` → `FileRecord` (status=`pending`) |
| Ingestion | `Watcher` | Subscribe to FS change events (watchdog + native), enqueue changed paths; polling fallback. | `start(targets, queue: WorkQueue) -> None`; `stop()` | FS events → `WorkItem` |
| Feature Extraction | `Tierer` | Map a `FileRecord` to a processing tier (1–4) by extension/MIME. Pure, no I/O beyond MIME sniff. | `tier_of(rec: FileRecord) -> Tier` | `FileRecord` → `Tier` |
| Feature Extraction | `FeatureExtractor` | Per tier, extract metadata, text snippet, OCR text, or thumbnail. Never reads more than the configured cap. | `extract(rec, tier) -> FileFeatures` | `FileRecord`,`Tier` → `FileFeatures` |
| Local Model | `EmbeddingService` | Lazy-load CLIP + bge; embed text and images; cache vectors. Owns the two embedding spaces (G2). Offline after load. | `embed_text(str) -> Vec`; `embed_image(Path) -> Vec`; `ensure_models(consent: bool)` | text/image → `Vec` (tagged by space) |
| Classification | `TaxonomyStore` | Load `categories.yaml`, embed each category's prompts in **both** spaces, cache them. Leaf categories only (G5). | `load(path) -> Taxonomy`; `prompt_vecs(space) -> dict[CatId, list[Vec]]` | yaml → `Taxonomy`, `CategoryPrompt[]` |
| Classification | `RuleEngine` | Apply extension/MIME/path-keyword rules; high-confidence early exits. | `apply(rec, features) -> RuleVerdict \| None` | features → `RuleVerdict?` |
| Classification | `Classifier` | Combine rule verdict + embedding argmax; compute calibrated confidence `f(cosine)` (G4); assign `category`/`needs_review`. | `classify(rec, features) -> Classification` | features → `Classification` |
| Classification | `PreferenceStore` | Record accept/reject/override events; adjust rule weights / prompt sets over time (§4.4). | `record(event: FeedbackEvent)`; `bias(catId) -> float` | `FeedbackEvent` → weight deltas |
| Organization | `Planner` | Turn `Classification` + thresholds into a proposed `MoveOp` set. Dry-run by default; nothing executed here. | `plan(cls: list[Classification]) -> Plan` | `Classification[]` → `Plan` (proposed `MoveOp[]`) |
| Organization | `Executor` | Apply approved `MoveOp`s with chosen semantics (`trash`/`copy`/`symlink`/`hard`), write op log. Atomic per op. | `apply(plan: Plan, mode: MoveMode) -> list[OpLogEntry]` | approved `Plan` → `OpLogEntry[]` |
| Organization | `History` | Append-only op log; `undo` replays verified inverse moves (G10). | `append(entry)`; `undo(n: int) -> UndoReport` | `OpLogEntry` → filesystem reversal |
| Persistence | `Database` | SQLite schema + migrations; CRUD for `FileRecord`, cache lookups by cache key (G6). | `upsert(rec)`; `get_by_cachekey(k) -> FileRecord?`; `migrate()` | all records ↔ SQLite |
| Config | `ConfigLoader` | Load/validate `scan` + `categories.yaml`; surface weak-guarantee warnings (G6, G8). | `load() -> AppConfig` | files → `AppConfig` |
| Interface | `Cli` | `typer` entrypoint: `status`,`scan`,`classify`,`preview`,`apply`,`undo`,`classify --review`. Headless service driver. | `main()` | user → orchestration calls |

Interface invariants:
- `Scanner` and `FeatureExtractor` are the **only** modules that touch file
  bytes. `EmbeddingService` is the **only** module that loads ML weights.
- `Executor` is the **only** module that mutates the filesystem; everything
  upstream is pure/proposal-only. This makes dry-run (TC-SAFE-1) structurally
  guaranteed, not a runtime flag.
- No module imports `Cli`; the CLI is a top-level composition root only.

---

## 3. Data Models

Typed fields with ownership and lifecycle. Types are Python-flavored
(`pyproject` stack, §11). `Vec = list[float]` (or `np.ndarray[float32]`),
tagged with the model space that produced it.

### DM-FileRecord (owner: `Database`)
The durable per-file row (§6 baseline-scan schema).

| Field | Type | Notes |
|---|---|---|
| `id` | `int` (PK) | DB-assigned. |
| `path` | `str` (abs) | Current location. Updated by `Executor` after a move. |
| `size` | `int` bytes | From `os.stat`. |
| `mtime` | `float` | Epoch seconds. |
| `ctime` | `float` | Epoch seconds. |
| `content_hash` | `str \| None` | SHA-256; present iff dedupe/cache-by-content enabled (G6). |
| `extension` | `str` | Lowercased, no dot. |
| `mime` | `str` | Sniffed. |
| `tier` | `int` (1–4) | From `Tierer`. |
| `text_snippet` | `str \| None` | First N KB / N tokens; only Tier 2–3. |
| `embedding` | `Vec \| None` | Tagged with `embedding_space`. |
| `embedding_space` | `'clip' \| 'bge' \| None` | Which model produced `embedding` (G2). |
| `category` | `str \| None` | Leaf category id (G5). |
| `confidence` | `float \| None` | Calibrated `f(cosine)` ∈ [0,1] (G4). |
| `status` | `enum` | `pending \| classified \| confirmed \| needs_review`. |

**Lifecycle:** `Scanner` creates with `status=pending` → `FeatureExtractor`/
`EmbeddingService` fill features → `Classifier` sets `category`,`confidence`,
`status` → user approval sets `confirmed` (via `Executor`/`History`). Cache-first
(§6): if cache key unchanged, the row is reused and re-classification is skipped.

### DM-CacheKey (owner: `Database`)
`(path, size, mtime)` by default; `(content_hash)` when content-cache enabled
(G6). Used to satisfy §6 "skip unchanged."

### DM-FileFeatures (owner: `FeatureExtractor`, transient)
| Field | Type | Notes |
|---|---|---|
| `metadata` | `dict[str, Any]` | EXIF/size/path hints. |
| `text` | `str \| None` | Extracted/OCR'd text, capped. |
| `thumbnail` | `Path \| None` | For Tier 3 image embedding. |
| `modality` | `'text' \| 'image' \| 'none'` | Selects the embedding space (G2). |

Lifecycle: created per-classify pass, not persisted except `text_snippet`.

### DM-CategoryPrompt (owner: `TaxonomyStore`)
| Field | Type | Notes |
|---|---|---|
| `cat_id` | `str` | Leaf id, e.g. `documents/invoices`. |
| `match` | `list[str]` | Prompt strings (§5). |
| `rules` | `RuleSpec \| None` | regex / ext / path. |
| `min_confidence` | `float` | Auto-move gate (§7.2). |
| `destination` | `str` (rel) | Target path; parents derived from it (G5). |
| `prompt_vecs_clip` | `list[Vec]` | Cached CLIP-space embeddings (G2). |
| `prompt_vecs_bge` | `list[Vec]` | Cached bge-space embeddings (G2). |

Lifecycle: loaded once from `categories.yaml`, prompt vectors cached on first
run, invalidated when `match` text or the model checkpoint changes.

### DM-Classification (owner: `Classifier`, transient → persisted into FileRecord)
| Field | Type | Notes |
|---|---|---|
| `cat_id` | `str \| None` | argmax leaf, or `None` → needs_review. |
| `cosine` | `float` | Raw top-1 similarity. |
| `confidence` | `float` | Calibrated (G4). |
| `source` | `'rule' \| 'embedding' \| 'hybrid'` | Provenance for auditability (§9). |

### DM-MoveOp / DM-Plan (owner: `Planner`)
| Field | Type | Notes |
|---|---|---|
| `src` | `str` | Current path. |
| `dst` | `str` | Computed from `destination` + leaf. |
| `mode` | `'trash' \| 'copy' \| 'symlink' \| 'hard'` | Default `trash` (G1). |
| `approved` | `bool` | False until user confirms (§7.1). |

`Plan` = ordered `list[MoveOp]` + summary counts; pure proposal, no side effects.

### DM-OpLogEntry (owner: `History`, durable append-only)
| Field | Type | Notes |
|---|---|---|
| `op_id` | `int` | Monotonic. |
| `ts` | `float` | Epoch. |
| `src_before` / `dst_after` | `str` | For inverse replay. |
| `mode` | `MoveMode` | As executed. |
| `hash_before` | `str \| None` | Verified before undo (G10). |
| `reversible` | `bool` | False for `hard` without trash. |

Lifecycle: written by `Executor`, consumed by `History.undo`; never mutated.

### DM-FeedbackEvent (owner: `PreferenceStore`)
`{file_id, from_cat, to_cat, action: accept|reject|override, ts}` → adjusts
rule weights / prompt bias (§4.4).

---

## 4. Test Contracts

Acceptance tests mapped **one-to-one** to specific claims in `ARCHITECTURE.md`.
Each test names the canonical claim it verifies. "Pass" is the literal criterion.

| ID | Canonical claim (source) | Test / Acceptance criterion |
|---|---|---|
| TC-PRIV-1 | §9 "No network calls during normal operation" | Run a full `scan`+`classify`+`apply` cycle with network namespace blocked (no sockets). Cycle completes with zero outbound connections. First-run model fetch is excluded only when explicitly invoked as a separate consented setup step (G8). |
| TC-PRIV-2 | §9 "open-source classifiers, transparent prompt sets… auditable" | Every `Classification` row carries a `source` provenance and references a prompt set readable from `categories.yaml`. Assert no opaque/hidden category source. |
| TC-SAFE-1 | §7.1 "Dry-run by default" | Calling `classify`+`preview` without an explicit `apply` mutates **zero** files on disk (assert filesystem hash of all targets unchanged). Structural: only `Executor.apply` can mutate. |
| TC-SAFE-2 | §7.2 "Only files with confidence ≥ min_confidence can be auto-moved" | Seed two files: one at conf=min+ε (planned for move), one at min−ε (must be `needs_review`, no `MoveOp`). Assert plan contents exactly. Requires calibrated `confidence` (G4). |
| TC-SAFE-3 | §3 Tier 4 "never auto-moved" / §14 mixed types → needs_review | Feed a no-extension file, a password-protected zip, and a nested-attachment PDF. All land `needs_review` with **no** `MoveOp`. |
| TC-SAFE-4 | §7.3 move semantics (resolved per G1) | Default mode produces recoverable `trash` ops; `--hard` required for unrecoverable move. Assert default `MoveOp.mode == 'trash'`. (Flags the §7.3 contradiction explicitly.) |
| TC-SAFE-5 | §7.4 "single undo reverses last N actions" | After applying N moves, `undo(N)` restores every file to `src_before`; verify by hash. Under a concurrent modification, the affected op aborts and the rest still revert (G10). |
| TC-SAFE-6 | §7.5 / §7.6 "exclusion rules; never root" | `.git`, `node_modules`, `~/Library/Mail` are skipped by default; assert no records produced for them. Assert process never calls a privilege-escalation API. |
| TC-TIER-1 | §3 tiering by extension/MIME | A `.zip`→Tier1, `.py`→Tier2, `.jpg`→Tier3, no-ext→Tier4. Assert `Tierer.tier_of` exact mapping; Tier1 performs **no** ML inference. |
| TC-CACHE-1 | §6 "Cache-first… skip if size/mtime/hash unchanged" | Classify a file, then re-run with unchanged file: second pass performs zero embedding calls (mock `EmbeddingService`, assert call count == 0). |
| TC-MODEL-1 | §4.2(3) cosine argmax — corrected by G2 | A text file is scored **only** against `prompt_vecs_bge`; an image **only** against `prompt_vecs_clip`. Assert no cross-space comparison occurs (instrument `EmbeddingService`). |
| TC-MODEL-2 | §4.2 "argmax of cosine similarities" | Given a synthetic prompt set with one obviously-matching category, `Classifier` returns that `cat_id` as top-1. |
| TC-PERF-1 | §8 Tier-1 "< 1 ms per file" | Microbench Tier-1 metadata classification over 10k synthetic records; median < 1 ms/file. |
| TC-PERF-2 | §8 "crawl 1M files < 2 min" (with G9 split) | Stat-only crawl of a 1M-entry tree completes < 2 min; classification explicitly excluded from this budget. |
| TC-PERF-3 | §8 "incremental update < 5 s end-to-end" | Touch one doc; watcher→classify→preview round-trip median < 5 s. |
| TC-CONF-1 | §5 per-category `min_confidence` (needs G4) | `confidence` is in [0,1] and comparable across categories: two categories with different prompt-set sizes both honor the same numeric threshold. |
| TC-EXT-1 | §10 plugin interfaces | A no-op `Classifier` plugin can be registered and swapped without editing core modules (load via entry point); core still runs. |
| TC-REVIEW-1 | §3/§7.2 needs_review resolution (G7) | `classify --review` on a `needs_review` file accepts a manual label, emits a `MoveOp`, and records a `FeedbackEvent`. Assert the file exits `needs_review`. |

---

## 5. M1 Acceptance Criteria — Prototype Classifier

Maps to `ARCHITECTURE.md` §13: *"M1 — Prototype classifier: Manual Python
script: walk dir, embed text with BGE, embed images with CLIP, print top-1
category per file."* M1 is a **single throwaway script**, not the packaged
library (that is M2). No DB, no watcher, no moves.

### Scope (in)
- One script `m1_classify.py`, run as `python m1_classify.py <dir>`.
- Recursively walk `<dir>` (stat-only ingestion; honor a hardcoded exclude list:
  `.git`, `node_modules`, dotfiles).
- Tier routing (Tier 1/2/3 only; Tier 4 → print `needs_review`, no embedding).
- Text files (Tier 2): extract first N KB → `bge-small-en-v1.5` embedding.
- Image files (Tier 3): `clip-ViT-B-32` embedding (per G3; **not** CurrencyCLIP).
- A small **hardcoded** taxonomy (≈6 leaf categories) with prompt strings,
  embedded in **both** CLIP and bge spaces (per G2).
- Print one line per file: `path → top1_category (cosine=…)`.

### Scope (out — explicitly deferred to M2+)
SQLite, watcher, move/undo/Executor, confidence calibration, `categories.yaml`,
preference learning, OCR, performance tuning, CLI subcommands.

### Done criteria (all must hold)
1. **Runs offline after model load.** With network blocked post-download, the
   script classifies every supported file with zero network calls (TC-PRIV-1 at
   prototype scope). Model download is a separate, explicit one-time step.
2. **Correct space isolation (G2).** Text files are scored only against bge-space
   prompt vectors; images only against CLIP-space prompt vectors. No
   cross-space cosine is ever computed. (Verifiable by instrumentation.)
3. **Top-1 output for every walked file.** Each non-excluded file prints exactly
   one line; Tier-4/unsupported files print `needs_review` instead of a category
   and are **not** embedded.
4. **Tiering correct on a fixture set.** A fixture dir containing at least one
   `.txt`/`.md`, one `.py`, one `.jpg`/`.png`, one `.zip`, and one no-extension
   file routes to tiers 2,2,3,1,4 respectively.
5. **Deterministic argmax.** Given the hardcoded taxonomy and a fixed file, the
   printed top-1 category is stable across runs (same model, same input).
6. **Sanity accuracy gate.** On a labelled fixture set of ≥20 files spanning the
   ≈6 categories, top-1 accuracy ≥ 70%. (Prototype bar only; informs the §12
   benchmarking work, not a production target.)
7. **No filesystem mutation.** The script never moves, renames, copies, or
   deletes any file (read-only). Asserted by comparing a recursive hash of
   `<dir>` before and after the run.
8. **No crash on malformed input.** Unreadable/corrupt/zero-byte files are
   reported as `needs_review` and skipped, not fatal.

### M1 explicitly does NOT certify
Confidence thresholds (G4), move safety (§7), undo (§7.4), incremental caching
(§6), or any performance target (§8). Those are M2/M3 acceptance gates.

---

*End of ARCHITECTURE-EXTENSION.md*
</content>
</invoke>
