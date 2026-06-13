# AI File Organizer — System Report & User Guide

**Repository:** `github.com:sanskarpal/Cortex.git`
**Status:** All planned milestones (M1–M5) complete · 111 tests passing
**Canonical design documents:** `ARCHITECTURE.md` (system design) and `ARCHITECTURE-EXTENSION.md` (engineering contract: module interfaces, data models, acceptance tests, and the G1–G10 gap resolutions). This report explains how the implemented system works; where it and the design documents disagree, the design documents win.

---

## 1. What this system is

A **privacy-first, locally-running AI file organizer**. You point it at a folder (or several), it classifies every file into a semantic category — invoices, source code, screenshots, personal photos, prose documents, archives — and then proposes (never silently performs) a reorganization into a clean folder structure.

Three principles drive every design decision:

1. **Privacy.** Everything runs on your machine. No file content, metadata, or embedding ever leaves the device. The ML models are open-weight checkpoints that are downloaded once, with your explicit consent, and run offline afterwards.
2. **Safety.** The classifier is never trusted blindly. Every destructive action is dry-run by default, gated behind a calibrated confidence threshold, executed with recoverable "trash" semantics, logged to an append-only operation log, and reversible with a state-verified `undo`.
3. **Honesty about uncertainty.** Files the system cannot confidently classify are *held back* in a `needs_review` state rather than guessed at. A human resolves them, and each resolution feeds a preference-learning loop that improves future runs.

---

## 2. High-level architecture

The system is a pipeline of small, single-responsibility modules. Data flows left to right; **only one module in the entire system is allowed to mutate the filesystem** (the Executor), which makes "dry-run by default" a structural guarantee rather than a runtime flag.

```
            INGESTION                 FEATURE EXTRACTION          CLASSIFICATION                ORGANIZATION
┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ ingest.py    Scanner    │   │ features.py  Extractor   │   │ rules.py     RuleEngine │   │ planner.py   Planner    │
│  - walk dirs (stat-only)│──▶│  - capped text reads     │──▶│ classify.py  Classifier │──▶│  - dry-run Plan         │
│  - tier 1-4 routing     │   │  - image path handoff    │   │ calibrate.py Confidence │   │ executor.py  Executor   │
│  - exclusions, dotfiles │   │  - never crashes on      │   │ taxonomy.py  Categories │   │  - trash-default moves  │
│ watcher.py   Watcher    │   │    corrupt/unreadable    │   │ embedding.py Two models │   │ history.py   Undo log   │
│  - polling change feed  │   │    input                 │   │ preferences.py Feedback │   │  - state-verified undo  │
└─────────────────────────┘   └──────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
                                                                          ▲
                              ┌──────────────────────────┐                │
                              │ database.py  SQLite index│◀───────────────┘
                              │ config.py    categories  │     persistence + configuration
                              │ cli.py       entrypoint  │     (cli.py is the only composition root)
                              │ tui.py       dashboard   │
                              └──────────────────────────┘
```

All modules communicate only through typed data contracts defined in `src/organizer/types.py` (`FileRecord`, `FileFeatures`, `Classification`, `MoveOp`, `Plan`, `OpLogEntry`, …). No module imports the CLI; the CLI wires everything together.

---

## 3. How each stage works

### 3.1 Ingestion — `ingest.py`

`scan(root)` walks the directory tree using **stat-only** operations: it never opens a file's contents. For each regular file it records path, size, mtime, extension, and a MIME guess, then assigns a **processing tier**:

| Tier | Meaning | Examples | Treatment |
|---|---|---|---|
| 1 — METADATA | No content analysis useful | `.zip`, `.dmg`, `.exe`, fonts | Metadata only, never embedded |
| 2 — TEXT | Lightweight text extraction | `.txt`, `.md`, `.py`, `.json`, `.pdf` | First 4 KB → text embedding |
| 3 — VISION | Image understanding needed | `.jpg`, `.png`, `.heic`, `.webp` | Image → vision embedding |
| 4 — REVIEW | Ambiguous or risky | no extension, unknown type, ≥ 2 GiB | **Never auto-moved**, flagged `needs_review` |

Excluded automatically: `.git`, `node_modules`, `__pycache__`, `.venv`, all dotfiles and dot-directories, plus anything listed under `scan.excludes` in the config.

`watcher.py` provides an optional change feed: a polling thread (stdlib only, no native dependencies) snapshots `(size, mtime)` per file every second and enqueues `created` / `modified` / `deleted` events. The baseline snapshot emits nothing, so pre-existing files don't generate noise. The watcher never mutates anything and never classifies — it only observes.

### 3.2 Feature extraction — `features.py`

`extract(record)` produces a `FileFeatures` object according to the tier:

- **Tier 2 (text):** reads at most 4 KB (hard cap — this is the only content read in the pipeline), decodes UTF-8 with replacement, and tags the features `modality=TEXT`.
- **Tier 3 (image):** records the image path for the vision encoder; pixels are read only by the embedding model itself. Tagged `modality=IMAGE`.
- **Tiers 1 and 4:** `modality=NONE` — downstream stages will not embed these.

Extraction **never raises**. Corrupt, unreadable, permission-denied, and zero-byte files come back with an `error` field set, which routes them to `needs_review`.

### 3.3 The two-model embedding layer — `embedding.py`

This is the heart of the classifier, and the source of its most important invariant.

Two different models produce embeddings in **two incompatible vector spaces**:

- **Text:** `BAAI/bge-small-en-v1.5` (~130 MB) — embeds text snippets and category prompts.
- **Images:** `clip-ViT-B-32` (~600 MB) — embeds images and category prompts.

A cosine similarity between a bge vector and a CLIP vector is mathematically meaningless. The system therefore embeds **every category's prompts twice** — once per space — and a file is only ever scored against prompt vectors from its own modality's space. This invariant (gap G2 in the design docs) is enforced by a single lookup table (`SPACE_FOR_MODALITY`) and verified by instrumented tests.

The embedding service is a swappable interface with two backends:

- **`FakeEmbeddingService` (default):** deterministic, hash-based, dependency-free vectors. No downloads, fully offline, used by the entire test suite and by any run without `--real`. Not semantic — it exists so the pipeline's *mechanics* can run and be tested anywhere.
- **`SentenceTransformerEmbeddingService` (`--real`):** the actual bge + CLIP models. Heavy imports happen lazily, and loading requires an explicit `--consent` flag: model download is treated as a one-time *setup* step, distinct from normal operation, preserving the "no network calls during normal operation" guarantee.

### 3.4 Classification — `rules.py` → `classify.py`

Classification is a **hybrid, layered** decision:

**Layer 1 — Rules (zero compute, high confidence).** `RuleEngine` checks the category rules from `config/categories.yaml`:
- extension exact match → confidence 0.95 (e.g. `.pdf` → `documents/invoices`)
- path keyword substring match → confidence 0.85 (e.g. a file under `/invoices/`)

Every rule verdict carries a provenance string (`extension:pdf`, `path_keyword:invoice`) so every automated decision is auditable. If a rule fires, the embedding layer is skipped entirely.

**Layer 2 — Embeddings.** If no rule fires, the file's content embedding is compared (cosine similarity) against every category's prompt embeddings *in the matching space*. Each category's score is the **max** cosine across its prompts; the file's preference-adjusted argmax category wins.

**Layer 3 — Preference bias.** Each category's score is adjusted by a learned per-category bias in `[-0.10, +0.10]` accumulated from your past feedback (see §3.7). The bias shifts both *which* category wins and *how confident* the result is, while the reported cosine stays raw for auditability.

**Layer 4 — Calibrated confidence gate.** Raw cosine is not a probability, and bge cosines (~0.5–0.7 for good matches) live in a completely different range than CLIP cross-modal cosines (~0.2–0.3). `calibrate.py` fixes this with a per-space **temperature softmax** over all category scores: confidence = the probability mass on the winning category. This makes one threshold mean the same thing in both spaces. With gating enabled (`--gate`), any file whose confidence falls below the per-space threshold is routed to `needs_review` instead of being confidently wrong.

Measured effect on a 14-file public-sample evaluation with real models: ungated accuracy was 10/13; with gating, every auto-classified file was correct and the three ambiguous photos were held for review — **zero confidently-wrong decisions**.

> The current thresholds (bge: temp 0.015 / min-conf 0.40; CLIP: temp 0.015 / min-conf 0.60) were tuned on that small public-sample set against the 13-category default taxonomy. They are sensible defaults, not production calibration; retune if you change the taxonomy materially.

### 3.5 Persistence — `database.py`

A SQLite index (default `.organizer/index.db`) stores one row per file: path, size, mtime, optional content hash, extension, MIME, tier, category, confidence, and a lifecycle status (`pending → classified | needs_review → confirmed`). It provides **cache-first** behavior: if a file's `(path, size, mtime)` — or content hash, when content caching is enabled — is unchanged since its last classification, the expensive embedding step is skipped entirely on re-runs.

### 3.6 Organization — `planner.py`, `executor.py`, `history.py`

This stage is where the safety model lives:

1. **`Planner.plan()` is pure.** It turns classifications into a `Plan` — a list of proposed `MoveOp`s plus a `needs_review` list — and touches nothing on disk. Destination paths follow the category structure (`organized/documents/invoices/…`); name collisions get deterministic numeric suffixes.
2. **`Executor.apply()` is the only filesystem mutator in the system.** Its defaults are maximally safe: `dry_run=True` (does nothing), and every op must be explicitly `approved`. Four move semantics are supported:
   - `trash` (default): move the file; if something already occupies the destination, the incumbent goes to the system Trash first — nothing is ever silently overwritten.
   - `copy`: original stays in place.
   - `symlink`: original untouched; organized tree is links.
   - `hard`: plain move, marked irreversible in the log; requires explicit opt-in.
   Each op is individually wrapped: one failure skips that op and continues the batch. Before acting, the executor records a SHA-256 of the source for later verification.
3. **`History`** is an append-only JSONL operation log. `undo(n)` replays the last *n* reversible ops in reverse, but **verifies filesystem state before each inverse move**: the file must still be at its destination, the original location must be free, and the recorded hash must match. Any mismatch (e.g. you edited the file after the move) skips *that op only* and reports it, rather than corrupting state. In end-to-end testing, applying moves and undoing them restored every file byte-identically.

### 3.7 Preference learning — `preferences.py`

When you resolve a `needs_review` file with the `review` command, the system records a `FeedbackEvent` (override into the chosen category) in an append-only log. Accumulated events produce a per-category bias:

- accept of a category: +0.01
- reject: −0.01
- override *into* a category: +0.02; override *away* from one: −0.01
- total clamped to ±0.10

On every subsequent `classify`/`preview`/`apply` run the CLI snapshots this bias table once and feeds it into the scorer (§3.4 layer 3). The loop is closed: your corrections measurably shift future classifications toward your personal semantics.

### 3.8 Interfaces — `cli.py` and `tui.py`

`cli.py` is the composition root — the only file that wires modules together. Subcommands: `status`, `scan`, `classify`, `preview`, `apply`, `undo`, `review`, `tui`.

`tui.py` is a read-only Textual dashboard showing workspace health (file counts by status, average confidence, op-log and feedback counters) and a category-distribution bar chart. It never classifies and never mutates files.

---

## 4. Step-by-step: how to run it

### 4.1 Prerequisites

- **Python 3.11+** (developed and tested on 3.12).
- macOS, Linux, or Windows (developed on macOS; the watcher's polling design is fully cross-platform).
- ~1 GB of disk for the optional real models.

### 4.2 Setup

```bash
# 1. Clone
git clone git@github.com:sanskarpal/Cortex.git
cd Cortex

# 2. (Recommended) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Core dependencies (small)
pip install send2trash PyYAML

# 4. Optional — the real ML models (large download, one-time)
pip install sentence-transformers Pillow

# 5. Optional — the TUI dashboard
pip install textual

# 6. Verify everything works (offline, no models needed)
python -m pytest tests/ -q          # expect: 111 passed
```

### 4.3 Quick smoke test (no models, fully offline)

The default embedding backend is a deterministic fake — semantically meaningless but mechanically complete. Good for verifying the install:

```bash
python m1_classify.py ~/Downloads        # prints "path -> category (cosine=…)" per file
```

### 4.4 The real workflow

Replace `~/Downloads` with whatever folder you want to organize. **Nothing is moved until step 5, and even then only what you saw in the preview.**

```bash
# 1. Index the folder (stat-only, fast, touches nothing)
python src/organizer/cli.py scan ~/Downloads

# 2. Classify with the real models.
#    --consent authorizes the one-time model download (~750 MB total);
#    afterwards everything runs offline. --gate enables the confidence
#    gate so uncertain files are held for review instead of guessed.
python src/organizer/cli.py classify ~/Downloads --real --consent --gate

# 3. Check the state of the index
python src/organizer/cli.py status

# 4. PREVIEW the reorganization — a dry run. Prints every proposed move
#    and every held-back file. Nothing on disk changes.
python src/organizer/cli.py preview ~/Downloads --dest ~/Organized --real --consent --gate

# 5. Apply it. Default mode is "trash": recoverable, never clobbers.
python src/organizer/cli.py apply ~/Downloads --dest ~/Organized --real --consent --gate

# 6. Changed your mind? Reverse the last N moves (state-verified):
python src/organizer/cli.py undo -n 13
```

### 4.5 Resolving held-back files (and teaching the system)

Files the system wasn't confident about stay where they were and are listed as `REVIEW` in the preview. Label them yourself:

```bash
python src/organizer/cli.py review ~/Downloads/mystery.pdf documents/contracts --dest ~/Organized
```

This moves the file *and* records your decision in `.organizer/preferences.jsonl`. Future runs bias toward your corrections automatically.

### 4.6 The dashboard

```bash
python src/organizer/cli.py tui          # q = quit, r = refresh
```

Shows live workspace health and category distribution for the current index. Point it at a different index with `--db path/to/index.db`.

### 4.7 Customizing categories

Edit `config/categories.yaml`. Each category has:

```yaml
- cat_id: documents/invoices        # leaf id; also the destination subfolder
  match:                            # natural-language prompts for the embedding match
    - "an invoice with a total amount due"
    - "a billing statement from a vendor"
  min_confidence: 0.60              # per-category auto-move gate
  rules:                            # optional zero-compute fast paths
    extensions: [".pdf"]
    path_keywords: ["invoice", "billing"]
```

Add categories freely — the prompts are embedded once per space at startup and cached. If you change the taxonomy substantially, consider re-tuning the calibration constants in `src/organizer/calibrate.py` (see the note in §3.4).

### 4.8 Useful flags reference

| Flag | Commands | Effect |
|---|---|---|
| `--real` | classify/preview/apply | Use real bge + CLIP models instead of the offline fake |
| `--consent` | with `--real` | Authorize the one-time model download/load |
| `--gate` | classify/preview/apply | Enable the calibrated confidence gate (recommended) |
| `--cache` | classify | Skip files unchanged since last classification |
| `--mode {trash,copy,symlink,hard}` | apply/review | Move semantics (default `trash`) |
| `--db`, `--log`, `--prefs`, `--config` | most | Override the default state-file locations (`.organizer/…`, `config/categories.yaml`) |

---

## 5. Safety model — what protects your files

| Guarantee | Mechanism |
|---|---|
| Nothing moves without you seeing it first | Dry-run is the structural default; `preview` shows the exact plan; only `apply` executes |
| Only one code path can touch files | `Executor.apply` is the sole filesystem mutator; planner/classifier are pure |
| Uncertain ≠ guessed | Tier-4 files and sub-threshold confidence → `needs_review`, never auto-moved |
| Moves are recoverable | Default `trash` semantics; destination conflicts go to system Trash, never overwritten |
| Everything is reversible | Append-only op log with per-op source hash; `undo` verifies state before every inverse move and skips (not corrupts) anything that changed since |
| Sensitive areas excluded | `.git`, `node_modules`, dotfiles, and configured sensitive paths are never scanned |
| No privilege escalation | The tool only ever operates inside user-owned paths |
| Privacy | All inference local; model download is consent-gated setup; zero telemetry; auditable provenance on every decision |

---

## 6. Testing and verified behavior

- **111 automated tests**, all offline and hermetic (the fake embedding backend means no test ever needs a network or a model download). They map one-to-one to the acceptance contracts in `ARCHITECTURE-EXTENSION.md` §4 (TC-SAFE-*, TC-MODEL-*, TC-CACHE-1, TC-TIER-1, …).
- **Real-model validation** was performed against 14 downloaded public sample files (real photographs, desktop screenshots, open-source code, public-domain prose — no personal files): with gating, 10/10 auto-classified correctly, 0 wrong, 3 ambiguous photos correctly held for review; apply + undo restored all 14 files byte-identically.
- **Performance spot-check:** tier routing measured at ~0.0001 ms/file (budget: 1 ms). Real-model classification averaged ~71 ms/file on CPU. Large-scale crawl benchmarks (1M files) have not been run.

### 6.1 Large-scale accuracy evaluation (61-file public bench)

A second, larger evaluation (`testbench/eval.py`, regenerable — bench downloads are gitignored) ran the real models against **61 labeled public files**: real source code from cpython / requests / flask / express / jquery, six Project Gutenberg books, twelve Lorem Picsum photographs, eight Wikimedia desktop screenshots, five zip archives, eight synthetic invoices/receipts (labeled as synthetic — genuinely public invoice corpora don't exist), and four adversarial edge cases (no extension, zero-byte, corrupt PNG, unknown extension).

Three passes isolate the contribution of each layer:

| Pass | Overall correct | Auto-move precision | Wrong moves | Held for review |
|---|---|---|---|---|
| **A — Full system** (rules → embedding → gate) | **60/61 (98%)** | **100%** (56/56) | **0** | 1 |
| B1 — Embedding only, gated | 64% | 90% | 4 | 18 |
| B2 — Embedding only, raw argmax | 72% | 77% | 12 | 5 |

Key findings:

- **The safety metric held: zero wrong moves end-to-end.** All four edge cases were correctly held for review, and the evaluation verified zero filesystem mutation.
- **The confidence gate earns its keep:** on embeddings alone it cut wrong moves from 12 to 4; the full system eliminated the rest.
- **Most residual embedding errors are taxonomy ambiguity, not model failure** — e.g. a beach photograph scored as `photos/travel` when labeled `photos/personal`. Reasonable people would disagree on those labels too.
- **Caveat on the 98%:** bench filenames carry keywords (`invoice_1.txt`, `screenshot_3.png`), so the rule layer decided 50 of 61 files. The embedding-only passes (B1/B2) are the fairer measure of raw model quality.

The evaluation also **surfaced and fixed two real defects** (commit `e44b774`):

1. **Over-broad rule extensions** in `config/categories.yaml`: a `.txt` extension rule routed *every* text file to `documents/invoices` and `.png/.jpg` routed every image to `documents/receipts` (first-writer-wins), silently bypassing the embedding classifier. Rules now carry only format-unambiguous extensions (`.py`, `.js`, `.zip`, `.mp3`, `.dmg`, …); generic content extensions fall through to embeddings.
2. **Crash on corrupt images**: an undecodable PNG crashed the real CLIP backend mid-pass. The classifier now wraps embedding calls and routes undecodable content to `needs_review`.

### Known limitations

1. Calibration thresholds were tuned on a small sample set — treat recall as indicative, not guaranteed. Precision (not moving things wrongly) is the design priority and held at 100% in both evaluations (14-file and 61-file).
2. Image classification across 13 categories is conservative: lower-confidence photos land in `needs_review` rather than being guessed. The CLIP confidence threshold was tuned on a 29-photo labeled set (pets / landmarks / screenshots) to recall ~52% of photos at 100% precision (zero wrong auto-moves); the rest are held for review. This is the intended safety-over-recall tradeoff, but it still means manual review for a meaningful share of photo-heavy folders.
3. The watcher polls (1 s default) rather than using native FSEvents/inotify, and no daemon currently consumes its event queue — incremental re-classification is manual (`classify --cache`).
4. PDFs are classified by their pymupdf-extracted text layer; scanned PDFs without a text layer are held for review (OCR is not implemented).
5. Preference learning is a linear, clamped bias — deliberately simple; embedding-space adaptation is future work.
6. **Windows is not supported** — macOS and Linux only. The CLI exits with a clear error on Windows.

---

## 7. File map

```
Cortex/
├── ARCHITECTURE.md              # canonical design
├── ARCHITECTURE-EXTENSION.md    # engineering contract (gaps G1-G10, test contracts)
├── m1_classify.py               # M1 standalone prototype script
├── config/categories.yaml       # editable taxonomy + scan config
├── docs/REPORT.md               # this document
├── src/organizer/
│   ├── types.py                 # shared data contracts (all modules speak these)
│   ├── ingest.py                # Scanner + tier routing (stat-only)
│   ├── watcher.py               # polling change feed
│   ├── features.py              # capped, crash-proof feature extraction
│   ├── embedding.py             # two-space embedding service (fake + real backends)
│   ├── taxonomy.py              # built-in 6-category prompt set (M1)
│   ├── config.py                # categories.yaml loader + validation
│   ├── rules.py                 # extension/keyword rule engine
│   ├── classify.py              # space-isolated scoring + bias + gate
│   ├── calibrate.py             # per-space softmax confidence + thresholds
│   ├── preferences.py           # feedback log + clamped bias
│   ├── database.py              # SQLite index, cache-first skip
│   ├── planner.py               # pure dry-run plan builder
│   ├── executor.py              # the ONLY filesystem mutator
│   ├── history.py               # append-only op log + verified undo
│   ├── cli.py                   # composition root: 8 subcommands
│   └── tui.py                   # read-only Textual dashboard
└── tests/                       # 111 tests, fully offline
```
