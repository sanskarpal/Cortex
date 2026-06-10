# ARCHITECTURE.md
## AI File Organizer — System Architecture Plan

Version: 0.1 — Research & Design Document (No Implementation Yet)

---

## 1. Problem Summary

Users accumulate large numbers of files across local folders or whole devices.
Manual organization is time-consuming and inconsistent. The goal is to build
a **privacy-first, locally-running** system that:

- scans selected targets (folder / folders / whole device),
- classifies each file into semantic categories (e.g. invoice, receipt,
  source code, personal photo, design asset, document, archive, etc.),
- then reorganizes or proposes a reorganization into a cleaner structure
  aligned with the user's preferences.

**Key constraint:** Everything should be able to run on-device, without
uploading file contents to any external service (to avoid privacy risks).

---

## 2. High-Level Architecture

The system is split into four major layers:

1. **Ingestion Layer** — scans filesystem and yields candidate files.
2. **Feature Extraction Layer** — for each file, extract lightweight signals
   (extension, MIME, metadata, text content, embeddings).
3. **Classification Layer** — maps feature vectors / metadata to semantic
   category labels using a hybrid rule + ML pipeline.
4. **Organization Layer** — applies user-defined or system-suggested move/rename
   actions, with safety guarantees.

```

+------------------+     +------------------------+     +----------------------+     +----------------------+
| Ingestion Layer  | --> | Feature Extraction     | --> | Classification       | --> | Organization Layer   |
| - walk dirs      |     | - extension, MIME      |     | - rules + heuristics |     | - dry-run / confirm   |
| - watch events   |     | - metadata (EXIF etc)  |     | - embedding-based    |     | - move / rename / sym |
| - dedupe         |     | - OCR / text extraction|     | - confidence scoring |     |   links              |
| - filter out     |     | - image thumbnails     |     | - category taxonomy  |     | - rollback            |
|   system files   |     | - selected chunks for  |     | - user preference    |     | - history / undo      |
+------------------+     |   embedding when needed|     |   learning           |     +----------------------+
                         +------------------------+     +----------------------+
                                    |                             |
                                    v                             v
                         +-------------------+
                         | Local Model Layer |
                         | - CLIP / SigLIP   |
                         | - embedding model |
                         | - tiny classifier |
                         |   (optional)      |
                         +-------------------+
```

---

## 3. File-Type Coverage Strategy

Not all files need the same depth of analysis. We classify files into a
**processing tier** based on extension/MIME:

### Tier 1 — Metadata / heuristic only
- Archives (`.zip`, `.tar.gz`, `.7z`, `.rar`)
- Binaries / executables (`.app`, `.dmg`, `.exe`, `.dylib`)
- Disk images, fonts, system files

Classification for these is mostly based on extension and metadata (size,
creation date, path hints). **No ML inference is needed.**

### Tier 2 — Lightweight content extraction
- Plain text / Markdown / source code (`.py`, `.js`, `.ts`, `.md`, etc.)
- PDFs with embedded text
- `.csv`, `.json`, `.yaml`

We extract a small amount of content (first N KB / first N tokens) and run
it through a **text embedding model** for semantic similarity to category
prompts. This layer can also do syntax analysis for code files to detect
language.

### Tier 3 — Deep content extraction / vision
- Images (`.jpg`, `.png`, `.heic`, `.webp`, `.raw`)
- Scanned PDFs / documents without text layer (need OCR)
- Screenshots
- Video / audio (optional; can be deferred to v2)

We use a **vision-language model** (CLIP or similar) to compute an embedding
from the image and match it against textual category prompts. Optionally we
run OCR on text-heavy images to refine classification.

### Tier 4 — Ambiguous / risky
- Files without extension, unknown MIME, very large files
- Password-protected archives
- Executables / scripts from untrusted sources

These get a "needs review" label and are **never auto-moved**.

---

## 4. Classification Approach (Research-Based Selection)

### 4.1 Hybrid Rule Engine + ML Classifier

A pure ML classifier is not enough for a local, correctness-sensitive
tool. We build a **pipeline with multiple signals** and a **confidence
scoring mechanism**.

### 4.2 Signal Layers

1. **Rule layer (high confidence, low compute)**
   - File extension → primary category guess
   - MIME type
   - Path keywords (`/invoices/`, `/receipts/`, `tax_2024.pdf`)
   - Gitignored / system directories are excluded automatically

2. **Content signal layer (medium-high confidence, medium compute)**
   - For documents / text: extract text → embedding → cosine similarity
     against user-defined and built-in category prompt sets.
   - For images: CLIP / SigLIP embedding → cosine similarity.
   - For code: language detection + filename + class/function names.

3. **Embedding-based classifier (variable confidence, higher compute)**
   - We define a **fixed taxonomy of categories**, each represented by a set
     of textual prompts.
   - We embed each prompt once using a **local embedding model** (e.g.
     `BAAI/bge-small-en-v1.5`) and cache them.
   - We embed the extracted file content / image once and reuse that vector.
   - Classification = argmax of cosine similarities between file embedding and
     cached category prompt embeddings.

4. **User adaptation layer (long-term, optional)**
   - Track which files user moves / accepts / rejects.
   - Fine-tune category prompt embeddings or train a small adapter over
     cached embeddings to reflect personal semantics ("Invoices for client A
     vs B").

### 4.3 Recommended Model Components (Local)

| Purpose | Recommended (Initial) | Toy / Prototype alternative |
|---|---|---|
| Vision embedding | `BAAI/CurrencyCLIP` or open CLIP (ViT-B/32) | `clip-ViT-B-32` (Sentence Transformers) |
| Text embedding | `BAAI/bge-small-en-v1.5` | `all-MiniLM-L6-v2` |
| Image OCR | `tesseract` (easy local install) or `pytesseract` | N/A — tesseract is standard |
| Text extraction | `pymupdf`, `pdfplumber`, `python-pptx`, `docx2txt` | `pypandoc` |
| Image thumbnails | `Pillow`, `pyheif` (macOS HEIC) | `Pillow` only |
| Metadata | `Pillow.ExifTags`, `hachoir`, `mutagen` for audio/video | `os.stat` only |

Rationale:
- All models listed are **open-weight** or rule-based, runnable without GPU
  (CPU is slower but works), and do not require network.
- `bge-small-en-v1.5` is ~100MB and fast enough on modern laptops for
  thousands of files.
- CLIP is heavier (200–500MB depending on variant), so use it only for
  image-phase classification, not for text docs.
- Tiered processing + caching ensures we never re-embed unchanged files.

### 4.4 Why CLIP is a good fit here

- Zero-shot: does not require retraining to add new categories.
- Multimodal: same model can understand text AND images, giving consistent
  category semantics across file types.
- Open implementations exist via Hugging Face and Sentence Transformers,
  making it straightforward to embed.

Limitations to account for:
- Inference cost on CPU is nontrivial for large image libraries. Mitigation:
  only re-classify on change events (see Section 6).
- Accuracy drops on abstract categories or domains where training data was
  limited. Mitigation: prompt engineering (multiple prompts per category) +
  confidence thresholds.

---

## 5. Classification Taxonomy (Good Default)

A reasonable initial taxonomy (fully customizable by user):

```
documents/
  invoices/
  receipts/
  contracts/
  tax/
  personal/
  ids/
photos/
  screenshots/
  personal/
  travel/
  events/
  memes/
code/
  python/
  javascript/
  rust/
  configs/
  notebooks/
design/
  images/
  ui/
  figma-export/
media/
  music/
  videos/
  podcasts/
education/
  courses/
  notes/
  certificates/
downloads/
  archives/
  installers/
  uncategorized/
```

This taxonomy lives in a config file (`categories.yaml`) that users can
extend. Each category has:

* `match` — list of prompt strings used for embedding comparison
* `rules` — optional regex / extension / path rules
* `min_confidence` — threshold before an automated move is allowed
* `destination` — relative target path

---

## 6. Incremental & Event-Driven Processing

Scanning an entire device on every run is wasteful. The architecture uses:

- **Initial baseline scan** — one-time full inventory that writes a local
  SQLite database:
  - file path
  - size, mtime, ctime
  - content hash (optional, for strong dedup)
  - extension, MIME
  - extracted text snippet (optional)
  - embedding vector (when computed)
  - assigned category + confidence
  - processing status (pending / classified / confirmed / needs_review)

- **Change watcher** — FSEvents / inotify / polling depending on platform
  (macOS SFW at first), enqueueing changed files into a small work queue.
  This keeps the system reactive without rescanning everything.

- **Batch + throttle** — embedding inference is done in small async batches,
  respecting user CPU usage limits. A configurable concurrency / rate limit
  keeps the UI responsive.

- **Cache-first** — if size, mtime, and hash haven't changed since last
  classification, skip entirely.

---

## 7. Safety Model

Moving files based on a classifier is risky. The architecture enforces:

1. **Dry-run by default**
   The first run after any category change shows a preview of moves. The
   user explicitly approves.

2. **Confidence thresholding**
   Only files with `confidence >= min_confidence` for that category can be
   auto-moved. Others go to `needs_review` and remain untouched.

3. **Move semantics: soft by default**
   By default, files are MOVED. Optional:
   - `copy` — leave original in place, organize a copy elsewhere.
   - `symlink` — create organized symlinks without touching originals.
   - `trash` — originals are moved to system trash, recoverable.

4. **Atomic & reversible**
   All reorganization is recorded in an operation log. A single `undo`
   reverses the last N actions by replaying inverse moves.

5. **Exclusion rules**
   - User explicitly excludes certain directories (e.g. `.git`, node_modules,
     system folders).
   - Sensitive directories (e.g. `~/Library/Mail`) are excluded by default.

6. **Permissions**
   The tool should never request superuser / root; it only operates inside
   user-owned paths.

---

## 8. Performance Targets (Soft)

| Scenario | Target |
|---|---|
| File system crawl (1M files) | < 2 minutes on modern SSD |
| Tier 1 classification | < 1ms per file (pure Python / metadata) |
| Tier 2 (text embedding) | 20–200ms per file depending on length |
| Tier 3 (CLIP image) | 200–600ms per image on CPU, faster with Metal/GPU |
| Incremental update on file change | < 5s end-to-end for typical doc update |

These are aspirational; should guide hardware guidance and model choice.

---

## 9. Privacy Guarantees

- All processing is local. No file contents, embeddings, or metadata leave
  the machine unless the user explicitly opts into a cloud sync feature.
- The design should be auditable: open-source classifiers, transparent
  prompt sets, and an offline-first codebase.
- No network calls during normal operation. Any optional remote sync is
  always a separate opt-in plugin.
- For users who want extra privacy:
  - Category prompts are plaintext and editable in `categories.yaml`.
  - The model configs are community-sourced Hugging Face checkpoints; users
    can replace them with other open models.
  - No telemetry by default.

---

## 10. Extensibility Points

The architecture is built as a pipeline with well-defined interfaces so that
new classification strategies can be added without rewriting the core:

- **Ingester plugin** — add new filesystem sources (SMB mounts, cloud
  drives, archives).
- **Extractor plugin** — support for new file types (video frames, audio
  transcripts, CAD files).
- **Classifier plugin** — swap CLIP for your own vision model, or add a
  custom fine-tuned text classifier.
- **Organizer plugin** — support for other OS conventions (Linux `xdg`,
  Windows Libraries) or custom fan-out rules.
- **UI / interaction** — the core can be driven from CLI, TUI (Textual /
  Rich), or a minimal desktop UI. This document does not specify UI; the
  agent layer should be the headless service + CLI entrypoint.

---

## 11. Implementation Stack Recommendation

| Concern | Technology |
|---|---|
| Language | Python 3.11+ (GLib / ML / ecosystem strength) |
| Packaging | `pyproject.toml`, installable with pip |
| Embedding runtime | Hugging Face `transformers` + `torch` (CPU mode via `torch` or `mlx` on macOS) |
| Vision models | Hugging Face `transformers` CLIP pipelines, or `sentence-transformers` CLIP |
| OCR | `pytesseract` / `tesseract` binary |
| Text extraction | `pymupdf`, `pdfplumber`, `python-pptx`, `docx2txt` |
| Metadata | `Pillow`, `hachoir`, `mutagen` |
| Database | `SQLite` via `sqlite3` / `alembic` for schema migrations |
| Filesystem watching | `watchdog` (cross-platform), supplemented by native APIs on macOS / Linux |
| CLI | `typer` or `click` |
| TUI (future) | `textual` |

Rust / Go could be used for the core watcher or high-performance walker,
but the ML stack is Python-native, so keeping the orchestrator in Python
for the prototype is the path of least resistance.

---

## 12. Research Gaps & Next Questions

This plan reflects a research-based recommendation, not implementation:

1. **Model benchmarking needed:** We should benchmark `bge-small-en-v1.5`
   against `all-MiniLM-L6-v2` and maybe `gte-small` for latency vs.
   accuracy on **actual user documents** (invoices, code, personal notes).

2. **CLIP variant comparison:** Should we start with `clip-ViT-B-32` or
   a lighter architecture? On CPU-only machines, ViT-L/14 is painful;
   ViT-B/32 is usable but not fast. SigLIP or MobileCLIP could be a
   better first-class choice for desktops.

3. **Code classification approach:** There are mature code-specific
   models (`StarCoder`, `CodeBERT`) that could be cued for language
   detection and structural understanding. We need to decide whether to
   add this early or keep it at extension / `tree-sitter` parsing for v1.

4. **OCR strategy for scanned docs:** Tesseract works but accuracy is
   limited. We should evaluate whether to bundle a small doc model
   (e.g., `paddleocr` or `easyocr`) at the cost of disk footprint.

5. **User preference learning:** Is a simple preference log enough, or
   do we need a real fine-tune step? Likely start with rule weights and
   user corrections, graduate to embedding-space adaptation once we have
   enough labeled examples.

---

## 13. Milestone Plan (No Code)

| Milestone | Deliverable |
|---|---|
| M1 — Prototype classifier | Manual Python script: walk dir, embed text with BGE, embed images with CLIP, print top-1 category per file. |
| M2 — Core library | Python package: watch + scan + DB + classify for 3 file tiers. CLI with `status`, `classify`, `preview`, `apply`, `undo`. |
| M3 — Rule + embedding hybrid | Prompt/category overlay, confidence scoring, dry-run reorganization. |
| M4 — User preferences | `categories.yaml`, threshold tuning, basic feedback loop. |
| M5 — Desktop UI (optional) | Minimal TUI or tray app showing workspace health + category distribution. |

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Wrong classification leading to lost files | Dry-run default; soft-move options (symlink / copy); operation log with undo; exclude sensitive dirs. |
| Large model downloads / disk footprint | Make models optional / lazy-download on first run. Provide "minimal" profile without vision model. |
| CPU inference too slow for large libraries | Pre-classify only on new/changed files; skip unchanged cache; recommend minimum specs: 4 cores, 16GB RAM. |
| User categorizes by project context, not content | Allow manual override; expose custom category matchers; aim for a "human-in-the-loop" design from day one. |
| Complex file types (e.g. mixed video+audio, PDFs with nested attachments) | Leave these in `needs_review` rather than guessing. |
| Filesystem watching can be brittle | Implement blind polling fallback; treat watcher as a speed optimization, not a correctness guarantee. |

---

*End of ARCHITECTURE.md*
