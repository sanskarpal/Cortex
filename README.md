# Cortex — privacy-first AI file organizer

Cortex scans a folder, understands what each file *is* — an invoice, source code, a screenshot, a personal photo, a book — and reorganizes it into a clean category structure. Everything runs **locally**: two small open-weight models (bge text embeddings + CLIP vision), no cloud, no telemetry, nothing leaves your machine.

It is built safety-first:

- **Dry-run by default** — you always see the full plan before anything moves.
- **Confidence-gated** — files the AI isn't sure about are held for your review, never guessed at. In evaluation: **zero wrong auto-moves** across 75 labeled test files.
- **Recoverable** — moves use trash-backed semantics, never overwrite, and every action lands in an append-only log with a state-verified `undo`.
- **Learns from you** — manually labeling a held-back file feeds a preference bias into future runs.

> **Status: alpha.** Solid for technical users; see [docs/REPORT.md](docs/REPORT.md) §6 for honest evaluation results and known limitations (PDF text extraction is the big one).

## Install

```bash
git clone https://github.com/sanskarpal/Cortex.git
cd Cortex
pip install -e ".[ml,tui]"     # core + real AI models + dashboard
# or just `pip install -e .` for the lightweight rules-only core
```

Requires Python 3.11+. The ML extra pulls torch/sentence-transformers (~2 GB installed); model weights (~750 MB) download once, on first use, only with your explicit `--consent`.

## Quickstart

```bash
# 1. See what would happen (dry run — nothing moves)
organizer preview ~/Downloads --dest ~/Organized --real --consent --gate

# 2. Apply it (trash-safe, logged, reversible)
organizer apply ~/Downloads --dest ~/Organized --real --consent --gate

# 3. Changed your mind?
organizer undo -n 10

# 4. Label the files it wasn't sure about (teaches it your preferences)
organizer review ~/Downloads/whatever.pdf documents/contracts --dest ~/Organized

# 5. Dashboard
organizer tui
```

Without `--real --consent` the pipeline runs with a deterministic offline stub instead of the AI models — useful for testing the mechanics, not for actual organizing.

## Customize categories

Edit `config/categories.yaml` (or copy it and pass `--config`). Each category is a destination folder plus natural-language prompts the embeddings match against, optional extension/keyword fast-path rules, and a per-category confidence threshold.

## How it works

Four-stage local pipeline: stat-only scan → tiered feature extraction → hybrid classification (rules → dual-space embeddings → calibrated confidence gate → preference bias) → dry-run plan and trash-safe execution. Full architecture deep-dive and step-by-step guide: **[docs/REPORT.md](docs/REPORT.md)**; design contracts: `ARCHITECTURE.md` + `ARCHITECTURE-EXTENSION.md`.

## Development

```bash
pip install -e ".[all]"
python -m pytest tests/ -q     # 111 tests, fully offline, no model downloads
```

## License

MIT — see [LICENSE](LICENSE).
