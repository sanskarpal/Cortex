"""Config layer for M2: load/validate categories.yaml, surface weak-guarantee
warnings, and build CategoryPrompt objects in both embedding spaces.

Implements ConfigLoader (ARCHITECTURE-EXTENSION.md §2, Config row) and owns
AppConfig. Uses yaml.safe_load; enforces leaf-only cat_ids (G5); defaults
min_confidence=0.5 (§7.2); defaults destination=cat_id (G5); emits G6/G8
warnings into AppConfig.warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # PyYAML; installed per task contract

from organizer.embedding import EmbeddingService
from organizer.types import CategoryPrompt, EmbeddingSpace


@dataclass
class AppConfig:
    """Validated configuration produced by ConfigLoader.load().

    categories: dicts with keys cat_id/match/min_confidence/destination/rules.
    excludes / sensitive_excludes: scan exclusions (§7.5, TC-SAFE-6).
    move_mode: default semantics string (G1: "trash").
    content_cache: whether to use content-hash cache key (G6).
    warnings: non-fatal advisory strings (G6, G8) for callers to log.
    """

    categories: list[dict]
    excludes: list[str]
    sensitive_excludes: list[str]
    move_mode: str
    content_cache: bool
    warnings: list[str] = field(default_factory=list)


class ConfigLoader:
    """Load, validate, and transform categories.yaml into AppConfig."""

    _G6_WARNING = (  # G6: weaker (path,size,mtime) cache key when content_cache=false
        "G6: content_cache is disabled. Cache key falls back to "
        "(path, size, mtime), which is weaker than a content hash — "
        "files renamed or touched without content change may be re-classified."
    )
    _G8_WARNING = (  # G8: model download is setup, not normal operation (§9 vs §14)
        "G8: Local embedding models (BGE, CLIP) are not pre-loaded. "
        "First use triggers a one-time model download requiring explicit user "
        "consent. Offline use is guaranteed after models are present (TC-PRIV-1)."
    )

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> AppConfig:
        """Parse and validate categories.yaml; return AppConfig with warnings.

        Raises FileNotFoundError if the file is missing.
        Raises ValueError on any structural or semantic validation failure.
        """
        raw = self._parse_yaml()

        categories = self._validate_categories(raw.get("categories") or [])
        scan = raw.get("scan") or {}
        defaults = raw.get("defaults") or {}

        cfg = AppConfig(
            categories=categories,
            excludes=list(scan.get("excludes") or []),
            sensitive_excludes=list(scan.get("sensitive_excludes") or []),
            move_mode=str(defaults.get("move_mode") or "trash"),
            content_cache=bool(defaults.get("content_cache", False)),
        )
        self._attach_warnings(cfg)
        return cfg

    def build_category_prompts(
        self,
        cfg: AppConfig,
        embedder: EmbeddingService,
        spaces: tuple[EmbeddingSpace, ...] = (EmbeddingSpace.BGE, EmbeddingSpace.CLIP),
    ) -> list[CategoryPrompt]:
        """Build CategoryPrompts with prompt_vecs cached in all requested spaces.

        Mirrors taxonomy.build_taxonomy but driven by cfg.categories (yaml).
        Each prompt is embedded once per EmbeddingSpace via embed_prompt (G2).
        Hold the returned list; do not rebuild per file (§4.2).
        """
        taxonomy: list[CategoryPrompt] = []
        for entry in cfg.categories:
            cat = CategoryPrompt(cat_id=entry["cat_id"], match=entry["match"])
            # G2: populate both spaces so no cross-space cosine is ever computed.
            for space in spaces:
                cat.prompt_vecs[space] = [
                    embedder.embed_prompt(p, space) for p in entry["match"]
                ]
            taxonomy.append(cat)
        return taxonomy

    def _parse_yaml(self) -> dict[str, Any]:
        """Read yaml with safe_load (never yaml.load)."""
        if not self._path.exists():
            raise FileNotFoundError(f"categories.yaml not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                f"categories.yaml must be a top-level mapping, got {type(raw).__name__}"
            )
        return raw

    def _validate_categories(self, raw_cats: Any) -> list[dict]:
        """Validate categories list; apply field defaults; reject duplicates (§5/G5)."""
        if not isinstance(raw_cats, list):
            raise ValueError("'categories' must be a YAML sequence (list).")

        seen_ids: set[str] = set()
        validated: list[dict] = []

        for i, entry in enumerate(raw_cats):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"categories[{i}]: expected a YAML mapping, got {type(entry).__name__}"
                )

            # -- cat_id (G5: leaf path must contain "/") ----------------------
            cat_id = entry.get("cat_id")
            if not cat_id or not isinstance(cat_id, str):
                raise ValueError(f"categories[{i}]: 'cat_id' is required (non-empty str).")
            if "/" not in cat_id:
                raise ValueError(
                    f"categories[{i}]: cat_id '{cat_id}' has no '/' — "
                    f"all categories must be leaf ids (G5), e.g. 'documents/invoices'."
                )
            if cat_id in seen_ids:
                raise ValueError(f"categories[{i}]: duplicate cat_id '{cat_id}'.")
            seen_ids.add(cat_id)

            # -- match --------------------------------------------------------
            match = entry.get("match")
            if not match or not isinstance(match, list):
                raise ValueError(
                    f"categories[{i}] ('{cat_id}'): 'match' must be a non-empty list."
                )
            for j, m in enumerate(match):
                if not isinstance(m, str) or not m.strip():
                    raise ValueError(
                        f"categories[{i}] ('{cat_id}'): match[{j}] must be a non-empty str."
                    )

            # -- min_confidence (§7.2) ----------------------------------------
            raw_conf = entry.get("min_confidence")
            if raw_conf is None:
                min_confidence = 0.5
            else:
                try:
                    min_confidence = float(raw_conf)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"categories[{i}] ('{cat_id}'): 'min_confidence' must be a float, "
                        f"got {raw_conf!r}."
                    )
                if not (0.0 <= min_confidence <= 1.0):
                    raise ValueError(
                        f"categories[{i}] ('{cat_id}'): 'min_confidence' {min_confidence} "
                        f"is outside [0, 1]."
                    )

            # -- destination (G5: default = cat_id) ---------------------------
            destination = entry.get("destination") or cat_id

            # -- rules (optional dict) ----------------------------------------
            rules = entry.get("rules") or None
            if rules is not None and not isinstance(rules, dict):
                raise ValueError(
                    f"categories[{i}] ('{cat_id}'): 'rules' must be a YAML mapping."
                )

            validated.append({
                "cat_id": cat_id,
                "match": [m.strip() for m in match],
                "min_confidence": min_confidence,
                "destination": destination,
                "rules": rules,
            })

        return validated

    def _attach_warnings(self, cfg: AppConfig) -> None:
        """Append G6 and G8 weak-guarantee warnings to cfg.warnings."""
        if not cfg.content_cache:          # G6: weaker cache key
            cfg.warnings.append(self._G6_WARNING)
        cfg.warnings.append(self._G8_WARNING)  # G8: model download is setup
