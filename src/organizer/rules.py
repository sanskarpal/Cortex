"""Classification layer — RuleEngine (M3).

High-confidence, zero-compute early exits before the embedding classifier.
Implements the rule layer described in ARCHITECTURE.md §4.2(1) and the module
contract in ARCHITECTURE-EXTENSION.md §2 (RuleEngine row).

Provenance strings in RuleVerdict.reason satisfy TC-PRIV-2 (§9 auditability):
every automated decision carries a readable trace of which signal fired.
"""

from __future__ import annotations

from dataclasses import dataclass

from organizer.types import FileFeatures, FileRecord


@dataclass
class RuleVerdict:
    """Result of a successful rule-layer match.

    cat_id:     Leaf category id (e.g. "documents/invoices").
    confidence: Fixed per-signal constant (§4.2): 0.95 for an extension match,
                0.85 for a path-keyword match. Deliberately uncalibrated here —
                the Classifier layer may apply G4 calibration on top.
    reason:     Machine-readable provenance tag, e.g. "extension:pdf" or
                "path_keyword:invoice". Required for TC-PRIV-2 / §9 auditability.
    """

    cat_id: str
    confidence: float
    reason: str


class RuleEngine:
    """Apply extension/path-keyword rules; emit a high-confidence verdict or None.

    Pure, no I/O, no ML — suitable for the Tier-1 throughput target (TC-PERF-1).

    Args:
        categories: Raw validated category dicts from AppConfig.categories. Each
                    dict may carry an optional 'rules' sub-dict with:
                        extensions    (list[str]) — e.g. [".pdf", ".docx"]
                        path_keywords (list[str]) — e.g. ["invoice", "billing"]
                    Categories without a usable 'rules' dict are ignored.

    Lookup tables are precomputed once in __init__; first writer wins on any
    collision, so category order in the config is the tie-break.
    """

    _EXT_CONFIDENCE = 0.95
    _KW_CONFIDENCE = 0.85

    def __init__(self, categories: list[dict]) -> None:
        # ext_map: lowercased extension (no leading dot) -> cat_id
        self._ext_map: dict[str, str] = {}
        # kw_map: lowercased path keyword -> cat_id
        self._kw_map: dict[str, str] = {}

        for cat in categories:
            cat_id = cat.get("cat_id")
            rules = cat.get("rules")
            # Ignore ruleless / malformed categories cleanly (§4.2: rules optional).
            if not cat_id or not isinstance(rules, dict):
                continue

            # Extensions: strip leading dot, lowercase; first writer wins.
            for raw_ext in rules.get("extensions") or []:
                ext = str(raw_ext).lstrip(".").lower()
                if ext and ext not in self._ext_map:
                    self._ext_map[ext] = cat_id

            # Path keywords: lowercase; first writer wins.
            for raw_kw in rules.get("path_keywords") or []:
                kw = str(raw_kw).lower()
                if kw and kw not in self._kw_map:
                    self._kw_map[kw] = cat_id

    def apply(
        self,
        rec: FileRecord,
        features: FileFeatures | None = None,  # noqa: ARG002 — reserved for future signals
    ) -> RuleVerdict | None:
        """Return a RuleVerdict on the first matching rule, else None.

        Lookup precedence (extension beats keyword, per §4.2 layer 1):
          1. Extension exact-match  -> confidence 0.95, reason "extension:<ext>"
          2. Path keyword substring -> confidence 0.85, reason "path_keyword:<kw>"

        Keyword matching is a case-insensitive substring test against the full
        path string, so both directory components (e.g. "/invoices/") and the
        filename (e.g. "tax_2024.pdf") are searched. None means "no rule fired" —
        the caller falls through to the embedding classifier.
        """
        ext = rec.extension.lower()  # FileRecord contract: already no leading dot
        if ext and ext in self._ext_map:
            return RuleVerdict(
                cat_id=self._ext_map[ext],
                confidence=self._EXT_CONFIDENCE,
                reason=f"extension:{ext}",
            )

        path_lower = str(rec.path).lower()
        for kw, cat_id in self._kw_map.items():
            if kw in path_lower:
                return RuleVerdict(
                    cat_id=cat_id,
                    confidence=self._KW_CONFIDENCE,
                    reason=f"path_keyword:{kw}",
                )

        return None
