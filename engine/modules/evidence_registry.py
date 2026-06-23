"""
Evidence Registry
-----------------
Lightweight in-memory registry for claims and supporting evidence.

Responsibilities:
- Register claims with category, evidence, source_module and confidence
- Store structured claim records
- Provide helpers for duplicate detection and retrieval

This is intentionally simple and synchronous; integration with persistence
or external stores will be added in later sprints.
"""
from typing import List, Dict, Any

SUPPORTED_CATEGORIES = {
    'feature_drift', 'target_leakage', 'calibration', 'missing_values',
    'outliers', 'importance_drift', 'slice_degradation', 'label_noise', 'root_cause'
}


class EvidenceRegistry:
    def __init__(self):
        # store list of claim records
        self._records: List[Dict[str, Any]] = []

    def register(self, claim: str, category: str, evidence: List[str], source_module: str, confidence: int = None) -> Dict[str, Any]:
        """Register a claim and return the structured record.

        Args:
            claim: Human-readable claim text
            category: Claim category (e.g., 'feature_drift')
            evidence: List of evidence strings
            source_module: Originating analysis engine name
            confidence: Optional numeric confidence (0-100)
        """
        rec = {
            'claim': claim,
            'category': category,
            'evidence': evidence or [],
            'source_module': source_module,
            'confidence': int(confidence) if confidence is not None else None,
        }
        self._records.append(rec)
        return rec

    def get_claims(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of registered claim records."""
        return list(self._records)

    def find_duplicates(self) -> List[str]:
        """Return list of claim texts that appear more than once."""
        seen = {}
        dupes = []
        for r in self._records:
            text = (r.get('claim') or '').strip()
            seen[text] = seen.get(text, 0) + 1
        for text, count in seen.items():
            if count > 1:
                dupes.append(text)
        return dupes

    def clear(self):
        self._records.clear()

    def __len__(self):
        return len(self._records)
