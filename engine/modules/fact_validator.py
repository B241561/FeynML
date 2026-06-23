"""
Fact Validator
--------------
Validate claim records produced by EvidenceRegistry or other engines.

Responsibilities:
- Detect unsupported claim categories
- Detect missing evidence
- Detect missing source_module
- Detect duplicate claims

Return shape:
{
  "valid": bool,
  "issues": [str, ...]
}

This validator is lightweight and non-throwing; it collects issues and
returns them for downstream handling.
"""
from typing import List, Dict, Any

SUPPORTED_CATEGORIES = {
    'feature_drift', 'target_leakage', 'calibration', 'missing_values',
    'outliers', 'importance_drift', 'slice_degradation', 'label_noise', 'root_cause'
}


def validate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a list of claim records.

    Args:
        records: List of dicts with keys ['claim','category','evidence','source_module','confidence']

    Returns:
        dict with keys 'valid' (bool) and 'issues' (list of strings)
    """
    issues: List[str] = []
    if not isinstance(records, list):
        return {'valid': False, 'issues': ['Records must be a list']}

    # duplicate detection
    seen = {}
    for r in records:
        claim = (r.get('claim') or '').strip()
        if claim:
            seen[claim] = seen.get(claim, 0) + 1

    for claim, count in seen.items():
        if count > 1:
            issues.append(f"Duplicate claim detected: {claim}")

    for idx, r in enumerate(records):
        claim = r.get('claim') if isinstance(r, dict) else None
        category = r.get('category') if isinstance(r, dict) else None
        evidence = r.get('evidence') if isinstance(r, dict) else None
        source = r.get('source_module') if isinstance(r, dict) else None

        if not claim:
            issues.append(f"Record #{idx} missing claim text")
            continue

        if not category:
            issues.append(f"Claim missing category: {claim}")
        else:
            if category not in SUPPORTED_CATEGORIES:
                issues.append(f"Unsupported claim category: {category} for claim: {claim}")

        if not evidence or (isinstance(evidence, list) and len(evidence) == 0):
            issues.append(f"Claim missing evidence: {claim}")

        if not source:
            issues.append(f"Claim missing source_module: {claim}")

    valid = len(issues) == 0
    return {'valid': valid, 'issues': issues}
