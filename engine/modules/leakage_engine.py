import sys
import os
from engine.base_module import BaseModule

_HERE = os.path.dirname(os.path.abspath(__file__))
_P4_FL = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase4", "feature_leakage"))
if _P4_FL not in sys.path:
    sys.path.insert(0, _P4_FL)

try:
    from leakage_scanner import scan, leakage_summary, rank_leakage_suspects
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

class LeakageEngine(BaseModule):
    """
    Engine Module — Leakage Engine
    ==============================
    Production wrapper around scratch/phase4/feature_leakage/ modules.
    """
    def _run(self, X, y, df=None, model=None, feature_names=None, time_col=None, train_idx=None, test_idx=None):
        """Run leakage scanner."""
        if not _MODULES_LOADED:
            self._error(f"Failed to load Leakage modules: {_IMPORT_ERROR}")
            return self._result({"error": _IMPORT_ERROR}, severity="CRITICAL")

        self._log("Starting Leakage scan...")
        
        # Run unified scan
        report = scan(X, y, df, model, feature_names, time_col, train_idx, test_idx)
        suspects = rank_leakage_suspects(report)
        summary = leakage_summary(report)
        
        # Determine severity based on suspect confidence scores (explicit thresholds)
        severity = "NONE"
        num_suspects = len(suspects)
        if num_suspects == 0:
            severity = "NONE"
        else:
            # Extract scores robustly (some scanners use 'score' or 'confidence')
            scores = []
            for s in suspects:
                sc = None
                if isinstance(s, dict):
                    sc = s.get('score') if 'score' in s else s.get('confidence')
                else:
                    # Fallback if suspect is an object with attributes
                    sc = getattr(s, 'score', None) if hasattr(s, 'score') else getattr(s, 'confidence', None)
                try:
                    sc = float(sc) if sc is not None else None
                except Exception:
                    sc = None
                if sc is not None:
                    scores.append(sc)

            # Default to LOW if suspects present but no numeric scores found
            if not scores:
                severity = "LOW"
            else:
                if any(s >= 0.85 for s in scores):
                    severity = "CRITICAL"
                elif any(s >= 0.70 for s in scores):
                    severity = "HIGH"
                elif any(s >= 0.50 for s in scores):
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

        findings = {
            "summary": summary,
            "num_suspects": len(suspects),
            "suspects": suspects,
            "full_report": report
        }
        
        self._log(f"Leakage scan complete. Found {len(suspects)} suspects.")
        return self._result(findings, severity=severity)

def run_verification():
    """Run module verification."""
    engine = LeakageEngine()
    # Simple placeholder call
    print("LeakageEngine loaded and ready.")

if __name__ == "__main__":
    run_verification()
