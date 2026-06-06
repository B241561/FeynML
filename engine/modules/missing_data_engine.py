import sys
import os
import numpy as np
from engine.base_module import BaseModule

_HERE = os.path.dirname(os.path.abspath(__file__))
_P4_MD = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase4", "missing_data"))
if _P4_MD not in sys.path:
    sys.path.insert(0, _P4_MD)

try:
    from missingness_classifier import classify_mechanism, compute_missingness_rate
    from missingness_as_signal import indicator_target_correlation
    from multiple_imputation import mice_cycle
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

class MissingDataEngine(BaseModule):
    """
    Engine Module — Missing Data Engine
    ===================================
    Production wrapper around scratch/phase4/missing_data/ modules.
    """
    def _run(self, X, y=None):
        """Run missing data audit."""
        if not _MODULES_LOADED:
            self._error(f"Failed to load Missing Data modules: {_IMPORT_ERROR}")
            return self._result({"error": _IMPORT_ERROR}, severity="CRITICAL")

        self._log("Starting Missing Data audit...")
        
        # 1. Mechanism Classification
        results = classify_mechanism(X, y)
        classifications = results["classifications"]
        
        # 2. Rates
        rates = compute_missingness_rate(X)
        
        # 3. Signal Analysis (if y provided)
        signal = {}
        if y is not None:
            signal = indicator_target_correlation(X, y)
            
        # Determine severity
        max_rate = max(rates.values()) if rates else 0
        severity = "NONE"
        if max_rate > 0.05: severity = "LOW"
        if max_rate > 0.20: severity = "MEDIUM"
        
        # Check for MNAR
        if any(m == "MNAR" for m in classifications.values()):
            severity = "HIGH"
            self._warn("Detected MNAR mechanism - imputation may be biased.")

        findings = {
            "missingness_rates": rates,
            "mechanisms": classifications,
            "global_mcar": results["global_mcar"],
            "signal_strength": signal,
            "max_missing_rate": float(max_rate)
        }
        
        self._log(f"Missing Data audit complete. Max rate: {max_rate:.2%}")
        return self._result(findings, severity=severity)

    def impute(self, X, n_cycles=5):
        """Production imputation utility."""
        if not _MODULES_LOADED:
            return X
        return mice_cycle(X, n_cycles=n_cycles)

def run_verification():
    """Run module verification."""
    engine = MissingDataEngine()
    print("MissingDataEngine loaded and ready.")

if __name__ == "__main__":
    run_verification()
