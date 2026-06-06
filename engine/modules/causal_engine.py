import sys
import os
import numpy as np
import pandas as pd
from engine.base_module import BaseModule

_HERE = os.path.dirname(os.path.abspath(__file__))
_P4_CAUSAL = os.path.abspath(os.path.join(_HERE, "..", "..", "scratch", "phase4", "causal"))
if _P4_CAUSAL not in sys.path:
    sys.path.insert(0, _P4_CAUSAL)

try:
    from dag_builder import DAG
    from confounder_detector import find_all_confounders, should_adjust_for
    from simpsons_paradox import detect_simpsons_paradox
    from potential_outcomes import propensity_score, ipw_ate, overlap_check
    from diff_in_diff import did_estimate, parallel_trends_test
    from dowhy_integration import full_causal_analysis, DOWHY_AVAILABLE
    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)

class CausalEngine(BaseModule):
    """
    Engine Module — Causal Engine
    =============================
    Production wrapper around scratch/phase4/causal/ modules.
    """
    def _run(self, df, treatment, outcome, dag_str=None, common_causes=None, group_col=None, post_col=None):
        """
        Run causal analysis.
        
        Parameters:
            df: pd.DataFrame
            treatment: str - Treatment column name
            outcome: str - Outcome column name
            dag_str: Optional DOT string for DAG
            common_causes: Optional list of confounders
            group_col: Optional for Simpson's Paradox / DiD
            post_col: Optional for DiD
            
        Returns:
            dict - Structured results
        """
        if not _MODULES_LOADED:
            self._error(f"Failed to load Causal modules: {_IMPORT_ERROR}")
            return self._result({"error": _IMPORT_ERROR}, severity="CRITICAL")

        self._log(f"Starting Causal analysis for {treatment} -> {outcome}...")
        
        findings = {}
        
        # 1. Simpson's Paradox Check
        if group_col and group_col in df.columns:
            self._log(f"Checking Simpson's Paradox relative to {group_col}...")
            sp_res = detect_simpsons_paradox(df[treatment].values, df[outcome].values, df[group_col].values)
            findings["simpsons_paradox"] = sp_res
            if sp_res["detected"]:
                self._warn(f"Simpson's Paradox detected for group {group_col}!")
        
        # 2. DiD Check
        if post_col and post_col in df.columns:
            self._log("Running Difference-in-Differences...")
            y_pre_t = df[(df[treatment] == 1) & (df[post_col] == 0)][outcome]
            y_post_t = df[(df[treatment] == 1) & (df[post_col] == 1)][outcome]
            y_pre_c = df[(df[treatment] == 0) & (df[post_col] == 0)][outcome]
            y_post_c = df[(df[treatment] == 0) & (df[post_col] == 1)][outcome]
            
            if len(y_pre_t) > 0 and len(y_post_t) > 0:
                did_val = did_estimate(y_pre_t, y_post_t, y_pre_c, y_post_c)
                findings["did_estimate"] = float(did_val)
        
        # 3. Effect Estimation (IPW)
        if common_causes:
            self._log(f"Estimating ATE using IPW (adjusting for {common_causes})...")
            X = df[common_causes]
            T = df[treatment]
            Y = df[outcome]
            
            ps = propensity_score(X, T)
            ate = ipw_ate(Y, T, ps)
            overlap = overlap_check(ps)
            
            findings["ipw_ate"] = float(ate)
            findings["propensity_overlap"] = overlap
            if not overlap["is_valid"]:
                self._warn("Poor propensity overlap detected. ATE estimate may be unreliable.")

        # 4. DoWhy Integration
        if DOWHY_AVAILABLE:
            self._log("Running DoWhy analysis...")
            try:
                dowhy_res = full_causal_analysis(df, treatment, outcome, dag_str, method="backdoor.linear_regression")
                findings["dowhy"] = dowhy_res
            except Exception as e:
                self._error(f"DoWhy analysis failed: {str(e)}")

        severity = "NONE"
        if findings.get("simpsons_paradox", {}).get("detected"):
            severity = "HIGH"
        elif not findings.get("propensity_overlap", {}).get("is_valid", True):
            severity = "MEDIUM"

        return self._result(findings, severity=severity)

def run_verification():
    """Run module verification."""
    engine = CausalEngine()
    print("CausalEngine loaded and ready.")

if __name__ == "__main__":
    run_verification()
